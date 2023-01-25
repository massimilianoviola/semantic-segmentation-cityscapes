import albumentations as A
import cv2
from glob import glob
import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_cityscapes import *
from epoch import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# ======== CONFIGURATION ======== #

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# p_* stands for "path"
# s_* stands for "string" (i.e. all other strings than paths)
# n_* stands for n-dim size (0-dim: number of objects, 1-dim+: shape)
S_EXPERIMENT = "DeepLabV3+_EfficientNetB4_CE"
P_DIR_DATA = "/Workspace/Datasets/Cityscapes"
P_DIR_CKPT = os.path.join("/Workspace", S_EXPERIMENT, "Checkpoints")
P_DIR_LOGS = os.path.join("/Workspace", S_EXPERIMENT, "Logs")
P_DIR_EXPORT = os.path.join("/Workspace", S_EXPERIMENT, "Export")
S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S_NAME_ENCODER = "efficientnet-b4"
S_NAME_WEIGHTS = "imagenet"
N_EPOCH_MAX = 60
N_SIZE_BATCH_TRAINING = 8  # training batch size
N_SIZE_BATCH_VALIDATION = 4  # validation batch size
N_SIZE_BATCH_TEST = 1  # test batch size
N_SIZE_PATCH = 512  # patch size for random crop
N_STEP_LOG = 5  # evaluate on validation set and save model every N iterations
N_WORKERS = 16  # to be adapted for each system
# other notations:
# l_* stands for "list"
# i_* stands for an index, e.g.: for i_object, object in enumerate(l_object):
# d_* stands for "dict"
# k_* stands for "key" (of a dictionary item)


# ======== SETUP ======== #

# setup model
model = smp.DeepLabV3Plus(
    encoder_name = S_NAME_ENCODER,
    encoder_weights = S_NAME_WEIGHTS,
    in_channels = 3,
    classes = 20,
)
# U-Net model
#model = smp.Unet(
#    encoder_name = S_NAME_ENCODER,
#    encoder_weights = S_NAME_WEIGHTS,
#    in_channels = 3,
#    classes = 20,
#)
# To enable multi-GPU training. Set device_ids accordingly.
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
# setup input normalization
preprocess_input = get_preprocessing_fn(
    encoder_name = S_NAME_ENCODER,
    pretrained = S_NAME_WEIGHTS,
)
# setup input augmentations (for cropped and full images)
transform_crop = A.Compose([
    A.RandomCrop(N_SIZE_PATCH, N_SIZE_PATCH),
#    A.HorizontalFlip(p=0.25),
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])
transform_full = A.Compose([
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])
# setup datasets
dataset_training = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "train",
    mode = "fine",
    transform = transform_crop,
    device = S_DEVICE,
)
dataset_validation = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "val",
    mode = "fine",
    transform = transform_full,
    device = S_DEVICE,
)
dataset_test = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "test",
    mode = "fine",
    transform = transform_full,
    device = S_DEVICE,
)
# setup data loaders
loader_training = DataLoader(
    dataset_training,
    batch_size = N_SIZE_BATCH_TRAINING,
    shuffle = True,
    num_workers = N_WORKERS,
)
loader_validation = DataLoader(
    dataset_validation,
    batch_size = N_SIZE_BATCH_VALIDATION,
    shuffle = False,
    num_workers = N_WORKERS,
)
loader_test = DataLoader(
    dataset_test,
    batch_size = N_SIZE_BATCH_TEST,
    shuffle = False,
    num_workers = N_WORKERS,
)
# setup loss
loss = torch.nn.CrossEntropyLoss()
loss.__name__ = "ce_loss"
# To use SMP losses
#loss = smp.losses.DiceLoss(mode="multiclass")
#loss.__name__ = "dice_loss"
# setup optimizer
optimizer = torch.optim.Adam([
    dict(params = model.parameters(), lr = 5e-4),
])
# setup learning rate scheduler
# (here cosine decay that reaches 1e-6 after N_EPOCH_MAX)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer = optimizer,
    T_max = N_EPOCH_MAX,
    eta_min = 1e-6,
)
# setup Tensorboard logs writer
os.makedirs(os.path.join(P_DIR_LOGS, "Training"), exist_ok = True)
os.makedirs(os.path.join(P_DIR_LOGS, "Validation"), exist_ok = True)
writer_training = SummaryWriter(
    log_dir = os.path.join(P_DIR_LOGS, "Training")
)
writer_validation = SummaryWriter(
    log_dir = os.path.join(P_DIR_LOGS, "Validation")
)


# ======== TRAINING ======== #

# initialize training instance
epoch_training = Epoch(
    model,
    s_phase = "training",
    loss = loss,
    optimizer = optimizer,
    device = S_DEVICE,
    verbose = True,
    writer = writer_training,
)
# initialize validation instance
epoch_validation = Epoch(
    model,
    s_phase = "validation",
    loss = loss,
    device = S_DEVICE,
    verbose = True,
    writer = writer_validation,
)
# start training phase
os.makedirs(P_DIR_CKPT, exist_ok = True)
max_score = 0
# iterate over epochs
for i in range(1, N_EPOCH_MAX + 1):
    print(f"Epoch: {i} | LR = {round(scheduler.get_last_lr()[0], 8)}")
    d_log_training = epoch_training.run(loader_training, i_epoch = i)
    iou_score = round(d_log_training["iou_score"] * 100, 2)
    print(f"IoU = {iou_score}%")
    print()
    # log validation performance
    if i % N_STEP_LOG == 0:
        d_log_validation = epoch_validation.run(loader_validation, i_epoch = i)
        iou_score = round(d_log_validation["iou_score"] * 100, 2)
        print(f"IoU = {iou_score}%")
        # save model if better than previous best
        if max_score < iou_score:
            max_score = iou_score
            torch.save(
                model,
                os.path.join(P_DIR_CKPT, f"best_model_epoch_{i:0>4}.pth")
            )
            print("Model saved!")
        print()
    scheduler.step()
writer_training.close()
writer_validation.close()


# ======== TEST ======== #

print("\n==== TEST PHASE====\n")
# create export directory
os.makedirs(P_DIR_EXPORT, exist_ok = True)
# load best model
p_model_best = sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[-1]
print(f"Loading following model: {p_model_best}")
model = torch.load(p_model_best)
# initialize test instance
test_epoch = Epoch(
    model,
    s_phase = "test",
    loss = loss,
    p_dir_export = P_DIR_EXPORT,
    device = S_DEVICE,
    verbose = True,
)
test_epoch.run(loader_test)

# remove intermediate checkpoints
for model_checkpoint in sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[:-1]:
    os.remove(model_checkpoint)
