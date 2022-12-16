import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as A
import cv2
import os
from glob import glob
from epoch import *
from dataset_cityscapes import *
from torch.utils.tensorboard import SummaryWriter


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


# ======== CONFIGURATION ======== #

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# p_* stands for "path"
S_EXPERIMENT = "EfficientNetB0"
P_DIR_DATA = "/data/othrys4-space/data/henr_co/Image_Video_Understanding/Datasets/Cityscapes"
P_DIR_CKPT = os.path.join(
    "/data/othrys4-space/data/henr_co/Image_Video_Understanding",
    S_EXPERIMENT,
    "Checkpoints",
)
P_DIR_LOGS = os.path.join(
    "/data/othrys4-space/data/henr_co/Image_Video_Understanding",
    S_EXPERIMENT,
    "Logs",
)
P_DIR_EXPORT = os.path.join(
    "/data/othrys4-space/data/henr_co/Image_Video_Understanding",
    S_EXPERIMENT,
    "Export",
)
# s_* stands for "string" (i.e. all other strings than paths)
S_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
S_NAME_ENCODER = "efficientnet-b0"
S_NAME_WEIGHTS = "imagenet"
# n_* stands for n-dim size (0-dim: number of objects, 1-dim+: shape)
N_EPOCH_MAX = 100  # 50
N_SIZE_BATCH_TRAINING = 6  # training batch size
N_SIZE_BATCH_VALIDATION = 3  # validation batch size
N_SIZE_BATCH_TEST = 1  # test batch size
N_SIZE_PATCH = 512  # patch size for random crop
N_STEP_LOG = 1  # evaluate on validation set and save model every N iterations
N_WORKERS = 16  # 16 works best on an RTX Titan, to be adapted for each system
# other notations:
# l_* stands for "list"
# i_* stands for an index, e.g.: for i_object, object in enumerate(l_object):
# d_* stands for "dict"
# k_* stands for "key" (of a dictionary item)

# TIP: all these notations are those I use in my own code to make it extra clear
# what kind of object each variable is and therefore how to use it.
# It becomes especially handy to avoid getting confused when writing new names,
# as for example pluralizing names comes with risks of typos, e.g.:
#   for matrix in matrices:
#       ...
# or:
#   for matrix in matrixes:
#       ...
# (both are valid English plurals of "matrix", but no one ever uses the same)
# Instead I find it less error-prone and confusing to write:
#   for matrix in l_matrix:
#       ...
# And as for dictionaries:
#   for k_matrix, matrix in d_matrix.items():
#       ...
# Of course we can use any other naming convention (or even none at all ;)


# ======== SETUP ======== #

# setup model
model = smp.Unet(
    encoder_name = S_NAME_ENCODER,
    encoder_weights = S_NAME_WEIGHTS,
    in_channels = 3,
    classes = 20,
)
# setup input normalization
preprocess_input = get_preprocessing_fn(
    encoder_name = S_NAME_ENCODER,
    pretrained = S_NAME_WEIGHTS,
)
# setup input augmentations (for cropped and full images)
transform_crop = A.Compose([
    A.RandomCrop(N_SIZE_PATCH, N_SIZE_PATCH),
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
)
dataset_validation = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "val",
    mode = "fine",
    transform = transform_full,
)
dataset_test = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "test",
    mode = "fine",
    transform = transform_full,
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
loss.__name__ = 'ce_loss'
# setup optimizer
optimizer = torch.optim.Adam([
    dict(params = model.parameters(), lr = 1e-3),
])
# setup learning rate scheduler
# (here exponential decay that reaches initial_lr / 1000 after N_EPOCH_MAX)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer = optimizer,
    gamma = 1e-2 ** (1 / N_EPOCH_MAX),
    last_epoch = -1,
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
    # TIP: I'm using an "f-string" below, introduced in Python 3.6
    # they're much more convenient than the old .format() method I find ;)
    print(f'Epoch: {i} | LR = {round(scheduler.get_last_lr()[0], 8)}')
    d_log_training = epoch_training.run(loader_training, i_epoch = i)
    iou_score = round(d_log_training['iou_score'] * 100, 2)
    print(f'IoU = {iou_score}%')
    print()
    # log validation performance
    if i % N_STEP_LOG == 0:
        d_log_validation = epoch_validation.run(loader_validation, i_epoch = i)
        iou_score = round(d_log_validation['iou_score'] * 100, 2)
        print(f'IoU = {iou_score}%')
        # save model if better than previous best
        if max_score < iou_score:
            max_score = iou_score
            torch.save(
                model,
                os.path.join(P_DIR_CKPT, f'best_model_epoch_{i:0>4}.pth')
            )
            print('Model saved!')
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
print(f'Loading following model: {p_model_best}')
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
