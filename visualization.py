import albumentations as A
import cv2
import glob
import numpy as np
import torch
import os
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm
from dataset_cityscapes import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# ======== CONFIGURATION ======== #

S_NAME_CITY = "berlin" # if using test dataset, one of: berlin, bielefeld, bonn, leverkusen, mainz, munich
S_NAME_ENCODER = "efficientnet-b4"
S_NAME_WEIGHTS = "imagenet"
P_DIR_MODEL = "/Workspace/DeepLabV3+_EfficientNetB4_CE/Checkpoints/best_model_epoch_0060.pth"
P_DIR_DATA = "/Workspace/Datasets/Cityscapes"
P_VIDEO_DIR = "/Workspace/Videos/"
os.makedirs(P_VIDEO_DIR, exist_ok = True)
P_OUTPUT_VIDEO= os.path.join(P_VIDEO_DIR, f"{S_NAME_CITY}.avi")  # output video name

N_IMAGES = 20
FPS = 2

# cityscapes image size
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1024
# legend size
LEGEND_WIDTH = IMAGE_WIDTH * 2
LEGEND_HEIGHT = 100
# image and model description size
DESC_WIDTH = IMAGE_WIDTH * 2
DESC_HEIGHT = 50

S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== SETUP ======== #

# setup model
model = torch.load(P_DIR_MODEL, map_location=S_DEVICE)
model.eval()
# setup input normalization
preprocess_input = get_preprocessing_fn(
    encoder_name = S_NAME_ENCODER,
    pretrained = S_NAME_WEIGHTS,
)
# play with it! add a transformation to see the generalization performance
transform_weather = A.Compose([
    #A.RandomRain(brightness_coefficient=0.9, blur_value=3, p=1),
    #A.RandomSnow(snow_point_lower=0.3, snow_point_upper=0.5, p=1),
    #A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.05, p=1),
])
transform_full = A.Compose([
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])
# setup test dataset
dataset_test = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "test",
    mode = "fine",
    transform = transform_full,
    device= S_DEVICE,
)


# ======== DISPLAY ======== #

# create legend with names and colors of the classes as a 2x10 colored grid with text
legend = np.zeros((LEGEND_HEIGHT, LEGEND_WIDTH, 3), dtype=np.uint8)
# add the classes using dataset mapping and info
for class_trainid, class_id in enumerate(dataset_test.th_i_lut_trainid2id[:20]):
    color = dataset_test.classes[class_id[0]].color
    class_name = dataset_test.classes[class_id[0]].name if class_trainid != 19 else "background"
    # add the corresponding color to the background
    cv2.rectangle(legend,
                  (class_trainid%10 * LEGEND_WIDTH//10, class_trainid//10 * LEGEND_HEIGHT//2),
                  ((1 + class_trainid%10) * LEGEND_WIDTH//10, (1 + class_trainid//10) * LEGEND_HEIGHT//2),
                  color, -1)
    # add class name on top of the colored rectangle
    textsize = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    x = int((0.5 + class_trainid%10) * LEGEND_WIDTH/10 - textsize[0]/2)
    y = int((0.5 + class_trainid//10) * LEGEND_HEIGHT/2 + textsize[1]/2)
    cv2.putText(legend, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# convert legend from RGB to BGR for later use
legend = cv2.cvtColor(legend, cv2.COLOR_RGB2BGR)

# all the images in test for a specific city, to create a video
image_paths = sorted(glob.glob(dataset_test.images_dir + f"/{S_NAME_CITY}/*.png"))
out = cv2.VideoWriter(P_OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"XVID"),
    FPS, (LEGEND_WIDTH, IMAGE_HEIGHT + LEGEND_HEIGHT + DESC_HEIGHT)
)
for img in tqdm(image_paths[:N_IMAGES]):
    # normal image on the left
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # add weather augmentation
    image_weather = transform_weather(image = image)["image"]
    # predict image
    with torch.inference_mode():
        model_input = transform_full(image = image_weather)["image"]
        model_input = torch.from_numpy(model_input).unsqueeze(0)
        model_input = model_input.to(S_DEVICE)
        logits = model(model_input)
        prediction = logits.argmax(axis = 1)
    # convert prediction values from train_id to color mask
    prediction_color = lut.lookup_chw(
        td_u_input = prediction.byte(),
        td_i_lut = dataset_test.th_i_lut_trainid2color,
    ).permute((1, 2, 0)).cpu().numpy()
    # superimpose prediction to input image
    blend = cv2.addWeighted(image_weather, 0.4, prediction_color, 0.6, 0.0)
    # tile the blend image to the input image
    img_concat = cv2.hconcat([image_weather, blend])
    # convert from RGB to BGR
    img_concat = cv2.cvtColor(img_concat, cv2.COLOR_RGB2BGR)
    # add legend on top of the two
    img_concat = cv2.vconcat([legend, img_concat])
    
    # add description
    background = np.full((DESC_HEIGHT, DESC_WIDTH, 3), 100, dtype=np.uint8)
    desc = f"City: {S_NAME_CITY}    Model: {P_DIR_MODEL}    Frame: {img}"
    textsize = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    x = int(DESC_WIDTH/2 - textsize[0]/2)
    y = int(DESC_HEIGHT/2 + textsize[1]/2)
    cv2.putText(background, desc, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # put it under the images
    img_concat = cv2.vconcat([img_concat, background])
    
    out.write(img_concat)
    
out.release()
