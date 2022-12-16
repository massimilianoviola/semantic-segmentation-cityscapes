import torchvision
import numpy as np
import cv2
import pdb


class DatasetCityscapesSemantic(torchvision.datasets.Cityscapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_type = "semantic")
        # TODO: check validity of comment below
        # TIP: I removed the whole target_type index search as the class is now
        # hard-coded for semantic segmentation (somehow the parameters passing
        # of *args and **kwargs failed with py3.10 and the solution was
        # equivalent to removing the passing of target_type)
        self.colormap_id2trainid = self._generate_colormap_id2trainid()
        self.colormap_trainid2id = self._generate_colormap_trainid2id()

    def _generate_colormap_id2trainid(self):
        colormap = {}
        for class_ in self.classes:
            if class_.train_id in (-1, 255):
                continue
            colormap[class_.train_id] = class_.id
        return colormap

    def _generate_colormap_trainid2id(self):
        colormap = {}
        colormap[0] = 255
        for class_ in self.classes:
            if class_.train_id in (-1, 255):
                continue
            colormap[class_.id] = class_.train_id
        return colormap

    # formerly called _convert_to_segmentation_mask()
    def convert_id2trainid(self, image_id):
        height, width = image_id.shape[:2]
        # generate default segmentation mask filled with class 19 (background)
        image_trainid = np.full((height, width), len(self.colormap_id2trainid))
        # overwrite pixel with values corresponding to each class mask
        for class_trainid, class_id in self.colormap_id2trainid.items():
            image_trainid[image_id == class_id] = class_trainid
        return image_trainid
    
    def convert_trainid2id(self, image_trainid):
        height, width = image_trainid.shape[:2]
        # generate default segmentation mask filled with class 19 (background)
        image_id = np.zeros((height, width, 1), dtype = np.float32)
        # overwrite pixel with values corresponding to each class mask
        for class_id, class_trainid in self.colormap_trainid2id.items():
            image_id[image_trainid == class_trainid] = class_id
        return image_id

    def convert_trainid2color(self, image_trainid):
        height, width = image_trainid.shape[:2]
        image_color = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, class_trainid in self.colormap_trainid2id.items():
            image_color[image_trainid.squeeze() == class_trainid] = self.classes[class_id].color
        return image_color
    
    def convert_trainid2color_nchw(self, image_trainid):
        n_patch, _, height, width = image_trainid.shape
        image_color = np.zeros((n_patch, 3, height, width), dtype=np.uint8)
        for class_id, class_trainid in self.colormap_trainid2id.items():
            for i_channel in range(3):
                image_color[:, i_channel][image_trainid.squeeze() == class_trainid] = self.classes[class_id].color[i_channel]
        return image_color

    def __getitem__(self, index):
        # read images
        p_image = self.images[index]
        p_target = self.targets[index][0]  # 0 is index of semantic target type
        image = cv2.imread(p_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = cv2.imread(p_target, cv2.IMREAD_UNCHANGED)
        # convert target values from indices (id) to training indices (train_id)
        target = self.convert_id2trainid(target)
        if self.transform is not None:
            transformed = self.transform(image = image, mask = target)
            image = transformed["image"]
            target = transformed["mask"]
        return image, target, p_image, p_target
