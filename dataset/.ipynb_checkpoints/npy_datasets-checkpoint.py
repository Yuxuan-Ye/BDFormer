from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2


class NPY_datasets_multitask(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets_multitask, self)
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(path_Data + 'val/images/')
            images_list = sorted(images_list, key=lambda x: int(x.split('.')[0]))
            masks_list = os.listdir(path_Data + 'val/masks/')
            masks_list = sorted(masks_list, key=lambda x: int(x.split('.')[0]))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255       # expand_dims: W*H -> W*H*C(C=1)
        contour = np.expand_dims(cv2.Canny(msk.astype(np.uint8), 0, 1), axis=2) / 255
        # kernel = np.ones((5, 5), np.uint8)
        kernel = np.ones((9, 9), np.uint8)
        contour = cv2.dilate(contour, kernel).reshape([256, 256, 1])
        img, msk, contour = self.transformer((img, msk, contour))
        return img, msk, contour

    def __len__(self):
        return len(self.data)

class PH2_datasets_multitask(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(PH2_datasets_multitask, self)
        images_list = os.listdir(path_Data + 'images/')
        images_list = sorted(images_list, key=lambda x: int(x.split('.')[0]))
        masks_list = os.listdir(path_Data + 'masks/')
        masks_list = sorted(masks_list, key=lambda x: int(x.split('.')[0]))
        self.data = []
        for i in range(len(images_list)):
            img_path = path_Data + 'images/' + images_list[i]
            mask_path = path_Data + 'masks/' + masks_list[i]
            self.data.append([img_path, mask_path])
        self.transformer = config.test_transformer


    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255       # expand_dims: W*H -> W*H*C(C=1)
        contour = np.expand_dims(cv2.Canny(msk.astype(np.uint8), 0, 1), axis=2) / 255
        kernel = np.ones((5, 5), np.uint8)
        contour = cv2.dilate(contour, kernel).reshape([256, 256, 1])
        # contour = cv2.Canny(msk.astype(np.uint8), 0, 1) / 255
        img, msk, contour = self.transformer((img, msk, contour))
        return img, msk, contour

    def __len__(self):
        return len(self.data)

