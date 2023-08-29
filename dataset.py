# Custom dataset
from PIL import Image, ImageChops, ImageEnhance
import os
import time
import torch.utils.data as data
import random
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def DataAugmentation(image_dir,subfolder='train',Data_augmentation=True):
    input_path = os.path.join(image_dir, subfolder)
    if Data_augmentation:
        file_list1 = glob(input_path + '/' + '**.jpg')
        for i in range(len(file_list1)):
            img_path = file_list1[i]
            img = Image.open(img_path)
            name = img_path.split('/')[-1]
            mask = img.crop((0, 0, img.width // 2, img.height))
            img1 = img.crop((img.width // 2, 0, img.width, img.height))
            img1 = ImageEnhance.Brightness(img1)
            brightness = 1.07
            img1 = img1.enhance(brightness)
            img1 = ImageEnhance.Brightness(img1)
            brightness = 0.87
            img1 = img1.enhance(brightness)
            img1 = ImageEnhance.Color(img1)
            color = 0.8
            img1 = img1.enhance(color)
            img1 = ImageEnhance.Contrast(img1)
            contrast = 0.8
            img1 = img1.enhance(contrast)
            enh_sha = ImageEnhance.Sharpness(img1)
            sharpness = 3.0
            img1 = enh_sha.enhance(sharpness)
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            mask = ImageEnhance.Brightness(mask)
            brightness = 1.07
            mask = mask.enhance(brightness)
            mask = ImageEnhance.Brightness(mask)
            brightness = 0.87
            mask = mask.enhance(brightness)
            mask = ImageEnhance.Color(mask)
            color = 0.8
            mask = mask.enhance(color)
            mask = ImageEnhance.Contrast(mask)
            contrast = 0.8
            mask = mask.enhance(contrast)
            mask = ImageEnhance.Sharpness(mask)
            sharpness = 3.0
            mask = mask.enhance(sharpness)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            htitch1 = np.hstack((mask, img1))
            im = Image.fromarray(htitch1)
            save_path = input_path + '/da_' + name
            im.save(save_path)
            pass
        print('Complete data augmentation')





class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder='train', direction='AtoB', transform=None, resize_scale=None,
                 crop_size=None, fliplr=False, Data_augmentation=True):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn)

        if self.direction == 'AtoB':
            input = img.crop((0, 0, img.width // 2, img.height))
            target = img.crop((img.width // 2, 0, img.width, img.height))
        elif self.direction == 'BtoA':
            input = img.crop((img.width // 2, 0, img.width, img.height))
            target = img.crop((0, 0, img.width // 2, img.height))
        # preprocessing
        if self.resize_scale:
            input = input.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            input = input.crop((x, y, x + self.crop_size, y + self.crop_size))
            target = target.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)

        return input, target



    def __len__(self):
        return len(self.image_filenames)


