import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from data import custom_transforms as tr

class FBPSegmentation(data.Dataset):
    NUM_CLASSES = 25

    def __init__(self, args, root="/data1/gyl/RS_DATASET/FBP", split="train"):
        
        self.root = root
        self.split = split
        self.args = args

        self.images_base = os.path.join(self.root, self.split, 'rgb_images')
        self.annotations_base = os.path.join(self.root, self.split, 'gid_labels')
        self.ignore_index = 255
        self.class_names = ['unlabeled', 'industrial area', 'paddy field', 'irrigated field', 'dry cropland',
                 'garden land', 'arbor forest', 'shrub forest', 'park', 'natural meadow', 'artificial meadow', 
                 'river', 'urban residential', 'lake', 'pond', 'fish pond',
                 'snow', 'bareland', 'rural residential', 'stadium',
                 'square', 'road', 'overpass', 'railway station', 'airport'
                 ]
        self.class_nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        self.img_list = os.listdir(self.images_base)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        lbl_name = img_name.split('.')[0]+'_24label.png'
        img_path = os.path.join(self.images_base, self.img_list[index])
        lbl_path = os.path.join(self.annotations_base, lbl_name)
        _img = Image.open(img_path)
        _target = Image.open(lbl_path)
        sample = {'image': _img, 'label':_target}
        
        if self.split == 'train':
            train_set = self.transform_tr(sample)
            return train_set
        elif self.split == 'val':
            val_set = self.transform_val(sample)
            return val_set
        '''
        elif self.split == 'test':
            test_set = self.transform_ts(sample)
            return test_set
        '''
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            tr.RandomCrop(crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=self.args.crop_size),
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)