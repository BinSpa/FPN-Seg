import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from data import custom_transforms as tr

class URURSegmentation(data.Dataset):
    NUM_CLASSES = 8

    def __init__(self, args, root="/home/deep/nas/gyl/RS_DATASET/URUR", split="train"):
        
        self.root = root
        self.split = split
        self.args = args

        self.images_base = os.path.join(self.root, self.split, 'image')
        self.annotations_base = os.path.join(self.root, self.split, 'label')
        self.ignore_index = 255
        self.class_names = ["background", "building", "farmland", "greenhouse", "woodland", "bareland", "water", "road"]
        self.class_nums = [0,1,2,3,4,5,6,7]
        self.img_list = os.listdir(self.images_base)

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.images_base, self.img_list[index])
        lbl_path = os.path.join(self.annotations_base, img_name)
        _img = Image.open(img_path).convert('RGB')
        _target = Image.open(lbl_path)
        sample = {'image': _img, 'label':_target}
        
        if self.split == 'train':
            train_set = self.transform_tr(sample)
            return train_set
        elif self.split == 'val':
            val_set = self.transform_val(sample)
            return val_set
        elif self.split == 'test':
            test_set = self.transform_ts(sample)
            return test_set
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            tr.RandomCrop(crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=self.args.crop_size),
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)