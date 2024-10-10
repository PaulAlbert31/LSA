import os
import copy
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io.image import decode_jpeg, read_file

from mypath import Path

class webvision_dataset(Dataset): 
    def __init__(self,  mode, num_classes, transform=None, transform_cont=None, transform_ssl=None):
        self.root = Path.db_root_dir("webvision")
        self.transform = transform
        self.mode = mode
        self.num_class = num_classes
        self.transform_cont = transform_cont
        self.transform_ssl = transform_ssl 
        self.clean_noisy = None
        
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_classes:
                    self.val_imgs.append(img)
                    self.val_labels.append(target)
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_classes:
                    train_imgs.append(img)
                    self.targets.append(target)
            self.data = train_imgs

            self.targets = np.array(self.targets)
            print(len(self.data))
                            
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.data[index]
            target = self.targets[index]
            img_path = os.path.join(self.root, img_path)

            if self.transform_cont is None:
                #Speed up tracking and val data augmentation
                #by performing them at the batch level outside of the dataloader
                img = read_file(img_path)
                img = decode_jpeg(img, device="cpu")
                return {'image':self.transform(img), 'target':target, 'index':index}
            else:
                img = Image.open(img_path)
            
            img_t = self.transform(img)
            sample = {'image':img_t, 'target':target, 'index':index}
            if self.transform_ssl is not None:
                img_ = self.transform(img)
                sample['image_'] = img_           

            if self.transform_ssl is not None:
                sample['image1'] = self.transform_ssl(img)
            if self.transform_cont is not None:
                sample['image2'] = self.transform_cont(img)
        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
            path = self.root+'val_images_256/'+img_path

            img = read_file(path)
            img = decode_jpeg(img, device="cpu")
            
            sample = {'image':self.transform(img), 'target':target, 'index':index}
            
        return sample
           
    def __len__(self):
        if self.mode!='test':
            return len(self.data)
        else:
            return len(self.val_imgs)    
