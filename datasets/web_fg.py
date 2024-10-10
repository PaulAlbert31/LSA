import os
import copy
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.io.image import decode_png, decode_jpeg, read_file

from mypath import Path

class fg_web_dataset(Dataset): 
    def __init__(self, transform, mode, transform_cont=None, transform_ssl=None, which='web-bird'):
        classes_n = {'web-bird': 200, 'web-car':196, 'web-aircraft':100}
        self.num_class = classes_n[which]
        self.root = Path.db_root_dir(which)
        self.transform = transform
        self.mode = mode
        self.transform_cont = transform_cont
        self.transform_ssl = transform_ssl
        self.clean_noisy = None
        
        if self.mode=='test':
            with open(os.path.join(self.root, 'val-list.txt')) as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = []
            for line in lines:
                img, target = line.split(':')
                target = int(target)                
                self.val_imgs.append(os.path.join(self.root, img))
                self.val_labels.append(target)
            self.val_labels = np.array(self.val_labels)
            print(len(self.val_labels))
        else:    
            with open(os.path.join(self.root, 'train-list.txt')) as f:
                lines=f.readlines()    
            train_imgs = []
            self.targets = []
            for line in lines:
                img, target = line.split(':')
                target = int(target)
                train_imgs.append(os.path.join(self.root, img))
                self.targets.append(target)
                    
            self.data = train_imgs
            self.targets = np.array(self.targets)
            print(len(self.data))
                            
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.data[index]
            target = self.targets[index]
            if self.transform_cont is not None:            
                img = Image.open(img_path).convert('RGB')
                
                img_t = self.transform(img)                
                img_ = self.transform(img)
                            
                sample = {'image':img_t, 'image_':img_, 'target':target, 'index':index}
                
                if self.transform_ssl is not None:
                    sample['image1'] = self.transform_ssl(img)
                if self.transform_cont is not None:
                    sample['image2'] = self.transform_cont(img)
            else:
                img = read_file(img_path)
                #Can be used to fix some corrupted images
                """
                try:
                    img = decode_jpeg(img, mode=torchvision.io.ImageReadMode.RGB)
                except:
                    img = Image.fromarray(np.array(Image.open(img_path).convert('RGB')))
                    img.save(img_path, "JPEG")
                    
                    img = read_file(img_path)
                    img = decode_jpeg(img, mode=torchvision.io.ImageReadMode.RGB)                  
                """
                img = decode_jpeg(img, mode=torchvision.io.ImageReadMode.RGB)
                img = self.transform(img)
                sample = {'image':img, 'target':target, 'index':index}
                        
            return sample
            
        elif self.mode=='test':
            
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
            img = read_file(img_path)
            img = decode_jpeg(img, device="cpu", mode=torchvision.io.ImageReadMode.RGB)
                          
            sample = {'image':self.transform(img), 'target':target, 'index':index}
                       
            return sample
           
    def __len__(self):
        if self.mode!='test':
            return len(self.data)
        else:
            return len(self.val_imgs)    
