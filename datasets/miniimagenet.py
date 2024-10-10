import os
import json
import copy
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io.image import decode_jpeg, read_file

from mypath import Path

class MiniImagenet(Dataset):
    # including hard labels & soft labels
    def __init__(self, data, labels, transform=None, transform_cont=None, transform_ssl=None, clean_noisy=None):
        self.data, self.targets =  data, labels
        self.transform = transform
        self.transform_cont = transform_cont
        self.transform_ssl = transform_ssl
        self.clean_noisy = clean_noisy
        self.num_class = 100
        if clean_noisy is None:
            clean_noisy = torch.ones(len(self.data)).bool()
        
        is_noisy = clean_noisy
        r = torch.arange(len(data))        
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform_cont is not None:
            img = Image.open(img)
            
            img_t = self.transform(img)
            img_ = self.transform(img)

            sample = {'image':img_t, 'image_':img_, 'target':target, 'index':index}
            if self.transform_ssl is not None:
                sample['image1'] = self.transform_ssl(img)
            if self.transform_cont is not None:
                sample['image2'] = self.transform_cont(img)
        else: #Speed up tracking and val data augmentations by performing them at the batch level outside of the dataloader
            img = read_file(img)
            img = decode_jpeg(img, device="cpu")
            sample = {'image':self.transform(img), 'target':target, 'index':index}            
        if self.clean_noisy is not None:
            sample['clean_noisy'] = self.clean_noisy[index]
            
        return sample

    def __len__(self):
        return len(self.data)


def make_dataset(root=Path.db_root_dir('miniimagenet_preset'), noise_ratio="0.4", noise_type="red", seed=1):
    np.random.seed(42)
    nclass = 100
    img_paths = []
    labels = []
    clean_noisy = []
    clean_anno = json.load(open(os.path.join(root, "mini-imagenet-annotations.json")))["data"]
    anno_dict = {}
    for anno in clean_anno:
        anno_dict[anno[0]['image/id']] = 1-int(anno[0]['image/class/label/is_clean'])

    n = 500
        
    for split in ["training", "validation"]:
        if split == "training":
            class_split_path = os.path.join(root, 'split', f'{noise_type}_noise_nl_{noise_ratio}')
        else:
            train_num = len(img_paths)
            class_split_path = os.path.join(root, 'split', 'validation')
            
        with open(class_split_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                name, c = l.split(' ')
                if split=='training':
                    img_paths.append(os.path.join(root, 'all_images_resampled_84', name))
                else:
                    img_paths.append(os.path.join(root, 'validation', f'{int(c)}', name))
                labels.append(int(c))
                if name[0] != "n" and split == "training":
                    clean_noisy.append(1)
                elif split == "training":
                    clean_noisy.append(0)

    labels = np.array(labels)
    train_paths = img_paths[:train_num]
    train_labels = labels[:train_num]
    val_paths = img_paths[train_num:]
    val_labels = labels[train_num:]
    clean_noisy = torch.tensor(clean_noisy, dtype=torch.bool)
    print(clean_noisy.sum() / len(clean_noisy))
            
    return train_paths, train_labels, val_paths, val_labels, None, None, clean_noisy
