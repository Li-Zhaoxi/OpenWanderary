import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import cv2



def get_imglist(data_dir, fold, phase):

    
    filenames = pd.read_csv(os.path.join(data_dir, 'filenames.csv'))

    imagenames = filenames.values[:, 0]
    gtnames = filenames.values[:, 1]
    
    testnum = int(len(filenames) * 0.2)
    teststart = fold * testnum
    testend = (fold + 1) * testnum

    if phase == 'test':
        imgnames = imagenames[teststart:testend]
        gtnames = gtnames[teststart:testend]
    else:
        imagenames = np.concatenate([imagenames[:teststart], imagenames[testend:]], axis=0)
        gtnames = np.concatenate([gtnames[:teststart], gtnames[testend:]], axis=0)
        valnum = int(len(gtnames) * 0.2)
        if phase == 'train':
            imgnames, gtnames = imagenames[valnum:], gtnames[valnum:]
        elif phase == 'val':
            imgnames, gtnames = imagenames[:valnum], gtnames[:valnum]
        else:
            raise ValueError('phase should be train or val or test')
    return imgnames, gtnames


class DataFolder(data.Dataset):
    def __init__(self, root_dir, phase, fold, gan_aug=False, data_transform=None):
        """
        :param root_dir: 
        :param data_transform: data transformations
        :param phase: train, val, test
        :param fold: fold number, 0, 1, 2, 3, 4
        :param gan_aug: whether to use gan augmentation
        """
        super(DataFolder, self).__init__()
        self.data_transform = data_transform
        self.gan_aug = gan_aug
        self.phase = phase
        self.fold = fold 
        self.root_dir = root_dir
        self.imgnames, self.gtnames = get_imglist(self.root_dir, self.fold, self.phase)
        # self.imgs_aug, self.masks_aug, _ = get_imglist(os.path.join(self.root_dir, 'aug'), self.fold, self.phase)

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        imgname = self.imgnames[idx]
        gtname = self.gtnames[idx]

        imgpath = os.path.join(self.root_dir, "images", imgname)
        gtpath = os.path.join(self.root_dir, "labels", gtname)

        img = cv2.imread(imgpath)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        mask = cv2.imread(gtpath, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
                
        if self.data_transform is not None:
            transformed = self.data_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return {'image': img, 'label': mask, 'name': imgname}
    