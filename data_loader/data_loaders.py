import cv2
from torchvision.utils import make_grid

import scipy.io
import os
from PIL import Image

import numpy as np
from torch.utils import data
from torchvision import transforms as T

from base import BaseDataLoader
import csv

class CUBDataset(data.Dataset):
    data_dir = 'data/CUB_200/images'
    data_list = 'data/CUB_200/images.txt'
    bounding_box = 'data/CUB_200/bounding_boxes.txt'
    def __init__(self, transform, box_crop=True):
        self.transform = transform
        self.crop = box_crop
        
        with open(self.data_list, 'r') as fd:
            self.data = fd.readlines()
        
        with open(self.bounding_box, 'r') as fd:
            self.box = fd.readlines()


    def __getitem__(self, index):
        im_n = self.data[index].split()[-1]
        # print(im_n)
        im = Image.open(os.path.join(self.data_dir, im_n))
        im = im.convert('RGB')

        def cvt(str):
            return int(float(str))

        if self.crop:
            x, y, w, h = self.box[index].split()[1:] 
            x, y, w, h = cvt(x), cvt(y), cvt(w), cvt(h)
            im = im.crop((x,y,x+w,y+h))
        # print(im.getpixel((0,0)))
        # print(im.size)
        im = self.transform(im)
        # print(im.size())
        
        return im

    def __len__(self):
        return len(self.data)


class CUBLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=0, im_res=64, **kwargs):

        # Normalize to -1 ~ 1        
        trfm = T.Compose([
            T.Resize((im_res, im_res)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])

        self.dataset = CUBDataset(trfm, **kwargs)
    
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)

    def get_batchsize(self):
        return self.batch_size


if __name__ == '__main__':
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])

    trfm = T.Compose([
        T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
            ])
    dset = CUBDataset(trfm)
    print(len(dset))
    im = make_grid([dset[i] for i in range(16)], nrow=4, normalize=True)
    print(im.size())
    npimg = im.numpy().transpose(1, 2, 0)*255
    print(npimg.shape)
    cv2.imwrite('data.png', npimg)

    # data, label = dset[0]
    # print(data.size(), label)

    # loader = CUBLoader(10, mode='train')
    # print(len(loader))
    # for data, lable in loader:
    #     print(data.size())
    #     break
