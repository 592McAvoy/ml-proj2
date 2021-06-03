import cv2
from torchvision.utils import make_grid
import torch

import scipy.io
import os
from PIL import Image

import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from base import BaseDataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
class CelebADataset(data.Dataset):
    dir_path = 'data/CelebA/img_align_celeba/img_align_celeba'
    
    def __init__(self, transform, N=10000):
        self.transform = transform
        self.N = N
        # imgs = []
        # for i in range(N):
        #     im_n = '{:06d}.jpg'.format(i+1)
        #     # print(im_n)
        #     im = Image.open(os.path.join(self.dir_path, im_n))
        #     im = im.crop((0, 28, 178, 198))
        #     im = im.convert('RGB')
        #     imgs.append(im)
        # self.imgs = imgs

    def __getitem__(self, index):
        im_n = '{:06d}.jpg'.format(index+1)
        # print(im_n)
        im = Image.open(os.path.join(self.dir_path, im_n))
        im = im.convert('RGB')
        # im = self.imgs[index]

        im = self.transform(im)
        
        label = 0
        
        return im, int(label)

    def __len__(self):
        return self.N

class CelebALoader(BaseDataLoader):
    dir_path = 'data/CelebA/copy'
    def __init__(self, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=8, im_res=64, **kwargs):
        # Normalize to -1 ~ 1
        trfm = T.Compose([
            T.CenterCrop((178, 178)),
            T.Resize((im_res, im_res)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])

        self.dataset = ImageFolder(self.dir_path, trfm)
        # self.dataset = CelebADataset(trfm, **kwargs)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)



if __name__ == '__main__':
    trfm = T.Compose([
            # T.CenterCrop((178, 178)),
            T.Resize((64, 64)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
    dset = CelebADataset(trfm)
    print(len(dset))

    im = make_grid([dset[i][0] for i in range(32)], nrow=8, normalize=True)
    print(im.size())
    npimg = im.numpy().transpose(1, 2, 0)*255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    print(npimg.shape)
    
    cv2.imwrite('celebA.png', npimg)

    # dloader = CelebALoader(16)
    # print(len(dloader))
    # for im, label in dloader:
    #     print(im.size())
    #     im = make_grid(im, nrow=4, normalize=True)
    #     npimg = im.numpy().transpose(1, 2, 0)*255
    #     print(npimg.shape)
    #     cv2.imwrite('celebA.png', npimg)

    #     break

    # data, label = dset[0]
    # print(data.size(), label)

    # loader = CUBLoader(10, mode='train')
    # print(len(loader))
    # for data, lable in loader:
    #     print(data.size())
    #     break
