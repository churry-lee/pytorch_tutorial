#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import tarfile
from torchvision.datasets.utils import download_and_extract_archive


DATASET_YEAR_DICT = {
    '2012': [
        {
            'url': 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
            'filename': 'VOCtrainval_11-May-2012.tar',
            'md5': '6cd6e144f989b92b3379bac3b3de84fd',
            'base_dir': os.path.join('VOCdevkit', 'VOC2012')
        },
        {
            'url': 'http://pjreddie.com/media/files/VOCtest_11-May-2012.tar',
            'filename': 'VOCtest_11-May-2012.tar',
            'md5': '6cd6e144f989b92b3379bac3b3de84fd',
            'base_dir': os.path.join('VOCdevkit', 'VOC2012')
        }
    ],
    '2007': [
        {
            'url': 'http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
            'filename': 'VOCtrainval_06-Nov-2007.tar',
            'md5': 'c52e279531787c972589f7e41ab4ae64',
            'base_dir': os.path.join('VOCdevkit', 'VOC2007')
        },
        {
            'url': 'http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar', 
            'filename': 'VOCtest_06-Nov-2007.tar', 
            'md5': 'b6e924de25625d8de591ea690078ad9f',
            'base_dir': os.path.join('VOCdevkit', 'VOC2007')
        }
    ]
}


class VOCDownLoader(object):
    def __init__(self, root: str, year: str, image_set: str='train'):
        self.year = year
        self.image_set = image_set

        if image_set == 'train':
            idx = 0
        elif image_set == 'test':
            idx = 1

        dataset_year_dict = DATASET_YEAR_DICT[year][idx]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]
        
        base_dir = dataset_year_dict["base_dir"]

        self.root = os.path.join(root, 'VOC', year, image_set)
        if not os.path.isdir(self.root):
            os.makedirs(self.root, exist_ok=True)
        voc_file = os.path.join(self.root, self.filename)
        if not os.path.exists(voc_file):
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
        else:
            base_dir = base_dir.split('/')
            self.voc_root = os.path.join(self.root, base_dir[0])
            if not os.path.exists(self.voc_root):
                ap = tarfile.open(voc_file)
                ap.extractall(self.root)
                ap.close()
                print('Extracting complete')
            print('Files already downloaded')
            
train = VOCDownLoader(root='./data', year='2007', image_set='train')
print(train.root, train.year, train.image_set)
