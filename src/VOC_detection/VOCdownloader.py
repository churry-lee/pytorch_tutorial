#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from torchvision.datasets.utils import download_and_extract_archive
import os
import tarfile
import collections
from PIL import Image
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from typing import List, Dict, Tuple, Any

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
    def __init__(self, root: str, year: str, image_set: str='train', pre_processing: bool=False):
        self.year = year
        self.image_set = image_set
        self.pre_processing = pre_processing

        if image_set == 'train':
            idx = 0
        elif image_set == 'test':
            idx = 1

        dataset_year_dict = DATASET_YEAR_DICT[year][idx]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]
        
        self.root = os.path.join(root, 'VOC', year, image_set)
        
        base_dir = dataset_year_dict["base_dir"]
        base_dir = base_dir.split('/')
        voc_root = os.path.join(self.root, base_dir[0])

        if not os.path.isdir(self.root):
            os.makedirs(self.root, exist_ok=True)
        voc_file = os.path.join(self.root, self.filename)
        if not os.path.exists(voc_file):
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
        else:
            if not os.path.exists(voc_root):
                ap = tarfile.open(voc_file)
                ap.extractall(self.root)
                ap.close()
                print('Extracting complete')
            print('Files already downloaded')
       
        splits_dir = os.path.join(voc_root, base_dir[1], "ImageSets", "Main")
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()] 
       
        image_dir = os.path.join(voc_root, base_dir[1], "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, base_dir[1], "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]

        assert len(self.images) == len(self.targets)
    
    def __len__(self):
        return len(self.images)
            

class VOCDetect(VOCDownLoader):
    @property
    def annotations(self) -> List[str]:
        return self.targets

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.pre_processing:
            image = pre_processing(image)
        target = self.parse_voc_xml(ET_parse(self.annotations[idx]).getroot())
        return image, target
    
    def parse_voc_xml(self, node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        #print(f'node: {node}')
        children = list(node)
        #print(f'children: {children}')
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict