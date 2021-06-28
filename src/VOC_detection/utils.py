#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from typing import List, Dict, Tuple, Any

def imshow(img, norm: bool=False):
    if norm:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    return img

def parse_from_target(target):
        voc_dict: Dict = {}

        classes: List = []
        bboxes: List = []

        for i in range(len(target['annotation']['object'])):
            bbox: List = []
            _class = target['annotation']['object'][i]['name']
            bndbox = target['annotation']['object'][i]['bndbox']
            width = target['annotation']['size']['width']
            height = target['annotation']['size']['height']
            for key, val in bndbox.items():
                bbox.append(int(val))
            classes.append(_class)
            bboxes.append(bbox)
        
        voc_dict['labels'] = classes
        voc_dict['bboxes'] = bboxes
        voc_dict['size'] = (int(width), int(height))
        return voc_dict

import torchvision.transforms.functional as FT

def resize(image, bboxes, image_size, dims: Tuple=(300, 300), return_percent_coords=True):
    width, height = image_size[0], image_size[1]

    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
    new_bboxes: List = []
    for bbox in bboxes:
        bbox = torch.tensor(bbox)
        new_box = bbox / old_dims  # percent coordinates
        new_box = new_box.tolist()[0]

        for i, rate in enumerate(new_box):
            if i % 2 == 0:
                new_box[i] = rate * dims[0]
            elif i % 2 == 1:
                new_box[i] = rate * dims[1]
        new_bboxes.append(new_box)
    new_bboxes = torch.tensor(new_bboxes)

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_bboxes: List = []
        for bbox in bboxes:
            bbox = torch.tensor(bbox)
            new_box = bbox * new_dims
            new_bboxes.append(new_box.tolist()[0])
        new_bboxes = torch.tensor(new_bboxes)

    return new_image, new_bboxes