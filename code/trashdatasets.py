import os
import numpy as np
import torch
import PIL.Image
from xml.dom.minidom import parse


class DualClassTrashDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all images while sorting them to ensure alignment
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "labelXML"))))
        self.labelDictionary = {'plastic': 1, 'other': 2}
        self.reduce_list = ['unknown', 'metal', 'wood', 'fishing', 'cloth', 'paper', 'rubber', 'platstic', 'papper',
                            'bio', 'rov']  # 'timestamp',

    def __getitem__(self, idx):
        # here, we load images and the bboxes
        img_path = os.path.join(self.root, "Images", self.imgs[idx])  #
        bbox_xml_path = os.path.join(self.root, "labelXML", self.bbox_xml[idx])  #

        img = PIL.Image.open(img_path).convert("RGB")

        # read the files: the voc format stores files in the xml format
        dom = parse(bbox_xml_path)
        # Get document Element Object
        data = dom.documentElement
        # Get Objects
        objects = data.getElementsByTagName('object')
        # get bounding box coordinates
        boxes = []
        labels = []

        for object_ in objects:
            # Get the contents of the label
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # this gives the label value

            if name == 'timestamp':
                continue

            if name in self.reduce_list:
                name = 'other'

            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)

            boxes.append([xmin, ymin, xmax, ymax])
            # img = PIL.Image.open(img_path).convert("RGB")
            labels.append(np.int(self.labelDictionary[name]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}  # a target dictionary containing all the boxes and labels
        # since we are training a detection network, there is no target [masks] = masks
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)