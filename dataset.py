import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from random import shuffle
import cv2
import time


def augmenter(image):
    return transforms.RandomHorizontalFlip(p=0.5)(
        transforms.ColorJitter(contrast=0.25)(
            transforms.RandomAffine(
                0, translate=(0.03, 0.03))(image)))


def process_image(image):
    means = [0.485, 0.456, 0.406]
    inv_stds = [1/0.229, 1/0.224, 1/0.225]

    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    for channel, mean, inv_std in zip(image, means, inv_stds):
        channel.sub_(mean).mul_(inv_std)
    return image

categories = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat',
              'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird',
              'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake',
              'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch',
              'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant',
              'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier',
              'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
              'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
              'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator',
              'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard',
              'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
              'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet',
              'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase',
              'wine glass', 'zebra']

category_dict_classification = dict((category, count) for count, category in enumerate(categories))
category_dict_sequential = dict((category, count) for count, category in enumerate(categories))
category_dict_sequential['<end>'] = len(categories)
category_dict_sequential['<start>'] = len(categories) + 1
category_dict_sequential['<pad>'] = len(categories) + 2
category_dict_sequential_inv = dict((value, key)
                                    for key, value in category_dict_sequential.items())

class COCOMultiLabel(Dataset):
    def __init__(self, train, classification, image_path):
        super(COCOMultiLabel, self).__init__()
        self.train = train
        if self.train == True:
            self.coco_json = json.load(open('coco_train.json', 'r'))
            self.max_length = 18 + 2 # highest number of labels for one image in training
            self.image_path = image_path + '/train2014/'
        elif self.train == False:
            self.coco_json = json.load(open('coco_val.json', 'r'))
            self.max_length = 15 + 2
            self.image_path = image_path + '/val2014/'

        else:
            assert 0 == 1
        assert classification in [True, False]
        self.classification = classification
        self.fns = self.coco_json.keys()


    def __len__(self):
        return len(self.coco_json)

    def __getitem__(self, idx):
        json_key = self.fns[idx]
        categories_batch = self.coco_json[json_key]['categories']
        image_fn = self.image_path + json_key

        image = Image.open(image_fn)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.train:
            try:
                image = augmenter(image)
            except IOError:
                print "augmentation error"
        transform=transforms.Compose([
                           transforms.Resize((288, 288)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                           ])
        try:
            image = transform(image)        
        except IOError:
            return None

        # labels
        labels = []
        labels_classification = np.zeros(len(categories), dtype=np.float32)
        labels.append(category_dict_sequential['<start>'])
        for category in categories_batch:
            labels.append(category_dict_sequential[category])
            labels_classification[category_dict_classification[category]] = 1

        labels.append(category_dict_sequential['<end>'])
        for _ in range(self.max_length - len(categories_batch) - 1):
            labels.append(category_dict_sequential['<pad>'])

        labels = torch.LongTensor(labels)
        labels_classification = torch.from_numpy(labels_classification)
        label_number = len(categories_batch) + 2 # including the <start> and <end>

        if self.classification:
            return_tuple = (image, labels_classification)
        else:
            return_tuple = (image, labels, label_number, labels_classification)
        return return_tuple
