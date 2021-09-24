import os
import numpy as np
from torch.utils import data
from PIL import Image
import json


class IDDADataset(data.Dataset):

    def __init__(self, root, scenario, mean, crop_size, remap=False, ignore_index=255):
        self.root = root
        self.scenario_path = os.path.join(self.root, scenario)
        self.crop_size = crop_size
        self.ignore_index = ignore_index
        self.mean = mean
        self.remap = remap
        self.files = []

        self.img_ids = [i_id.replace(".jpg", "") for i_id in os.listdir(os.path.join(self.scenario_path, "RGB"))]
        self.info = json.load(open(os.path.join(self.root, "README", "idda_info.json"), 'r'))
        self.class_mapping = self.info['idda2cityscapes_trainid']

        for name in self.img_ids:
            image_path = os.path.join(self.scenario_path, "RGB/%s.jpg" % name)
            label_path = os.path.join(self.scenario_path, "Semantic/%s.png" % name)
            self.files.append({
                "image": image_path,
                "label": label_path,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file = self.files[index]

        # open image and label file
        image = Image.open(file['image']).convert('RGB')
        label = Image.open(file['label'])
        name = file['name']

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # remap the semantic label if remap is not None
        if self.remap is not None:
            label_copy = self.ignore_index * np.ones(label.shape, dtype=np.float32)
            for k, v in self.class_mapping:
                label_copy[label == k] = v
            label = label_copy

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name

