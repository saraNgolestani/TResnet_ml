import tqdm
import torchvision.transforms as transforms
import os
import torch
import torchvision.datasets as datasets
from src.helper_functions.pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import cv2
import pickle
import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
import os
import tqdm
from torchvision import transforms
from pytorch_lightning.core.datamodule import LightningDataModule

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((80), dtype=torch.long)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
        # print(self.cat2cat)


def get_dataloaders():
    batch_size = 128
    workers = 2
    num_classes = 80
    image_size = 228
    data = '/local/scratch1/makbn/sara/data'
    # COCO Data loading
    instances_path_val = os.path.join(data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(data, 'annotations/instances_train2014.json')
    data_path_val = f'{data}/val2014'  # args.data
    data_path_train = f'{data}/train2014'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((image_size, image_size)),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False, drop_last=True)
    dataloaders = {'train': train_dl, 'val': val_dl}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    return dataloaders, dataset_sizes


class COCODatasetLightning(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.workers = 2
        self.num_classes = 80
        self.image_size = 224
        self.data_path = '/home/sara.naserigolestani/hydra-tresnet/data/coco'
        self.batch_size = 128

        instances_path_val = os.path.join(self.data_path, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(self.data_path, 'annotations/instances_train2014.json')
        data_path_val = f'{self.data_path}/val2014'  # args.data
        data_path_train = f'{self.data_path}/train2014'  # args.data
        self.train_dataset = CocoDetection(data_path_train,
                                      instances_path_train,
                                      transforms.Compose([
                                          transforms.Resize((self.image_size, self.image_size)),
                                          transforms.ToTensor(),
                                          # normalize,
                                      ]))
        self.val_dataset = CocoDetection(data_path_val,
                                           instances_path_val,
                                           transforms.Compose([
                                               transforms.Resize((self.image_size, self.image_size)),
                                               transforms.ToTensor(),
                                               # normalize,
                                           ]))

    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_dl = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=True, drop_last=True)
        return train_dl

    def val_dataloader(self):
        val_dl = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=True, drop_last=True)
        return val_dl




















