from typing import Union, Tuple, List

import numpy as np
import torch
import json
from pytorch_lightning.core.datamodule import LightningDataModule
import argparse
import os
from torchvision import transforms
from collections import defaultdict
from torchvision import datasets as datasets
import itertools
from PIL import Image
import pickle
import random
from torch.utils.data import Dataset, DataLoader



def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class UAV:
    def __init__(self, annFile_path):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        json_file = open(annFile_path)
        if json_file is not None:
            file_data = json.load(json_file)
            self.dataset = file_data
            self.creatIndex()

    def creatIndex(self):
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])


        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']   in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


class UAVArialDetection(datasets.coco.CocoDetection):
    def __init__(self,args,  root, annFile, transform=None, target_transform=None):
        self.root = root
        self.uav = UAV(annFile)
        self.ids = list(self.uav.imgToAnns.keys())
        print(f'ids len:{len(self.ids)}')
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = args.num_classes
        with open('/home/sara.naserigolestani/hydra-tresnet/saved_cat2cat.pkl', 'rb') as f:
            loaded_cat2cat = pickle.load(f)
        self.cat2cat = loaded_cat2cat
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uav = self.uav
        img_id = self.ids[index]
        ann_ids = uav.getAnnIds(imgIds=img_id)
        target = uav.loadAnns(ann_ids)

        output = torch.zeros((self.num_classes), dtype=torch.long)
        for obj in target:
            if obj['category_id'] == 1:
                obj['category_id'] = 3
            elif obj['category_id'] == 2:
                obj['category_id'] = 6
            elif obj['category_id'] == 3:
                obj['category_id'] = 1
            elif obj['category_id'] == 4:
                obj['category_id'] = 10
            elif obj['category_id'] == 5:
                obj['category_id'] = 4
            elif obj['category_id'] == 6:
                obj['category_id'] = 11
            elif obj['category_id'] == 7:
                obj['category_id'] = 8
            elif obj['category_id'] == 8:
                obj['category_id'] = 7
            output[self.cat2cat[obj['category_id']]] = 1
        target = output
        path = uav.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class UAVDatasetLightning(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.workers = 1
        self.num_classes = args.num_classes
        self.image_size = 224
        self.data_path = '/home/sara.naserigolestani/hydra-tresnet/data/uav/aerial_yolo'
        self.batch_size = args.batch_size
        self.args = args
        instances_path_val = os.path.join(self.data_path, 'valid/fixed_annotations2.json')
        instances_path_train = os.path.join(self.data_path, 'train/fixed_annotations2.json')
        data_path_val = f'{self.data_path}/valid'  # args.data
        data_path_train = f'{self.data_path}/train'  # args.data
        self.train_dataset = self.load_data_from_file(data_path=data_path_train, instances_path=instances_path_train,
                                                      sampling_ratio=1, seed=0)
        self.val_dataset = self.load_data_from_file(data_path=data_path_val, instances_path=instances_path_val,
                                                    sampling_ratio=1, seed=0)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_dl = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=True, drop_last=True, num_workers= self.workers, worker_init_fn=seed_worker, generator=g)
        return train_dl

    def val_dataloader(self):
        val_dl = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            pin_memory=True, drop_last=True,num_workers= self.workers, worker_init_fn=seed_worker, generator=g)

        print(f'size of dataset: {len(self.val_dataset)}')
        print(f'size of dataloader: {len(val_dl)}')

        return val_dl

    def test_dataloader(self):
        val_dl = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            pin_memory=True, drop_last=True ,num_workers= self.workers, worker_init_fn=seed_worker, generator=g)

        print(f'size of dataset: {len(self.val_dataset)}')
        print(f'size of dataloader: {len(val_dl)}')

        return val_dl

    def load_data_from_file(self, data_path, instances_path, sampling_ratio=1.0, seed=0):
        if sampling_ratio == 1.0:
            print(f'loading the whole dataset from: {data_path}')
            return UAVArialDetection(self.args, data_path,
                                     instances_path,
                                     transforms.Compose([
                                         transforms.Resize((self.image_size, self.image_size)),
                                         transforms.ToTensor(),
                                         # normalize,
                                     ]))
        else:
            print(f'loading a subset(%{sampling_ratio * 100}) of dataset from: {data_path}')
            whole_set = UAVArialDetection(self.args, data_path,
                                          instances_path,
                                          transforms.Compose([
                                              transforms.Resize((self.image_size, self.image_size)),
                                              transforms.ToTensor(),
                                              # normalize,
                                          ]))
            subset_size = int(len(whole_set) * sampling_ratio)
            random.seed(seed)
            subset_indices = random.sample(list(range(len(whole_set))), subset_size)
            subset = torch.utils.data.Subset(whole_set, subset_indices)
            print(f'subset size: {len(subset)}')
            return subset
