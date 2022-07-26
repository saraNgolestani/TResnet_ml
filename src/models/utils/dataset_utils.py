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

        output = torch.zeros((90), dtype=torch.long)
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


def get_dataloaders(args):
    batch_size = args.batch_size
    workers = args.num_workers
    num_classes = args.num_classes
    image_size = args.input_size
    data = '/home/sara.naserigolestani/hydra-tresnet/data/coco'
    # COCO Data loading
    instances_path_val = os.path.join(data, 'annotations/instances_val2017.json')
    instances_path_train = os.path.join(data, 'annotations/instances_train2017.json')
    data_path_val = f'{data}/val2017'  # args.data
    data_path_train = f'{data}/train2017'  # args.data
    val_dataset = load_data_from_file(data_path=data_path_val, instances_path=instances_path_val,
                                      sampling_ratio=args.dataset_sampling_ratio, seed=0, image_size=image_size)
    train_dataset = load_data_from_file(data_path=data_path_train, instances_path=instances_path_train,
                                        sampling_ratio=args.dataset_sampling_ratio, seed=0, image_size=image_size)
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


def load_data_from_file(data_path, instances_path, sampling_ratio=1.0, seed=0, image_size=224):
    if sampling_ratio == 1.0:
        print(f'loading the whole dataset from: {data_path}')
        return CocoDetection(data_path,
                             instances_path,
                             transforms.Compose([
                                 transforms.Resize((image_size, image_size)),
                                 transforms.ToTensor(),
                                 # normalize,
                             ]))
    else:
        print(f'loading a subset(%{sampling_ratio * 100}) of dataset from: {data_path}')
        whole_set = CocoDetection(data_path,
                                  instances_path,
                                  transforms.Compose([
                                      transforms.Resize((image_size, image_size)),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
        subset_size = int(len(whole_set) * sampling_ratio)
        random.seed(seed)
        subset_indices = random.sample(list(range(len(whole_set))), subset_size)
        subset = torch.utils.data.Subset(whole_set, subset_indices)
        print(f'subset size: {len(subset)}')

        return subset


def get_weighted_labels(phase='train'):
    dataloaders, dataset_sizes = get_dataloaders()
    pbar = tqdm.tqdm(dataloaders[phase], desc=f'phase:{phase}')
    n_samples = []
    n = np.zeros(80)
    for _, labels in pbar:
        for j in labels:
            for i in range(len(j)):
                j_array = np.array(j)
                n[i] = n[i] + j_array[i]
    return n








