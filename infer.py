import torch
from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse
import torch.nn as nn
import time
from datetime import datetime
import copy
import torch.nn.functional as F
import tqdm
from src.models.utils.score_utils import Statistics, compute_scores, compute_scores_and_th
from src.models.utils.dataset_utils import get_dataloaders
import warnings
import wandb
from sklearn.exceptions import UndefinedMetricWarning
from pytorch_lightning import seed_everything
import numpy as np
import random


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir')
parser.add_argument('--save_path', default='saved_models')
parser.add_argument('--tresnet_unit_size', default='L', choices=['M', 'L', 'XL'], help='TResNet model size')
parser.add_argument('--model_type', default='tresnet', choices=['basic', 'tresnet'], help='model types')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--wandb_name', default='tresnet')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--remove_aa_jit', action='store_true', default=True)
parser.add_argument('--wandb_name', default='tresnet_bestTHapproach')
parser.add_argument('--num_gpu', default=1, type=int, help="number of gpu")
parser.add_argument('--train_precision', default=16, type=int, help="model precision")
parser.add_argument('--max_epoch', default=300, type=int, help="max number of epochs")
parser.add_argument('--dataset_sampling_ratio', default=1.0, type=float, help="sampling ratio of dataset")
parser.add_argument('--seed', default=0, type=int, help="seed for randomness")
parser.add_argument('--lr', default=5e-4, type=float, help="learning rate")
parser.add_argument('--load_from_path', default=False, type=bool, help='whether to load from an old model statics or not')


def set_seed(seed=0):
    seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def train_model(model, dataloaders, criterion, optimizer, scheduler, device,  num_epochs=100, scaler=None):
    print('start training')
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_precision = 0.0
    global_step = 0
    save_path = args.save_path + '/Tresnet_coco2017.pt'
    TH = 0.45

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            cum_stats = Statistics()
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            all_preds = []
            all_actuals = []
            all_loss = []
            running_loss = 0.0
            running_step = 0
            pbar = tqdm.tqdm(dataloaders[phase], desc=f'phase:{phase}')
            # Iterate over data.
            temp_sigmoid = F.sigmoid
            for inputs, labels in pbar:
                global_step += 1
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        predictions = model(inputs)
                        all_actuals.extend(labels.cpu().tolist())
                        all_preds.extend(predictions.detach().cpu().tolist())
                        loss = criterion(predictions, labels)
                        all_loss.append(loss)
                        # backward + optimize only if in training phase
                    if phase == 'train':
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        optimizer.zero_grad()

                    preds = (predictions.detach() >= TH)
                    # statistics
                    current_loss = loss.item() * inputs.size(0)
                    running_loss += current_loss
                    if phase == 'val':
                        scores, TH = compute_scores_and_th(preds.cpu(), labels.cpu(), TH)
                    else:
                        scores, _ = compute_scores_and_th(preds.cpu(), labels.cpu(), TH)
                    cum_stats.update(float(current_loss), precision=scores, best_th=TH)
                    running_step += 1

                    pbar.set_description(f'phase:{phase}\t L:{cum_stats.loss(): .4f}\t'
                                         f'A:{cum_stats.precision(): .3f}\t')

                wandb.log({f'{phase}_loss_perstep': (cum_stats.loss()),
                           f'{phase}_mAP_perstep':(cum_stats.precision()),
                           'learning_rate': scheduler.get_last_lr()[0]})
            scores, TH = compute_scores_and_th(all_preds, all_actuals)
            wandb.log({f'{phase}_loss': (sum(all_loss)/len(all_loss)),
                       f'{phase}_mAP': (100 * (sum(scores) / len(scores))),
                       'learning_rate': scheduler.get_last_lr()[0]})
        if phase == 'val' and (100 * (sum(scores) / len(scores))) > best_precision:
            best_precision = cum_stats.precision()
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Precision: {:4f}'.format(best_precision))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():

    # parsing args
    set_seed(0)
    args = parser.parse_args()
    wandb.init(project="tresnet",
               name=args.wandb_name)

    wandb.config = {

        "learning_rate": args.lr,

        "train_batch_size": args.batch_size,

        "tresnet_unit_size": args.tresnet_unit_size

    }
    print('get device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}\tmodel type: {args.model_type}\t')

    scaler = torch.cuda.amp.GradScaler()
    dataloaders, data_sizes = get_dataloaders(args=args)

    # setup model
    print('creating model...')
    model = create_model(args).cuda()
    print('done\n')

    print('creat loss, optimizer and scheduler function...')
    # classes_weights = np.load('/home/sara.naserigolestani/classes_weights.npy')
    # tensor_weights = torch.from_numpy(classes_weights)
    # tensor_weights = tensor_weights.to(device, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)


# actual validation process

    if args.load_from_path:
        model.load_state_dict(torch.load(args.save_path))
    print('doing training...')
    best_model = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, scheduler=step_lr_scheduler,
                             device=device, num_epochs=args.num_epochs, scaler=scaler)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    torch.save(model.state_dict(), 'model/final_model_{}'.format(dt_string))


if __name__ == '__main__':
    set_seed(0)
    args = parser.parse_args()
    main()
