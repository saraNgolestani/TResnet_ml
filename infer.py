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
from src.models.utils.score_utils import Statistics, compute_scores
from src.models.utils.dataset_utils import get_dataloaders


# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--remove_aa_jit', action='store_true', default=False)


def train_model(model, dataloaders, criterion, optimizer, scheduler, device,  num_epochs=100, scaler=None):

    print('start training')
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_precision = 0.0
    global_step = 0
    save_path = 'model/Tresnet_M_single_imagenet.pt'
    TH = 0.5

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        cum_stats = Statistics()
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_step = 0
            pbar = tqdm.tqdm(dataloaders[phase], desc=f'phase:{phase}')
            # Iterate over data.
            # temp_sigmoid = F.sigmoid
            for inputs, labels in pbar:
                global_step += 1
                if (global_step%1) == 0:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.float)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                            predictions = model(inputs)
                            pred_loss = criterion(predictions, labels)
                            loss = pred_loss
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
                    scores = compute_scores(preds.cpu(), labels.cpu())
                    cum_stats.update(float(current_loss), *scores)

                    running_step += 1

                    pbar.set_description(f'phase:{phase}\t L:{cum_stats.loss(): .4f}\t'
                                         f'A:{cum_stats.precision(): .3f}\t F1:{cum_stats.f1(): .4f}%')

            if phase == 'val' and cum_stats.precision() > best_precision:
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
    args = parser.parse_args()
    print('get device')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    # setup model
    print('creating model...')
    model = create_model(args).cuda()
    print('done\n')

    # setup data loader
    print('creating data loader...')
    dataloaders, dataset_sizes = get_dataloaders()
    print('done\n')

    print('creat loss, optimizer and scheduler function...')
    classes_weights = np.load('/home/sara.naserigolestani/classes_weights.npy')
    tensor_weights = torch.from_numpy(classes_weights)
    tensor_weights = tensor_weights.to(device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


# actual validation process
    print('doing training...')
    best_model = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, scheduler=step_lr_scheduler,
                             device=device, num_epochs=args.num_epochs, scaler=scaler)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    torch.save(model.state_dict(), 'model/final_model_{}'.format(dt_string))


if __name__ == '__main__':
    main()
