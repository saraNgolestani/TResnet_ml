
from src.models import create_model
import argparse
from src.models.utils.dataset_utils import COCODatasetLightning
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import wandb
import warnings
from pytorch_lightning.callbacks import LearningRateMonitor
warnings.filterwarnings('always')
# torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--remove_aa_jit', action='store_true', default=True)
parser.add_argument('--wandb_name', default='lr_monitor_tresnet_mgpu')

checkpoint_callback = ModelCheckpoint(
    monitor='val acc on epoch',
    dirpath='saved_models',
    filename='model-{epoch:03d}-{val_acc:.2f}',
    save_top_k=2,
    mode='max'
)




def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    args = parser.parse_args()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(project=args.wandb_name, entity="sara_ngln")

    wandb_logger.experiment.config.update({
        "val_zoom_factor": args.val_zoom_factor,
        "batch_size": args.batch_size,
        "num_gpus": 2,
        "num_nodes": args.num_epochs,
    })

    run()
    model = create_model(args)
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], max_epochs=100, num_nodes=1, gpus=2, accelerator='gpu', devices=2)
    train_dl = COCODatasetLightning().train_dataloader()
    val_dl = COCODatasetLightning().val_dataloader()
    trainer.fit(model, train_dl, val_dl)




