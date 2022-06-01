
from src.models import create_model
import argparse
from src.models.utils.dataset_utils import COCODatasetLightning
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


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

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='saved_models',
    filename='model-{epoch:03d}-{val_acc:.2f}',
    save_top_k=2,
    mode='max'
)
args = parser.parse_args()
model = create_model(args)
trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=100, num_nodes=1, gpus=4, accelerator='ddp', )
train_dl = COCODatasetLightning().train_dataloader()
val_dl = COCODatasetLightning().val_dataloader()
trainer.fit(model, train_dl, val_dl)




