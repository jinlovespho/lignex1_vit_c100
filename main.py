import argparse

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image 
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
import torchsummary
from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
import wandb

from torchprofile import profile_macs 
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", default=False, help="API Key for WandB")
parser.add_argument("--dataset", default="c100", type=str, help="[c10, c100, svhn]")
parser.add_argument('--dataset_path', type=str)
parser.add_argument("--num-classes", default=100, type=int)
parser.add_argument("--model-name", type=str, )
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=256, type=int)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--min-lr", default=5e-6, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=400, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--warmup-epoch", default=10, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="ViT-small-method4_c100")
parser.add_argument("--exp-name", type=str)
parser.add_argument("--vit-type", type=str)
parser.add_argument("--head-mix-method", type=int)
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None
        
        self.train_throughput=[]
        self.val_throughput=[]
        

    def forward(self, x):
        
        # macs = profile_macs(self.model, x)
        # macs2, params2 = profile(self.model, inputs=(x, ))
        # print(macs/1e9)
        # print(macs2/1e9)
        
        # flops = FlopCountAnalysis(self.model, x)
        # print(flops.total()/1e9)
        # # print(flop_count_table(flops))
        # breakpoint()
        
        
        # patch size 4
        # image size 32 
        
        # print('macs: ', macs/1e9)
        # print('macs2: ', macs2/1e9)
        # print('flops: ', flops.total()/1e9)


        # breakpoint()
        
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
                    
            # Training Throughput
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            out = self.model(img)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            self.train_throughput.append(curr_time)
                
            # out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        # save_image(img, 'img_t.jpg', normalize=True)

        return loss

    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        
        img, label = batch
        
        # Validation Throughput
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        out = self.model(img)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        self.val_throughput.append(curr_time)
        
        
        # out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # save_image(img, 'img_v.jpg', normalize=True)

        return loss
    
    def validation_epoch_end(self, outputs):
        # print('validation')
        
        # n1=np.array(self.train_throughput)
        # n2=np.array(self.val_throughput)
        # breakpoint()
        # self.train_throughput.clear()
        # self.val_throughput.clear()
        
        
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)
    
    def _log_image(self, image):
        pass
        # grid = torchvision.utils.make_grid(image, nrow=4)
        # self.logger.experiment.log_image(grid.permute(1,2,0))
        # print("[INFO] LOG IMAGE!!!")

if __name__ == "__main__":
    args.experiment_name = get_experiment_name(args)
    if args.api_key:
        print("[INFO] Log with wandb!")
        logger = pl.loggers.WandbLogger(
            save_dir="logs",
            project=args.project_name,
            name=args.experiment_name,
            config=args
        )
        refresh_rate = 0
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(
            save_dir="logs",
            name=args.experiment_name
        )
        refresh_rate = 1
    net = Net(args)

    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, weights_summary="full", progress_bar_refresh_rate=refresh_rate, num_sanity_val_steps=0)
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
