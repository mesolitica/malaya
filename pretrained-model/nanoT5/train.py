from omegaconf import open_dict
import hydra
import torch
import time
import transformers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)
from glob import glob
import os


class Module(LightningModule):
    def __init__(self, args, config):
        super().__init__()
        self.model = get_model(args, config)
        if args.model.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()
        if args.model.flash_attention:
            from optimum.bettertransformer import BetterTransformer
            self.model = BetterTransformer.transform(self.model)
        if args.model.compile:
            self.model = torch.compile(self.model)
        self.args = args

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.model, self.args)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.args)
        scheduler = {"scheduler": self.lr_scheduler, "interval": "step", "frequency": 1}
        return (
            [self.optimizer],
            [scheduler],
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "Losses/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        total_norm = 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log(
            "total_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    wandb_logger = WandbLogger(project=os.environ.get('WANDB_PROJECT', 'lightning_project'))
    config = get_config(args)
    model = Module(args, config)
    tokenizer = get_tokenizer(args)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='step',
        mode='max',
        dirpath=args.model_directory,
        every_n_train_steps=args.checkpoint.every_steps,
        filename='model-{epoch:02d}-{step}',
    )
    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        max_steps=args.optim.total_steps,
        gradient_clip_val=args.optim.grad_clip,
        accelerator='gpu',
        devices=num_gpus,
        limit_val_batches=100,
        precision=args.precision,
        log_every_n_steps=5,
        accumulate_grad_batches=args.optim.grad_acc,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        strategy='ddp',
    )

    checkpoints = glob(os.path.join(args.model_directory, 'model-*'))
    if len(checkpoints):
        checkpoint = sorted(x, key=lambda x: int(x.split('-')[-1]), reverse=True)[0]
    else:
        checkpoint = None

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=None,
        ckpt_path=checkpoint,
    )


if __name__ == "__main__":
    main()
