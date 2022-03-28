import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig


from src.models.models import get_model
from src.datasets import datasets
import src.utils.hydra_utils as hydra_utils
from src.utils.common import override_weights


log = hydra_utils.get_logger(__name__)


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):

    if "seed" in config.trainer:
        seed_everything(config.trainer.seed)

    hydra_utils.extras(config)
    hydra_utils.print_config(config, resolve=True)


    # setup dataset
    log.info("initializing dataset")
    train_dataset = datasets.get_dataset(config, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.num_workers,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None
    )
    val_dataset = datasets.get_dataset(config, "val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None
    )

    # setup model
    log.info("initializing model")
    model_dynamic_cfg = {
        "num_samples": len(train_dataset),
    }
    if hasattr(train_dataset, "dimensions"):
        model_dynamic_cfg = {
            "dimensions": train_dataset.dimensions
        }
    model = get_model(config, **model_dynamic_cfg)

    log.info("setup checkpoint callback")
    # setup checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=None,  # the path is set by hydra in config.yaml TODO: add flexibility?
        monitor="val_loss",  # TODO: add monitor
        save_top_k=50,
        period=1,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if config.trainer.weight_only:
        checkpoint = None
    else:
        checkpoint = config.trainer.checkpoint

    # start training
    trainer = pl.Trainer(
        gpus=config.trainer.gpus,
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=5,
        resume_from_checkpoint=checkpoint,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        precision=16,
    )
    # load pretrained data
    if (config.trainer.checkpoint is not None) and config.trainer.weight_only:
        pretrained_weights = torch.load(
            config.trainer.checkpoint
        )['state_dict']
        override_weights(
            model, pretrained_weights, keys=['decoder']
        )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
