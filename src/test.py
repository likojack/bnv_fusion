import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig

from src.models.models import MODELS
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
    test_dataset = datasets.get_dataset(config, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.eval_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None
    )

    # setup model
    log.info("initializing model")
    model = MODELS[config.model.name].load_from_checkpoint(
        config.trainer.checkpoint,
        **{
            "cfg": config
        }
    )

    # start training
    trainer = pl.Trainer(
        gpus=config.trainer.gpus,
    )
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
