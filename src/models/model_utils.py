from src.utils.import_utils import import_from


def set_optimizer_and_lr(cfg, parameters):
    """

    Args:
        config
        parameters to be optimized

    Return:
        A tuple of an optimizer and a learning rate dict
        a learning rate dict:
            {
                'scheduler': lr_scheduler, # The LR scheduler instance (required)
                'interval': 'epoch', # The unit of the scheduler's step size
                'frequency': 1, # The frequency of the scheduler
                'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
                'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
                'strict': True, # Whether to crash the training if `monitor` is not found
                'name': None, # Custom name for LearningRateMonitor to use
            }
    """

    optimizer = import_from(
        module=cfg.optimizer._target_package_,
        name=cfg.optimizer._class_
    )(parameters, lr=cfg.optimizer.lr.initial)
    lr_scheduler = import_from(
        module="torch.optim.lr_scheduler",
        name=cfg.optimizer.lr.scheduler
    )(
        optimizer=optimizer,
        **cfg.optimizer.lr_scheduler
    )

    return optimizer, lr_scheduler
