import os
import os.path as osp
import logging
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class DatasetBase(Dataset):
    """ Abstract base dataset class
    """
    def __init__(self, cfg, subset):
        self.dataset_dir = cfg.dataset.data_dir
        self.dataset_name = cfg.dataset.name
        if cfg.dataset.categories is None:
            categories = os.listdir(cfg.dataset.data_dir)
            categories = [c for c in categories
                          if osp.isdir(osp.join(cfg.dataset.data_dir, c))]
        else:
            categories = cfg.dataset.categories

        self.file_list = []
        for c_idx, c in enumerate(categories):
            subpath = osp.join(cfg.dataset.data_dir, "splits", c)
            if not osp.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = osp.join(subpath, subset + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.file_list += [
                {'category': c, 'model': m}
                for m in models_c
            ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self):
        raise NotImplementedError()
