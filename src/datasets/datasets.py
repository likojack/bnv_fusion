datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def get_dataset(cfg, subset):
    dataset = datasets[cfg.dataset.name](cfg, subset)
    return dataset
