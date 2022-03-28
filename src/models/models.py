MODELS = {}


def register(name):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator


def get_model(cfg, **kwargs):
    model = MODELS[cfg.model.name](cfg, **kwargs)
    return model
