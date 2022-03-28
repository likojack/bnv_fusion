from importlib import import_module


def import_from(module, name):
    """ import "name" from "module" 
    """

    return getattr(import_module(module), name)
