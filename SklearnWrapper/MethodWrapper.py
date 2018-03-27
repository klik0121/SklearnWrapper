from abc import ABC

class MethodWrapper(object):
    """Abstract class for wrapping one of the sklearn methods"""
    wrappers = {}

    @classmethod
    def __init_subclass__(cls, name:str, **kwargs):
        cls.wrappers[name] = cls

