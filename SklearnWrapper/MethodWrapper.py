from abc import ABC

class MethodWrapper(object):
    """Abstract class for wrapping one of the sklearn methods"""
    wrappers = {}

    def __init__(self):
        self.animation_delay = -1

    @classmethod
    def __init_subclass__(cls, name:str, **kwargs):
        cls.wrappers[name] = cls
