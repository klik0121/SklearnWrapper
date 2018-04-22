from abc import ABC

class MethodWrapper(object):
    """Abstract class for wrapping one of the sklearn methods"""
    wrappers = {}

    def __init__(self):
        self.animation_delay = -1
    
    def set_animation_delay(self, value:str):
        self.animation_delay = float(value)

    @classmethod
    def __init_subclass__(cls, name:str, **kwargs):
        cls.wrappers[name] = cls

