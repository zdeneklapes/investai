# -*- coding: utf-8 -*-
def reload_module(module):
    """Reloads a module"""
    import importlib

    importlib.reload(module)
