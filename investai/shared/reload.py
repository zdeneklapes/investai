# -*- coding: utf-8 -*-
# This function reload the module
def reload_module(module):
    import importlib

    importlib.reload(module)
