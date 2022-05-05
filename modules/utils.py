#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: utils.py
@time:2022/05/04
@description:
"""


def register(name=None, registry=None):
    """
    将某个函数获这某个类注册到某各地方，装饰器函数
    :param name: 注册的函数别名
    :param registry: 注册保存的对象
    :return: registered fun
    """

    def decorator(fn, registration_name=None):
        module_name = registration_name or fn.__name__
        if module_name in registry:
            raise LookupError(f"module {module_name} already registered.")
        registry[module_name] = fn
        return fn

    return lambda fn: decorator(fn, name)
