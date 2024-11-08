"""
dinamic import of all projects classes into the current directory
ex.: from package_name import Class1, Class2

without the below code, we would have to do a manual import
ex.: from package_name.module1 import Class1

Author: Leonardo
Date: 2024-11-07
"""
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)