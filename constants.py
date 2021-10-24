import os
import sys


gettrace = getattr(sys, 'gettrace', None)
EPSILON = 1e-6
DEBUG = gettrace is not None and gettrace() is not None
if DEBUG:
    print('Debugging')

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
RAW_MESHES = f'{DATA_ROOT}/meshes/'
OUT = f'{DATA_ROOT}/out/'