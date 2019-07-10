import sys
import os
from pathlib import Path

from PIL import Image
import numpy as np

sys.path.append('/home/dprasad/notebooks/SPIRAL-tensorflow')
sys.path.append('/home/dprasad/notebooks/SPIRAL-tensorflow/libs/mypaint')
#import envs.mnist

from tqdm import tqdm
import tensorflow as tf
from PIL import Image, ImageDraw
from collections import defaultdict

from lib import surface, tiledsurface, brush
from envs.mypaint_utils import *
import matplotlib.pyplot as plt

'''
Libraries have been installed in:
   /usr/local/lib

If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the '-LLIBDIR'
flag during linking and do at least one of the following:
   - add LIBDIR to the 'LD_LIBRARY_PATH' environment variable
     during execution
   - add LIBDIR to the 'LD_RUN_PATH' environment variable
     during linking
   - use the '-Wl,-rpath -Wl,LIBDIR' linker flag
   - have your system administrator add LIBDIR to '/etc/ld.so.conf'

export PYTHONPATH=PYTHONPATH:/usr/local/lib/mypaint
'''

def test_simple():
    from config import get_args
    args = get_args(group_name='environment')
    from envs import Simple
    print(args.conditional)
    print(args.episode_length)
    print(args)
    env = Simple(args)

    for ep_idx in range(10):
        step = 0
        env.reset()

        while True:
            action = env.random_action()
            print("[Step {}] ac: {}".format(step, action))
            state, reward, terminal, info = env.step(action)
            step += 1
            
            if terminal:
                print("Ep #{} finished.".format(ep_idx))
                env.save_image("simple{}.png".format(ep_idx))
                break



if __name__=="__main__":
    import utils as ut
    from config import get_args
    from envs import MNIST, SimpleMNIST
    args = get_args()
    ut.train.set_global_seed(args.seed)

    env = args.env.lower()
    env = 'mnist'
    if env == 'mnist':
        env = MNIST(args)
    elif env == 'simple_mnist':
        env = SimpleMNIST(args)
    else:
        raise Exception("Unkown environment: {}".format(args.env))

    for ep_idx in range(10):
        step = 0
        env.reset()

        while True:
            action = env.random_action()
            print("[Step {}] ac: {}".format(
                    step, env.get_action_desc(action)))
            state, reward, terminal, info = env.step(action)
            env.save_image("mnist{}_{}.png".format(ep_idx, step))
            
            if terminal:
                print("Ep #{} finished ==> Reward: {}".format(ep_idx, reward))
                break

            step += 1
