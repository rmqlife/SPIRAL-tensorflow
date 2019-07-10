# from __future__ import print_functions
from colorenv import ColorEnv, PaintMode

import os
from os.path import join, basename
import numpy as np
import sys

sys.path.append('/home/dprasad/notebooks/SPIRAL-tensorflow')
sys.path.append('/home/dprasad/notebooks/SPIRAL-tensorflow/libs/mypaint')
from lib import surface, tiledsurface, brush
import cv2

def test_canvas_size():
    rect = [0, 0, 100, 200]
    s = tiledsurface.Surface()
    s.flood_fill(0, 0, (255, 255, 255), (0, 0, 100, 200), 0, s)
    s.begin_atomic()
    strips = next(surface.scanline_strips_iter(s, rect))
    print(next(strips).shape)

class args:
  jump=False
  curve=True
  screen_size=64
  location_size=64
  color_channel=3
  brush_path='assets/brushes/dry_brush.myb'
  data_dir='data/train_brushes'
  stroke_number=100

def single_strokes(args):
    env=ColorEnv(args, paint_mode=PaintMode.STROKES_ONLY)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    
    ims = list()
    for i in range(args.stroke_number):
        env.reset()    
        action = actions[i]
        
        transfered_action = np.zeros(12)
        transfered_action[:6] = action[:6]
        transfered_action[-3:] = action[-3:]
        
        env.draw(transfered_action)
        print(transfered_action)
        im = env.image.astype('uint8')
        if i<10:    
            filename = "{}_{}.png".format(brush_name, i)
            cv2.imwrite(join(args.data_dir, filename), im)
            print(filename)
        else:
            print(i)
        ims.append(im)
    np.savez(join(args.data_dir,'strokes'), images = np.array(ims), actions = actions)

        
if __name__=="__main__":
    from util import get_filelist
    brush_paths = get_filelist('assets/train_brushes', ext=['myb'])
    
    actions = np.random.uniform(size=(args.stroke_number,12))
    for brush_path in brush_paths:
        brush_name = basename(brush_path).split('.')[0]
        args.brush_path = brush_path
        args.data_dir = 'data/train_brushes/'
        single_strokes(args)
    
