#!/usr/bin/env python
# coding: utf-8
import numpy as np; import csv; from glob import glob; import os; import sys; 
from shutil import move; from tqdm import tqdm; import os.path as osp  
import matplotlib.pyplot as plt; from PIL import Image 
import cv2 
from scipy.misc import imread, imsave 
from skimage.io import imread as skimread 
import random 
import itertools 
import h5py
import matplotlib.pyplot as plt 
from skimage.measure import compare_ssim, compare_psnr
from skimage.util import img_as_ubyte as imby
from skimage import color
import colorsys 
from collections import OrderedDict, Counter 
from itertools import islice 

np.random.seed(seed=12345)

import configparser

# Tracker specific
from config.retina_cfg import cfg_re150 as retina_cfg
from config.det_cfg import det_cfg
from obj_detect import objDetect
from flow_tracker import Tracker
import argparse
# create input

parser = argparse.ArgumentParser(description='Testing the tracking')
parser.add_argument('--base_dir',
                    required=True,
                    type=str, help='Base directory for the dataset')
parser.add_argument('--save_dir',
                    required=True,
                    type=str, help='path to save the results')
parser.add_argument('--frame_dim', '--image-dimensions', default=(1080, 1920),
                    type=tuple, help='Training Image dimensions')


args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
all_results = []
traj_fname = osp.join(args.save_dir + "_trajectory.txt")


def create_realpair(real_test, file_len=None):
    img_pairs = []
    if not file_len:
        file_len = len(sorted(glob(osp.join(real_test, "*.jpg"))))
    for pair in range(1, file_len):
        imname1 = osp.join(real_test, "{:06d}".format(pair)+'.jpg')
        imname2 = osp.join(real_test, "{:06d}".format(pair+1)+'.jpg')
        img_pairs.append((imname1, imname2))
    return img_pairs


frame_shape = (args.frame_dim[0], args.frame_dim[1], 3)
frame_pair = create_realpair(args.base_dir)
print("Total length is " + str(len(frame_pair)))
objdetect = objDetect(retina_cfg, det_cfg)
tracker_cfg = {}
tracker_cfg['im_shape'] = frame_shape
tracker_cfg['inactive_patience'] = 30

tracker = Tracker(objdetect, tracker_cfg, save_dir=args.save_dir,
                 use_public=False, strict_public=False, only_public=False,
                 save_frames=True)

for im0, im1 in tqdm(frame_pair):
    cur_im = imread(im0)
    tracker.step(cur_im)
    
cur_result = tracker.get_results()
with open(traj_fname, "w+") as of:
    writer = csv.writer(of, delimiter=',')
    for i, track in cur_result.items():
        for frame, bb in track.items():
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            writer.writerow([frame, i+1, x1+1, y1+1, x2-x1+1,
                             y2-y1+1, -1, -1, -1, -1])