#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import csv
import logging
import os
import os.path as osp
from glob import glob

import numpy as np
import yaml
from matplotlib.pyplot import imread
from tqdm import tqdm

from head_detection.data import cfg_mnet, cfg_res50_4fpn, cfg_res152
from obj_detect import HeadHunter
from tracker import Tracker

from collections import defaultdict

np.random.seed(seed=12345)

Parser = argparse.ArgumentParser(description='Testing the tracker on MOT style')
Parser.add_argument('--base_dir',
                    required=True, 
                    type=str, help='Base directory for the dataset')
Parser.add_argument('--save_path',
                    required=True,
                    type=str, help='Directory to save the results')
Parser.add_argument('--cfg_file', 
                    default='./config/config.yaml',
                    type=str, help='path to config file')
Parser.add_argument('--start_ind', 
                    default=0,
                    type=int, help='should I skip any seq?')
Parser.add_argument('--save_frames', 
                    default=False,
                    type=bool, help='should I save frames?')
Parser.add_argument('--dataset', 
                    default='all',
                    type=str, help='Train/Test/All')

Parser.add_argument('--detector', 
                    default='det',
                    type=str, help='Directory where public detection are saved')


args = Parser.parse_args()
log = logging.getLogger('Head Tracker on MOT style data')
log.setLevel(logging.DEBUG)


# Get parameters from Config file
with open(args.cfg_file, 'r') as stream:
    CONFIG = yaml.safe_load(stream)
det_cfg = CONFIG['DET']['det_cfg']
backbone = CONFIG['DET']['backbone']
tracktor_cfg = CONFIG['TRACKTOR']
motion_cfg = CONFIG['MOTION']
tracker_cfg = CONFIG['TRACKER']
gen_cfg = CONFIG['GEN']
# is_save = gen_cfg['save_frames']

# Initialise network configurations
if backbone == 'resnet50':
    net_cfg = cfg_res50_4fpn
elif backbone == 'resnet152':
    net_cfg = cfg_res152
elif backbone == 'mobilenet':
    net_cfg = cfg_mnet
else:
    raise ValueError("Invalid Backbone")


def read_public_det(det):
    det_dict = defaultdict(list)
    with open(det, 'r') as dfile:
        for i in dfile.readlines():
            cur_det = [float(k) for k in i.strip('\n').split(',')]
            det_dict[int(cur_det[0])].append([cur_det[2],
                                              cur_det[3],
                                              cur_det[4]+cur_det[2],
                                              cur_det[3]+cur_det[5],
                                              cur_det[6]/100.])
    return det_dict


# Get sequences of MOT Dataset
datasets = ('HT-train', 'HT-test') if args.dataset == 'all' else (args.dataset, )
for dset in datasets:
    mot_dir = osp.join(args.base_dir, dset)
    mot_seq = os.listdir(mot_dir)[args.start_ind:]
    mot_paths = sorted([osp.join(mot_dir, seq) for seq in mot_seq])

    # Create the required saving directories
    if args.save_frames:
        save_paths = [osp.join(args.save_path, seq) for seq in mot_seq]
        _ = [os.makedirs(i, exist_ok=True) for i in save_paths]
        assert len(mot_paths) == len(save_paths)

    all_results = []


    for ind, mot_p in enumerate(tqdm(mot_paths)):
        seqfile = osp.join(mot_p, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seqfile)
        c_width = int(config['Sequence']['imWidth'])
        c_height = int(config['Sequence']['imHeight'])
        seq_length = int(config['Sequence']['seqLength'])
        seq_ext = config['Sequence']['imExt']
        seq_dir = config['Sequence']['imDir']
        cam_motion = bool(config['Sequence'].get('cam_motion', False))
        seq_name = config['Sequence']['name']
        traj_dir = osp.join(args.save_path, dset, mot_seq[ind])
        os.makedirs(traj_dir, exist_ok=True)
        traj_fname = osp.join(traj_dir, 'pred.txt')
        log.info("Total length is " + str(seq_length))
        im_shape = (c_height, c_width, 3)
        im_path = osp.join(mot_p, seq_dir)
        seq_images = sorted(glob(osp.join(im_path, '*'+seq_ext)))
        # Create detector and traktor
        detector = HeadHunter(net_cfg, det_cfg, im_shape, im_path).cuda()
        save_dir = save_paths[ind] if args.save_frames else None

        # Read Public detection if necessary
        if tracker_cfg['use_public'] is True:
            print("using " + args.detector)
            det_file = args.detector + '.txt'
            det_dict = read_public_det(osp.join(mot_p, 'det', det_file))
        tracker = Tracker(detector, tracker_cfg, tracktor_cfg, motion_cfg, im_shape,
                        save_dir=save_dir,
                        save_frames=args.save_frames, cam_motion=cam_motion,
                        public_detections=det_dict)

        for im0 in tqdm(seq_images):
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
                    writer.writerow([frame, i, x1, y1, x2-x1+1,
                                    y2-y1+1, 1, 1, 1, 1])
