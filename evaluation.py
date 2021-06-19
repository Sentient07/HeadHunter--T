#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from collections import defaultdict, OrderedDict
from os import path as osp

import numpy as np
import torch
from cycler import cycler as cy

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm
from tqdm import tqdm

import configparser
import csv
import os
import os.path as osp
import argparse

import numpy as np
from im_utils import compute_centroid



def rint(x):
    return int(round(float(x)))

# In[2]:


#################################
## Notes on the data structure ##
##  1) seq -> list containing dict
##  2) data -> A dict that is an element of list seq
##  3) data -> Keys are 'gt'
##  4) data['gt'] -> Dict containing track ids as keys and BB as values
 
#################################


def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)    
    # iterate through frames
    for i, (prediction, data) in enumerate(tqdm(zip(results, seq))):
        gt = data['gt'] # data is GT data at Nth frame
        ignore = data['ignore']
        gt_ids = []
        gt_boxes = []
        # For the current frame, get ID and box information
        ig_boxes, ig_ids = [], []
        for (i_id, i_val), (gt_id, box) in zip(ignore.items(), gt.items()):
            assert i_id == gt_id
            if i_val > 0:
                ig_ids.append(i_id)
                ig_boxes.append(box)
                continue
            gt_ids.append(gt_id)
            gt_boxes.append(box)


        gt_boxes = np.stack(gt_boxes, axis=0)
        # x1, y1, x2, y2 --> x1, y1, width, height
        gt_centroids = compute_centroid(gt_boxes)
        gt_boxes = np.stack((gt_boxes[:, 0],
                             gt_boxes[:, 1],
                             gt_boxes[:, 2],
                             gt_boxes[:, 3]),
                            axis=1)

        pred_i = prediction['gt']
        pred_ids = []
        pred_boxes = []
        for p_id, box in pred_i.items():
            pred_ids.append(p_id)
            pred_boxes.append(box)
        pred_boxes = np.stack(pred_boxes, axis=0)
        # x1, y1, x2, y2 --> x1, y1, width, height
        pred_boxes = np.stack((pred_boxes[:, 0],
                                pred_boxes[:, 1],
                                pred_boxes[:, 2],
                                pred_boxes[:, 3]),
                                axis=1)
        dmat = mm.distances.iou_matrix(ig_boxes, pred_boxes, max_iou=0.6)
        notnan_ind = np.where(~np.isnan(dmat))[1]
        n_pred_boxes = np.delete(pred_boxes, notnan_ind, axis=0)
        n_pred_ids = np.delete(np.array(pred_ids), notnan_ind)
        distance = mm.distances.iou_matrix(gt_boxes, n_pred_boxes, max_iou=0.6)
        mot_accum.update(
            gt_ids,
            n_pred_ids,
            distance,
            gt_centroids=gt_centroids)

    return mot_accum


def fetch_gt(config, gt_file):
    seqLength = int(config['Sequence']['seqLength'])
    ignore_ar = {}
    boxes = {}
    dets = {}
    total = []
    for i in range(1, seqLength+1):
        boxes[i] = OrderedDict()
        ignore_ar[i] = OrderedDict()
        dets[i] = []

    with open(gt_file, "r") as inf:
        reader = csv.reader(inf, delimiter=',')
        for row in tqdm(reader):
            ignore = 0
            conf_cond = float(row[6]) > 0 and float(row[8]) > 0
            class_cond = int(float(row[7])) == 1
            if not conf_cond or not class_cond:
                ignore = 1
            x1 = float(row[2])
            y1 = float(row[3])
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = float(row[4])
            y2 = float(row[5])
            bb = np.array([x1,y1,x2,y2], dtype=np.float32)
            boxes[int(row[0])][int(row[1])] = bb
            ignore_ar[int(row[0])][int(row[1])] = ignore

    for i in range(1, seqLength+1):
        sample = {'gt':boxes[i],
                  'ignore':ignore_ar[i]}
        total.append(sample)
    return total

def fetch_predictions(config, pred_file):
    seqLength = int(config['Sequence']['seqLength'])
    visibility = {}
    boxes = {}
    dets = {}
    total = []
    for i in range(1, seqLength+1):
        boxes[i] = {}
        visibility[i] = {}
        dets[i] = []
        
    with open(pred_file, "r") as inf:
        reader = csv.reader(inf, delimiter=',')
        for row in tqdm(reader):
            # class person, certainity 1, visibility >= 0.25
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = float(row[2])
            y1 = float(row[3])
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = float(row[4])
            y2 = float(row[5])
            bb = np.array([x1,y1,x2,y2], dtype=np.float32)
            boxes[int(row[0])][int(row[1])] = bb
            visibility[int(row[0])][int(row[1])] = float(row[8])
                
    for i in range(1, seqLength+1):
        sample = {'gt':boxes[i],
                  'vis':visibility[i]}
        total.append(sample)
    return total


def evaluate_mot_accums(accums, names=['Final'], generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums, 
        metrics=mm.metrics.motchallenge_metrics, 
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)
    return str_summary, summary

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--gt_dir', type=str, help='path to GT directory')
    parser.add_argument('--pred_dir', type=str, help='Path to prediction directory')
    parser.add_argument('--dataset', type=str, help='should I skip any seq?')

    args = parser.parse_args()

    if args.dataset == 'train':
        datasets = [osp.join(args.gt_dir, 'train')]

    elif args.dataset == 'mot':
        datasets = [osp.join(i, j) for i in next(os.walk(args.gt_dir))[1] \
                    for j in os.listdir(osp.join(args.gt_dir, i)) if 'MOT' in j]
    else:
        all_seq = os.listdir(osp.join(args.gt_dir, args.dataset))
        datasets = [osp.join(args.dataset, i) for i in all_seq]

    results_file = osp.join(args.pred_dir, 'results.txt')
    print("Saving results to " + str(results_file))
    summary_dict = defaultdict(list)
    std_metrics = ['mota', 'idf1', 'ideucl', 'mostly_tracked', 'mostly_lost',
                    'num_false_positives', 'num_misses', 'num_switches']
    with open(results_file, 'w') as rf:
        # Construct paths
        for dset in datasets:
            gt_dir = osp.join(args.gt_dir, dset)
            pred_dir = osp.join(args.pred_dir, dset)
            mot_seq = sorted(os.listdir(gt_dir))


            seq_file = osp.join(gt_dir, 'seqinfo.ini')
            gt_file = osp.join(gt_dir, 'gt', 'gt.txt')
            config = configparser.ConfigParser()
            config.read(seq_file)
            seq_name = config['Sequence']['name']
            pred_file = osp.join(pred_dir, 'pred.txt')
            print("Results for sequence : " + seq_name + " is : ")
            gt_list = fetch_gt(config, gt_file)
            pred_list = fetch_predictions(config, pred_file)
            str_summary, summary = evaluate_mot_accums([get_mot_accum(pred_list, gt_list)])
            for key in std_metrics:
                summary_dict[key].append(summary[key])
            rf.write(str_summary)
    print('\n')
    print([(k, np.mean(v)) for k, v in summary_dict.items()])
