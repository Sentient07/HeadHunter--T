import numpy as np; import csv; from glob import glob; import os; import sys; import imageio 
import matplotlib.pyplot as plt; from PIL import Image
import cv2
import random
import itertools
import h5py
import colorsys
from collections import Counter
from scipy.misc import imread, imsave
from skimage.io import imread as skimread
from skimage.transform import resize
from skimage.measure import compare_ssim, compare_psnr
import base64
import pycocotools.mask as cocomask
import os.path as osp
from itertools import permutations, combinations_with_replacement


#fetch images
def fetch_ims(im_path, ext="*.png"):
    return sorted(glob(osp.join(im_path, ext)))

def fetch_allpath(base_pth, ext_pth, extension="*.png", drop=False):
    final_list = []
    for j in ext_pth:
        cur_pth = osp.join(base_pth, j)
        cur_list = fetch_ims(cur_pth, extension)
        if drop:
            cur_list.pop(0)
        final_list.extend(cur_list)
    return final_list


def constraint_boxes(detections, im_shape):
    (startX, startY, endX, endY, score) = detections
    startX, startY = max(0, startX), max(0, startY)
    max_x, max_y = im_shape[1], im_shape[0]
    endX, endY = min(endX, max_x-1), min(endY, max_y-1)
    detections = (startX, startY, endX, endY, score)
    return detections


def check_area(detections, im_shape):
    if len(detections) < 1:
        raise ValueError("Invalid detection length")
    (startX, startY, endX, endY) = detections[:4]
    box_width = (endX-startX)
    box_height = (endY-startY)
    area =  box_width * box_height
    if area < 10 or box_height < 2 or box_width < 2:
        return False
    return True


def get_area(boxes):
    """
    Area of BB
    """
    boxes = np.array(boxes)
    if len(boxes.shape) != 2:
        area = np.product(boxes[2:4] - boxes[0:2])
    else:
        area = np.product(boxes[:, 2:4] - boxes[:, 0:2], axis=1)
    return area


def get_refined_detection(detections, im_shape, conf):
    """
    Constraint the output of detector to lie within the image.
    Also check if the detection is valid by measuring area of BB.

    detections : [[x_min, y_min, x_max, y_max, score]]
    im_shape : (H, W, 3)
    """
    refined_detection = []
    for dets in detections:
        score = dets[-1]
        if score<conf:
            continue
        dets = constraint_boxes(dets, im_shape)
        if check_area(dets, im_shape) is False:
            continue
        refined_detection.append(dets)
    refined_detection = np.array(refined_detection)
    return refined_detection


def is_outside(detections, im_shape, thresh=0):
    dets = [int(i) for i in detections]
    startX, startY, endX, endY = [dets[i] for i in range(0, 4)]
    if startX-thresh<0 or startY-thresh<0 or endX+thresh>=im_shape[1] or endY+thresh>=im_shape[0]:
        return True


def coord_shift(std_coord):
    """
    Convert (x_min, y_min, x_max, y_max) to 
    (x_centroid, y_centroid, aspect_ratio, height)
    """
    std_coord = np.array(std_coord)
    x_c, y_c = compute_centroid(std_coord)
    height = std_coord[3] - std_coord[1]
    width = std_coord[2] - std_coord[0]
    aspect_ratio = height/width
    return np.array([x_c, y_c, aspect_ratio, height])

def inverse_coord_shift(kalman_coord):
    """
    opposite of `def coord_shift`
    """
    x_c, y_c, aspect_ratio, height = kalman_coord[:4]
    width = height / aspect_ratio
    x_min = x_c - width / 2
    x_max = x_c + width / 2
    y_min = y_c - height / 2
    y_max = y_c + height / 2
    std_coord = np.array([x_min, y_min, x_max, y_max])
    return np.int_(np.round(std_coord))


def box_from_centroid(boxes):
    """
    Convert (x_centroid, y_centroid, width, height) to 
    (x_min, y_min, x_max, y_max)
    """
    boxes = np.array(boxes)
    if len(boxes.shape) > 1:
        mins = boxes[:, 0:2] - boxes[:, 2:4]//2
        maxes = boxes[:, 0:2] + boxes[:, 2:4]//2
        boxes = np.c_[mins, maxes]
    else:
        mins = boxes[0:2] - boxes[2:4]//2
        maxes = boxes[0:2] + boxes[2:4]//2
        boxes = np.r_[mins, maxes]
    return boxes


def compute_centroid(box):
    box = np.array(box)
    if len(box.shape) == 1:
        centroid = (box[0:2] + box[2:4])/2
    else:
        centroid = (box[:, 0:2] + box[:, 2:4])/2
    return centroid


def plot_boxes(cur_frame, head_map, body_map={}, text=True):
    plotting_im = cur_frame.copy()
    for index, (t_id, t_dim) in enumerate(head_map.items()):
        (startX, startY, endX, endY) = [int(i) for i in t_dim]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cur_centroid = tuple([(startX+endX)//2,
                              (startY+endY)//2])
        if text:
            cv2.putText(plotting_im, str(t_id), cur_centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for index, (t_id, t_dim) in enumerate(body_map.items()):
        (startX, startY, endX, endY) = [int(i) for i in t_dim]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
    return plotting_im


def scatter_particles(box, cent_disp=(-3,0,3), max_particles=100):
    """
    This method is used to initialise bounding boxes around initial detection for particles.
    """
    box = np.array(box)
    init_val = [i for i in combinations_with_replacement(list(cent_disp), 4)]
    disp_list = []
    for k in init_val:
        for i in permutations(k):
            disp_list.append(i)
    unique_disp_list = np.unique(np.array(disp_list), axis=0)
    scattered_particles = box + unique_disp_list
    t_particles = []
    for part in scattered_particles:
        if np.any((part[2:4] - part[0:2])<=0):
            continue
        t_particles.append(part)

    if len(t_particles) > max_particles:
        overflow = len(t_particles) - max_particles
        t_particles = t_particles[overflow//2:overflow//2+max_particles]
    return np.array(t_particles).astype(np.float32)


def get_neighbour_loc(box, p_noise, min_hw=10, h_variance=0.25,
                      ar_range=(0.7, 1.4), max_particles=100):
    """

    ARCHIVED!!! Test code not using anymore!

    box : (x_min, y_min, x_max, y_max)
    noise : `int`. initial position uncertainity
    vel_noise : Uncertainity in velocity estimation. 
                TODO : remove hard coding
    min_hw : Don't allow boxes to be smaller than this
    h_variance : amount by which h varies. in decimal
                 TODO : remove hardcoding
    ar_range : Aspect ratio range

    """
    box = np.array(box)
    final_box = []
    x_c, y_c = compute_centroid(box)
    h,w = box[2:4] - box[0:2]

    h_min = max(min_hw, h-h*h_variance)
    h_max = h+h*h_variance
    p_noise = int(p_noise)
    aspect_ratio = w/h
    aspect_ratio_range = np.linspace(ar_range[0]*aspect_ratio,
                                     ar_range[1]*aspect_ratio, 4)
    centroid_range = np.linspace(-p_noise, p_noise+1, 4).astype(np.int32)
    h_range = np.linspace(int(h_min), int(h_max+1), 4).astype(np.int32)
    
    for i in centroid_range:
        for j in centroid_range:
            for cur_h in h_range:
                for ar in aspect_ratio_range:
                    cur_w = cur_h * ar
                    if cur_h<min_hw or cur_w<min_hw:
                        continue
                    c_box = box_from_centroid([x_c+i, y_c+j, cur_w, cur_h])
                    final_box.append(c_box)
    final_box = np.array(final_box)
    if len(final_box) > max_particles:
        idx = np.round(np.linspace(0, len(final_box) - 1, max_particles)).astype(int)
        final_box = final_box[idx]
    # If boxes are really small, just add the original one and return
    if len(final_box) < 1:
        np.append(final_box, box)
    return final_box


def matrix_histcmp(vec1, vec2):
    """
    vec1 : shape : M
    vec2 : shape : N
    """
    hist_dist = []
    for h1 in vec1:
        h1_dist = []
        for h2 in vec2:
            h1_dist.append(cv2.compareHist(h1, h2,
                           cv2.HISTCMP_BHATTACHARYYA))
        hist_dist.append(h1_dist)
    return np.array(hist_dist)

def compute_new_hsv(im):
    """
    Illuminance and Gamma invariant HSV
    """
    eps = 1e-10
    r,g,b = np.array(cv2.split(im)) + eps
    traditional_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    numerator = np.log(r) - np.log(g)
    denominator = np.log(r) + np.log(g) - 2*np.log(b) + eps
    new_hue = np.clip(np.round(numerator/denominator).astype(np.uint8), 0, 180)
    new_hsv = np.zeros_like(traditional_hsv).astype(np.uint8)
    new_hsv[:, :, 0] = new_hue
    new_hsv[:, :, 1] = traditional_hsv[:, :, 1]
    new_hsv[:, :, 2] = traditional_hsv[:, :, 2]
    return new_hsv

def compute_histogram(im, kernel=True):
    """
    im : cropped patch
    kernel : Will perform circular masking 
    """
    x,y = im.shape[:2]
    if kernel:
        mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(y, x))
        mask = mask[:, :, np.newaxis]
    else:
        mask = np.ones((x,y,1)).astype(np.uint8)
    hsv_im = compute_new_hsv(im)
    channels = [0,1,2]
    hist_size = [8, 8, 8]
    hist_range = [0, 180, 0, 256, 0, 256]
    # hist_hue = cv2.calcHist([hsv_im], [0],
    #                         mask, [30], [0, 180], False)
    # cv2.normalize(hist_hue,hist_hue,0,255,cv2.NORM_MINMAX)
    hist = cv2.calcHist([hsv_im], channels,
                     None, hist_size, hist_range)
    image_hist = cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    return image_hist



def check_yposition(public_box, head_box):
    """
    Check if Public box has Y coordinate > head box's
    """
    # Higher Y centroid condition
    h_centroid = compute_centroid(head_box)
    box_centroid = compute_centroid(public_box)
    if box_centroid[1] > h_centroid[1]:
        return True
    return False


def warp_pos(pos, warp_matrix):
    """
    Warping position for camera motion compensation
    """
    import torch
    p1 = torch.Tensor([pos[0], pos[1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[2], pos[3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1)
