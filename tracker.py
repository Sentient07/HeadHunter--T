# flow_tracker

import os
import sys
from collections import Counter
from glob import glob
from tqdm import tqdm
import numpy as np
import os.path as osp
import cv2
import json
import pycocotools.mask as cocomask
from munkres import Munkres, print_matrix, make_cost_matrix

from track import Track
from im_utils import *
from scipy.misc import imread, imsave
from scipy.spatial.distance import cdist
from copy import deepcopy
from operator import itemgetter
import base64
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision.ops.boxes import clip_boxes_to_image, nms

munkres_inst = Munkres()


class Tracker:
    """The main tracking file, here is where NO magic happens."""
    # only track pedestrian

    def __init__(self, obj_detector, tracker_cfg, tracktor_cfg, motion_cfg,
                 im_shape, save_dir=None, save_frames=False,
                 cam_motion=True, public_detections=None):

        self.obj_detector = obj_detector

        self.inactive_patience = tracker_cfg.get('inactive_patience')
        self.use_reid = tracker_cfg.get('use_reid', True)

        self.im_shape = im_shape

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.results = {}
        self.frame_number = 0
        self.last_image = None
        self.save_frames = save_frames
        self.save_dir = save_dir

        # Tracktor related
        self.regression_thresh = tracktor_cfg['regression_thresh']
        self.detection_confidence = tracktor_cfg['detection_confidence']
        self.detection_nms_thresh = tracktor_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracktor_cfg['regression_nms_thresh']
        self.device = torch.device("cuda")

        # Motion Model
        self.cam_motion = cam_motion
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.number_of_iterations = motion_cfg['n_iterations']
        self.termination_eps = motion_cfg['termination_eps']

        # Reid
        self.lambd = tracker_cfg.get('lambd', 0.9)
        
        #PF Related
        self.n_particles = tracker_cfg['n_particles']
        
        self.use_public = False
        if public_detections:
            print("Using public detection")
            self.public_detections = public_detections
            self.use_public = True


    def reset(self, hard=False):
        self.tracks = []
        self.inactive_tracks = []
        self.frame_number = 0

        if hard:
            self.track_num = 0
            self.results = {}


    def get_track(self, cur_id):
        for tr in self.tracks:
            if tr.id == cur_id:
                return tr

    def get_lost_track(self, cur_id):
        for tr in self.inactive_tracks:
            if tr.id == cur_id:
                return tr

    def compute_crop(self, tr_pos):
        ## Crop the head for gathering HSV features.
        max_y, max_x = self.im_shape[:2]
        xmin, ymin, xmax, ymax = [int(round(i)) for i in tr_pos[:4]]
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(max_x, xmax), min(max_y, ymax)
        cropped_target = self.cur_im[ymin:ymax, xmin:xmax, :]
        return cropped_target

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        self.inactive_tracks += tracks

    def regress_lost_track(self):
        """
        Move tracks that are inactive using PF's CVA
        """
        for lt in self.inactive_tracks:
            lt.step()
        temp_lost = []
        for lt in self.inactive_tracks:
            if not check_area(lt.pos, self.im_shape):
                continue
            if lt.kill_flag:
                continue
            if lt.count_inactive > self.inactive_patience:
                continue
            temp_lost.append(lt)
        self.inactive_tracks = temp_lost

    def track_found_again(self, track):
        # reset inactive count and add to self.track
        self.inactive_tracks = [it for it in self.inactive_tracks if it not in [track]]
        self.tracks.append(track)

    def add(self, new_id_map):
        obj_ids = list(new_id_map.keys())
        assert 0 not in obj_ids # Check if background is being added
        n_dets = len(obj_ids)

        for nd_i, (cur_id, box_loc) in enumerate(new_id_map.items()):
            assert self.get_track(cur_id) is None
            histogram = compute_histogram(self.compute_crop(box_loc))
            # crop_tar = self.compute_crop(box_loc)
            # mask_loc = self.mask_from_box(box_loc)
            n_track = Track(track_id=cur_id, time_stamp=self.frame_number,
                            pos=box_loc,
                            count_inactive=0,
                            inactive_patience=self.inactive_patience,
                            im_shape=self.im_shape, hist_vec=histogram,
                            max_particles=self.n_particles)
            self.tracks.append(n_track)
        self.track_num += n_dets


    def read_publicdet(self):
        """
        Read from a specified detection file.
        """
        detections = np.array(self.public_detections.get(int(self.frame_number)))
        boxes = detections[:, :4].astype(np.int16)
        scores = detections[:, -1].astype(np.float32)
        return np.c_[boxes, scores]


    def filter_regressed_tracks(self, all_score, all_pos, mean_scores, mean_pos):
        """
        Filter the particles based on mean score. whether or not to regress based 
        on mean score. Also checks image coordinate sanity.
        TODO : Cleanup the method to eliminate for loop through particles.
               Rather do it as np.ndarray.
        """
        existing_ids = [t.id for t in self.tracks]
        f_best_scores = []
        f_all_scores = []
        f_all_pos = []
        f_best_pos = []
        filtered_ids = []
        iter_zip = zip(existing_ids, all_score, all_pos, mean_scores, mean_pos)

        for i, (t_i, as_i, apos_i, bs_i, bpos_i) in enumerate(iter_zip):
            # check area of box
            if check_area(bpos_i, self.im_shape) is False:
                continue
            if bs_i > self.regression_thresh:
                f_all_scores.append(as_i)
                f_best_scores.append(bs_i)
                f_best_pos.append(bpos_i)
                filtered_ids.append(t_i)
                f_all_pos.append(apos_i)

        f_best_scores = torch.tensor(f_best_scores,
                                    dtype=torch.float32,
                                    device=self.device)
        f_best_pos = torch.tensor(f_best_pos,
                                    dtype=torch.float32,
                                    device=self.device)
        filtered_ids = np.array(filtered_ids)
        f_all_scores = np.array(f_all_scores)
        f_all_pos = np.array(f_all_pos)

        return f_all_scores, f_all_pos, f_best_scores, f_best_pos, filtered_ids


    def regress_single_track(self, regress_pos):
        t_box, t_score = self.obj_detector.regress_boxes(self.cur_im, regress_pos)
        t_box = clip_boxes_to_image(t_box, self.im_shape[:-1]).cpu().numpy()
        return t_score, t_box

    def regress_particles(self, active_particles, n_particles):
        """Regress the position of the tracks and also checks their scores."""
        t_score, t_box = self.regress_single_track(active_particles)
        # Remove last one as we need everything from penultimate
        split_indices = np.cumsum(n_particles)[:-1]
        # how much to add to each split to get overall argmax
        amax_ind_offest = np.r_[0, split_indices].astype(np.int32)
        # splitting each particles into a list
        # List because each track can have unequal particles
        # and cannot be made into a np.array
        split_scores = np.split(t_score, split_indices)
        split_pos = np.split(t_box, split_indices)
        split_amax = np.array([np.argmax(i) for i in split_scores])
        best_ind = split_amax + amax_ind_offest
        mean_scores = t_score[best_ind]
        mean_pos = t_box[best_ind]
        return split_scores, split_pos, mean_scores, mean_pos


    def track_match(self):
        new_id_map = {}
        regress_matches = {}
        max_id = self.track_num

        prev_ids = [t.id for t in self.tracks]
        inactive_ids = [lt.id for lt in self.inactive_tracks]
        
        # Align the position of existing particles
        if self.cam_motion:
            aligned_particles = self.align_particles([t.get_particles() for t in self.tracks])
            [t.align_particles(a_p) for (t, a_p) in zip(self.tracks, aligned_particles)]

        # Align the positions of inactive track
        for lt in self.inactive_tracks:
            lt.pos = self.align([lt.pos])[0]

        # Check scores of active track
        active_particles = [torch.tensor(t.predict_particles(), dtype=torch.float32,
                            device=self.device) for t in self.tracks]

        t_n_particles = [t.roiPF.created_particles for t in self.tracks]
        t_active_particles = torch.cat(active_particles, axis=0)
        
        all_scores, all_pos, mean_scores, mean_pos = self.regress_particles(t_active_particles,
                                                                            t_n_particles)
        f_all_scores, f_all_pos, f_best_scores, f_best_pos, match_ids = self.filter_regressed_tracks(all_scores,
                                                                                                     all_pos,
                                                                                                     mean_scores,
                                                                                                     mean_pos)
        # PERFORM NMS
        keep_tracks = nms(f_best_pos, f_best_scores, self.regression_nms_thresh).detach().cpu().numpy()
        nms_ids = match_ids[keep_tracks] # np.ndarray
        nms_pos = f_best_pos[keep_tracks] # torch.array
        nms_scores = f_best_scores[keep_tracks] # torch.array
        nms_val = np.c_[nms_pos.detach().cpu().numpy(),
                        nms_scores.detach().cpu().numpy()] # np.ndarray
        nms_all_scores = f_all_scores[keep_tracks] # np.ndarray
        nms_all_pos = f_all_pos[keep_tracks] # np.ndarray

        # IMPORTANT : nms_ids, nms_pos, nms_scores have index wise correspondence
        assert len(nms_ids) == len(nms_pos) == len(nms_scores) == len(nms_all_scores) == len(nms_all_pos)

        for (m_id, m_pos, a_score, a_pos) in zip(nms_ids, nms_val, nms_all_scores, nms_all_pos):
            t = self.get_track(m_id)
            t.update_position(m_pos[:4], a_score, a_pos)

        # Copy the signature of matched map
        matched_map = {k:v for k,v in zip(nms_ids, nms_val[:, :4])}

        # Update for matched IDs
        for old_id, new_pos in zip(nms_ids, nms_val[:, :4]):
            cur_track = self.get_track(old_id)
            histogram = compute_histogram(self.compute_crop(new_pos))
            cur_track.update_track(self.frame_number, hist_vec=histogram)

        # Find new tracks. Set high confidence for existing tracks
        det_pos_gpu = torch.tensor(self.boxes, dtype=torch.float32, device=self.device)
        combined_pos = torch.cat([nms_pos, det_pos_gpu])
        combined_scores = torch.cat(
                    [2*torch.tensor(np.ones(nms_scores.shape[0]), dtype=torch.float32, device=self.device),
                     torch.tensor(self.det_scores, dtype=torch.float32, device=self.device)])

        keep_det = nms(combined_pos, combined_scores, self.detection_nms_thresh).cpu().numpy().tolist()
        new_ind = [i - len(nms_pos) for i in keep_det if i >= len(nms_pos)]
        new_boxes = det_pos_gpu[new_ind].cpu().numpy()
        new_scores = self.det_scores[new_ind]

        # Check if new is old
        if len(inactive_ids) > 0 and len(new_boxes) > 0:
            regress_matches, matched_ind = self.appearance_match(new_boxes)
            new_boxes = [v for i,v in enumerate(new_boxes) if i not in matched_ind]

        for new_b in new_boxes:
            max_id += 1
            new_id_map[max_id] = new_b

        for old_id, new_pos in regress_matches.items():
            ls_t = self.get_lost_track(old_id)
            assert ls_t is not None
            self.track_found_again(ls_t)
            histogram = compute_histogram(self.compute_crop(new_pos))
            ls_t.update_position(new_pos, all_scores=None, all_pos=None)
            ls_t.update_track(self.frame_number, hist_vec=histogram, rematch=True)

        lost_ids = list(set(prev_ids) - set(nms_ids))
        lost_tracks = [self.get_track(i) for i in lost_ids]
        self.tracks_to_inactive(lost_tracks)

        matched_map = {**matched_map, **regress_matches}
        return matched_map, new_id_map


    def appearance_match(self, boxes, thresh=1.):
        """
        boxes : remaining boxes, (xmin,ymin,xmax,ymax)
        lambd : amount to weigh distance compared to colour match
        thresh : max val to consider a rematch.
                 max possible distance = 1., max allowed Hist dist = 0.3
                 => 0.2+0.3 = 0.5
        returns:
            rematch_map : dict<id:new_pos>
            box_ind : list[index of unmatched box]
        """
        rematch_map = {}
        box_ind = []
        # regress_pos = self.align([lt.pos for lt in self.inactive_tracks])
        regress_pos = np.asarray([lt.pos for lt in self.inactive_tracks])
        regress_id = [lt.id for lt in self.inactive_tracks]
        regress_cent = compute_centroid(regress_pos)
        box_cent = compute_centroid(boxes)
        range_ar = 2*np.max(boxes[:, 2:4] - boxes[:, 0:2], axis=1).reshape(-1, 1)
        dist_matrix = cdist(box_cent, regress_cent, metric='cityblock')
        dist_matrix = dist_matrix / range_ar
        dist_cond = dist_matrix < 1.
        dist_matrix = dist_matrix * dist_cond + (1-dist_cond)*1e20
        # Appearance matrix
        box_hist = [compute_histogram(self.compute_crop(b_pos)).flatten() for b_pos in boxes]
        box_hist = np.array(box_hist)
        regress_hist = np.asarray([lt.hist_vec.flatten() for lt in self.inactive_tracks])
        appearance_mat = matrix_histcmp(box_hist, regress_hist)
        # remove far away appearance info
        appearance_mat = appearance_mat * dist_cond + (1-dist_cond)*1e20
        if self.use_reid:
            cost_matrix = (1-self.lambd)*appearance_mat + self.lambd * dist_matrix
        else:
            cost_matrix = dist_matrix
        max_indexes = munkres_inst.compute(cost_matrix.tolist())
        for row, col in max_indexes:
            if cost_matrix[row][col] <= thresh:
                rematch_map[regress_id[col]] = boxes[row]
                box_ind.append(row)
        return rematch_map, box_ind



    def step(self, blob):

        self.frame_number += 1
        self.cur_im = blob

        if self.use_public:
            assert self.public_detections is not None
            detections = self.read_publicdet()
        else:
            boxes, scores = self.obj_detector.predict_box(blob)
            detections = torch.cat((boxes, torch.unsqueeze(scores, 1)), 1).cpu().numpy()
        
        # Perform NMS within image, to remove too cluttered head
        refined_det = get_refined_detection(detections, self.im_shape, self.detection_confidence)
        self.boxes, self.det_scores = refined_det[:, :4], np.squeeze(refined_det[:, 4:])

        self.detections = np.c_[self.boxes, self.det_scores]

        #### Perform CMC ####
        if self.frame_number > 1:
            self._compute_warp_matrix()

        #####################
        # Association #
        #####################
        matched_map = None
        matched_mask = np.zeros(blob.shape[:-1]).astype(np.int16)
        if len(self.tracks) > 0:
            assert self.last_image is not None
            self.regress_lost_track()
            prev_ids = [t.id for t in self.tracks]
            inactive_ids = [lt.id for lt in self.inactive_tracks]
            matched_map, new_id_map = self.track_match()
            # for lost tracks, check if inactive for too long, then kill it
            self.inactive_tracks = [
                t for t in self.inactive_tracks if t.count_inactive <= self.inactive_patience
            ]
        else:
            init_ids = list(range(1, len(self.boxes)+1))
            new_id_map = {n_id:n_pos for (n_id, n_pos) in zip(init_ids, self.boxes)}
            matched_map = new_id_map
        # Add new detections
        self.add(new_id_map)

        if self.use_public:
            matched_boxes = list(matched_map.values())

        if self.save_frames:
            assert self.save_dir is not None
            plotted_im = plot_boxes(blob, matched_map)
            imsave(osp.join(self.save_dir, "{:06d}".format(self.frame_number) + '.jpg'), plotted_im)

        ####################
        # Generate Results #
        ####################
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            position = t.pos
            self.results[t.id][self.frame_number] = np.concatenate([position, np.array([1.])])
        self.last_image = blob


    def _compute_warp_matrix(self):

        assert self.frame_number > 1
        im1 = deepcopy(self.last_image)
        im2 = deepcopy(self.cur_im)
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.number_of_iterations,
                    self.termination_eps)

        cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix,
                                            self.warp_mode, criteria,
                                            inputMask=None, gaussFiltSize=5)
        self.warp_matrix = torch.from_numpy(warp_matrix)
        self.motion_vector = np.tile(warp_matrix[:, 2],2)


    def align(self, position):
        """Aligns the positions of active and inactive tracks depending on camera motion.
            Code borrowed from Tim Meinhardt
            """
        position_gpu = torch.tensor(position, dtype=torch.float32, device=self.device)
        aligned_pos = []
        for pos in position_gpu:
            aligned_pos.append(warp_pos(pos, self.warp_matrix).numpy().tolist())
            # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data
        if len(aligned_pos)>0:
            return np.array(aligned_pos).reshape(-1, 4)
        else:
            return np.array(aligned_pos)

    def align_particles(self, particles):
        """
        particles
        -----------
        list(np.ndarray(N, 4))
        """
        aligned_particles = []
        for particle in particles:
            aligned_particles.append(self.align(particle))

        return aligned_particles


    def get_results(self):
        return self.results
