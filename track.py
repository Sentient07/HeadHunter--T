import os
import sys

from collections import Counter
from glob import glob
from tqdm import tqdm

import numpy as np
import os.path as osp

from im_utils import *
from scipy.misc import imsave
from copy import deepcopy
from particleFilter import RoIPF



class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, track_id, time_stamp, pos,
                count_inactive, inactive_patience,
                im_shape, hist_vec, max_particles=100):
        
        """
        id : ID number through the process. Unique for each det
        time_stamp : Current Timestamp(Frame number)
        pos : Position in the frame(X_min, Y_min, X_max, Y_max)
        score : detection confidences
        count_inactive : Nos of time_stamp for which this frame has been inactive
        inactive_patience : Nos of time_stamps for which this track is allowed to be
                            inactive
        im_shape : Shape of the incoming frame. (H, W, 3)
        last_pos : `pos` at time_stamp-N(hopefully N > 1)
        
        """
        self.id = track_id
        self.time_stamp = time_stamp
        self.pos = pos
        self.p_noise = min(self.pos[3] - self.pos[1],
                            self.pos[2] - self.pos[0]) / 4
        self.count_inactive = count_inactive
        self.inactive_patience = inactive_patience
        self.im_shape = im_shape
        self.last_pos = []
        self.max_particles = max_particles

        # Initialise Particle Filter
        self.init_filter()
        # Appearance
        self.hist_vec = hist_vec
        self.past_hist = [hist_vec]
        self.hist_count = 0

        # Attributes
        self.kill_flag = False
        self.exit_thresh = 50 # Image border


    def init_filter(self):
        self.roiPF = RoIPF(self.max_particles, self.pos)

    def avg_hist(self):
        return np.mean(self.past_hist, axis=0)

    def predict_particles(self):
        return self.roiPF.predict()

    def get_n_particles(self):
        return self.roiPF.created_particles

    def get_particles(self):
        return self.roiPF.get_all_particles()

    def align_particles(self, aligned_particles):
        self.roiPF.align_particles(aligned_particles)

    def update_position(self, best_pos, all_scores=None, all_pos=None):
        """
        When rematching, all_scores is None and hence particles need to be
        resampled. 

        """
        self.pos = best_pos
        self._update_lastpos()
        if all_scores is not None and all_pos is not None:
            self.score = np.max(all_scores)
            self.roiPF.update(best_pos, all_scores, all_pos)
            if self.roiPF.neff() < self.roiPF.created_particles // 3:
                self.roiPF.resample_particles(self.pos)

    def update_track(self, time_stamp, hist_vec, rematch=False):
        """
        time_stamp : Current Timestamp(Frame number)
        pos : Position in the frame(X_min, Y_min, X_max, Y_max)
        """
        if rematch:
            self.last_pos = []
            self.init_filter()
        self.time_stamp = time_stamp
        self.past_hist.append(hist_vec)
        self.hist_vec = hist_vec
        self.reset_count_inactive()
        # self.last_pos_copy = deepcopy(self.last_pos)

    def _update_lastpos(self):
        if isinstance(self.last_pos, list):
            self.last_pos.append(self.pos)
            self.last_pos = np.array(self.last_pos)
        else:
            self.last_pos = np.concatenate([self.last_pos, np.array([self.pos])])
    
    def step(self):
        self.count_inactive += 1
        self.time_stamp += 1
        if not self._does_it_exit():
            self._update_lastpos()
            self.pos = self._predict_position()


    def get_velocity(self):
        return self.roiPF.velocity

    def _does_it_exit(self):
        if len(self.last_pos) < 1 :
            self.inactive_patience = 5
            return False
        cur_centroid = compute_centroid(self.pos)
        cur_vel = self.get_velocity()
        cur_wh = self.pos[2:4] - self.pos[0:2]
        # Check if it's leaving the frame
        # TODO : Keep the inactive patience equally high rather than killing it right away.
        eventual_centroid = cur_centroid + (self.inactive_patience * cur_vel)
        eventual_box = box_from_centroid(np.r_[eventual_centroid, cur_wh])
        if is_outside(eventual_box, self.im_shape, thresh=20):
            self.kill_flag = True
            self.count_inactive += 30
            return True
        return False
    
    def reset_count_inactive(self):
        self.count_inactive = 0

    def _predict_position(self):

        cur_state = self.pos
        cur_centre = compute_centroid(cur_state)
        cur_wh = self.roiPF.get_mean_wh()
        new_centre = cur_centre + self.get_velocity()
        new_pos = box_from_centroid(np.r_[new_centre, cur_wh])
        return new_pos
