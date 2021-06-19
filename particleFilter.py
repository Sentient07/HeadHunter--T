#!/usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt; from PIL import Image 
from scipy.stats import norm
from im_utils import get_neighbour_loc, compute_centroid, get_area, scatter_particles, box_from_centroid


class RoIPF:

    def __init__(self, max_particles, init_state):
        self.n_particles = max_particles
        self.created_particles = None
        self.init_state = init_state
        self.past_vel = []
        self.create_particles(init_state)
        assert self.created_particles is not None
        # States are x, y, w, h of the BBox.
        self.velocity = np.array([0., 0.])
        self.pos = init_state

        self.h_past = []
        self.w_past = []


    def create_particles(self, init_state):
        """
        Draw BB centered within circle centered at (C_x, C_y) of radius R
        
        Initialise n_particles
        """
        self.particles = scatter_particles(init_state, max_particles=self.n_particles)
        self.created_particles = len(self.particles)
        self.weights = np.array([1.0/self.created_particles]*self.created_particles)


    def get_vel(self, new_position):
        prev_cent = compute_centroid(self.pos)
        cur_cent = compute_centroid(new_position)
        cur_vel = cur_cent - prev_cent
        return cur_vel


    def align_particles(self, aligned_particles):
        self.particles = aligned_particles


    def _update_mean_wh(self, pos):
        w,h = pos[2:4] - pos[0:2]
        self.h_past.append(h)
        self.w_past.append(w)


    def get_mean_wh(self):
        if len(self.w_past) == 0:
            return self.pos[2:4] - self.pos[0:2]
        else:
            return np.r_[np.mean(self.w_past), np.mean(self.h_past)]


    def update(self, new_position, all_score, all_pos):
        """
        1) Update velocity and position of particle
        2) Update weight of partilces corresponding to prev loc
        3) Update positions of particles
        """
        cur_vel = self.get_vel(new_position)
        self.update_velocity(cur_vel)
        self.pos = new_position
        self._update_mean_wh(new_position)
        particle_weight = np.ones_like(all_score) - all_score # Score to distance
        self.update_weights(particle_weight)
        # With updated velocity, compute new locations
        self.particles = all_pos


    def update_velocity(self, velocity):
        self.past_vel.append(velocity)
        mean_vel = np.average(self.past_vel, axis=0)
        self.velocity = mean_vel
        

    def get_all_particles(self):
        return self.particles


    def get_position(self):
        mean_pos = np.average(self.particles, weights=self.weights, axis=0)
        return mean_pos


    def predict(self):
        particle_centre = compute_centroid(self.particles)
        particle_hw = self.particles[:, 2:4] - self.particles[:, 0:2]
        n_particle_centre = particle_centre + self.velocity
        n_particles_chw = np.c_[n_particle_centre, particle_hw]
        return box_from_centroid(n_particles_chw)
        # self.particles = box_from_centroid(n_particles_chw)


    def update_weights(self, weights, dist_std=0.15):
        #Generating a temporary array for the input position
        # PDF evaluated at 0 since our we want a gaussian centered around
        # the centroid
        self.weights *= norm(weights, dist_std).pdf(0)
        self.weights += 1.e-300 #avoid zeros
        self.weights /= sum(self.weights) #normalize


    def estimate(self):
        # Weighted average
        # pos = self.particles[:, :2]
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var  = np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        return mean, var


    def neff(self):
        return 1. / np.sum(np.square(self.weights))


    def resample_particles(self, best_particle=None):
        if best_particle is None:
            best_particle = self.get_top_particles(n=1)[1][0]
        self.create_particles(best_particle)


    def returnParticlesCoordinates(self, index=-1):
        if(index<0):
            return self.particles.astype(int)
        else:
            return self.particles[index,:].astype(int)


    def drawParticles(self, frame, color=[0,0,255], radius=2):
        for x_particle, y_particle, _, _ in self.particles.astype(int):
            frame = cv2.circle(frame, (x_particle, y_particle),
                               radius, color, -1) #RED: Particles
        return frame


    def get_top_particles(self, n=1):
        idx = (-self.weights).argsort()[:n]
        return idx, self.particles[idx, :]
