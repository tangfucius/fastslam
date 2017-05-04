import numpy as np
import random

class FastSLAM:
    def __init__(self, init_pos, num_particles, landmarks):
        self.particles = []
        self.robot = Robot(init_pos)
        self.landmarks = landmarks
        self.num_particles = num_particles
        self.create_particles()

    def create_particles(self):
        pass

    def run(self):
        while True:
            # receive robot controls
            objs = self.robot.sense()
            for p in self.particles:
                p.update(objs)
            self.particles = self.resample_particles()

    def resample_particles(self):
        return self.particles


