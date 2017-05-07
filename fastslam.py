import numpy as np
import random
from particle import *
from copy import deepcopy

class FastSLAM:
    def __init__(self, init_pos, num_particles, landmarks):
        self.particles = []
        self.robot = Robot(init_pos, landmarks)
        self.landmarks = landmarks
        self.num_particles = num_particles
        self.create_particles()

    def create_particles(self):
        for i in range(self.num_particles):
            p = Particle(self.robot.pose)
            self.particles.append(p)

    def run(self, controls):
        for ctrl in controls:
            self.robot.move(ctrl)
            print self.robot.pose
            # receive robot controls
            obss = self.robot.sense()
            print obss
            for obs in obss:
                for p in self.particles:
                    p.update(obs)
                print [p.weight for p in self.particles]
                self.particles = self.resample_particles()
            #print self.mean_pos()

    def resample_particles(self):
        new_particles = []
        weight = [p.weight for p in self.particles]
        index = int(random.random() * self.num_particles)
        beta = 0.0
        mw = max(weight)
        for i in range(self.num_particles):
            beta += random.random() * 2.0 * mw
            while beta > weight[index]:
                beta -= weight[index]
                index = (index + 1) % self.num_particles
            new_particle = deepcopy(self.particles[index])
            new_particle.weight = 1.0
            new_particles.append(new_particle)
        return new_particles

    def mean_pos(self):
        mean_x = 0.0
        mean_y = 0.0
        for p in self.particles:
            mean_x += p.pose.x
            mean_y += p.pose.y
        return mean_x/self.num_particles, mean_y/self.num_particles

if __name__=="__main__":
    lms = []
    for i in range(10):
        lm = Landmark(mu=np.array([[i],[i]]), sigma=np.eye(2))
        lms.append(lm)
        lm = Landmark(mu=np.array([[i*math.cos(math.pi/6)],[i*math.sin(math.pi/6)]]), sigma=np.eye(2))
        lms.append(lm)

    p0 = Pose(0, 0, math.pi/4)
    fs = FastSLAM(p0, 10, lms)
    controls = [Pose(1,1,0),]*5
    fs.run(controls)

