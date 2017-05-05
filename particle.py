import numpy as np
import random
import math

class Pose:
    def __init__(self, x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __add__(self, newpose):
        return Pose(self.x+newpose.x, self.y+newpose.y, (self.theta+newpose.theta)%(2*np.pi))

    def pos(self):
        return np.array([[self.x],[self.y]])

class Landmark:
    #mu should be a 2x1 np array
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.count = 1

    def update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

class Particle:
    tol = 1e-4

    def __init__(self, pose):
        self.pose = pose
        self.landmarks = []
        self.motion_noise = 0.5
        self.theta_noise = 0.1
        self.obs_noise_cov = np.array([[0.1, 0], [0, (3.0*math.pi)**2]])
        self.weight = 1.0 

    def move_noise(self):
        x_noise = random.gauss(0, self.motion_noise)
        y_noise = random.gauss(0, self.motion_noise)
        t_noise = random.gauss(0, self.theta_noise)
        return Pose(x_noise, y_noise, t_noise)

    def move(self, mv):
        self.pose = self.pose+mv+self.move_noise()

    def update(self, obs):
        #obs is a (r, theta) tuple
        if self.landmarks:
            prob, ix, pred_obs, G, Q = self.check_association(obs)
            if prob<self.tol:
                self.create_landmark(obs)
            else:
                self.update_landmark(obs, ix, pred_obs, G, Q)
        else:
            self.create_landmark(obs)

    def g(self, mu):
        pass

    def g_inv(self, obs):
        pass

    def jacobian(self, landmark):
        mu = landmark.mu
        d2 = np.sum((self.pos()-mu)**2)
        d = math.sqrt(d2)
        dx = mu[0][0] - self.x
        dy = mu[1][0] - self.y
        G = np.array([[dx/d, dy/d][-dy/d2, dx/d2]])
        Q = G.T.dot(landmark.sigma).dot(G)+self.obs_noise_cov

    def update_landmark(self, obs, ix, pred_obs, G, Q):
        landmark = self.landmarks[ix]
        K = landmark.sigma.dot(G).dot(np.linalg.inv(Q))
        new_mu = landmark.mu + K.dot((obs-pred_obs).T)
        new_sigma = (np.eye(2)-K.dot(G)).dot(landmark.sigma)
        landmark.update(new_mu, new_sigma)

    def create_landmark(obs):
        mu = self.g_inv(obs)
        G_inv = np.linalg.inv(self.jacobian(mu))
        sigma = G_inv.dot(self.obs_noise_cov).dot(G_inv.T)
        landmark = Landmark(mu, sigma)
        self.landmarks.append(landmark)

    def check_association(self, obs):
        pass
