import numpy as np
import random
import math

def mahalanobis(x, mean, cov):
    det = math.sqrt(2*math.pi*np.linalg.det(cov))
    num = np.exp(-0.5*(x-mean).T.dot(np.linalg.inv(cov)).dot(x-mean))
    ret = num/det
    return ret[0][0]

class Pose:
    def __init__(self, x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __add__(self, newpose):
        return Pose(self.x+newpose.x, self.y+newpose.y, (self.theta+newpose.theta)%(2*math.pi))

    def pos(self):
        return np.array([[self.x],[self.y]])

    def __str__(self):
        return "({}, {}, {})".format(str(self.x), str(self.y), str(self.theta))

class Landmark:
    #mu should be a 2x1 np array
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.count = 1

    def update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __str__(self):
        return "Landmark mu: {}, sigma: {}".format(np.array_str(self.mu),  np.array_str(self.sigma))

class Particle:
    tol = 1e-4

    def __init__(self, pose):
        self.pose = pose
        self.landmarks = []
        self.motion_noise = 0.2
        self.theta_noise = 0.1
        self.obs_noise_cov = np.array([[0.1, 0], [0, (3.0*math.pi/180)**2]])
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
        prob = self.tol
        if self.landmarks:
            prob, ix, pred_obs, G, Q = self.check_association(obs)
            if prob<self.tol:
                self.create_landmark(obs)
            else:
                self.update_landmark(obs, ix, pred_obs, G, Q)
        else:
            self.create_landmark(obs)
        self.weight = prob

    def g_inv(self, obs):
        dist, theta = obs
        lm_x = self.pose.x + dist * math.cos(theta)
        lm_y = self.pose.y + dist * math.sin(theta)
        return np.array([[lm_x],[lm_y]])

    def jacobian(self, mu):
        dx = mu[0][0] - self.pose.x
        dy = mu[1][0] - self.pose.y
        d2 = dx**2+dy**2
        d = math.sqrt(d2)
        pred_obs = np.array([[d], [math.atan2(dy, dx)]])
        G = np.array([[dx/d, dy/d],[-dy/d2, dx/d2]])
        return pred_obs, G

    def update_landmark(self, obs, ix, pred_obs, G, Q):
        landmark = self.landmarks[ix]
        K = landmark.sigma.dot(G).dot(np.linalg.inv(Q))
        new_mu = landmark.mu + K.dot((obs-pred_obs).T)
        new_sigma = (np.eye(2)-K.dot(G)).dot(landmark.sigma)
        landmark.update(new_mu, new_sigma)
        landmark.count+=1

    def create_landmark(self, obs):
        mu = self.g_inv(obs)
        _, G = self.jacobian(mu)
        G_inv = np.linalg.inv(G)
        sigma = G_inv.dot(self.obs_noise_cov).dot(G_inv.T)
        landmark = Landmark(mu, sigma)
        self.landmarks.append(landmark)

    def check_association(self, obs):
        prob = 0.0
        pred_obs = None
        ix = -1
        G = None
        Q = None
        obs = np.array(obs)[:, np.newaxis]
        #print(len(self.landmarks))
        for i, landmark in enumerate(self.landmarks):
            p_obs, _G = self.jacobian(landmark.mu)
            _Q = _G.T.dot(landmark.sigma).dot(_G)+self.obs_noise_cov
            print((obs-p_obs).T)
            w = mahalanobis(obs, p_obs, _Q)
            if w>prob:
                prob = w
                pred_obs = p_obs
                G = _G
                Q = _Q
                ix = i
        return prob, ix, pred_obs, G, Q

if __name__=="__main__":
    pose = Pose(0, 0, math.pi/4)
    p = Particle(pose)
    movement = Pose(1, 1, math.pi/20)
    p.move(movement)
    print(p.pose)
    obs = (5, math.pi/9)
    p.update(obs)
    for lm in p.landmarks:
        print(lm)
    print(p.check_association(obs))
    # cov = np.array([[ 0.18409051, 0.0315773 ],[ 0.0315773,0.01785957]])
    # print("det:", np.linalg.det(cov))
    # print("man:", mahalanobis(obs, obs, cov))
