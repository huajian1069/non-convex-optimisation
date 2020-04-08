import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib import patches

def get_distance(self):
    if np.any(self.distance_arg == None) or np.any(self.distance_val == None):
        self.distance_arg = np.linalg.norm(self.arg - self.optimal.reshape(1,1,2), axis=(1,2))
        self.distance_val = np.linalg.norm(self.val - self.optimum, axis=1)
    return self.distance_arg, self.distance_val

def plot_distance(self):
    self.get_distance()
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    self.plot_distance_common(ax1, self.val.shape[0]-1)
    
def plot_distance_common(self, ax1, i):
    ax1.plot(np.arange(i), self.distance_arg[1:i+1], color='green', label='Frobenius norm \nof parameters')
    ax1.set_xlim(0, self.val.shape[0])
    ax1.set_ylim(np.min(self.distance_arg)*0.9, np.max(self.distance_arg)*1.1)
    ax1.set_xlabel('iteration', fontsize=15)
    ax1.set_ylabel('distance in domain', color='green', fontsize=15)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    ax2.plot(np.arange(i), self.distance_val[1:i+1], color='red', label='L2 norm \nof func value')
    ax2.set_ylim(np.min(self.distance_val)*0.9, np.max(self.distance_val)*1.1)
    ax2.set_ylabel('distance in codomain', color='red', fontsize=15)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right') 
    
def print_mean_variance(self):
    # print mean and variance of each iteration
    for i, a in enumerate(self.stats['var']):
        print('iter=',i, 'mean=', self.stats['mean'][i].T , 'var=\n', a@a.T, '\n')
        
def print_evaluations_per_iteration(self):
    for i, iter_ in enumerate(self.stats['evals_per_iter']):
        print('iter=', i, '\n', iter_.squeeze(),'\n') 
    
def print_arguments_before_and_after_move(self):
    for i, iter_ in enumerate(self.stats['x_vs_original']):
        print('iter=', i, '\nbefore\n', iter_[:2], '\nafter\n', iter_[2:], '\n') 
            
def generate_point_cloud(self, sigma, alpha, beta, adjust, points):
    self.num = points.shape[0]
    self.res = np.zeros((self.num, ))
    self.points = points
    
    for i in range(self.num):
        val, arg, stats = cma_es_general(self.points[i].reshape(2,1), sigma, alpha, beta, adjust, self.func, self.dfunc, self.optimal, self.optimum)
        if(stats['status'] == 'd'):
            self.res[i] = 1
        elif(stats['status'] == 'l'):
            self.res[i] = 0.5
        else:
            self.res[i] = 0

def plot_prob_vs_radius(self, *args):
    def count_global_min(res, points):        
        distance = np.linalg.norm(points, axis=1)
        idx = np.argsort(distance)
        dis_ascending = distance[idx]
        res_ascending = res[idx]
        prob = np.zeros((self.num, ))
        for i in range(self.num):
            prob[i] = np.sum(res_ascending[:i+1] == 0) / (i + 1) 
        return dis_ascending, prob
    argc = len(args)
    assert argc%2 == 0
    pair_cnt = int(argc / 2)
    dis_ascendings = np.zeros((self.num, pair_cnt + 1))
    probs = np.zeros((self.num, pair_cnt + 1))
    dis_ascendings[:,0], probs[:,0] = count_global_min(self.res, self.points)
    for i in range(pair_cnt):
        dis_ascendings[:,i+1], probs[:,i+1] = count_global_min(args[i*2], args[i*2+1])
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, np.max(dis_ascendings))
    ax.set_ylim(0, 2)
    ax.set_xlabel('distance from origin', fontsize=13)
    ax.set_ylabel('prob of global minminum', fontsize=13)
    for i in range(pair_cnt+1):
        ax.plot(dis_ascendings[:,i], probs[:,i])
    plt.show()
    
def plot_cloud_point(self):
    fig = plt.figure(figsize=(7,7))
    '''
    # one quadrant
    x1 = np.hstack((self.points[:,0], self.points[:,1]))
    y1 = np.hstack((self.points[:,1], self.points[:,0]))
    res1 = np.hstack((self.res, self.res))
    # two qudrant
    x2 = np.hstack((x1, -x1))
    y2 = np.hstack((y1, y1))
    res2 = np.hstack((res1, res1))
    # four qudrant
    x = np.hstack((x2, -x2))
    y = np.hstack((y2, -y2))
    hue = np.hstack((res2, res2))
    '''
    x = self.points[:,0]
    y = self.points[:,1]
    hue = self.res
    p = sns.scatterplot(x=x, y=y, color="r", hue=hue, hue_norm=(0, 1), legend=False)