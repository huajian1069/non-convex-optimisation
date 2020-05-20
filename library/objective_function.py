import numpy as np
from abc import ABC, abstractmethod

import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

class objective_func(ABC):
    @abstractmethod
    def func(self, x):
        pass
    @abstractmethod
    def dfunc(self, x):
        pass
    @abstractmethod
    def get_optimal(self):
        pass
    @abstractmethod
    def get_optimum(self):
        pass
    def visualise1d(self, lim, n):
        ''' 
            lim: the visualisation scope [-lim, lim] in each dimension
            n: the number of points used to interpolate between [-lim, lim]
        '''
        xs = np.linspace(-lim, lim, n)
        fs = []
        for x in xs:
            fs.append(self.func(x))
        plt.plot(xs, fs)
    def visualise2d(self, lim, n):
        x, y = np.linspace(-lim, lim, n), np.linspace(-lim, lim, n)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros(xx.shape)
        for j in range(n):
            for i in range(n):
                zz[j, i] = self.func(np.array([x[i], y[j]]))
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        sc = ax.scatter(x=xx.ravel(), y=yy.ravel(), c=zz.ravel())
        ax.scatter(x=[self.optimal[0]], y=[self.optimal[1]], c='red', marker='x')
        plt.colorbar(sc)
        fig.show()
        return ax
    def visualise3d(self, lim, n):
        x, y = np.linspace(-lim, lim, n), np.linspace(-lim, lim, n)
        z = []
        for i in y:
            z_line = []
            for j in x:
                z_line.append(self.func(np.array([j,i])))
            z.append(z_line)
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y),  \
                              go.Scatter3d(x=[self.optimal[0]], y=[self.optimal[1]], z=[self.optimum])])
        fig.update_layout(autosize=False,
                          scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
    def visulise_gradient(self, lim, n):
        x, y = np.linspace(-lim, lim, n), np.linspace(-lim, lim, n)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros((n, n, 2))
        for j in range(len(y)):
            for i in range(len(x)):
                zz[j, i, :] = self.dfunc(np.array([x[i], y[j]]))
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.quiver(xx,yy,zz[:,:,0],zz[:,:,1])
        ax.scatter(x=[self.optimal[0]], y=[self.optimal[1]], c='red', marker='x')
        fig.show()
        return ax
    def visualise2d_section(self, pos, dire):
        ''' 
            pos: the position of cross-section
            dire: along this direction/dimension to get cross-section
        '''
        fig = plt.figure(figsize=(4,4))
        xs = np.linspace(-self.lim, self.lim, 301)
        fs = []
        if dire == 'x':
            for x in xs:
                fs.append(self.func(np.array([x, pos])))
        else:
            for x in xs:
                fs.append(self.func(np.array([pos, x])))
        plt.plot(xs, fs)
        fig.show()
    def visualize2d_section_gradient(self, pos, dire):
        fig = plt.figure(figsize=(4,4))
        xs = np.linspace(-self.lim, self.lim, 300)
        dfs = []
        if dire == 'x':
            for x in xs:
                dfs.append(self.dfunc(np.array([x, pos])))
        else:
            for x in xs:
                dfs.append(self.dfunc(np.array([pos, x])))
        dfs = np.array(dfs)
        plt.plot(xs, dfs[:,0])
        plt.plot(xs, dfs[:,1])
        fig.show()
    
class ackley(objective_func):
    def __init__(self, dim=2):
        self.optimum = 0
        self.lim = 25
        self.dim = dim
        self.optimal = np.zeros((self.dim, ))
    def func(self, x):
        '''
        the period of local minimum along each axis is 1, integer coordinate (1,1), (2,3)... 
        x and y is interchangeable
        global minimum is 0 with arguments x=y=0
        local minimums far away from orgin are 20
        supremum is 20 + e - 1/e = 22.35
        symmetric along x=0, y=0, y=x lines
        disappearing global gradient when far away from optimal
        '''
        arg1 = -0.2 * np.sqrt(np.power(x, 2).mean())
        arg2 = np.cos(2*np.pi*x).mean()
        return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    def dfunc(self, x):
        if np.linalg.norm(x) == 0:
            return x
        arg1 = -0.2 * np.sqrt(np.power(x, 2).mean())
        arg2 = np.cos(2*np.pi*x).mean()
        g = lambda xx: -0.8 * xx / arg1 * np.exp(arg1) / self.dim + 2 * np.pi * np.sin(2 * np.pi * xx) * np.exp(arg2) / self.dim
        return g(x)
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    
class bukin(objective_func):
    '''
    non-disappearing gradient
    large gradient and uncontinuous gradient around ridge/local optimal
    optimum: 0
    optimal: (-10, 1)
    '''
    def __init__(self):
        self.optimal = np.array([-10, 1])
        self.optimum = 0
        self.lim = 15
    def func(self, x):
        return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)
    def dfunc(self, x):
        arg1 = x[1] - 0.01 * x[0]**2
        arg2 = 50 / np.sqrt(np.abs(arg1)) * np.sign(arg1) if arg1 != 0 else 0
        return np.array([- 0.02 * x[1] * arg2 + 0.01 * np.sign(x[0] + 10), arg2])
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum

class eggholder(objective_func):
    # evaluated domain: 
    def __init__(self):
        self.optimal = np.array([522, 413])
        self.optimum = 0
        self.lim = 550
    def func(self, x):
        if np.abs(x[0]) > self.lim or np.abs(x[1]) > self.lim:
            return 2e3
        arg1 = x[0]/2 + (x[1] + 47) 
        arg2 = x[0]   - (x[1] + 47)
        f = lambda xx: np.sin(np.sqrt(np.abs(xx)))
        return -(x[1] + 47) * f(arg1) - x[0] * f(arg2) + 976.873
    def dfunc(self, x):
        if np.abs(x[0]) > self.lim or np.abs(x[1]) > self.lim:
            return np.array([0, 0])
        arg1 = x[0]/2 + (x[1] + 47) 
        arg2 = x[0]   - (x[1] + 47)
        g = lambda xx: np.cos(np.sqrt(np.abs(xx)))/np.sqrt(np.abs(xx))/2*np.sign(xx)
        f1 = (x[1] + 47) * g(arg1)
        f2 = x[0] * g(arg2)
        return np.array([-f1/2 - np.sin(arg2) - f2, \
                         -f1   - np.sin(arg1) + f2])
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    
class tuned_ackley(objective_func):
    # evaluated domain: circle with radius 19
    def __init__(self, lim=22, dim=2):
        self.optimum = 0
        self.lim = lim
        self.dim = dim
        self.optimal = np.zeros((self.dim, ))
    def func(self, x):
        '''
        the period of local minimum along each axis is 1, integer coordinate (1,1), (2,3)... 
        x and y is interchangeable
        global minimum is 0 with arguments x=y=0
        symmetric along x=0, y=0, y=x lines
        disappearing global gradient when far away from optimal
        '''
        if np.linalg.norm(x) > self.lim:
            return 5e1
        arg1 = -0.2 * np.sqrt(np.power(x, 2).mean())
        arg2 = 0.5 * np.cos(np.pi*x).mean()
        return -20. * np.exp(arg1) - 0.1 * arg1**4 * np.exp(arg2) + 20.
    def dfunc(self, x):
        if np.linalg.norm(x) == 0:
            return x
        elif np.linalg.norm(x) > self.lim:
            return np.zeros((self.dim, ))
        arg1 = -0.2 * np.sqrt(np.power(x, 2).mean())
        arg2 = 0.5 * np.cos(np.pi*x).mean()
        g = lambda xx: -0.8 * xx / arg1 * np.exp(arg1) / self.dim + np.pi/20 * arg1**4 * np.sin(np.pi * xx) * np.exp(arg2) / self.dim \
                         - 4 * xx/6250 * np.exp(arg2) * np.power(x, 2).sum() / self.dim**2
        return g(x)
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    def visualise2d_section(self, pos, dire):
        super().visualise2d_section(pos, dire)
        plt.plot([-25, 25], [15.67, 15.67], label='y=15.67')
        plt.plot([-25, 25], [3.63, 3.66], label='y=3.66')
        plt.plot([12.96, 12.96], [0, 50], label='x=12.96')
        plt.plot([22, 22], [0, 50], label='x=22')
        plt.legend()
