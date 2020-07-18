import numpy as np
import torch
from abc import ABC, abstractmethod

#import seaborn as sns
import matplotlib.pyplot as plt

class objective_func(ABC):
    @abstractmethod
    def func(self, x):
        pass
    def dfunc(self, x):
        out = self.func(x)
        out.backward()
        return x.grad
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
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
    
class ackley(objective_func):
    '''
    the period of local minimum along each axis is 1, integer coordinate (1,1), (2,3)... 
    x and y is interchangeable
    global minimum is 0 with arguments x=y=0
    local minimums far away from orgin are 20
    supremum is 20 + e - 1/e = 22.35
    symmetric along x=0, y=0, y=x lines
    disappearing global gradient when far away from optimal
    '''
    def __init__(self, dim=2):
        self.optimum = 0
        self.lim = 5
        self.dim = dim
        self.optimal = torch.zeros((self.dim, ), device=torch.device('cuda:0'))
        self.x = None
        self.out = None
    def func(self, x):

        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = torch.cos(2*np.pi*x).mean() 
        return -20. * torch.exp(arg1) - torch.exp(arg2) + 20. + np.e
    def dfuncR(self, x):
        if torch.norm(x) < 1e-3:
            return torch.zeros((self.dim, ))
        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = torch.cos(2*np.pi*x).mean()
        g = lambda xx: -0.8 * xx / arg1 * torch.exp(arg1) / self.dim + 2 * np.pi * torch.sin(2 * np.pi * xx) * torch.exp(arg2) / self.dim
        return g(x)

    
class bukin():
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
        self.x = x
        self.out = 100 * torch.sqrt(torch.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * torch.abs(x[0] + 10)
        return self.out
    def dfuncR(self, x):
        arg1 = x[1] - 0.01 * x[0]**2
        arg2 = 50 / torch.sqrt(torch.abs(arg1)) * torch.sign(arg1) if arg1 != 0 else 0
        return torch.tensor([- 0.02 * x[0] * arg2 + 0.01 * torch.sign(x[0] + 10), arg2])
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum

class eggholder(objective_func):
    # evaluated domain: 
    def __init__(self):
        self.optimal = torch.tensor([522, 413])
        self.optimum = 0
        self.lim = 550
    def func(self, x):
        if torch.abs(x[0]) > self.lim or torch.abs(x[1]) > self.lim:
            return torch.tensor([2e3], requires_grad=True)
        arg1 = x[0]/2 + (x[1] + 47) 
        arg2 = x[0]   - (x[1] + 47)
        f = lambda xx: torch.sin(torch.sqrt(torch.abs(xx)))
        self.x = x
        self.out = -(x[1] + 47) * f(arg1) - x[0] * f(arg2) + 976.873
        return self.out
    def dfuncR(self, x):
        if torch.abs(x[0]) > self.lim or torch.abs(x[1]) > self.lim:
            return torch.tensor([0, 0])
        arg1 = x[0]/2 + (x[1] + 47) 
        arg2 = x[0]   - (x[1] + 47)
        g = lambda xx: torch.cos(torch.sqrt(torch.abs(xx)))/torch.sqrt(torch.abs(xx))/2*torch.sign(xx)
        f1 = (x[1] + 47) * g(arg1)
        f2 = x[0] * g(arg2)
        return torch.tensor([-f1/2 - torch.sin(torch.sqrt(torch.abs(arg2))) - f2, \
                         -f1 - torch.sin(torch.sqrt(torch.abs(arg1))) + f2])
    def get_optimal(self):
        return self.optimal
    def get_optimum(self):
        return self.optimum
    
class tuned_ackley():
    # evaluated domain: circle with radius 19
    def __init__(self, lim=22, dim=2):
        self.optimum = 0
        self.lim = lim
        self.dim = dim
        self.optimal = torch.zeros((self.dim, ))
    def func(self, x):
        '''
        the period of local minimum along each axis is 1, integer coordinate (1,1), (2,3)... 
        x and y is interchangeable
        global minimum is 0 with arguments x=y=0
        symmetric along x=0, y=0, y=x lines
        disappearing global gradient when far away from optimal
        '''
        if torch.norm(x) > self.lim:
            return torch.tensor([5e1], requires_grad=True)
        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = 0.5 * torch.cos(np.pi*x).mean()
        self.x = x
        self.out = -20. * torch.exp(arg1) - 0.1 * arg1**4 * torch.exp(arg2) + 20.
        return self.out
    def dfuncR(self, x):
        if torch.norm(x) < 1e-3:
            return torch.zeros((self.dim,))
        elif torch.norm(x) > self.lim:
            return torch.zeros((self.dim, ))
        arg1 = -0.2 * torch.sqrt(torch.pow(x, 2).mean())
        arg2 = 0.5 * torch.cos(np.pi*x).mean()
        g = lambda xx: -0.8 * xx / arg1 * torch.exp(arg1) / self.dim + np.pi/20 * arg1**4 * torch.sin(np.pi * xx) * torch.exp(arg2) / self.dim \
                         - 4 * xx/6250 * torch.exp(arg2) * torch.power(x, 2).sum() / self.dim**2
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
