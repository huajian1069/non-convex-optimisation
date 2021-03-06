import numpy as np
import time
from library.optimiser import *
from library.post_analysis import *
from library.experiments import *

def do_multiple_exps(obj, opt, zoneParas):
    # init multiple experiments
    mexps = multiple_experiment()

    # init one experiment
    exp = single_experiment()
    
    # One experiment: setup objective function and optimiser
    exp.set_objective(obj)
    exp.set_optimizer(opt)

    # Multiple experiment: setup one expreiment
    mexps.set_single_exp(exp)
    # Multiple experiment: setup sample zone    
    mexps.set_sample_zone(zoneParas)
    
    data = mexps.do()
    
    return data

class single_experiment:
    def __init__(self, tol=0.1):
        self.tol = tol
    def set_objective(self, objective_func):
        self.objective_func = objective_func

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def do(self):
        optimal, optimum, statistics = self.optimizer.optimise(self.objective_func)
        #dist_arg = np.linalg.norm(optimal.detach().cpu().numpy() - self.objective_func.get_optimal())
        #dist_val = np.linalg.norm(optimum.detach().cpu().numpy() - self.objective_func.get_optimum())
        if  dist_arg < self.tol \
        or  dist_val < self.tol:
            statistics['status'] = 'global minimum'
        elif statistics['status'] != 'diverge':
            statistics['status'] = 'local minimum'
        print("distance domain, codomain: ", dist_arg, dist_val)
        if self.optimizer.verbose:
            print("Result: ", statistics['status'])
            print("found minimum: {}, minimum position: {}, evals: {}".format(optimum, torch.norm(optimal).item(), statistics['evals']))
        if self.optimizer.record == False:
            return statistics['status'], optimum, optimal, statistics['evals']
        else:
            statistics['optimal'] = self.objective_func.get_optimal()
            statistics['optimum'] = self.objective_func.get_optimum()
            statistics['found_optimal'] = optimal
            statistics['found_optimum'] = optimum
            return statistics
            
class multiple_experiment:
    def set_sample_zone(self, paras):
        self.paras = paras
        self.origin = paras['origin']
        self.edge = paras['edge']
        self.step = paras['step']
        self.sym = paras['sym'] if 'sym' in self.paras.keys() else False
        self.size = paras['size']
    def set_single_exp(self, exp):
        self.exp = exp
    def do(self):
        # get derivative parameters
        num_x = int(self.edge[0] / self.step)
        num_y = int(self.edge[1] / self.step)
        abs_edge = np.zeros((2,))
        abs_edge[0] = self.origin[0] + self.edge[0]
        
        # initlise matrix to record results
        data = {}
        data['convergence'] = np.zeros((num_y, num_x))
        data['cost'] = np.zeros_like(data['convergence'])
        data['evals'] = np.zeros_like(data['convergence'])
        data['mask'] = np.ones_like(data['convergence'], dtype=np.uint8)
        data['x'] = np.zeros_like(data['convergence'])
        data['y'] = np.zeros_like(data['convergence'])     
        if self.sym and num_x == num_y:   
            total_num = (num_x + 1) * num_x / 2
        else:
            total_num = num_x * num_y
        
        start = time.time()
        for i, x in enumerate(np.arange(self.origin[0], abs_edge[0], self.step)):
            if self.sym:
                abs_edge[1] = self.origin[1] + x + self.step 
            else:
                abs_edge[1] = self.origin[1] + self.edge[1]
            for j, y in enumerate(np.arange(self.origin[1], abs_edge[1], self.step)):
                points = np.random.rand(self.size, 2) * self.step + np.array([x, y]) 
                data['mask'][num_y-1-j, i] = 0
                data['x'][num_y-1-j, i] = x
                data['y'][num_y-1-j, i] = y
                # calculate the probility of getting global minimum 
                res = np.zeros((self.size, ))
                costs = np.zeros_like(res)
                evals = np.zeros_like(res)
                for k in range(self.size):
                    self.exp.optimizer.x0 = torch.tensor(points[k].reshape(2,1), requires_grad=True)
                    status, costs[k], _, evals[k] = self.exp.do()
                    if(status == 'global minimum'):
                        res[k] = 1
                data['convergence'][num_y-1-j, i] = np.mean(res)
                data['cost'][num_y-1-j, i] = np.mean(costs)
                data['evals'][num_y-1-j, i] = np.mean(evals)
            if self.sym:
                completed_num = (i + 2) * (i + 1) / 2
            else:
                completed_num = (i+1) * num_y 
            num_current = int(num_y - (abs_edge[1]-self.origin[1])/self.step)
            print("cost: {}, prob: {}".format(data['cost'][num_current, i], data['convergence'][num_current, i]) )
            print("complete: {} / {} ".format(int(completed_num), int(total_num)))
        end = time.time()
        if self.sym:
            num = self.edge[0] / self.step
            num = (num + 1) * num / 2
        else:
            num = num_x * num_y
        data['paras'] = self.paras
        print("avg probility of convergence: ", data['convergence'].sum() / num)
        print("avg cost: ", data['cost'].sum() / num)
        print("avg evals per exp: ", data['evals'].sum() / num)
        print("total time: {},  time per eval:{}\n".format(end - start, (end - start)/num/data['evals'].sum()))
        return data