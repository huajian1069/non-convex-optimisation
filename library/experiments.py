import numpy as np
from library.optimiser import *
from library.post_analysis import *
from library.experiments import *

class single_experiment:
    def set_objective(self, objective_func):
        self.objective_func = objective_func

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def do(self):
        optimal, optimum, statistics = self.optimizer.optimise(self.objective_func)
        if np.linalg.norm(optimal - self.objective_func.get_optimal()) < 1e-1 \
        or np.linalg.norm(optimum - self.objective_func.get_optimum()) < 1e-1:
            statistics['status'] = 'global minimum'
        elif statistics['status'] != 'diverge':
            statistics['status'] = 'local minimum'
        if self.optimizer.verbose:
            print("Result: ", statistics['status'])
            print("found minimum: {}, minimum position: {}, evals: {}".format(optimum, optimal.ravel(), statistics['evals']))
        if self.optimizer.record == False:
            return statistics['status'], optimum, statistics['evals']
        else:
            self.analyser = post_analysis(statistics, self.objective_func)
            
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
        avg_res = np.zeros((num_y, num_x))
        avg_cost = np.zeros_like(avg_res)
        avg_evals = np.zeros_like(avg_res)
        mask = np.ones_like(avg_res, dtype=np.uint8)
        position_x = np.zeros_like(avg_res)
        position_y = np.zeros_like(avg_res)
        
        for i, x in enumerate(np.arange(self.origin[0], abs_edge[0], self.step)):
            if self.sym:
                abs_edge[1] = self.origin[1] + x + self.step 
            else:
                abs_edge[1] = self.origin[1] + self.edge[1]
            for j, y in enumerate(np.arange(self.origin[1], abs_edge[1], self.step)):
                points = np.random.rand(self.size, 2) * self.step + np.array([x, y]) 
                mask[num_y-1-j, i] = 0
                position_x[num_y-1-j, i] = x
                position_y[num_y-1-j, i] = y
                # calculate the probility of getting global minimum 
                res = np.zeros((self.size, ))
                costs = np.zeros_like(res)
                evals = np.zeros_like(res)
                for k in range(self.size):
                    self.exp.optimizer.x0 = points[k].reshape(2,1)
                    status, costs[k], evals[k] = self.exp.do()
                    if(status == 'global minimum'):
                        res[k] = 1
                avg_res[num_y-1-j, i] = np.mean(res)
                avg_cost[num_y-1-j, i] = np.mean(costs)
                avg_evals[num_y-1-j, i] = np.mean(evals)
        data = {'x': position_x, 
                'y': position_y,
                'mask': mask,
                'res': avg_res,
                'cost': avg_cost,
                'evals': avg_evals}
        self.analyser = post_analysis_multiple(self.paras, data)
        return data