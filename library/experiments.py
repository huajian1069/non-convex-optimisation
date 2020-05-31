import numpy as np
from library.optimiser import *
from library.post_analysis import *
from library.experiments import *

class single_experiment:
    def __init__(self, tol=0.1):
        self.tol = tol
    def set_objective(self, objective_func):
        self.objective_func = objective_func

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def do(self):
        optimal, optimum, statistics = self.optimizer.optimise(self.objective_func)
        if np.linalg.norm(optimal - self.objective_func.get_optimal()) < self.tol \
        or np.linalg.norm(optimum - self.objective_func.get_optimum()) < self.tol:
            statistics['status'] = 'global minimum'
        elif statistics['status'] != 'diverge':
            statistics['status'] = 'local minimum'
        if self.optimizer.verbose:
            print("Result: ", statistics['status'])
            print("found minimum: {}, minimum position: {}, evals: {}".format(optimum, optimal.ravel(), statistics['evals']))
        if self.optimizer.record == False:
            return statistics['status'], optimum, statistics['evals']
        else:
            statistics
            
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
        if self.sym and num_x == num_y:   
            total_num = (num_x + 1) * num_x / 2
        else:
            total_num = num_x * num_y
           
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
            if self.sym:
                completed_num = (i + 1) * i / 2 + j + 1
            else:
                completed_num = i * num_y + j + 1
            print("cost: {}, prob: {}".format(avg_cost[num_y-1-j, i], avg_res[num_y-1-j, i]) )
            print("complete: {} / {} ".format(int(completed_num), int(total_num)))
        if self.sym:
            num = self.edge[0] / self.step
            num = (num + 1) * num / 2
        else:
            num = num_x * num_y
        print("avg probility of converge: ", data['res'].sum() / num)
        print("avg cost: ", data['cost'].sum() / num)
        print("avg evals per exp: ", data['evals'].sum() / num)
        print("\n")
        data = {'x': position_x, 
                'y': position_y,
                'mask': mask,
                'res': avg_res,
                'cost': avg_cost,
                'evals': avg_evals
                'paras': self.paras}
        return data