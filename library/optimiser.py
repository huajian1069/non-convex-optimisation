import numpy as np
import torch
from abc import ABC, abstractmethod


class optimizer(ABC):
    @abstractmethod
    def set_parameters(self, para):
        '''
        input: parameters, in dictionary
        '''
        pass
    @abstractmethod
    def optimise(self, obj):
        '''
        input: objective function class
        output: empirical found optimal, optimum, and statistics of procedure information
        '''
        pass
    
class adjust_optimizer(optimizer):
    def adjust(self, x0, obj):
        self.x0 = x0
        return self.optimise(obj)
    
class cma_es(adjust_optimizer):
    def __init__(self, dim=2):
        self.dim = dim
        paras = {'x0': torch.zeros((dim,)),
                 'std': torch.ones((dim,)) * 3, 
                 'tol': 1e-5, 
                 'adjust_func': do_nothing(), 
                 'record': False, 
                 'verbose': False}
        self.set_parameters(paras)
    def set_parameters(self, paras):
        self.paras = paras
        self.x0 = paras['x0'] 
        self.std = paras['std']
        self.tol = paras['tol']
        self.adjust_func = paras['adjust_func']
        self.max_iter = 400 if 'max_iter' not in paras.keys() else paras['max_iter']
        # set none to use default value 
        self.cluster_size = None if 'cluster_size' not in paras.keys() else paras['cluster_size']
        self.survival_size = None if 'survival_size' not in paras.keys() else paras['survival_size']
        self.record = True if 'record' not in paras.keys() else paras['record']
        self.verbose = True if 'verbose' not in paras.keys() else paras['verbose']
    def optimise(self, obj):
        '''
        @param obj: objective function class instance
        return arg: found minimum arguments
               val: found minimum value
               stats: collection of recorded statistics for post-analysis
        '''                  
        def update_mean(x):
            return (weights @ x).reshape(dim, 1)
        def update_ps(ps, sigma, C, mean, mean_old):
            return (1 - cs) * ps + torch.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - mean_old) / sigma 
        def update_pc(pc, sigma, ps, mean, mean_old):
            hsig = (torch.norm(ps) / torch.sqrt(1 - (1 - cs)**(2 * iter_/lambda_)) / chiN < 1.4 + 2/(dim + 1)).int()
            return (1 - cc) * pc + hsig * torch.sqrt(cc * (2 - cc) * mueff) * (mean - mean_old) / sigma
        def update_C(C, pc, x, mean_old, sigma):
            hsig = (torch.norm(ps) / torch.sqrt(1 - (1 - cs)**(2 * iter_/lambda_)) / chiN < (1.4 + 2/(dim + 1))).int()
            artmp = (1 / sigma) * (x - mean_old.reshape(1, dim))
            return (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T @ torch.diag(weights) @ artmp
        def update_sigma(sigma, ps):
            return sigma * torch.exp((cs / damps) * (torch.norm(ps)/ chiN - 1))
        def is_not_moving(arg, val, pre_arg, pre_val, tol):
            dis_arg = torch.norm(arg - pre_arg, dim=1).mean()
            dis_val = torch.abs(val - pre_val).mean()
            return (dis_arg < tol and dis_val < tol) 

        if self.verbose:
            print("\n\n*******starting optimisation from intitial mean: ", self.x0.squeeze().detach.numpy())
        # User defined input parameters 
        dim = self.dim
        sigma = 0.3
        D = self.std / sigma
        self.mean = self.x0.reshape(dim, 1)
        # the size of solutions group
        lambda_ = 4 + int(3 * np.log(dim)) if self.cluster_size == None else self.cluster_size  
        # only best "mu" solutions are used to generate iterations
        mu = int(lambda_ / 2) if self.survival_size == None else self.survival_size
        # used to combine best "mu" solutions                                               
        weights = np.log(mu + 1/2) - torch.log(torch.arange(mu, dtype=torch.float) + 1) 
        weights = (weights / torch.sum(weights)).float()    
        mueff = 1 / torch.sum(weights**2) 

        # Strategy parameter setting: Adaptation
        # time constant for cumulation for C
        self.cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)  
        # t-const for cumulation for sigma control
        self.cs = (mueff + 2) / (dim + mueff + 5)  
        # learning rate for rank-one update of C
        self.c1 = 2 / ((dim + 1.3)**2 + mueff)    
        # and for rank-mu update
        self.cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))  
        # damping for sigma, usually close to 1  
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/( dim + 1)) - 1) + cs                                                                 

        # Initialize dynamic (internal) strategy parameters and constants
        # evolution paths for C and sigma
        self.pc = torch.zeros((dim, 1))     
        self.ps = torch.zeros((dim, 1)) 
        # B defines the coordinate system
        B = torch.eye(int(dim))       
        # covariance matrix C
        C = B * torch.diag(D**2) * B.T 
        # C^-1/2 
        invsqrtC = B * torch.diag(D**-1) * B.T   
        # expectation of ||N(0,I)|| == norm(randn(N,1)) 
        chiN = dim**0.5 * (1 - 1/(4 * dim) + 1 / (21 * dim**2))  

        # --------------------  Initialization --------------------------------  
        x, x_old, f = torch.zeros((lambda_, dim)), torch.zeros((lambda_, dim)), torch.zeros((lambda_,))
        stats = {}
        inner_stats = {}
        stats['inner'] = []
        stats['val'], stats['arg'] = [], []
        stats['x_adjust'] = []
        iter_eval, stats['evals_per_iter'] = torch.zeros((lambda_, )), []
        inner_stats = [{}] * lambda_
        stats['mean'], stats['std'] = [], []
        stats['status'] = None
        iter_, eval_ = 0, 0
        # initial data in record
        for i in range(lambda_):
            x[i,:] = (mean + torch.randn(dim, 1)).squeeze()
            f[i] = obj.func(x[i])
        idx = torch.argsort(f)
        x_ascending = x[idx]
        if self.record:
            stats['inner'].append(inner_stats.detach().numpy())
            stats['arg'].append(x_ascending.detach().numpy())
            stats['val'].append(f[idx].detach().numpy())
            stats['mean'].append(mean.detach().numpy())
            stats['std'].append(sigma * B @ torch.diag(D))
            stats['evals_per_iter'].append(torch.ones((lambda_,)).detach().numpy())
            stats['x_adjust'].append(np.vstack((x.T.clone().detach().numpy(), x.T.clone().detach().numpy())))
        arg = x_ascending
        val = f[idx]
        pre_arg = x_ascending
        pre_val = f[idx]
        best_val = 1e4
        best_arg = None
        
        # optimise by iterations
        try:
            while iter_ < self.max_iter:
                iter_ += 1
                # generate candidate solutions with some stochastic elements
                for i in range(lambda_):
                    x[i] = (mean + sigma * B @ torch.diag(D) @ torch.randn(dim, 1)).squeeze()
                    x_old[i] = x[i]
                    x[i], f[i], inner_stats[i] = self.adjust_func.adjust(x[i].clone().detach().requires_grad_(True), obj)
                    eval_ += inner_stats[i]['evals']
                    iter_eval[i] = inner_stats[i]['evals']
                # sort the value and positions of solutions 
                idx = torch.argsort(f)
                x_ascending = x[idx]

                # update the parameter for next iteration
                mean_old = mean
                mean = update_mean(x_ascending[:mu])
                ps =   update_ps(ps, sigma, C, mean, mean_old)
                pc =   update_pc(pc, sigma, ps, mean, mean_old)
                sigma = update_sigma(sigma, ps)
                C =    update_C(C, pc, x_ascending[:mu], mean_old, sigma)
                C = torch.triu(C) + torch.triu(C, 1).T
                D, B = torch.eig(C, eigenvectors=True)
                D = torch.sqrt(D[:,0])
                invsqrtC = B @ torch.diag(D**-1) @ B

                # record data during process for post analysis
                if self.record:
                    stats['inner'].append(inner_stats.clone().detach().numpy())
                    stats['arg'].append(x_ascending.detach().numpy())
                    stats['val'].append(f[idx].detach().numpy())
                    stats['mean'].append(mean.detach().numpy())
                    stats['std'].append((sigma * B @ np.diag(D)).detach().numpy())
                    stats['evals_per_iter'].append(iter_eval.clone().detach().numpy())
                    stats['x_adjust'].append(np.vstack((x_old.T.clone().detach().numpy(), x.T.clone().detach().numpy())))
                # stopping condition    
                arg = x_ascending
                val = f[idx]
                if best_val > val[0]:
                    best_val = val[0]
                    best_arg = arg[0]              
                # check the stop condition
                if torch.max(D) > (torch.min(D) * 1e6):
                    stats['status'] = 'diverge'
                    print('diverge, concentrate in low dimension manifold')
                    break
                if is_not_moving(arg, val, pre_arg, pre_val, self.tol) :
                    break
                pre_arg = arg
                pre_val = val
        except np.linalg.LinAlgError as err:
            stats['status'] = 'diverge'
            print('diverge, raise LinAlgError!')
        finally:
            if self.verbose:
                print('eigenvalue of variance = {}'.format(D))
                print('total iterations = {}, total evaluatios = {}'.format(iter_, eval_))
                print('found minimum position = {}, found minimum = {}'.format(best_arg.detach().numpy(), best_val.detach().numpy()))

        # carry statistics info before quit
        if self.record:
            stats['arg'] = np.array(stats['arg'])
            stats['val'] = np.array(stats['val'])
            stats['mean'] = np.array(stats['mean'])
            stats['std'] = np.array(stats['std'])
            stats['evals_per_iter'] = np.array(stats['evals_per_iter'])
            stats['x_adjust'] = np.array(stats['x_adjust'])
        stats['evals'] = eval_
        return best_arg, best_val, stats
    
class do_nothing(adjust_optimizer):
    def __init__(self, dim=2, verbose=False):
        self.stats = {}
        self.stats['status'] = None
        self.stats['evals'] = 1
        self.verbose = False
        self.record = False
        self.x0 = torch.zeros((dim,))
        self.verbose = verbose
    def set_parameters(self, paras):
        self.verbose = paras['verbose']
        self.record = paras['record']
    def optimise(self, obj):
        if self.verbose:
            print("\n*******starting optimisation from intitial point: ", self.x0.squeeze().detach().numpy())
        return self.x0, obj.func(self.x0), self.stats
    
class round_off(adjust_optimizer):
    def __init__(self, dim=2, verbose=False):
        self.stats = {}
        self.stats['status'] = None
        self.stats['evals'] = 1
        self.verbose = False
        self.record = False
        self.x0 = torch.zeros((dim,))
        self.verbose = verbose
    def set_parameters(self, paras):
        self.verbose = paras['verbose']
        self.record = paras['record']
    def optimise(self, obj):
        if self.verbose:
            print("\n\n*******starting optimisation from intitial point: ", self.x0.squeeze().detach().numpy())
        return np.round(self.x0), obj.func(self.x0), self.stats
    
class adam(adjust_optimizer):
    def __init__(self, alpha=0.01, verbose=False, dim=2):
        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-11
        self.max_iter = 10000
        self.tol = 1e-3
        self.verbose = verbose
        self.record = False
        self.x0 = torch.zeros((dim,))
        
    def set_parameters(self, paras):
        self.paras = paras
        self.x0 = paras['x0']
        self.alpha = paras['alpha']
        self.beta_1 = paras['beta_1']
        self.beta_2 = paras['beta_2']
        self.epsilon = paras['epsilon']
        self.max_iter = paras['max_iter']
        self.tol = paras['tol']
        self.verbose = True if 'verbose' not in paras.keys() else paras['verbose']
        self.record = False if 'record' not in paras.keys() else paras['record']
        
    def optimise(self, obj):
        m_t = 0 
        v_t = 0 
        eval_cnt = 0
        x = self.x0
        stats = {}
        stats['status'] = None
        stats['gradient_before_after'] = []
        stats['arg'] = []
        stats['val'] = []
        if self.record:
            stats['arg'].append(x.clone().detach().numpy())
            stats['val'].append(obj.func(x).detach().numpy())
            stats['gradient_before_after'].append([obj.dfunc(x).detach().numpy(), obj.dfunc(x).detach().numpy()])
        if self.verbose:
            print("\n\n*******starting optimisation from intitial point: ", self.x0.squeeze().detach().numpy())
        while eval_cnt < self.max_iter:					#till it gets converged
            eval_cnt += 1
            x = x.clone().detach().requires_grad_(True)
            g_t = obj.dfunc(x)		#computes the gradient of the stochastic function
            m_t = self.beta_1*m_t + (1-self.beta_1)*g_t	#updates the moving averages of the gradient
            v_t = self.beta_2*v_t + (1-self.beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
            m_cap = m_t/(1-(self.beta_1**eval_cnt))		#calculates the bias-corrected estimates
            v_cap = v_t/(1-(self.beta_2**eval_cnt))		#calculates the bias-corrected estimates
            x_prev = x.clone()								
            est_df = (m_cap)/(torch.sqrt(v_cap)+self.epsilon)
            with torch.no_grad():
                x -= self.alpha * est_df 	#updates the parameters
            if self.record:
                stats['arg'].append(x.clone().detach().numpy())
                stats['val'].append(obj.func(x).detach().numpy())
                stats['gradient_before_after'].append([g_t.detach().numpy(), est_df.detach().numpy()])
            if(torch.norm(x-x_prev) < self.tol):		#checks if it is converged or not
                break
        if self.verbose:
            print('total evaluatios = {}'.format(eval_cnt))
            print('gradient at stop position = {},\nmodified graident = {}'.format(g_t, est_df))
            print('found minimum position = {}, found minimum = {}'.format(x.detach().numpy(), obj.func(x).detach().numpy()))
        stats['arg'] = np.array(stats['arg'])
        stats['val'] = np.array(stats['val'])
        stats['gradient_before_after'] = np.array(stats['gradient_before_after'])
        stats['evals'] = eval_cnt
        return x, obj.func(x), stats
    
class line_search(adjust_optimizer):
    def __init__(self, alpha=1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = 100
        self.tol = 1e-2
        self.verbose = False
        self.record = False
     
    def set_parameters(self, paras):
        self.paras = paras
        self.x0 = paras['x0']
        self.alpha = paras['alpha']
        self.beta = paras['beta']
        self.max_iter = paras['max_iter']
        self.tol = paras['tol']
        self.verbose = True if 'verbose' not in paras.keys() else paras['verbose']
        self.record = True if 'record' not in paras.keys() else paras['record']
    def optimise(self, obj):
        '''
        @param x0: initial point position
        @param alpha: initial step size
        @param beta: control the armijo condition
        @return x: point position after moving to local minimum
        '''
        x = self.x0
        alpha_ = self.alpha
        tao = 0.5
        fx = obj.func(x)
        p = - obj.dfunc(x)
        fnx = obj.func(x + alpha_ * p)
        eval_cnt = 3
        stats = {}
        stats['status'] = None
        stats['gradient'] = []
        stats['arg'] = []
        stats['val'] = []
        if self.record:
            stats['arg'].append(x.clone().detach().numpy())
            stats['val'].append(fx.detach().numpy())
            stats['gradient'].append(-p.detach().numpy())
        if self.verbose:
            print("\n*******starting optimisation from intitial point: ", self.x0.squeeze().detach().numpy())
        for k in range(self.max_iter):
            while fnx > fx + alpha_ * self.beta * (-p @ p):
                alpha_ *= tao
                fnx = obj.func(x + alpha_ * p)
                eval_cnt += 1
            with torch.no_grad():
                x += alpha_ * p
            fx = fnx
            x = x.clone().detach().requires_grad_(True)
            p = -obj.dfunc(x)
            fnx = obj.func(x + alpha_ * p)
            eval_cnt += 2
            if self.record:
                stats['arg'].append(x.clone().detach().numpy())
                #print(eval_cnt, stats['arg'])
                stats['val'].append(fx.detach().numpy())
                stats['gradient'].append(-p.detach().numpy())
            if torch.norm(p) < self.tol:
                break
        stats['evals'] = eval_cnt
        if self.verbose:
            print('total evaluatios = {}'.format(eval_cnt))
            print('gradient at stop position = {}'.format(-p.detach().numpy()))
            print('found minimum position = {}, found minimum = {}'.format(x.detach().numpy(), fx.detach().numpy()))
        stats['arg'] = np.array(stats['arg'])
        stats['val'] = np.array(stats['val'])
        stats['gradient'] = np.array(stats['gradient'])
        return x, fnx, stats
