import numpy as np
from abc import ABC, abstractmethod

class optimizer(ABC):
    @abstractmethod
    def set_parameters(self, para):
        '''
        input: parameters, in dictionary
        '''
        pass
    @abstractmethod
    def optimise(self, objective_cls):
        '''
        input: objective function class
        output: empirical found optimal, optimum, and statistics of procedure information
        '''
        pass
    
class adjust_optimizer(optimizer):
    def adjust(self, x0, obj):
        self.x0 = x0
        arg, val, stats = self.optimise(obj)
        return arg, val, stats['evals']
    
class cma_es(adjust_optimizer):
    def __init__(self):
        paras = {'x0': np.zeros((2,)),
                 'std': np.ones((2,)) * 3, 
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
            return (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - mean_old) / sigma 
        def update_pc(pc, sigma, ps, mean, mean_old):
            hsig = np.abs(ps) / np.sqrt(1 - (1 - cs)**(2 * iter_/lambda_)) / chiN < 1.4 + 2/(dim + 1)
            return (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - mean_old) / sigma
        def update_C(C, pc, x, mean_old, sigma):
            hsig = np.abs(ps) / np.sqrt(1 - (1 - cs)**(2 * iter_/lambda_)) / chiN < 1.4 + 2/(dim + 1)
            artmp = (1 / sigma) * (x - mean_old.reshape(1, dim))
            return (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T @ np.diag(weights) @ artmp
        def update_sigma(sigma, ps):
            return sigma * np.exp((cs / damps) * (np.linalg.norm(ps)/ chiN - 1))
        def is_not_moving(arg, val, pre_arg, pre_val, tol):
            dis_arg = np.linalg.norm(arg - pre_arg)
            dis_val = np.linalg.norm(val - pre_val)
            return (dis_arg < tol and dis_val < tol*1e5) or (dis_val < tol and dis_arg < tol*1e5) 

        if self.verbose:
            print("\n\n*******starting optimisation from intitial mean: ", self.x0.ravel())
        # User defined input parameters 
        dim = 2    
        sigma = 0.3
        D = self.std / sigma
        mean = self.x0

        # the size of solutions group
        lambda_ = 4 + int(3 * np.log(dim)) if self.cluster_size == None else self.cluster_size  
        # only best "mu" solutions are used to generate iterations
        mu = int(lambda_ / 2) if self.survival_size == None else self.survival_size
        # used to combine best "mu" solutions                                               
        weights = np.log(mu + 1/2) - np.log(np.arange(mu) + 1) 
        weights = weights / np.sum(weights)     
        mueff = np.sum(weights)**2 / np.sum(weights**2) 

        # Strategy parameter setting: Adaptation
        # time constant for cumulation for C
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)  
        # t-const for cumulation for sigma control
        cs = (mueff + 2) / (dim + mueff + 5)  
        # learning rate for rank-one update of C
        c1 = 2 / ((dim + 1.3)**2 + mueff)    
        # and for rank-mu update
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))  
        # damping for sigma, usually close to 1  
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/( dim + 1)) - 1) + cs                                                                 

        # Initialize dynamic (internal) strategy parameters and constants
        # evolution paths for C and sigma
        pc = np.zeros((dim, 1))     
        ps = np.zeros((dim, 1)) 
        # B defines the coordinate system
        B = np.eye(dim)       
        # covariance matrix C
        C = B * np.diag(D**2) * B.T 
        # C^-1/2 
        invsqrtC = B * np.diag(D**-1) * B.T   
        # expectation of ||N(0,I)|| == norm(randn(N,1)) 
        chiN = dim**0.5 * (1 - 1/(4 * dim) + 1 / (21 * dim**2))  

        # --------------------  Initialization --------------------------------  
        x, x_old, f = np.zeros((lambda_, dim)), np.zeros((lambda_, dim)), np.zeros((lambda_,))
        stats = {}
        stats['val'], stats['arg'] = [], []
        stats['x_adjust'] = []
        iter_eval, stats['evals_per_iter'] = np.zeros((lambda_, )), []
        stats['mean'], stats['std'] = [], []
        stats['status'] = None
        iter_, eval_ = 0, 0

        # initial data in record
        for i in range(lambda_):
            x[i] = (mean + np.random.randn(dim, 1)).ravel()
            f[i] = obj.func(x[i])
        idx = np.argsort(f)
        x_ascending = x[idx]
        if self.record:
            stats['arg'].append(x_ascending)
            stats['val'].append(f[idx])
            stats['mean'].append(mean)
            stats['std'].append(sigma * B @ np.diag(D))
            stats['evals_per_iter'].append(np.ones((lambda_,)))
            stats['x_adjust'].append(np.vstack((x.T.copy(), x.T.copy())))
        arg = x_ascending
        val = f[idx]
        pre_arg = x_ascending
        pre_val = f[idx]
        
        # optimise by iterations
        try:
            while iter_ < self.max_iter:
                iter_ += 1
                
                # generate candidate solutions with some stochastic elements
                for i in range(lambda_):
                    x[i] = (mean + sigma * B @ np.diag(D) @ np.random.randn(dim, 1)).ravel() 
                    x_old[i] = x[i]
                    x[i], f[i], eval_cnt = self.adjust_func.adjust(x[i], obj)
                    eval_ += eval_cnt
                    iter_eval[i] = eval_cnt
                # sort the value and positions of solutions 
                idx = np.argsort(f)
                x_ascending = x[idx]

                # update the parameter for next iteration
                mean_old = mean
                mean = update_mean(x_ascending[:mu])
                ps =   update_ps(ps, sigma, C, mean, mean_old)
                pc =   update_pc(pc, sigma, ps, mean, mean_old)
                sigma = update_sigma(sigma, ps)
                C =    update_C(C, pc, x_ascending[:mu], mean_old, sigma)
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eig(C)
                D = np.sqrt(D)
                invsqrtC = B @ np.diag(D**-1) @ B

                # record data during process for post analysis
                if self.record:
                    stats['arg'].append(x_ascending)
                    stats['val'].append(f[idx])
                    stats['mean'].append(mean)
                    stats['std'].append(sigma * B @ np.diag(D))
                    stats['evals_per_iter'].append(iter_eval.copy())
                    stats['x_adjust'].append(np.vstack((x_old.T.copy(), x.T.copy())))
                # stopping condition    
                arg = x_ascending
                val = f[idx]
                
                # check the stop condition
                if np.max(D) > (np.min(D) * 1e6):
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
                print('found minimum position = {}, found minimum = {}'.format(arg[0], val[0]))

        # carry statistics info before quit
        if self.record:
            stats['arg'] = np.array(stats['arg'])
            stats['val'] = np.array(stats['val'])
            stats['mean'] = np.array(stats['mean'])
            stats['std'] = np.array(stats['std'])
            stats['evals_per_iter'] = np.array(stats['evals_per_iter'])
            stats['x_adjust'] = np.array(stats['x_adjust'])
        stats['evals'] = eval_
        return arg[0], val[0], stats
    
class do_nothing(adjust_optimizer):
    def __init__(self, verbose=False):
        self.stats = {}
        self.stats['status'] = None
        self.stats['evals'] = 1
        self.verbose = False
        self.record = False
        self.x0 = np.array([10, 10]) 
        self.verbose = verbose
    def set_parameters(self, paras):
        self.verbose = paras['verbose']
        self.record = paras['record']
    def optimise(self, obj):
        if self.verbose:
            print("\n*******starting optimisation from intitial point: ", self.x0.ravel())
        return self.x0, obj.func(self.x0), self.stats
    
class round_off(adjust_optimizer):
    def __init__(self, verbose=False):
        self.stats = {}
        self.stats['status'] = None
        self.stats['evals'] = 1
        self.verbose = False
        self.record = False
        self.x0 = np.array([10, 10]) 
        self.verbose = verbose
    def set_parameters(self, paras):
        self.verbose = paras['verbose']
        self.record = paras['record']
    def optimise(self, obj):
        if self.verbose:
            print("\n\n*******starting optimisation from intitial point: ", self.x0.ravel())
        return np.round(self.x0), obj.func(self.x0), self.stats
    
class adam(adjust_optimizer):
    def __init__(self, verbose=False):
        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.max_iter = 10000
        self.tol = 1e-2
        self.verbose = verbose
        self.record = False
        self.x0 = np.array([10, 10]) 
        
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
        self.record = True if 'record' not in paras.keys() else paras['record']
        
    def optimise(self, obj):
        m_t = 0 
        v_t = 0 
        eval_cnt = 0
        x = self.x0.copy().reshape(2,)
        stats = {}
        stats['status'] = None
        if self.verbose:
            print("\n\n*******starting optimisation from intitial point: ", self.x0.ravel())
        while eval_cnt < self.max_iter:					#till it gets converged
            eval_cnt += 1
            g_t = obj.dfunc(x)		#computes the gradient of the stochastic function
            m_t = self.beta_1*m_t + (1-self.beta_1)*g_t	#updates the moving averages of the gradient
            v_t = self.beta_2*v_t + (1-self.beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
            m_cap = m_t/(1-(self.beta_1**eval_cnt))		#calculates the bias-corrected estimates
            v_cap = v_t/(1-(self.beta_2**eval_cnt))		#calculates the bias-corrected estimates
            x_prev = x								
            x = x - (self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)	#updates the parameters
            if(np.linalg.norm(x-x_prev) < self.tol):		#checks if it is converged or not
                break
        stats['evals'] = eval_cnt
        return x, obj.func(x), stats
    
class line_search(adjust_optimizer):
    def __init__(self, alpha=1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = 100
        self.tol = 1e-2
        self.stats = {}
        self.stats['status'] = None
        self.verbose = False
        self.record = False
        self.x0 = np.array([10, 10]) 
     
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
        x = self.x0.copy().reshape(2,)
        alpha_ = self.alpha
        tao = 0.5
        fx = obj.func(x)
        p = - obj.dfunc(x)
        fnx = obj.func(x + alpha_ * p)
        eval_cnt = 3
        if self.verbose:
            print("\n*******starting optimisation from intitial point: ", self.x0.ravel())
        for k in range(self.max_iter):
            while fnx > fx + alpha_ * self.beta * (-p @ p):
                alpha_ *= tao
                fnx = obj.func(x + alpha_ * p)
                eval_cnt += 1
            x += alpha_ * p
            fx = fnx
            p = -obj.dfunc(x)
            fnx = obj.func(x + alpha_ * p)
            eval_cnt += 2
            if np.linalg.norm(p) < self.tol:
                break
        self.stats['evals'] = eval_cnt
        return x, fnx, self.stats

class line_search_1step(adjust_optimizer):
    def __init__(self, alpha=1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = 4
        self.tol = 1e-2
        self.stats = {}
        self.stats['status'] = None
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
        x = self.x0.copy().reshape(2,)
        alpha_ = self.alpha
        tao = 0.5
        fx = obj.func(x)
        p = - obj.dfunc(x)
        eval_cnt = 2
        while obj.func(x + alpha_ * p) > fx + alpha_ * self.beta * (-p @ p):
            alpha_ *= tao
            eval_cnt += 1
        x += alpha_ * p
        self.stats['evals'] = eval_cnt
        return x, obj.func(x), self.stats