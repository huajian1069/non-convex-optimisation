import numpy as np
def cma_es_general(mean0, D, alpha, beta, adjust, func, dfunc, optimal, optimum):
    '''
    @param mean0: the initial mean of candidates
    @param D: control the initial variance of candidates
    @param alpha, beta: initial step size and control the armijo condition (only relevent for line search)
    @param adjust: adjust strategy to improve CMA-ES, do nothing or do line search
    @param func, dfunc: objective function and its derivative 
    @param optimal, optimum: optimal and optimum of objective function
    
    return r_val: record of func values during iterations
           r_arg: record of arguments during iterations
           stats: collection of the other recorded datas
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
        # User defined input parameters (need to be edited)
    
    print("*******starting soon, intitial mean: ********\n", mean0)
    dim = 2
    mean = mean0
    sigma = 0.3
    D = D / sigma
    tolerance = 1e-6
    max_iter = 400
    
    # Strategy parameter setting: Selection  
    lambda_ = 4 + int(3 * np.log(dim))       # the size of solutions group
    mu = int(lambda_ / 2)     # only best "mu" solutions are used to generate iterations
    weights = np.log(mu + 1/2) - np.log(np.arange(mu) + 1) 
    weights = weights / np.sum(weights)      # used to combine best "mu" solutions
    mueff = np.sum(weights)**2 / np.sum(weights**2) 
    
    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)  # time constant for cumulation for C
    cs = (mueff + 2) / (dim + mueff + 5)  # t-const for cumulation for sigma control
    c1 = 2 / ((dim + 1.3)**2 + mueff)    # learning rate for rank-one update of C
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))  # and for rank-mu update
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/( dim + 1)) - 1) + cs  # damping for sigma, usually close to 1                                                                 
        
    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros((dim, 1))     # evolution paths for C and sigma
    ps = np.zeros((dim, 1)) 
    B = np.eye(dim)                        # B defines the coordinate system
    C = B * np.diag(D**2) * B.T            # covariance matrix C
    invsqrtC = B * np.diag(D**-1) * B.T    # C^-1/2 
    chiN = dim**0.5 * (1 - 1/(4 * dim) + 1 / (21 * dim**2))  # expectation of ||N(0,I)|| == norm(randn(N,1)) 

    # --------------------  Initialization --------------------------------  
    x, x_old, f = np.zeros((lambda_, dim)), np.zeros((lambda_, dim)), np.zeros((lambda_,))
    r_val, r_arg = [], []
    r_x_adjust = []
    iter_eval, r_iter_eval = np.zeros((lambda_, )), []
    r_means, r_vars = [], []
    iter_, eval_ = 0, 0
    stats = {}
    stats['status'] = 'g'
    
    # initial data in record
    for i in range(lambda_):
        x[i] = (mean + np.random.randn(dim, 1)).ravel()
        f[i] = func(x[i])
    idx = np.argsort(f)
    x_ascending = x[idx]
    r_arg.append(x_ascending)
    r_val.append(f[idx])
    r_means.append(mean)
    r_vars.append(sigma * B @ np.diag(D))
    r_iter_eval.append(np.ones((lambda_,)))
    r_x_adjust.append(np.vstack((x.T.copy(), x.T.copy())))
    
    # optimise by iterations
    try:
        while iter_ < max_iter:
            iter_ += 1
            # generate candidate solutions with some stochastic elements
            for i in range(lambda_):
                x[i] = (mean + sigma * B @ np.diag(D) @ np.random.randn(dim, 1)).ravel() 
                x_old[i] = x[i]
                x[i], eval_cnt = adjust(x[i], alpha, beta, func, dfunc)
                f[i] = func(x[i])
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
            r_arg.append(x_ascending)
            r_val.append(f[idx])
            r_means.append(mean)
            r_vars.append(sigma * B @ np.diag(D))
            r_iter_eval.append(iter_eval.copy())
            r_x_adjust.append(np.vstack((x_old.T.copy(), x.T.copy())))

            # check the stop condition
            if np.max(D) > (np.min(D) * 1e6) or np.linalg.norm(r_arg[-1] - r_arg[-2]) < tolerance \
                        or np.linalg.norm(r_val[-1] - r_val[-2]) < tolerance:
                break
    except np.linalg.LinAlgError as err:
            stats['status'] = 'd'
            print('raise LinAlgError!')
    
    # analyse the result of optimisation at end
    if (np.max(D) > (np.min(D) * 1e2) and np.linalg.norm(mean) > (np.linalg.norm(mean0) * 1e1)) \
                or np.linalg.norm(mean) > (np.linalg.norm(mean0) * 1e2) or stats['status'] == 'd':
        print('diverge!!')
        stats['status'] = 'd'
    elif np.linalg.norm(x - optimal) < 1e-1 or np.linalg.norm(f - optimum) < 1e-1:
        print('Global minimum')
        stats['status'] = 'g'
    else:
        print('stuck in local minimum!!')
        stats['status'] = 'l'
    print('eigenvalue of variance = {}'.format(D))
    print('min = {}, total iterations = {}, total evaluatios = {}\n position = {} {}\n'.format(f[0], iter_, eval_, x_ascending[0, 0], x_ascending[0, 1]))
    
    # carry process info before quit
    stats['evals_per_iter'] = np.array(r_iter_eval)
    stats['mean'] = np.array(r_means)
    stats['var'] = np.array(r_vars)
    stats['x_adjust'] = np.array(r_x_adjust)
    return np.array(r_val), np.array(r_arg), stats

def line_search(x0, alpha, beta, f, deri_f):
    '''
    @param x0: initial point position
    @param alpha: initial step size
    @param beta: control the armijo condition
    @return x: point position after moving to local minimum
    '''
    x = x0.copy()
    alpha0 = alpha
    beta = 0.1
    tao = 0.5
    k_max = 100
    tolerance = 1e-2
    fx = f(x)
    p = - deri_f(x)
    fnx = f(x + alpha * p)
    evaluation_cnt = 4
    for k in range(k_max):
        while fnx > fx + alpha * beta * (-p @ p):
            alpha *= tao
            fnx = f(x + alpha * p)
            evaluation_cnt += 1
        x += alpha * p
        fx = fnx
        p = -deri_f(x)
        fnx = f(x + alpha * p)
        evaluation_cnt += 2
        if np.linalg.norm(p) < tolerance:
            break
    return x, evaluation_cnt

def do_nothing(x0, alpha, beta, f, deri_f):
    return x0, 1

def round_off(x0, alpha, beta, f, deri_f):
    return np.round(x0), 1

def line_search_1step(x0, alpha, beta, f, deri_f):
    '''
    @param x0: initial point position
    @param alpha: initial step size
    @param beta: control the armijo condition
    @return x: point position after moving to local minimum
    '''
    x = x0.copy()
    alpha0 = alpha
    beta = 0.1
    tao = 0.5
    tolerance = 1e-2
    fx = f(x)
    p = - deri_f(x)
    evaluation_cnt = 4
    while f(x + alpha * p) > fx + alpha * beta * (-p @ p):
        alpha *= tao
        evaluation_cnt += 1
    x += alpha * p
    return x, evaluation_cnt

