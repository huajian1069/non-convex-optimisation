def plot_distance(arg, val):
    distance_arg = np.linalg.norm(arg, axis=(1,2))
    distance_val = np.linalg.norm(val, axis=1)
    fig = plt.figure(figsize=(8, 4))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(np.arange(val.shape[0]), distance_arg, color='green', label='Frobenius norm \nof parameters')
    ax1.set_xlabel('iteration', fontsize=15)
    ax1.set_ylabel('distance in domain', color='green', fontsize=15)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()  
    ax2.plot(np.arange(val.shape[0]), distance_val, color='red', label='L2 norm \nof func value')
    ax2.set_ylabel('distance in codomain', color='red', fontsize=15)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    
def animate_moving_cluster(stats, num):
    def generate_points(stats, num):
        cluster_x = np.zeros((stats['mean'].shape[0], num, 2))
        for iter_ in range(stats['mean'].shape[0]):
            for n in range(num):
                cluster_x[iter_, n] = (stats['mean'][iter_] + stats['var'][iter_] @ np.random.randn(2, 1)).ravel()
        return cluster_x
    arg = generate_points(stats, num)
    arg_re = arg.reshape(-1, 2)
    x = np.array(arg_re[:,0])
    y = np.array(arg_re[:,1])
    eigVal_sqrts = np.linalg.norm(stats['var'], ord=2, axis=(1))
    def animate(i):
        plt.clf()
        ax = fig.add_subplot(1, 1, 1)
        mean = stats['mean'][i]
        eigVal_sqrt = eigVal_sqrts[i]
        eigVec = stats['var'][i] / eigVal_sqrt
        width, height = 2 * 3 * eigVal_sqrt
        angle = np.arctan2(eigVec[0,1], eigVec[0,0]) * 180 / np.pi
        e1 = patches.Ellipse(mean, width, height,
                             angle=-angle, linewidth=2, fill=False, zorder=2)
        ax.add_patch(e1)
        ax.scatter(mean[0], mean[1], c='black', s=15)
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        ax.set_xlim(np.min(x),  np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_title('iter=%d' % (i+1))
        #ax.grid(True)
        p = sns.scatterplot(x=x[int(i*num):int((i+1)*num)], y=y[int(i*num):int((i+1)*num)], 
                            color="r", hue=i, hue_norm=(0, arg.shape[0]), legend=False)
    fig = plt.figure(figsize=(8,4))
    ani = animation.FuncAnimation(fig, animate, frames=arg.shape[0], repeat=False, interval=500)
    plt.show()
    return ani

def animate_scatterplot_distance(arg, val, stats):
    arg_re = arg.reshape(-1, 2)
    x = np.array(arg_re[:,0])
    y = np.array(arg_re[:,1])
    distance_val = np.linalg.norm(val, axis=1)
    distance_arg = np.linalg.norm(arg, axis=(1,2))
    eigVal_sqrts = np.linalg.norm(stats['var'], ord=2, axis=(1))
    def animate(i):
        plt.clf()
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.set_xlim(np.min(x),  np.max(x))
        ax0.set_ylim(np.min(y), np.max(y))
        ax0.set_title('iter=%d, func distance=%.3f, domain distance=%.3f' % (i+1, distance_val[i+1], distance_arg[i+1]))
        mean = stats['mean'][i]
        eigVal_sqrt = eigVal_sqrts[i]
        eigVec = stats['var'][i] / eigVal_sqrt
        width, height = 2 * 3 * eigVal_sqrt
        angle = np.arctan2(eigVec[0,1], eigVec[0,0]) * 180 / np.pi
        e1 = patches.Ellipse(mean, width, height,
                             angle=-angle, linewidth=2, fill=False, zorder=2)
        ax0.add_patch(e1)
        ax0.scatter(mean[0], mean[1], c='black', s=15)
        ax0.axvline(c='grey', lw=1)
        ax0.axhline(c='grey', lw=1)
        ax0.set_xlim(np.min(x),  np.max(x))
        ax0.set_ylim(np.min(y), np.max(y))
        ax0.set_title('iter=%d' % (i+1))
        p = sns.scatterplot(x=x[int(i*6):int((i+1)*6)], y=y[int(i*6):int((i+1)*6)], color="r", hue=i, hue_norm=(0,val.shape[0]), legend=False)
        #p.tick_params(labelsize=17)
        #ax0.set_setp(p.lines,linewidth=7)

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.plot(np.arange(i+1), distance_arg[:i+1], color='green', label='Frobenius norm \nof parameters')
        ax1.set_xlim(0, val.shape[0])
        ax1.set_ylim(np.min(distance_arg)*0.9, np.max(distance_arg)*1.1)
        ax1.set_xlabel('iteration', fontsize=15)
        ax1.set_ylabel('distance in domain', color='green', fontsize=15)
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.legend(loc='lower left')
        
        ax2 = ax1.twinx()  
        ax2.plot(np.arange(i+1), distance_val[:i+1], color='red', label='L2 norm \nof func value')
        ax2.set_ylim(np.min(distance_val)*0.9, np.max(distance_val)*1.1)
        ax2.set_ylabel('distance in codomain', color='red', fontsize=15)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')
    fig = plt.figure(figsize=(8,4))
    ani = animation.FuncAnimation(fig, animate, frames=val.shape[0], repeat=False, interval=500)
    plt.show()
    return ani
'''
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
ani.save('Evolution.mp4', writer=writer)
'''

def print_mean_variance(stats):
    # print mean and variance of each iteration
    for i, a in enumerate(stats['var']):
        print('iter=',i, 'mean=', stats['mean'][i].T , 'var=\n', a@a.T, '\n')
        
def print_evaluations_per_iteration(stats):
    for i, iter_ in enumerate(stats['evals_per_iter']):
        print('iter=', i, '\n', iter_.squeeze(),'\n') 
    
def print_arguments_before_and_after_move(stats):
    for i, iter_ in enumerate(stats['x_vs_original']):
        print('iter=', i, '\nbefore\n', iter_[:2], '\nafter\n', iter_[2:], '\n') 
        
def generate_point_cloud(sigma, alpha, beta, adjust, func, dfunc, global_arg, global_val):
    def random_intial_mean(radius):
        rx = np.random.rand() * radius
        ry = np.random.rand() * radius 
        if(ry > rx):
            rx, ry = ry, rx
        return np.array([rx, ry])
    num = 500
    edge = 1000
    res = np.zeros((num, ))
    points = np.zeros((num, 2))
    result_strings = []
    exp = non_convex_optimisation(func, dfunc, global_arg, global_val)
    
    for i in range(num):
        points[i] = random_intial_mean(edge)
        record = points[i]
        exp.do_experiments(points[i].reshape(2,1), sigma, alpha, beta, adjust)
        val, arg, stats = exp.get_recorded_data()
        if(stats['status'] == 'd'):
            res[i] = 1
        elif(stats['status'] == 'l'):
            res[i] = 0.5
        else:
            res[i] = 0
    return res, points

def plot_prob_vs_radius(*args):
    def count_global_min(res, points):        
        distance = np.linalg.norm(points, axis=1)
        idx = np.argsort(distance)
        dis_ascending = distance[idx]
        res_ascending = res[idx]
        prob = np.zeros((num, ))
        for i in range(num):
            prob[i] = np.sum(res_ascending[:i+1] == 0) / (i + 1) 
        return dis_ascending, prob
    num = 500
    argc = len(args)
    assert argc%2 == 0
    pair_cnt = int(argc / 2)
    dis_ascendings = np.zeros((num, pair_cnt))
    probs = np.zeros((num, pair_cnt))
    for i in range(pair_cnt):
        dis_ascendings[:,i], probs[:,i] = count_global_min(args[i*2], args[i*2+1])
        
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, np.max(dis_ascendings))
    ax.set_ylim(0, 2)
    ax.set_xlabel('distance from origin', fontsize=13)
    ax.set_ylabel('prob of global minminum', fontsize=13)
    for i in range(pair_cnt):
        ax.plot(dis_ascendings[:,i], probs[:,i])
    
def plot_cloud_point(res, points):
    fig = plt.figure(figsize=(7,7))
    # one quadrant
    x1 = np.hstack((points[:,0], points[:,1]))
    y1 = np.hstack((points[:,1], points[:,0]))
    res1 = np.hstack((res, res))
    # two qudrant
    x2 = np.hstack((x1, -x1))
    y2 = np.hstack((y1, y1))
    res2 = np.hstack((res1, res1))
    # four qudrant
    x = np.hstack((x2, -x2))
    y = np.hstack((y2, -y2))
    hue = np.hstack((res2, res2))
    p = sns.scatterplot(x=x, y=y, color="r", hue=hue, hue_norm=(0, 1), legend=False)