import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib import patches
import seaborn as sns

class post_analy1d:
    def __init__(self, stats):
        self.stats = stats
        self.n = self.stats['val'].shape[0]
        #self.xs = np.linspace(np.min(self.stats['arg'])-2, np.max(self.stats['arg'])+2, 150)
        self.trace = stats['trace']
            
    def animate_moving_position(self):
        def animate(i):
            #plt.clf()
            ax = fig.add_subplot(1, 1, 1)    
            ax.set_title('iter=%d' % (i+1))
            p = sns.scatterplot(x=arg[i], y=val[i], color="red", hue=i,
                            hue_norm=(0, self.n), s=73,legend=False)
            plt.plot(self.xs, self.fs, c="green")
        if self.stats['arg'].shape[1] > 1:
            # only leave the first column, the smallest candidate
            arg = self.stats['arg'][:,0]
            val = self.stats['val'][:,0]
        else:
            arg = self.stats['arg']
            val = self.stats['val']
        fig = plt.figure(figsize=(8,4))
        ani = animation.FuncAnimation(fig, animate, frames=self.n-1, repeat=False, interval=500)
        return ani
    
    def animate_moving_cluster(self):
        def animate(i):
            plt.clf()
            ax = fig.add_subplot(1, 1, 1)    
            ax.set_title('iter=%d' % (i+1))
            ax.set_xlim(np.min(arg), np.max(arg))
            ax.set_ylim(np.min(val), np.max(val))
            ax.axvline(self.stats['mean'][i], c='blue', lw=1)
            ax.axvline(self.stats['mean'][i] - 1 * self.stats['std'][i], c='grey', lw=1)
            ax.axvline(self.stats['mean'][i] + 1 * self.stats['std'][i], c='grey', lw=1)
            p = sns.scatterplot(x=arg[i], y=val[i], color="red", hue=i,
                            hue_norm=(0, self.n), s=73,legend=False)
            plt.plot(self.xs, self.fs, c="green")
        arg = self.stats['arg'].squeeze()
        val = self.stats['val'].squeeze()
        fig = plt.figure(figsize=(8,4))
        ani = animation.FuncAnimation(fig, animate, frames=self.n-1, repeat=False, interval=500)
        return ani
    
    def plot_grandient_before_after(self):
        x = np.arange(stats['gradient_before_after'].shape[0]-1)  # the label locations
        width = 0.35  # the width of the bars
        num = stats['gradient_before_after'].shape[2]

        fig = plt.figure(figsize=(8, 4*num))
        for i in range(num):
            ax = fig.add_subplot(num, 1, i+1)
            ax.bar(x-width/2, self.stats['gradient_before_after'][1:, 0, i], width, color="b", label='original')
            ax.plot(x-width/2, self.stats['gradient_before_after'][1:, 0, i], color="b")

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('gradient')
            if i == 0:
                ax.set_title('gradient before and after adjustment')
            #ax.set_xticks(x)
            ax.set_yticks(np.linspace(-10, 10, 11))
            ax.axhline(c='grey', lw=1)
            ax.set_ylim(-14, 14)
            ax.legend()

            ax = ax.twinx()  
            rects2 = ax.bar(x+width/2, self.stats['gradient_before_after'][1:, 1, i], width, color="y", label='moving averge')
            ax.plot(x+width/2, self.stats['gradient_before_after'][1:, 1, i], color="y")
            ax.set_ylabel('gradient')
            #ax.set_xticks(x)
            ax.set_yticks(np.linspace(-1, 1, 11))
            ax.set_ylim(-1.4, 1.4)
            ax.legend(loc='lower right') 

class post_analysis_multiple_cloud():
    def __init__(self, stats):
        self.points = stats['points']
        self.res = stats['res']
        self.num = stats['res'].shape[0]
    def plot_prob_vs_radius(self, *args):
        def count_global_min(res, points):        
            distance = np.linalg.norm(points, axis=1)
            idx = np.argsort(distance)
            dis_ascending = distance[idx]
            res_ascending = res[idx]
            prob = np.zeros((self.num, ))
            for i in range(self.num):
                prob[i] = np.sum(res_ascending[:i+1] == 1) / (i + 1) 
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

class post_analysis_zone:
    def __init__(self, data):
        self.paras = data['paras']
        self.origin = paras['origin']
        self.edge = paras['edge']
        self.step = paras['step']
        
        self.x = data['x']
        self.y = data['y']
        self.prob = data['convergence']
        self.cost = data['cost']
        self.evals = data['evals']
        self.mask = data['mask']
        
        self.xlabel = np.arange(self.origin[0], self.origin[0] + self.edge[0], self.step) + round(self.step/2)
        self.ylabel = np.arange(self.edge[1] + self.origin[1], self.origin[1], -self.step) - round(self.step/2)
        
    def __setup_axis(self, ax):
        sns.axes_style("white")
        ax.set_xticklabels(self.xlabel)
        ax.set_yticklabels(self.ylabel)
        
    def plot_scatter(self):
        fig = plt.figure(figsize=(10,10))
        sc = plt.scatter(self.x.ravel(), self.y.ravel(), c=self.prob.ravel(), marker='o', vmin=0, vmax=1, s=35, cmap='YlGnBu')
        plt.colorbar(sc)
        
    def plot_hotmap_prob(self):
        fig = plt.figure(figsize=(13, 13))
        ax = sns.heatmap(self.prob, mask=self.mask, vmin=0, vmax=1, square=True,  cmap="YlGnBu")
        self.__setup_axis(ax)

    def plot_hotmap_cost(self):
        fig = plt.figure(figsize=(13, 13))
        ax = sns.heatmap(self.cost, mask=self.mask, vmin=0, square=True,  cmap="YlGnBu")
        self.__setup_axis(ax)

    def plot_hotmap_evals(self):
        fig = plt.figure(figsize=(13, 13))
        ax = sns.heatmap(self.evals, mask=self.mask, vmin=0, square=True,  cmap="YlGnBu")  
        self.__setup_axis(ax)


        
class post_analysis_single():
    def __init__(self, stats):
        self.stats = stats
        self.optimal = stats['optimal']
        self.optimum = stats['optimum']
    def print_mean_variance(self):
        # print mean and variance of each iteration
        for i, a in enumerate(self.stats['std']):
            print('iter=',i, 'mean=', self.stats['mean'][i].T , '\nvar=\n', a@a.T, '\n')

    def print_evaluations_per_iteration(self):
        for i, iter_ in enumerate(self.stats['evals_per_iter']):
            print('iter=', i, '\n', iter_.squeeze(),'\n') 

    def print_points_before_and_after_adjust(self):
        for i, iter_ in enumerate(self. stats['x_adjust']):
            print('iter=', i, '\nbefore\n', iter_[:2], '\nafter\n', iter_[2:], '\n') 

    def __cal_distance(self):
        self.distance_arg = np.linalg.norm(self.stats['arg'] - self.optimal.reshape(1,1,2), axis=(1,2))
        self.distance_val = self.stats['val']
            
    def plot_distance(self):
        self.__cal_distance()
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 1, 1)
        self.__plot_distance_common(ax1, self.stats['val'].shape[0]-1)
    
    def plot_moving_cluster(self):
        fig = plt.figure(figsize=(9, 9))
        row = col = 3
        unit = int(self.stats['val'].shape[0]/(row * col)/2)
        for i in range(row):
            for j in range(col):
                ax=fig.add_subplot(row, col, 1 + row * i + j)
                # draw x-axis and y-axis
                ax.axvline(c='grey', lw=1)
                ax.axhline(c='grey', lw=1)
                # draw the position of optimal 
                ax.scatter(self.optimal[0], self.optimal[1], c='red', s=15)
                ax.scatter(x=self.stats['arg'][unit * (row * i + j),:,0], y=self.stats['arg'][unit * (row * i + j),:,1], 
                           c=self.stats['val'][unit * (row * i + j)], vmin = 0, vmax = 10)
                ax.set_title("%d / %d"%(unit * (row * i + j), self.stats['arg'].shape[0]))
                plt.xlim([-4, 4])
                plt.ylim([-4, 4])

    def __plot_distance_common(self, ax1, i):
        ax1.plot(np.arange(i), self.distance_arg[1:i+1], color='green', label='Frobenius norm \nof parameters')
        ax1.set_xlim(0, self.stats['val'].shape[0])
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
        
    def __draw_ellipse(self, ax, mean, std):
        eigVal_sqrt = np.linalg.norm(std, ord=2, axis=(0))
        eigVec = std / eigVal_sqrt
        width, height = 2 * 3 * eigVal_sqrt
        angle = np.arctan2(eigVec[0,1], eigVec[0,0]) * 180 / np.pi
        e1 = patches.Ellipse(mean, width, height,
                             angle=-angle, linewidth=2, fill=False, zorder=2)
        ax.add_patch(e1)
        ax.scatter(mean[0], mean[1], c='black', s=15)

    def __setup_scatter(self, ax, i):
        '''
        set up the plot of CMA-ES candidates at i-th iteration
        '''
        # draw x-axis and y-axis
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        # draw the position of optimal 
        ax.scatter(self.optimal[0], self.optimal[1], c='red', s=15)
        # draw the trail of local minimum
        if 'trail' in self.stats.keys():
            ax.scatter(self.stats['trail'][0], self.stats['trail'][1], c='red', s=11)
        # draw candidates on scatter plot
        ax.set_xlim(np.min(self.stats['arg'][:,:,0]), np.max(self.stats['arg'][:,:,0]))
        ax.set_ylim(np.min(self.stats['arg'][:,:,1]), np.max(self.stats['arg'][:,:,1]))
        p = sns.scatterplot(x=self.stats['arg'][i,:,0], y=self.stats['arg'][i,:,1], color="r", hue=i, hue_norm=(0, self.stats['val'].shape[0]), legend=False)
        # draw ellipse representing 3 sigma areas of normal distribution
        self.__draw_ellipse(ax, self.stats['mean'][i], self.stats['std'][i])

    def animate_moving_cluster(self):
        def animate(i):
            plt.clf()
            ax = fig.add_subplot(1, 1, 1)    
            ax.set_title('iter=%d' % (i+1))
            self.__setup_scatter(ax, i+1)
        fig = plt.figure(figsize=(8,4))
        ani = animation.FuncAnimation(fig, animate, frames=self.stats['arg'].shape[0]-1, repeat=False, interval=500)
        return ani

    def animate_scatterplot_distance(self):
        def animate(i):
            plt.clf()
            # draw scatter and ellipse
            ax0 = fig.add_subplot(2, 1, 1)
            ax0.set_title('iter=%d, func_dist=%.3f,  arg_dist=%.3f, mean=(%.3f, %.3f)' % (i+1, self.distance_val[i+1], self.distance_arg[i+1], \
                                        self.stats['mean'][i+1,0], self.stats['mean'][i+1,1]))
            self.__setup_scatter(ax0, i+1)
            
            # plot distance
            ax1 = fig.add_subplot(2, 1, 2)
            self.__plot_distance_common(ax1, i+1)
        self.__cal_distance()
        fig = plt.figure(figsize=(8,4))
        ani = animation.FuncAnimation(fig, animate, frames=self.stats['val'].shape[0]-1, repeat=False, interval=500)
        return ani