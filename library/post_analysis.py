import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib import patches
#import seaborn as sns

class post_analy1d:
    def __init__(self, stats):
        self.stats = stats
        self.f = stats['func']
        self.n = stats['val'].shape[0]
        
    def plot_position_after_before(self):
        if self.stats['arg'].shape[1] > 1:
            # only leave the first column, the smallest candidate
            arg = self.stats['arg'][:,0]
            val = self.stats['val'][:,0]
        else:
            arg = self.stats['arg'].squeeze()
            val = self.stats['val'].squeeze()
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(1, 1, 1)    
        
        #p = sns.scatterplot(x=arg, y=val, color="red", hue=np.arange(self.n),
       #                     hue_norm=(0, self.n), s=73, legend=False)
        xs = np.linspace(np.min(arg), np.max(arg), 251)
        fs = []
        for x in xs:
             fs.append(ak_1d.func(torch.tensor(x)).item())
        plt.plot(xs, fs, c="green")
    
    def plot_grandient_before_after(self):
        x = np.arange(self.stats['gradient_before_after'].shape[0]-1)  # the label locations
        width = 0.35  # the width of the bars
        num = self.stats['gradient_before_after'].shape[2]

        fig = plt.figure(figsize=(8, 4*num))
        for i in range(num):
            ax = fig.add_subplot(num, 1, i+1)
            ax.bar(x-width/2, self.stats['gradient_before_after'][1:, 0, i], width, color="b", label='original')
            ax.plot(x-width/2, self.stats['gradient_before_after'][1:, 0, i], color="b")

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('original gradient')
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
            ax.set_ylabel('modified gradient')
            #ax.set_xticks(x)
            ax.set_yticks(np.linspace(-1, 1, 11))
            ax.set_ylim(-1.4, 1.4)
            ax.legend(loc='lower right') 
            
class post_analysis_zone:
    def __init__(self, data):
        self.paras = data['paras']
        self.origin = self.paras['origin']
        self.edge = self.paras['edge']
        self.step = self.paras['step']
        
        self.x = data['x']
        self.y = data['y']
        self.prob = data['convergence']
        self.cost = data['cost']
        self.evals = data['evals']
        self.mask = data['mask']
        
        self.xlabel = np.arange(self.origin[0], self.origin[0] + self.edge[0], self.step) + round(self.step/2)
        self.ylabel = np.arange(self.edge[1] + self.origin[1], self.origin[1], -self.step) - round(self.step/2)
        
    def __setup_axis(self, ax):
        #sns.axes_style("white")
        ax.set_xticklabels(self.xlabel)
        ax.set_yticklabels(self.ylabel)
        
    def plot_scatter(self):
        #fig = plt.figure(figsize=(10,10))
        sc = plt.scatter(self.x.ravel(), self.y.ravel(), c=self.prob.ravel(), marker='o', vmin=0, vmax=1, s=35, cmap='YlGnBu')
        plt.colorbar(sc)
        
    def plot_hotmap_prob(self):
        #fig = plt.figure(figsize=(13, 13))
        #ax = sns.heatmap(self.prob, mask=self.mask, vmin=0, vmax=1, square=True,  cmap="YlGnBu")
        self.__setup_axis(ax)
        return ax

    def plot_hotmap_cost(self, max_cost):
        #fig = plt.figure(figsize=(13, 13))
       # ax = sns.heatmap(self.cost, mask=self.mask, vmin=0, vmax=max_cost, square=True,  cmap="YlGnBu")
        self.__setup_axis(ax)
        return ax

    def plot_hotmap_evals(self):
        #fig = plt.figure(figsize=(13, 13))
        #ax = sns.heatmap(self.evals, mask=self.mask, vmin=0, square=True,  cmap="YlGnBu")  
        self.__setup_axis(ax)
        return ax
        
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
        shape = self.stats['arg'][0].shape
        if(len(shape) == 1):
            self.distance_arg = np.linalg.norm(self.stats['arg'] - self.stats['optimal'].reshape(shape).cpu().numpy(), axis=2).mean(axis=1)
            self.distance_val = self.stats['val']
        else:
            self.distance_arg = np.linalg.norm(stats['arg'] - stats['optimal'].cpu().numpy(), axis=(1,2))
            self.distance_val = np.linalg.norm(stats['val'], axis=1)
    def plot_distance(self):
        self.__cal_distance()
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 1, 1)
        self.__plot_distance_common(ax1, self.stats['val'].shape[0]-1)
    
    def plot_moving_cluster(self):
        fig = plt.figure(figsize=(9, 9))
        row = col = 3
        unit = self.stats['val'].shape[0]/(row * col)
        for i in range(row):
            for j in range(col):
                ax=fig.add_subplot(row, col, 1 + row * i + j)
                # draw x-axis and y-axis
                ax.axvline(c='grey', lw=1)
                ax.axhline(c='grey', lw=1)
                # draw the position of optimal 
                ax.scatter(self.optimal[0], self.optimal[1], c='red', s=15)
                ax.scatter(x=self.stats['arg'][int(unit * (row * i + j)),:,0], y=self.stats['arg'][int(unit * (row * i + j)),:,1], 
                           c=self.stats['val'][int(unit * (row * i + j))], vmin = 0, vmax = 10)
                # unify the x,y scope
                min_x = np.min(self.stats['arg'][:,:,0])
                min_y = np.min(self.stats['arg'][:,:,1])
                max_x = np.max(self.stats['arg'][:,:,0])
                max_y = np.max(self.stats['arg'][:,:,1])
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_x, max_y)
                ax.set_title("%d / %d"%(int(unit * (row * i + j)), self.stats['arg'].shape[0]))

    def __plot_distance_common(self, ax1, i):
        ax1.plot(np.arange(i), self.distance_arg[1:i+1], color='green', label='L2 norm \nof parameters')
        ax1.set_xlim(0, self.stats['val'].shape[0])
        ax1.set_ylim(np.min(self.distance_arg[1:])*0.9, np.max(self.distance_arg[1:])*1.1)
        ax1.set_xlabel('iteration', fontsize=15)
        ax1.set_ylabel('distance in domain', color='green', fontsize=15)
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  
        ax2.plot(np.arange(i), self.distance_val[1:i+1], color='red', label='L2 norm \nof func value')
        ax2.set_ylim(np.min(self.distance_val[1:])*0.9, np.max(self.distance_val[1:])*1.1)
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
        min_x = np.min(self.stats['arg'][:,:,0])
        min_y = np.min(self.stats['arg'][:,:,1])
        max_x = np.max(self.stats['arg'][:,:,0])
        max_y = np.max(self.stats['arg'][:,:,1])
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_x, max_y)
       # p = sns.scatterplot(x=self.stats['arg'][i,:,0], y=self.stats['arg'][i,:,1], color="r", hue=i, hue_norm=(0, self.stats['val'].shape[0]), legend=False)
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