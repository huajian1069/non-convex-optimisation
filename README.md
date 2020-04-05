# non-convex optimisation: cma-es with gradients
#### Huajian Qiu
#### April 2020


## Introduction
A mathematical optimisation problem has the general form 
<img src="/tex/3a169c9f9865027eac8933023e9c119e.svg?invert_in_darkmode&sanitize=true" align=middle width=64.84037339999999pt height=39.45205439999997pt/>f_0(x)<img src="/tex/31de87f3a6027223e21eed2d9d8c170f.svg?invert_in_darkmode&sanitize=true" align=middle width=66.48915405pt height=45.84475499999998pt/>f_i(x) \leq b_i, i=1, ..., m.
\end{flushleft}<img src="/tex/1af38fb0d8e5d62339c3d5afa030c86d.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2745991999999pt height=85.29680939999997pt/>\[ f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y), i=0, ..., m. \]<img src="/tex/25a29e4e63822d380f27033451c14bc0.svg?invert_in_darkmode&sanitize=true" align=middle width=20.091388349999992pt height=39.45205439999997pt/><img src="/tex/f7904a88553d63fbcf2f77ccc1b7b9f0.svg?invert_in_darkmode&sanitize=true" align=middle width=136.46069909999997pt height=22.831056599999986pt/><img src="/tex/91e60c833d1a915acc87680959e54753.svg?invert_in_darkmode&sanitize=true" align=middle width=858.3862847999999pt height=795.4337985pt/>\href{https://en.wikipedia.org/wiki/Ackley_function}{Ackley}<img src="/tex/d9beed8618f96b50b714f1ed2d961554.svg?invert_in_darkmode&sanitize=true" align=middle width=519.54883365pt height=22.831056599999986pt/>\href{https://www.sfu.ca/~ssurjano/optimization.html}{list}$.

Week 1,2: 

\item made some 2D scatter and 3D surface visualisation tools for Ackley function. 
\item wrote the code of pure CMA-ES in python according to Wiki Matlab version and CMA-ES combined with line search algorithm.
\item Interesting finding: experiments show CMA-ES-line-search performs much better than pure CMA-ES, especially when the initial mean of optimization variable candidates is far away from optimal.   

Week 3,4:
\item made animations about optimisation process: moving clusters of candidate parameters
\item observed the round-off effect of line search, therefore add a round-off version of CMA-ES. It is not valuable by itself, but it indicates the strong relationship between local optimal and global optimal. Maybe there exist a large class of real problem where a similar relationship also exists. Then the optimisation problem will be cast to a noise-reducing problem. The key to solve this class of optimisation is to identify noise(often behaved in form of local optimal/high frequency part) and recover global information(often behaved as global optimal/low frequency part). I am still not sure how to identify the existence of this prior knowledge in objective function and how to take advantage of this inspiration. One potential way: Fourier transform.    
\end{itemize}

week 5:

\item added the visualisation of 2D normal distribution as ellipse
\item refactored the code by class
\item drawed the point cloud of global optimum convergence, first nice enough work to be included in final report 


## Schedule
### short term
week 6:
\item test more objective functions
\item implement and test one-step line search CMA-ES
\item plot grid plot of convergence


### long term
- Look forward more theoerical guidance: read books and browse slides about convex optimisation. Easter holiday is a good chance to do this.
- Also, I wonder how does anyone else tackle this problem, especially in context of neural network training. I should read some papers.
- How does CMA-ES behave compared with other heuristic method? like particle swarm optimisation(PSO), Ant colony optimisation(ACO). Distributed Intelligent System on another repository is a good example to start.


## feedback
Open to hear any voice from you, you can write your ideas or any other comments by opening an issue. If you are interested to contribute to this project, welcome create your pull request. I will keep on updating this repository during spring semester 2020. Anyway, it is assuring to share, to be open, to have a little influence in the world.

