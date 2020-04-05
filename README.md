# non-convex optimisation: cma-es with gradients
#### Huajian Qiu
#### April 2020


## Introduction
A mathematical optimisation problem has the general form 
<img src="/tex/3a169c9f9865027eac8933023e9c119e.svg?invert_in_darkmode&sanitize=true" align=middle width=64.84037339999999pt height=39.45205439999997pt/>f_0(x)<img src="/tex/31de87f3a6027223e21eed2d9d8c170f.svg?invert_in_darkmode&sanitize=true" align=middle width=66.48915405pt height=45.84475499999998pt/>f_i(x) \leq b_i, i=1, ..., m.<img src="/tex/396da5880331120780464b0dfe5cc8bb.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=14.15524440000002pt/>

One relative simple case with nice property is called convex optimisation problem, in which the objective and constraint functions are convex, which means they satisfy the inequality
<img src="/tex/b2bb3b9d7da1e1b96c41a41142a0190e.svg?invert_in_darkmode&sanitize=true" align=middle width=309.00682559999996pt height=24.65753399999998pt/>

With <p align="center"><img src="/tex/27f85ccdbe791f9cc4cc190a35062da9.svg?invert_in_darkmode&sanitize=true" align=middle width=136.46069909999997pt height=14.611878599999999pt/></p>. Notice, the linear programming is a special case of convex optimisation: equality replaces the more general inequality.

Nevertheless, in the context of neural network training and computer version challenges, the objective function is almost always non-linear and non-convex. Traditional techniques for general non-convex problems involve compromises. Local optimisation methods, like gradient descent, provide no information about distance to global optimum. Global optimisation method has worst-case complexity growing exponentially with problem size. As the abstraction of real CV problems, we could assume the objective function is non-convex, differentiable, and there are no constraint functions. The pursuing of better optimisation method should be based on these assumptions. 

In this project, I work on improving a global optimisation method-covariance matrix adaptation evolution strategy(CMA-ES) by exploiting <img src="/tex/67a588026a5bf4ad24406363e2332a8b.svg?invert_in_darkmode&sanitize=true" align=middle width=121.73456624999997pt height=22.831056599999986pt/>. Evolution strategies (ES) are stochastic, derivative-free methods for numerical optimization of non-convex continuous optimization problems. But it seems a pity that it ignores the built-in differentiablity of objective function in the context of neural network training. Therefore, the aim of this project is to integrate the information of gradient, find a better way to adjust the moving of particles in CMA-ES, get more guarantee of reaching global optimum with reasonable time cost. 

Also, based on my previous background, the methodology used in this project relies more on getting insights of optimisation method by <img src="/tex/f6fd56b48d4eb1c2f8ff71646f46a533.svg?invert_in_darkmode&sanitize=true" align=middle width=103.02681675pt height=22.831056599999986pt/>. Works will be more based on coding, testing, numerical experiments rather than theoretical proof. But I will combine some theoretical reasoning when necessary.  

<img src="/tex/312968c973c88ce62b535c234f1224e4.svg?invert_in_darkmode&sanitize=true" align=middle width=139.67950094999998pt height=117.92674409999996pt/>
To simplify the development of new/improved optimisation method, we choose to begin with some common test functions of optimisation method as benchmark. <img src="/tex/08ca1db62f5a869305447893e11b13ed.svg?invert_in_darkmode&sanitize=true" align=middle width=401.82078749999994pt height=24.65753399999998pt/>  function is used in the first few weeks. Now I change to the other items on the <img src="/tex/8d02a4a7d91e85945be8f122716dd6de.svg?invert_in_darkmode&sanitize=true" align=middle width=403.39677344999996pt height=24.65753399999998pt/>.

Week 1,2: 

<p align="center"><img src="/tex/108f4057b9d7a355cd353ad7a3243290.svg?invert_in_darkmode&sanitize=true" align=middle width=675.84524535pt height=139.54338585pt/></p>
Week 3,4:
<p align="center"><img src="/tex/465c2235201bc4beaf1c530e8ead9f24.svg?invert_in_darkmode&sanitize=true" align=middle width=675.8452222499999pt height=202.10045955pt/></p>

week 5:
<p align="center"><img src="/tex/391adbb3d9e39253b9036bc954016de3.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=6.39273195pt/></p>


\section{Schedule}
\subsection{short term}
week 6:
<p align="center"><img src="/tex/391adbb3d9e39253b9036bc954016de3.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=6.39273195pt/></p>

\subsection{long term}


\end{document}
