\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amsfonts,amsthm,amssymb}
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{extramarks}
\usepackage{chngpage}
\usepackage{soul}
\usepackage[usenames,dvipsnames]{color}
\usepackage{graphicx,float,wrapfig}
\usepackage{ifthen}
\usepackage{listings}
\usepackage{courier}
\usepackage{hyperref}

\title{non-convex optimisation: cma-es with gradients}
\author{Huajian Qiu}
\date{April 2020}

\begin{document}

\maketitle

\section{Introduction}
A mathematical optimisation problem has the general form 
\begin{flushleft}
\hspace{4cm} minimize $f_0(x)$         \\
\hspace{4cm} subject to $f_i(x) \leq b_i, i=1, ..., m.$
\end{flushleft}
One relative simple case with nice property is called convex optimisation problem, in which the objective and constraint functions are convex, which means they satisfy the inequality
\[ f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y), i=0, ..., m. \]

With $\alpha + \beta = 1, \alpha, \beta \geq 0$. Notice, the linear programming is a special case of convex optimisation: equality replaces the more general inequality.

Nevertheless, in the context of neural network training and computer version challenges, the objective function is almost always non-linear and non-convex. Traditional techniques for general non-convex problems involve compromises. Local optimisation methods, like gradient descent, provide no information about distance to global optimum. Global optimisation method has worst-case complexity growing exponentially with problem size. As the abstraction of real CV problems, we could assume the objective function is non-convex, differentiable, and there are no constraint functions. The pursuing of better optimisation method should be based on these assumptions. 

In this project, I work on improving a global optimisation method-covariance matrix adaptation evolution strategy(CMA-ES) by exploiting \textbf{differentiablity}. Evolution strategies (ES) are stochastic, derivative-free methods for numerical optimization of non-convex continuous optimization problems. But it seems a pity that it ignores the built-in differentiablity of objective function in the context of neural network training. Therefore, the aim of this project is to integrate the information of gradient, find a better way to adjust the moving of particles in CMA-ES, get more guarantee of reaching global optimum with reasonable time cost. 

Also, based on my previous background, the methodology used in this project relies more on getting insights of optimisation method by \textbf{visualization}. Works will be more based on coding, testing, numerical experiments rather than theoretical proof. But I will combine some theoretical reasoning when necessary.  

\section{Progress}
To simplify the development of new/improved optimisation method, we choose to begin with some common test functions of optimisation method as benchmark. \href{https://en.wikipedia.org/wiki/Ackley_function}{Ackley}  function is used in the first few weeks. Now I change to the other items on the \href{https://www.sfu.ca/~ssurjano/optimization.html}{list}.

Week 1,2: 

\begin{itemize}
    \item made some 2D scatter and 3D surface visualisation tools for Ackley function. 
    \item wrote the code of pure CMA-ES in python according to Wiki Matlab version and CMA-ES combined with line search algorithm.
    \item Interesting finding: experiments show CMA-ES-line-search performs much better than pure CMA-ES, especially when the initial mean of optimization variable 
    candidates is far away from optimal.   
\end{itemize}
Week 3,4:
\begin{itemize}
    \item made animations about optimisation process: moving clusters of candidate parameters
    \item observed the round-off effect of line search, therefore add a round-off version of CMA-ES. It is not valuable by itself, but it indicates the strong relationship between local optimal and global optimal. Maybe there exist a large class of real problem where a similar relationship also exists. Then the optimisation problem will be cast to a noise-reducing problem. The key to solve this class of optimisation is to identify noise(often behaved in form of local optimal/high frequency part) and recover global information(often behaved as global optimal/low frequency part). I am still not sure how to identify the existence of this prior knowledge in objective function and how to take advantage of this inspiration. One potential way: Fourier transform.    
\end{itemize}

week 5:
\begin{itemize}
    \item 
\end{itemize}


\section{Schedule}
\subsection{short term}
week 6:
\begin{itemize}
    \item 
\end{itemize}

\subsection{long term}


\end{document}
