# non-convex optimisation: cma-es with gradients
#### Huajian Qiu
#### April 2020


## Introduction

In this project, I work on improving a global optimisation method-covariance matrix adaptation evolution strategy(CMA-ES) by exploiting **differentiablity**. In short, CMA-ES is very likely the most successful evolution stragety, with solid mathematics foundation. But it ignore the information of gradient, this is a pity in above optimisation problem. 

My stragety is to inject an inner optimiser into CMA-ES. Here is the demo of results when using this new optimiser on Ackley benchmark function.
Throughout all the semester, I tested several off-the-shelf optimiser and my proposed composite optimiser on BENCHMARK functions(like Ackley), rather than real computer vision problems.  

![cma-line](figures/cma-line-search.gif)

## library
After I discover my implemented CMA has a weakness that cannot optimise the value of regularisation term. I turn to use a CMA library,which is very flexible and reliable:
```
pip install git+https://github.com/CMA-ES/pycma.git@master
```
I implemented the line search by myself in directory ```library/optimisers.py```

## How to run experiemnts

an example to run CMA
```
# initial mean and std
x0, sigma0 = latent_vectors[32].cpu().numpy(), 0.05
es = cma.CMAEvolutionStrategy(x0, sigma0)
i = 0
Xs = []
fits = []
pure_drags = []
while i < 15:
    # generate a list of latent candidates from current mean and std
    X = es.ask()
    # calculate the physical loss(drag) for each latent candidate
    fit = [func_drag(x) for x in X]
    # calculate the regularisation loss for each latent candidate
    apenalty = [func2d(x) for x in X]
    print("penalty: ", apenalty)
    fitness = [fit[j]+apenalty[j] for j in range(len(X))]
    # update the mean and std for next iteration
    es.tell(X, fitness)
    Xs.append(X)
    fits.append(fit)
    np.save("../Compare_optimisers/new_cma/cma_withReg_dif_loc32.npy", Xs)
    np.save("../Compare_optimisers/new_cma/cma_fitness_withReg_dif_loc32.npy", fits)
    # display state of current iteration
    es.disp(1)
    i += 1
```

an example to run CMA-line search

```
x0, sigma0 = latent_vectors[254].cpu().numpy(), 0.05
es = cma.CMAEvolutionStrategy(x0, sigma0)
i = 0
Xs = []
fits = []
Xqueue = []
fitQueue = []
while i < 30:
    X = es.ask()
    # new generated candidates are feed into inner optimiser, and final latent and fitness is returned
    new_x_fit = [inner_opt(x, Xs, fits) for x in X]
    X, fitness = zip(*new_x_fit)
    Xqueue += X
    fitQueue += fitness
    es.tell(Xqueue[-20:], fitQueue[-20:])
    es.disp(1)
    i += 1
```
My most recent experiments are made in [cuda0Run.ipynb](cuda0Run.ipynb). Please refer it for the definitions of functions used in above code snippets.
## Reference

1. CMA-ES [https://en.wikipedia.org/wiki/CMA-ES]
2. Test Objective function [https://www.sfu.ca/~ssurjano/optimization.html]
3. Convex Optimization – Boyd and Vandenberghe [https://web.stanford.edu/~boyd/cvxbook/]
4. MeshSDF: Differentiable Iso-Surface Extraction [https://arxiv.org/abs/2006.03997]
5. [Reports_Semester_Project_CVlab.pdf](report/Reports_Semester_Project_CVlab.pdf)
6. [Reports_Summer_Internship_CVlab.pdf](report/Reports_Summer_Internship_CVlab.pdf)

