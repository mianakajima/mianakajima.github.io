---
layout: post
title: Stochastic Optimization of Electric Grid Forecasts
---

I am currently trying to reproduce a paper "Task-based End-to-end Model Learning in Stochastic Optimization" by Priya Donti, Brandon Amos and J. Zico Kolter.
In this paper, they propose a method to learn to predict something, taking into account the end task that the prediction would be used for.

This wouldn't be necessary if we could predict what we're trying to predict perfectly, but we know that's not always the case ;) Since our model won't be perfect, we want to make sure that the model performs well in the task it will be used for.

For example, the authors ran an experiment using this method on electric grid scheduling. I summarize how the electric grid scheduling is broken down using this method in the table below. The "Experiment" column shows the prediction and task associated with electric grid scheduling, while the "Method" column shows how we would find the solution to each experiment part.


||Experiment|Method|
|----------|----|----|
|  Prediction | Electric load  | Neural network/Probabilistic Model  |
|Task    | Minimize the cost of a certain generator schedule (ie., both overgenerating and underegenerating electricity can incur extra costs when operating the grid)  |  Stochastic Optimization |

In practice, we would first generate a neural network model to predict the electric demand/how much electricity we should generate. Then, we would find tune that model using stochastic optimization, to minize the cost on the grid.

Today, I want to walk through the stochastic optimization part of this method since it was not straight forward and think it's pretty cool :)

### What is Stochastic Optimization?

In optimization, we are trying to find the optimal solution to something - usually that means we are finding the minimum or maximum of a function. Stochastic optimization is not that different. Stochastic implies a component of randomness, and stochastic optimization is just that: **optimization of a function which includes random variables or constraints**. The randomness may be able to be described as coming from a known distribution, or sometimes it may be unknown.

In this paper's case, the function we are trying to minimize is the cost of using a certain generator schedule. In grid scheduling, usually how much energy to generate is decided for the whole 24 hours of the next day and then adjustments are made the day of as needed. We assume there are costs associated with both undergenerating and overgenerating.

The paper assumes the cost function looks like this:

$$  \min_{z \in \mathbb{R}^{24}} \sum_{i = 1}^{24} \mathbb{E}_{y~p(y\vert x;\theta)} [\gamma_s[y_i -z_i]_+ + \gamma_e[z_i - y_i]_+ + \frac{1}{2}(z_i - y_i)^2] $$

$$ s.t. \vert z_i - z_{i - 1} \vert \leq c_r \forall i$$

Where:
- $z_i$ : Electricity generation scheduled for every hour $i$
- $y_i$ : Demand, which we assume comes from a Gaussian distribution
- $\gamma_s$: The cost of undergenerating
- $\gamma_e$: The cost of overgenerating
- $c_r$: Ramping constant (we assume the energy generation cannot change excessively from one hour to the next)
- $[x]_+$ : Function meaning take the maximum of $x$ or 0.

Since we don't know actually how much demand for energy there will be and we must assume it's a random variable, this is a stochastic optimization problem.

The authors of this paper solve this problem using a "stochastic gradient approach". I understood this to mean gradient descent using the cost function above rather than standard cost functions, such as root mean square error. The specific steps are:

1. Say $x$ (features such as weather) and $y$ (the true demand) come from some true distribution, $D$. And we are able to create an initial model $z = f(x \vert \theta)$ that predicts the demand, $z$ from features $x$.
2. Sample $(x, y)$ from true distrubution $D$
3. Solve for $z$ that will minimize the above cost function using sample.
4. Now do gradient descent. Update the parameters $\theta$ by taking derivates of the cost function.
5. Repeat steps 2-4 for a chosen number of epochs.

Now the question arises: how do we solve for the best solution of $z$? For this, we can use an approach called [sequential quadratic programming](https://en.wikipedia.org/wiki/Sequential_quadratic_programming).

### Sequential Quadratic Programming

The solution to minimizing the cost function does not look straightforward! Here, we can try using sequential quadratic programming. The big picture is that we are using local quadratic approximations of our function in order to iteratively shimmy our estimate to the real minimum of the function.

We can approximate the function at our intial guess of $z^{(j)}$ using a Taylor approximation to the second degree. It looks like this:

$$ Define: f(z) =  \mathbb{E}_{y~p(y\vert x;\theta)} [\gamma_s[y_i -z_i]_+ + \gamma_e[z_i - y_i]_+ + \frac{1}{2}(z_i - y_i)^2]$$

$$ d = z^{(j + 1)} - z^{(j)}$$

$$ f \approx  \nabla f(z^{(j)})d + \frac{1}{2} d^T\nabla^2 f(z^{(j)})d  $$

We can now actually change our problem from finding $z$ which minimizes $f(z)$ to finding $d$ which minimizes the approximation of $f(z)$  as shown below.

$$  d = arg \min_{d} \frac{1}{2} d^T\nabla^2 f(z^{(j)})d + \nabla f(z^{(j)})d $$

We can see this follows the form of a quadratic program which has a general form:

$$ arg \min_{x}  \frac{1}{2} x^TQx + P^Tx$$

Where:

- $Q$ is a nxn matrix and $P$ is a n-dimensional vector

We want to do this because quadratic programs may be easier to solve than our initial function. Dependent on the characteristics of matrix $Q$, there are known ways to solve for the minimum of the equation. Some examples are [here](https://en.wikipedia.org/wiki/Quadratic_programming). This paper uses a Python package `qpth` to solve the quadratic program.

After solving for $d$, we can now find our new estimate $z^{(j + 1)} = z^{(j)} + d$.

We can iteratively solve for $z^{(j + 1)}$ until we converge, ie., until $\vert z^{(j + 1)} - z^{(j)} \vert < \delta$ for some small $\delta$.

## Tying It All Together

Whew! So that was a lot of math and lots of talk on iterations.

This process can be summed up as follows:

1. In a scenario where you have a problem where you may want to assess task-based accuracy (which is a stochastic programming problem) in addition to predictive accuracy, one solution may be to train a probablistic predictive model and fine tune it based on a cost function related to your task.
2. The fine tuning involves stochastic gradient desent of your inital model using the task-based cost function.
3. In the electric grid scheduling case, minimizng the cost function was complex so we had to use sequential quadratic programming.

And that's it for this week! :)
