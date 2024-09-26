# LinearRegression

This example project demonstrates how the gradient descent algorithm may be used to solve a linear regression problem.

This code demonstrates how a gradient descent search may be used to solve the linear regression problem of fitting a line to a set of points. In this problem, we wish to model a set of points using a line. The line model is defined by two parameters - the line's slope `m`, and y-intercept `b`. Gradient descent attemps to find the best values for these parameters, subject to an error function.

The `main` defines a set of parameters used in the gradient descent algorithm including an initial guess of the line slope and y-intercept, the learning rate to use, and the number of iterations to run gradient descent for. 

```python
initial_b = 0 # initial y-intercept guess
initial_m = 0 # initial slope guess
num_iterations = 1000
``` 

Using these parameters a gradient descent search is executed on a sample data set.

A more generalized solution has been adopted in `MultivariateLinearRegression.py` following the formulas below:

<img src='/img/Cost-Function.jpg'>

Where 

-> $θ_j$     : Weights of the hypothesis.
-> $h_θ{(x_i)}$ : predicted y value for ith input.
-> $i$     : Feature index number (can be $0, 1, 2, ......, n$).
-> $α$     : Learning Rate of Gradient Descent.
-> $J(\varTheta_0,\varTheta_1)$ : Loss function or error function

# Dependencies

Install all dependencies with `pip install -r requirements.txt`.

# Credits

[mattnedrich](https://github.com/mattnedrich) for the non generalized version of the script.

[Tan-Moy](https://github.com/Tan-Moy) for the multivariate version of the script

[geeksforgeeks](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) for the image