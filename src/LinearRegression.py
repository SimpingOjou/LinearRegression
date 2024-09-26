import numpy as np
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    '''
        Compute the error for a line given the points and the slope and y-intercept of the line.
    '''
    totalError = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        totalError += (y - (m * x + b)) ** 2

    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    '''
        Compute the gradient of the error function with respect to the slope and y-intercept.
    '''
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)

    return new_b, new_m

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    '''
        Run the gradient descent algorithm.
    '''
    b = starting_b
    m = starting_m

    for _ in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)

    return b, m

if __name__ == '__main__':
    points = np.genfromtxt("./src/data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000

    error = []
    b_list = []
    m_list = []

    for i in range(num_iterations):
        if i == 0:
            b, m = step_gradient(initial_b, initial_m, points, learning_rate)
            error.append(compute_error_for_line_given_points(b, m, points))
            b_list.append(b)
            m_list.append(m)

            print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, error[i]))
            print("Running...")

            continue

        b, m = step_gradient(b, m, points, learning_rate)
        error.append(compute_error_for_line_given_points(b, m, points))
        b_list.append(b)
        m_list.append(m)

        if i == num_iterations - 1:
            print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, error[i]))

    plt.figure(figsize=(12,8))
    plt.subplot(2, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], label='Data points')
    plt.plot(points[:, 0], initial_m * points[:, 0] + initial_b, color='red', label='Before')
    plt.plot(points[:, 0], m * points[:, 0] + b, color='green', label='After')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regression before vs after")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)

    plt.loglog(range(num_iterations), error, label='Error')

    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Error over iterations")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, (3,4))

    plt.plot(b_list, m_list, label='Path', marker='o')

    plt.xlabel("b")
    plt.ylabel("m")
    plt.title("Gradient descent path")
    plt.legend()
    plt.grid()

    plt.show()


