import matplotlib.pyplot as plt
import numpy as np


class Parabola:

    """
    A parabola.

    x_series: A list
    degree: Integer. Multiples of 2.
    """
    def __init__(self, x_series, degree=2):
        self.degree = degree
        self.x_series = x_series

    def input_series(self):
        return self.x

    def output_series(self):
        outputs = [o**self.degree for o in self.x]
        return outputs

    def derivative(self):
        derivs = [self.degree*o for o in self.x]
        return self.degree*self.x

    def __repr__(self):
        return f'Parabola({self.x_series!r}, {self.degree!r})'

    def plot_function(self):
        xspace = np.linspace(min(self.x_series), max(self.x_series), 1000)
        yspace = xspace**self.degree
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.plot(xspace, yspace)


class SGDOneVariable(Parabola):
    def __init__(self, theta, x_series, lr=0.001):
        self.theta = theta
        self.lr = lr
        self.x_series = x_series

    def converge(self, steps=100):
        for i in range(steps):
            self.theta = self.theta-(self.lr*self.theta)
        return self.theta

    def store_iterations(self, steps=100):
        theta_vals = []
        for i in range(steps):
            dx = 2*self.theta  # hard coded
            self.theta = self.theta-(self.lr*dx)
            theta_vals.append(self.theta)

        iters = {
            'inputs': self.x_series,
            'theta_vals': theta_vals
        }
        return iters


class SGDOneVariableLRDecay(SGDOneVariable):

    """
    The learning rate is decayed until the iteration  𝜏
    The learning rate on the iteration  k
    The learning rate after the iteration  𝜏  is kept constant.
    𝜖𝜏  is generally set to 1 % of the initial learning rate (𝜖0).
    """

    def __init__(self, theta, x_series, tau=20, lr=0.001, decay_ratio=0.01):
        super().__init__(theta, x_series, lr)
        self.tau = 20

        self.epsilon_tau = decay_ratio*lr

    def store_iterations(self, steps=100):

        tau = self.tau
        epsilon_zero = self.lr
        epsilon_tau = self.epsilon_tau
        theta_vals = []

        for k in range(steps):

            alpha = k/tau

            if k < self.tau:
                epsilon_k = ((1-alpha)*epsilon_zero) + (alpha*epsilon_tau)
                lrd = epsilon_k

            dx = 2*self.theta  # hard coded
            self.theta = self.theta-(lrd*dx)
            theta_vals.append(self.theta)

            iters = {
                'inputs': self.x_series,
                'theta_vals': theta_vals
                }
        return iters


class SGDVisOneVariable(SGDOneVariable):

    """
    iters: List of iteration dictionaries
    """

    def __init__(self, iters, plot_title=None):
        self.iters = iters
        self.plot_title = plot_title

    def export_frame(self, steps=20, n_back=20, path=None):

        xspace = np.linspace(-4, 4, 10000)  # hard coded
        yspace = xspace**2
        plt.ioff()
        fig, ax = plt.subplots(figsize=(16, 9))
        COLORS = ['#444444', 'green']

        loop_count = 0
        for i in self.iters:
            inputs = i['inputs']
            theta_vals = i['theta_vals']

            # scatter points
            x_vals = theta_vals[0:steps][-n_back:]
            y_vals = [i**2 for i in x_vals]  # hard coded
            array_len = len(x_vals)-1

            point_sizes = [i+10 for i in range(array_len)]
            point_sizes.append(array_len+80)

            ax.axis('off')
            ax.plot(xspace, yspace, color='#444444')
            point_colors = COLORS[loop_count]
            loop_count += 1

            ax.scatter(
                x_vals, y_vals, color=point_colors,
                s=point_sizes
                )

            ax.set_title(
                self.plot_title,
                fontdict={'fontsize': 20, 'fontweight': 'medium'}
                )

            plt.close()
        return fig
