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
    def __init__(self, theta, learning_rate, x_series):
        self.theta = theta
        self.learning_rate = learning_rate
        self.x_series = x_series
        
    def converge(self, steps = 100):
        for i in range(steps):
            self.theta = self.theta-(self.learning_rate*self.theta)
        return self.theta
    
    def store_iterations(self, steps = 100):
        theta_vals = []
        for i in range(steps):
            dx = 2*self.theta ## hard coded -----------------------------
            self.theta = self.theta-(self.learning_rate*dx)
            theta_vals.append(self.theta)
        
        iters = {
            'inputs': self.x_series,
            'theta_vals': theta_vals
        }
        return iters
    
    
class SGDVisOneVariable(SGDOneVariable):
    def __init__(self, iters):
        self.iters = iters
    
    def export_frame(self, steps = 20, n_back=20, path = None):
    
        inputs = self.iters['inputs']
        theta_vals = self.iters['theta_vals']
        xspace = np.linspace(min(inputs), max(inputs), 10000)
        yspace = xspace**2
        plt.ioff()
        fig, ax = plt.subplots()
        
        # scatter points
        x_vals = theta_vals[0:steps][-n_back:]
        y_vals = [i**2 for i in x_vals]
        array_len = len(x_vals)-1
        
        point_sizes = [i+5 for i in range(array_len)]
        point_sizes.append(array_len+50)
        
        point_colors = ['#444444' for i in range(array_len)]
        point_colors.append('black')
        
        ax.axis('off')
        ax.plot(xspace, yspace, color = 'black')
        ax.scatter(x_vals, y_vals, color = point_colors, s = point_sizes)
        # plt.savefig(path)
        plt.close()
        return fig