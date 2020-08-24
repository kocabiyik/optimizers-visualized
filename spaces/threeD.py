import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Himmelblau:
    
    """
    Himmelblau Test Function: https://en.wikipedia.org/wiki/Himmelblau%27s_function
    """
    
    def __init__(self,
                 x_initial = 0, y_initial = 0,
                 space_lim_min = -5, space_lim_max = 5
                
                ):
        self.x_initial = x_initial
        self.y_initial = y_initial
        self.x_space = np.arange(space_lim_min, space_lim_max, 0.25)
        self.y_space = np.arange(space_lim_min, space_lim_max, 0.25)
        
    def run_gd(self, learning_rate = 0.01, beta = 0.9, iteration = 50):
        
        vdx = 0
        vdy = 0
        
        for i in range(iteration):
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = (self.x_initial**2+self.y_initial-11)**2+(self.x_initial+self.y_initial**2-7)**2
            
                x_vals = [x]
                y_vals = [y]
                z_vals = [z]
                
            # partial derivatives
            dx = 4*x**3-4*x*y-42*x+4*x*y-14
            dy = 4*y**3+2*x**2-26*y+4*x*y-22
            
            # momentum
            vdx = beta * vdx + dx
            vdy = beta * vdy + dy
            
            # updates
            x = x-learning_rate*vdx
            y = y-learning_rate*vdy
            z = (x**2+y-11)**2+(x+y**2-7)**2+10
            
            # record steps
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
        
        steps = {
            'x_vals': x_vals,
            'y_vals': y_vals,
            'z_vals': z_vals
        }
        
        return steps
        
        
    def plot_steps(self, steps, azimuth = 10, elevation = 55, color_map = cm.Greys,
                  n_back = 20):
        
        fig = plt.figure(figsize=(20, 16))
        ax = fig.gca(projection='3d')
        ax.set_axis_off()
        
        X, Y = np.meshgrid(self.x_space, self.y_space)
        Z = (X**2+Y-11)**2+(X+Y**2-7)**2
        
        # surface
        surf = ax.plot_surface(X, Y, Z,
                               linewidth = 0, 
                               alpha = 0.6, 
                               rstride = 2, cstride = 2, 
                               cmap=color_map)
        
        snake_len = len(steps['x_vals'][-n_back:])
        point_sizes = [i for i in range(snake_len-1)]
        point_sizes.append(100) # the last point is bigger
        
        ax.scatter(steps['x_vals'][-n_back:],
                   steps['y_vals'][-n_back:],
                   steps['z_vals'][-n_back:],
                   c = 'black', marker="o", alpha=1, s = point_sizes)
        
        ax.view_init(azim=azimuth, elev=elevation)
        plt.rc('figure')