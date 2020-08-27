import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

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
        
    # Stochastic Gradient Descent
    def run_gd(self, epsilon = 0.01, alpha = 0.9, iteration = 50, nesterov=False):
        
        """
        epsilon: the learning rate
        alpha: momentum parameter
        """
        
        vx = 0
        vy = 0
        
        for i in range(iteration):
            
            # keep the initial values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = (self.x_initial**2+self.y_initial-11)**2+(self.x_initial+self.y_initial**2-7)**2
            
                x_vals = [x]
                y_vals = [y]
                z_vals = [z]
            
            # the Nesterov Momentum ----
            if nesterov:
                
                # apply interim update
                x_tilda = x+(alpha*vx)
                y_tilda = y+(alpha*vy)
                
                # compute gradient at interim point
                gx = 4*x_tilda**3-4*x_tilda*y_tilda-42*x_tilda+4*x_tilda*y_tilda-14
                gy = 4*y_tilda**3+2*x_tilda**2-26*y_tilda+4*x_tilda*y_tilda-22
                
                # compute velocity update
                vx = alpha * vx - epsilon * gx
                vy = alpha * vy - epsilon * gy
                
                # apply update
                x = x+vx
                y = y+vy
                z = (x**2+y-11)**2+(x+y**2-7)**2+10
            
            # the Standard Momentum ----
            else:
            
                # compute gradient
                gx = 4*x**3-4*x*y-42*x+4*x*y-14
                gy = 4*y**3+2*x**2-26*y+4*x*y-22
            
                # compute velocity update
                vx = alpha * vx - epsilon * gx
                vy = alpha * vy - epsilon * gy
            
                # apply update
                x = x+vx
                y = y+vy
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
    
    # Adagrad
    def run_adagrad(self, epsilon, delta=1.e-7, iteration = 10):
        
        """
        AdaGrad
        """
        
        
        rx = 0 # gradient accumulation variable
        ry = 0
        
        for i in range(iteration):
            
            # keep the initial values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = (self.x_initial**2+self.y_initial-11)**2+(self.x_initial+self.y_initial**2-7)**2
            
                x_vals = [x]
                y_vals = [y]
                z_vals = [z]
            

            # Compute gradient
            gx = 4*x**3-4*x*y-42*x+4*x*y-14
            gy = 4*y**3+2*x**2-26*y+4*x*y-22
            
            # Accumulate squared gradient 
            rx = rx + gx*gx
            ry = ry + gy*gy
            
            # Compute update
            delta_theta_x = -(epsilon*gx)/(delta+math.sqrt(rx))
            delta_theta_y = -(epsilon*gy)/(delta+math.sqrt(ry))
            
            # apply update
            x = x+delta_theta_x
            y = y+delta_theta_y
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
    
    def run_rmsprop(self, epsilon, rho = 0.9, delta=1.e-7, iteration = 10):
        
        """
        RMSProp
        rho: decay rate
        """
        
        
        rx = 0 # gradient accumulation variable
        ry = 0
        
        for i in range(iteration):
            
            # keep the initial values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = (self.x_initial**2+self.y_initial-11)**2+(self.x_initial+self.y_initial**2-7)**2
            
                x_vals = [x]
                y_vals = [y]
                z_vals = [z]
            

            # Compute gradient
            gx = 4*x**3-4*x*y-42*x+4*x*y-14
            gy = 4*y**3+2*x**2-26*y+4*x*y-22
            
            # Accumulate squared gradient 
            rx = rho*rx + (1-rho)*gx*gx
            ry = rho*ry+ (1-rho)*gy*gy
            
            # Compute update
            delta_theta_x = -(epsilon*gx)/(delta+math.sqrt(rx))
            delta_theta_y = -(epsilon*gy)/(delta+math.sqrt(ry))
            
            # apply update
            x = x+delta_theta_x
            y = y+delta_theta_y
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
    
    def run_adam(self, epsilon, rho = 0.9, alpha = 0.9, delta=1.e-7, iteration = 10):
        
        """
        Adam
        rho: decay rate
        """
        
        # initial velocities
        vx = 0
        vy = 0
  
        rx = 0 # gradient accumulation variable
        ry = 0
        
        for i in range(iteration):
            
            # keep the initial values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = (self.x_initial**2+self.y_initial-11)**2+(self.x_initial+self.y_initial**2-7)**2
            
                x_vals = [x]
                y_vals = [y]
                z_vals = [z]
            

            # compute interim update
            x_tilda = x+(alpha*vx)
            y_tilda = y+(alpha*vy)
            
            # compute gradient at interim point
            gx = 4*x_tilda**3-4*x_tilda*y_tilda-42*x_tilda+4*x_tilda*y_tilda-14
            gy = 4*y_tilda**3+2*x_tilda**2-26*y_tilda+4*x_tilda*y_tilda-22

            # Accumulate squared gradient 
            rx = rho*rx + (1-rho)*gx*gx
            ry = rho*ry+ (1-rho)*gy*gy    

            # compute velocity update
            vx = alpha * vx - epsilon * gx
            vy = alpha * vy - epsilon * gy                

            # apply update
            x = x+vx
            y = y+vy
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
        
        
    def plot_steps(self, iters, steps_until_n=10, azimuth = 10, elevation = 55, color_map = cm.Greys,
                  n_back = 20, plot_title = None, colors = ['black', 'green']):
        """
        iters: List of iteration dictionaries
        """
        
        plt.ioff()
        fig = plt.figure(figsize = (16,9))
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
        
        loop_counter=0
        for i in iters:
            steps = i
            
            x_vals = steps['x_vals'][:steps_until_n]
            y_vals = steps['y_vals'][:steps_until_n]
            z_vals = steps['z_vals'][:steps_until_n]
            
            snake_len = len(x_vals[-n_back:])
            point_sizes = [i for i in range(snake_len-1)]
            point_sizes.append(100) # the last point is bigger
            
            ax.scatter(x_vals[-n_back:],
                       y_vals[-n_back:],
                       z_vals[-n_back:],
                       c = colors[loop_counter], marker="o", alpha=1, s = point_sizes)
            loop_counter+=1
        
        ax.view_init(azim=azimuth, elev=elevation)
        ax.set_title(plot_title, fontdict={'fontsize': 20, 'fontweight': 'medium'})
        plt.close()
        
        return fig