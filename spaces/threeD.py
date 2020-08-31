import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Surface:
    
    def __init__(self, test_function, x=0, y=0):
        self.x = x
        self.y = y
        self.test_function=test_function
        
    def get_z_at(self, x, y):
        if self.test_function=='himmelblau':
            return (x**2+y-11)**2+(x+y**2-7)**2
        if self.test_function=='parabolic':
            return x**2+y**2
        if self.test_function=='matyas':
            return (0.26*(x**2+y**2))-(0.48*x*y)
        if self.test_function=='saddle':
            return x**2-y**2
    
    def get_gx_at(self, x, y):
        if self.test_function=='himmelblau':
            return 4*x**3-4*x*y-42*x+4*x*y-14
        if self.test_function=='parabolic':
            return 2*x
        if self.test_function=='matyas':
            return (2*0.26*x)-(0.48*y)
        if self.test_function=='saddle':
            return 2*x
        
    def get_gy_at(self, x, y):
        if self.test_function=='himmelblau':
            return 4*y**3+2*x**2-26*y+4*x*y-22
        if self.test_function=='parabolic':
            return 2*y
        if self.test_function=='matyas':
            return (2*0.26*y)-(0.48*x)
        if self.test_function=='saddle':
            return -2*y

class State(Surface):
    
    """Iterates and returns history
    
    """
    def __init__(self, x_initial = 0, y_initial = 0, test_function = 'parabolic', space_lim_min = -5, space_lim_max = 5, iteration=120):
        super().__init__(test_function)
        self.x_initial = x_initial
        self.y_initial = y_initial
        self.iteration = iteration
        self.x_space = np.arange(space_lim_min, space_lim_max, 0.25)
        self.y_space = np.arange(space_lim_min, space_lim_max, 0.25)

    # Stochastic Gradient Descent
    def run_gd(self, epsilon=0.01, alpha=0.9, nesterov=False):
        
        
        """Runs Gradient Descent algorithm with momentum and returns the parameter update history.
        
        Resource:
        https://www.deeplearningbook.org/contents/optimization.html#pf15
        
        Keyword Arguments:
        epsilon -- the learning rate (default 0.01)
        alpha -- momentum parameter (default 0.9)
        iteration -- number of parameter updates (default 50)
        nesterov -- whether the momentum is nesterov momentum or standard momentum (default False)
        
        Returns:
        A dictionary containing 3 lists: x_history, y_history, z_history
        """
        
        # initial velocities
        vx = 0
        vy = 0
        
        for i in range(self.iteration):
            
            # keep the initial parameter values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = self.get_z_at(x, y)
            
                x_history = [x]
                y_history = [y]
                z_history = [z]
            
            # the nesterov momentum
            if nesterov:
                
                # apply interim update
                x_tilda = x+(alpha*vx)
                y_tilda = y+(alpha*vy)
                
                # compute gradient at interim point
                gx = self.get_gx_at(x_tilda, y_tilda)
                gy = self.get_gy_at(x_tilda, y_tilda)
                
                # compute velocity update
                vx = alpha * vx - epsilon * gx
                vy = alpha * vy - epsilon * gy
                
                # apply update
                x = x+vx
                y = y+vy
                z = self.get_z_at(x, y)
            
            # the standard Momentum
            else:
                
                # compute gradient
                gx = self.get_gx_at(x, y)
                gy = self.get_gy_at(x, y)
            
                # compute velocity update
                vx = alpha * vx - epsilon * gx
                vy = alpha * vy - epsilon * gy
            
                # apply update
                x = x+vx
                y = y+vy      
                z = self.get_z_at(x, y)
                
            # record steps
            x_history.append(x)
            y_history.append(y)
            z_history.append(z)
        
        steps = {
            'x_history': x_history,
            'y_history': y_history,
            'z_history': z_history
        }
        
        return steps
    
    def run_adagrad(self, epsilon=0.001, delta=1.e-7):
        
        """Runs AdaGrad algorithm and returns the parameter update history.
        
        Resource:
        https://www.deeplearningbook.org/contents/optimization.html#pf22
        
        Keyword Arguments:
        epsilon -- the learning rate
        iteration -- number of parameter updates (default 50)
        delta -- small constant for numerical stability (default 1.e-7)
        
        Returns:
        A dictionary containing 3 lists: x_history, y_history, z_history
        """
        
        
        rx = 0 # gradient accumulation variable
        ry = 0
        
        for i in range(self.iteration):
            
            # keep the initial parameter values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = self.get_z_at(x, y)
            
                x_history = [x]
                y_history = [y]
                z_history = [z]
                delta_theta_x_history = [None]
                delta_theta_y_history = [None]
                rx_history = [None]
                ry_history = [None]

            # compute gradient
            gx = self.get_gx_at(x, y)
            gy = self.get_gy_at(x, y)
            
            # accumulate squared gradient 
            rx = rx + (gx*gx)
            ry = ry + (gy*gy)
            
            # compute update
            delta_theta_x = (-1)*(epsilon/(delta+math.sqrt(rx)))*gx
            delta_theta_y = (-1)*(epsilon/(delta+math.sqrt(ry)))*gy
            
            # apply update
            x = x+delta_theta_x
            y = y+delta_theta_y
            z = self.get_z_at(x, y)
                
            # record steps
            x_history.append(x)
            y_history.append(y)
            z_history.append(z)
            delta_theta_x_history.append(delta_theta_x)
            delta_theta_y_history.append(delta_theta_y)
            rx_history.append(rx)
            ry_history.append(ry)
        
        steps = {
            'x_history': x_history,
            'y_history': y_history,
            'z_history': z_history,
            'delta_theta_x_history': delta_theta_x_history,
            'delta_theta_y_history': delta_theta_y_history,
            'rx_history': rx_history,
            'ry_history': ry_history
        }
        
        return steps
    
    def run_rmsprop(self, epsilon=0.001, rho=0.9, delta=1.e-6):
        
        """Runs RMSProp algorithm and returns the parameter update history.
        
        Resource:
        https://www.deeplearningbook.org/contents/optimization.html#pf22
        
        Keyword Arguments:
        epsilon -- the learning rate
        rho -- decay rate (default 0.9)
        delta -- small constant for numerical stability (default 1.e-6)
        iteration -- number of parameter updates (default 50)
        
        Returns:
        A dictionary containing 3 lists: x_history, y_history, z_history
        """
        
        
        rx = 0 # gradient accumulation variable
        ry = 0
        
        for i in range(self.iteration):
            
            # keep the initial values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = self.get_z_at(x, y)
            
                x_history = [x]
                y_history = [y]
                z_history = [z]
                delta_x_history = [None]
                delta_y_history = [None]
                rx_history = [None]
                ry_history = [None]
            
            # compute gradient
            gx = self.get_gx_at(x,y)
            gy = self.get_gy_at(x,y)
            
            # accumulate squared gradient 
            rx = (rho*rx) + ((1-rho)*gx*gx)
            ry = (rho*ry) + ((1-rho)*gy*gy)
            
            # compute update
            delta_x = (-1)*(epsilon/math.sqrt(delta+rx))*gx
            delta_y = (-1)*(epsilon/math.sqrt(delta+ry))*gy
            
            # apply update
            x = x+delta_x
            y = y+delta_y
            z = self.get_z_at(x, y)
                
            # record steps
            x_history.append(x)
            y_history.append(y)
            z_history.append(z)
            delta_x_history.append(delta_x)
            delta_y_history.append(delta_y)
            rx_history.append(rx)
            ry_history.append(ry)
        
        steps = {
            'x_history': x_history,
            'y_history': y_history,
            'z_history': z_history,
            'delta_x_history': delta_x_history,
            'delta_y_history': delta_y_history,
            'rx_history': rx_history,
            'ry_history': ry_history
        }
        
        return steps
    
    def run_adam(self, epsilon=0.001, rho1 =0.9, rho2=0.999, alpha=0.9, delta=1.e-8):

        """Runs Adam algorithm and returns the parameter update history.
        
        Resource:
        https://www.deeplearningbook.org/contents/optimization.html#pf23
        
        Keyword Arguments:
        epsilon -- the learning rate
        rho -- decay rate (default 0.9)
        alpha -- momentum parameter (default 0.9)
        delta -- small constant for numerical stability (default 1.e-7)
        iteration -- number of parameter updates (default 50)
        
        Returns:
        A dictionary containing 3 lists: x_history, y_history, z_history
        """
        
        # Initialize 1st and 2nd moment variables s = 0, r = 0
        rx = 0
        ry = 0
        sx = 0
        sy = 0
        
        # Initialize time step t = 0
        t = 0
        
        for i in range(self.iteration):
            
            # keep the initial values
            if i == 0:
                x = self.x_initial
                y = self.y_initial
                z = self.get_z_at(x, y)
            
                x_history = [x]
                y_history = [y]
                z_history = [z]
            
             # compute gradient
            gx = self.get_gx_at(x,y)
            gy = self.get_gy_at(x,y)  
            
            t = t+1
            
            # Update biased ﬁrst moment estimate
            sx = rho1*sx+(1-rho1)*gx
            sy = rho1*sy+(1-rho1)*gy
            
            # Update biased second moment estimate: r ← ρ2r + (1 − ρ2)g  g
            rx = (rho2*rx) + ((1-rho2)*gx*gx)
            ry = (rho2*ry) + ((1-rho2)*gy*gy)
            
            # Correct bias in ﬁrst moment
            sx_head = sx/(1-rho1)
            sy_head = sy/(1-rho1)
            
            # Correct bias in second moment:
            rx_head = rx/(2-rho2)
            ry_head = ry/(2-rho2)
            
            # compute update
            delta_theta_x = (-1)*epsilon*(sx_head/(math.sqrt(rx_head)+delta))
            delta_theta_y = (-1)*epsilon*(sy_head/(math.sqrt(ry_head)+delta))
            
            # apply update
            x = x+delta_theta_x
            y = y+delta_theta_y
            z = self.get_z_at(x, y)
                
            # record steps
            x_history.append(x)
            y_history.append(y)
            z_history.append(z)
        
        steps = {
            'x_history': x_history,
            'y_history': y_history,
            'z_history': z_history
        }
        
        return steps
    
    def plot_steps(self, iters, steps_until_n=50, azimuth=10, elevation=55, color_map=cm.gray,
                  n_back=20, plot_title=None, colors=['black', 'green']):
        """Plots iteration history
        
        Keyword arguments:
        iters -- List of iteration dictionaries
        steps_until_n -- the last iteration to plot
        azimuth -- azimuth (default 10)
        elevation -- elevation ( default 55)
        color_map -- color map, a matplotlib ojbect
        n_back -- the last n iteration on the plot
        plot_title -- plot heading
        colors=['black', 'green'] -- color for each type of 
        
        Returns:
        A plot
        """
        
        plt.ioff()
        fig = plt.figure(figsize = (10,10))
        ax = fig.gca(projection='3d')
        ax.set_axis_off()
        
        X, Y = np.meshgrid(self.x_space, self.y_space)
        Z = self.get_z_at(X, Y)
        
        # surface
        surf = ax.plot_surface(X, Y, Z,
                               linewidth = 0, 
                               alpha = 0.6, 
                               rstride = 2, cstride = 2, 
                               cmap=color_map)
        
        loop_counter=0
        for i in iters:
            steps = i
            
            x_history = steps['x_history'][:steps_until_n]
            y_history = steps['y_history'][:steps_until_n]
            z_history = steps['z_history'][:steps_until_n]
            
            snake_len = len(x_history[-n_back:])
            point_sizes = [i for i in range(snake_len-1)]
            point_sizes.append(100) # the last point is bigger
            
            ax.scatter(x_history[-n_back:],
                       y_history[-n_back:],
                       z_history[-n_back:],
                       c = colors[loop_counter], marker="o", alpha=1, s = point_sizes)
            loop_counter+=1
        
        ax.view_init(azim=azimuth, elev=elevation)
        ax.set_title(plot_title, fontdict={'fontsize': 20, 'fontweight': 'medium'})
        plt.close()
        
        return fig