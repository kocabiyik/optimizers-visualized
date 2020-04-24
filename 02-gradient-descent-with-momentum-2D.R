library(Deriv)
library(tidyverse)
library(stringr)
library(latex2exp)
library(glue)

# settings
apply_momentum <- FALSE

# test function
f <- function(x) ((x^2-4*x+4)*(x^2+4*x+2))

# plotting the test function
library(ggplot2)
p <- ggplot(data = data.frame(x = 0), mapping = aes(x = x))
p + stat_function(fun = f) +
  xlim(-4,4)

# derivative of the test function
f_prime <- Deriv(f)

# run a gradient descent
x = 3.5
x_initial = x
x_updated_values <- numeric(0)
learning_rate = 0.001

vdx = 0
beta = 0.9
for (i in 1:240) {
  if (i==1) {x_updated_values[i] = x}
  if (i==1) {next}
  
  dx = f_prime(x)
  if(apply_momentum) {
    vdx = beta*vdx+dx
    x = x-learning_rate*vdx
  } else {
    x = x-learning_rate*dx
  }
  x_updated_values[i] = x
}

df <- data_frame(x = x_updated_values,
                 y = f(x_updated_values))

# plotting the parameter updates
x_on_iteration_i <- df$x[i]
y_on_iteration_i <- df$y[i]
p + stat_function(fun = f) +
  xlim(-4,4)+
  geom_point(data = df, aes(x, y))+
  geom_point(x = x_on_iteration_i, y = y_on_iteration_i, color = 'blue', size = 4, alpha = 0.5)+
  theme(legend.position="none")+
  ylab("f(x)")+
  ggtitle(TeX('$f(x) = (x^2-4x+4)(x^2+4x+2)$'),
          subtitle = glue("Gradient Descent with Learning Rate: {learning_rate} and beta: {beta}"))
