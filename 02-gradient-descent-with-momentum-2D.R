library(Deriv)
library(tidyverse)
library(stringr)
library(latex2exp)
library(glue)

# video specs ----
video_name = "gd-ggplot2"
framerate = 29
duration_in_sec = 10
wait_before_running_in_sec = 1
width = 1920
height = 1080

# gradient descent params ----
learning_rate <- 5e-04
x = 3.5
x_initial = x

# momentum terms, if applicable ----
apply_momentum = FALSE
beta = 0.95
x_updated_values <- numeric(0)
vdx = 0
beta = 0.95

# video generation settings ----
verbose = TRUE
plot_naming = "plot_%03d"

# video specs auto calculated ----
num_iter <- duration_in_sec*framerate
dir_name = str_c('frames_', video_name)
len_run = num_iter
len_wait = wait_before_running_in_sec*framerate

# directory settings ----
if (!dir.exists(dir_name)) { dir.create(dir_name) }

# test function
f <- function(x) ((x^2-4*x+4)*(x^2+4*x+2))

# derivative of the test function
f_prime <- Deriv(f)

# run gradient descent ----
for (i in 1:240) {
  if (i==1) {x_updated_values[i] = x}
  if (i==1) {next}
  
  dx = f_prime(x)
  if(apply_momentum) {
    vdx = beta*vdx+(1-beta)*dx
    x = x-learning_rate*vdx
  } else {
    x = x-learning_rate*dx
  }
  x_updated_values[i] = x
}

df <- data_frame(x = x_updated_values,
                 y = f(x_updated_values))

# plotting ----
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

# animation run ----

for (i in seq(len_wait+len_run)) {
  
  # verbose 
  if (as.logical(verbose)) cat(sprintf("\rGenerating plot %d/%d...", i,  len_wait+len_run), file = stderr())
  
  # plot
  plot_name <- sprintf(plot_naming, i)

  
  
  p <- ggplot(data = data.frame(x = 0), mapping = aes(x = x)) +
    stat_function(fun = f, color = '#444444') +
    xlim(-4,4)
  
  # wait ...
  if(i<len_wait) {
    p + geom_point(x = df$x[1], y = df$y[1],
                   color = 'white',
                   size = seq(from = 0, to = 4, length.out = len_wait)[i],
                   alpha = 0.5)+
      theme(legend.position="none")+
      #ylab("f(x)")+
      #ggtitle(TeX('$f(x) = (x^2-4x+4)(x^2+4x+2)$'),
      #        subtitle = glue("Gradient Descent with Learning Rate: {learning_rate} and beta: {beta}"))+
      theme_void()+
      theme(
        panel.background = element_rect(fill = "#000000",
                                        colour = "black"
        ))
    
  # run!
  } else {
    df_to_plot <- tail(head(df,i-len_wait+1),20)
    df_to_plot$point_alpha <- seq(from = 0, to = 0.2, length.out = nrow(df_to_plot))
    df_to_plot$point_size <- seq(from = 0, to = 0.2, length.out = nrow(df_to_plot))
    x_on_iteration_i <- tail(df_to_plot$x,1)
    y_on_iteration_i <- tail(df_to_plot$y,1)
    p + geom_point(data = df_to_plot, aes(x, y, alpha = point_alpha), color = 'white')+
      geom_point(x = x_on_iteration_i, y = y_on_iteration_i, color = 'white', size = 4, alpha = 0.5)+
      theme_void()+
      theme(legend.position="none")+
      theme(
        panel.background = element_rect(fill = "#000000",
                                        colour = "black"
      ))
      #ylab("f(x)") +
      #ggtitle(TeX('$f(x) = (x^2-4x+4)(x^2+4x+2)$'),
      #        subtitle = glue("Gradient Descent with Learning Rate: {learning_rate} and beta: {beta}"))   
  }
  
  ggsave(glue('{dir_name}/{plot_name}.png'), height = 90*1.1, width = 160*1.1, units = 'mm')
  
}

# generate the bash command for ffmpeg
cmd <- glue('ffmpeg  -framerate {framerate} -i {dir_name}/{plot_naming}.png -s:v {width}x{height} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -r 30 {video_name}.mp4')
cmd
