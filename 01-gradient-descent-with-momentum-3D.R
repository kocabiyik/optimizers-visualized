library(tidyverse)
library(stringr)
library(glue)
library(Deriv)

# test functions ----
source('00-test-functions-for-optimization.R')

# video specs ----
video_name = "gd-with-momentum2"
height = 1080
framerate = 29
duration_in_sec = 10

# gradient descent ----
learning_rate <- 5e-04
x_val <- -2
y_val <- -6

# momentum terms, if applicable
apply_momentum = TRUE
beta = 0.95
vdx = 0
vdy = 0

# video generation settings ----
verbose = TRUE
plot_naming = "plot_%03d"

# video specs auto calculated ----
width = height*(16/9)
num_iter <- duration_in_sec*framerate
dir_name = str_c('frames_', video_name)
len = num_iter

# directory settings ----
if (!dir.exists(dir_name)) { dir.create(dir_name) }

# test function ----
f <- Himmelblau

# domains of test function ----
x <- seq(-6, 6, length = 100)
y <- x
z <- outer(x, y, f)

# partial derivatives of test test function ----
dx <- Deriv(f, "x")
dy <- Deriv(f, "y")

# initialize gd steps
updates_x <- vector("numeric", length = num_iter)
updates_y <- vector("numeric", length = num_iter)
updates_z <- vector("numeric", length = num_iter)

for (i in 1:num_iter) {
  
  if(apply_momentum) {
    # momentum terms: vdx and vdy
    vdx = beta * vdx + dx(x_val, y_val)
    vdy = beta * vdy + dy(x_val, y_val)  
  } else {
    # without momentum
    x_val <- x_val - learning_rate * dx(x_val, y_val)
    y_val <- y_val - learning_rate * dy(x_val, y_val)
  }
  
  # updates
  x_val <- x_val - learning_rate * vdx
  y_val <- y_val - learning_rate * vdy
  z_val <- f(x_val, y_val)
  updates_x[i] <- x_val
  updates_y[i] <- y_val
  updates_z[i] <- z_val
}


# visualize
i = 50
par(bg = 'black')
plt <- persp(x, y, z, theta = 50 - i * 0.01, 
             phi = 20 + log(i), expand = 0.5, col = "#999999", 
             border = "#111111", axes = FALSE, box = FALSE, 
             ltheta = 100, shade = 0.9)

# adding points, representing gradient updates
n_back = 20
start_n = i

if (i<n_back) {
  start_n = 1
  point_colors = c(rep("#999999", i-1), "white")
  point_sizes = c(seq(1, 2, length.out = n_back-1), 2.5)
  point_sizes = tail(point_sizes, i)
  point_transparency = c(seq(from = 0.1, to = 0.9, length.out = n_back-1),1)
  point_transparency = tail(point_transparency)
} else {
  start_n = i-n_back+1
  point_colors = c(rep("#999999", n_back-1), "white")
  point_sizes = c(seq(0.1, 2, length.out = n_back-1), 2.5)
  point_transparency = c(seq(from = 0.1, to = 0.9, length.out = n_back-1),1)
}
points(
  trans3d(updates_x[start_n:i],
          updates_y[start_n:i], 
          updates_z[start_n:i],
          pmat = plt), pch = 16, cex = point_sizes,
  col = point_colors,
  alpha = point_sizes
)


# animation ----

for (i in seq_len(len)) {
  
  # verbose 
  if (as.logical(verbose)) cat(sprintf("\rGenerating plot %d/%d...", i,  len), file = stderr())
  
  # plot the surface
  plot_name <- sprintf(plot_naming, i)
  png(glue('{dir_name}/{plot_name}.png'), width = width, height = height)
  
  par(bg = 'black')
  plt <- persp(x, y, z, theta = 50 - i * 0.01, 
               phi = 20 + log(i), expand = 0.5, col = "#999999", 
               border = "#111111", axes = FALSE, box = FALSE, 
               ltheta = 100, shade = 0.9)
  
  # adding points, representing gradient updates
  n_back = 20
  start_n = i
  
  if (i<n_back) {
    start_n = 1
    point_colors = c(rep("#999999", i-1), "white")
    point_sizes = c(seq(1, 2, length.out = n_back-1), 2.5)
    point_sizes = tail(point_sizes, i)
    point_transparency = c(seq(from = 0.1, to = 0.9, length.out = n_back-1),1)
    point_transparency = tail(point_transparency)
  } else {
    start_n = i-n_back+1
    point_colors = c(rep("#999999", n_back-1), "white")
    point_sizes = c(seq(0.1, 2, length.out = n_back-1), 2.5)
    point_transparency = c(seq(from = 0.1, to = 0.9, length.out = n_back-1),1)
  }
  points(
    trans3d(updates_x[start_n:i],
            updates_y[start_n:i], 
            updates_z[start_n:i],
            pmat = plt), pch = 16, cex = point_sizes,
    col = point_colors,
    alpha = point_sizes
  )
  dev.off()
  
}

# verbose for success ----
if (as.logical(verbose)) cat("done!\n", file = stderr())

# generate the bash command for ffmpeg ----
cmd <- glue('ffmpeg  -framerate {framerate} -i {dir_name}/{plot_naming}.png -s:v {width}x{height} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -r 30 {video_name}.mp4')
cmd
