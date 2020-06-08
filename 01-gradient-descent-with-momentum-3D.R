library(tidyverse)
library(stringr)
library(glue)
library(Deriv)

# test functions ----
source('00-test-functions-for-optimization.R')

# video specs ----
video_name = "gd-with-momentum2"
height = 2160
framerate = 29
duration_in_sec = 10

# gradient descent ----
learning_rate <- 0.01
x_val <- -5
y_val <- 2

# momentum terms, if applicable ----
apply_momentum = TRUE
beta = 0.98
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
    vdx = beta * vdx + (1-beta)*dx(x_val, y_val)
    vdy = beta * vdy + (1-beta)*dy(x_val, y_val)  
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


plot_iteration <- function(i) {
  
  # theme
  par(bg = 'black', family = 'Ubuntu Light')
  
  # surface
  plt <- persp(x, y, z,
               theta = -0 - i * 0.01,   # angles defining the azimuthal direction. (higher value: object rotates clockwise)
               phi = 20 + log(i),       # the colatitude viewing direction. (higher value: camera moves up).
               expand = 0.5,            # a expansion factor applied to the z coordinates. [0,1]
               
               ltheta = 100,            # lighting
               lphi = 10,
               shade = 0.9,
               
               border = "darkgray",
               col = "#999999",
               axes = FALSE, box = FALSE
               )
  
  # adding points, representing gradient updates
  start_n = i
  n_back = 20
  
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
    col = point_colors
  )
  title("Gradient Descent with Momentum", col.main= "#555555", font=3, cex.main = 4)
  title(str_c("\n\n iteration:", as.character(i)), col.main= "#555555", font=3, cex.main = 3)
}

# see one frame ----
plot_iteration(50)
png(glue('test.png'), width = width, height = height)
par(bg = 'black')
plot_iteration(i)
dev.off()

# animation ----
for (i in seq_len(len)) {
  
  # verbose 
  if (as.logical(verbose)) cat(sprintf("\rGenerating plot %d/%d...", i,  len), file = stderr())
  
  # plot the surface
  plot_name <- sprintf(plot_naming, i)
  png(glue('{dir_name}/{plot_name}.png'), width = width, height = height)
  par(bg = 'black')
  plot_iteration(i)
  dev.off()
}

# verbose for success ----
if (as.logical(verbose)) cat("done!\n", file = stderr())

# generate the bash command for ffmpeg ----
cmd <- glue('ffmpeg  -framerate {framerate} -i {dir_name}/{plot_naming}.png -s:v {width}x{height} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -r 30 {video_name}.mp4')
cmd
