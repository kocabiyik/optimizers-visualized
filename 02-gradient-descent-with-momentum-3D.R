library(av)

# video specs ----
width = 1920
height = 1080
framerate = 29
verbose = TRUE
output = "gd-with-momentum.mp4"

# music ----
big_horns_intro <- system.file("samples/Big_Horns_Intro.mp3",  package = "av", mustWork = TRUE)
info <- av_video_info(big_horns_intro)
len <- framerate * round(info$duration)
res <- round(72 * min(width, height)/480)

# test function ----
f <- function(x, y) {
    (x^2 + y - 11)^2 + (x + y^2 - 7)^2 
  }

# domains of test function ----
x <- seq(-6, 6, length = 100)
y <- x
z <- outer(x, y, f)

# partial derivatives of test test function ----
dx <- function(x, y) {
    4 * x^3 - 4 * x * y - 42 * x + 4 * x * y - 14
  }

dy <- function(x, y) {
    4 * y^3 + 2 * x^2 - 26 * y + 4 * x * y - 22
  }
  
# gradient descent ----
num_iter <- len
learning_rate <- 5e-04
x_val <- 6
y_val <- 6
beta = 0.95
vdx = 0
vdy = 0
updates_x <- vector("numeric", length = num_iter)
updates_y <- vector("numeric", length = num_iter)
updates_z <- vector("numeric", length = num_iter)

for (i in 1:num_iter) {
  # momentum terms: vdx and vdy
  vdx = beta * vdx + dx(x_val, y_val)
  vdy = beta * vdy + dy(x_val, y_val)
  
  # updates
  x_val <- x_val - learning_rate * vdx
  y_val <- y_val - learning_rate * vdy
  z_val <- f(x_val, y_val)
  updates_x[i] <- x_val
  updates_y[i] <- y_val
  updates_z[i] <- z_val
}

# visualize before creating the video ---
persp(x, y, z, theta = -50 - i * 0.05, 
      phi = 20 + log(i), expand = 0.5, col = "lightblue", 
      border = "lightblue", axes = FALSE, box = FALSE, 
      ltheta = 100, shade = 0.9)
points(
  trans3d(updates_x[1:i],
          updates_y[1:i], 
          updates_z[1:i],
          pmat = plt), pch = 16, cex = c(rep(0.6, i - 1), 1.2),
  col = c(rep("white", i - 1), "black")
)

# animation ----
video <- av_capture_graphics(output = output, audio = big_horns_intro, 
                               {
                                 for (i in seq_len(len)) {
                                   
                                   # verbose 
                                   if (as.logical(verbose)) cat(sprintf("\rGenerating plot %d/%d...", i,  len), file = stderr())
                                   
                                   # plot the surface
                                   plt <- persp(x, y, z, theta = -50 - i * 0.05, 
                                                phi = 20 + log(i), expand = 0.5, col = "lightblue", 
                                                border = "lightblue", axes = FALSE, box = FALSE, 
                                                ltheta = 100, shade = 0.9)
                                   
                                   # adding points, representing gradient updates
                                   points(
                                     trans3d(updates_x[1:i],
                                             updates_y[1:i], 
                                             updates_z[1:i],
                                             pmat = plt), pch = 16, cex = c(rep(0.6, i - 1), 1.2),
                                     col = c(rep("white", i - 1), "black")
                                     )
                                   
                                 }
                                 
                                 # verbose for success
                                 if (as.logical(verbose)) cat("done!\n", file = stderr())
                               }
                             ,width = width, height = height, res = res, framerate = framerate, 
                             )
# find video ----
utils::browseURL(video)
