# https://en.wikipedia.org/wiki/Test_functions_for_optimization

# easom
Easom <- function(x, y) {
  -cos(x)*cos(y)*exp(-((x-pi)**2+(y-pi)**2))
  }

# himmelblau
Himmelblau <- function(x, y) {
  (x**2+y-11)**2+(x+y**2-7)**2
}

McCormick <- function(x, y) {
  (x-y)**2+1.5*x+2.5*y+1
}

Easom <- function(x,y) {
  -cos(x)*cos(y)*exp(-((x-pi)**2+(y-pi)**2))  
}

Saddle <- function(x,y) {
  x**2-y**2
}
