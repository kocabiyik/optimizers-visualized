library(tidyverse)
n = 5

X = tibble(x1=sample(1:10, n),
       x2=sample(1:10, n)) %>% 
  mutate(noise_x1 = list(rnorm(200)),
         noise_x2 = list(rnorm(200))) %>% 
  unnest(cols = c(noise_x1, noise_x2)) %>% 
  transmute(x1 = x1+noise_x1,
            x2 = x2+noise_x2) %>% 
  as.matrix()

# placeholder
centroids = X[1:n,]

# Set K
K = n;

# You need to return the following variables correctly.
idx = rep(0, nrow(X));

nrow = nrow(X);

for (i in 1:nrow) {
  x1 = X[i,1]
  x2 = X[i,2]
  for (j in 1:K) {
    # centroid points
    c1 = centroids[j,1]
    c2 = centroids[j,2]
  
    # initialize
    if (j == 1) {
      idx[i] = 1; 
      dist_cache = sqrt((x2-c2)^2+(x1-c1)^2);
    }
    
    # calculate distance
    calculated_dist = sqrt((x2-c2)^2+(x1-c1)^2);
    
    # update centroid point
    if (calculated_dist<dist_cache) {
      dist_cache = calculated_dist; 
      idx[i] = j
    }
  }
}

# visualize
library(tidyverse)
df <- tibble(x1=X[,1],
           x2=X[,2],
           centroid = as.factor(idx)
           )

df_centroids = tibble(centroid_x1 = centroids[,1],
                      centroid_x2 = centroids[,2],
                      centroid = as.factor(1:n))

df %>% left_join(df_centroids) %>% 
  ggplot(aes(x1,x2, color = centroid))+
  geom_point()+
  geom_segment(aes(x=x1,
                   xend=centroid_x1,
                   y=x2,
                   yend=centroid_x2
                   ),
               alpha = 0.3)+theme_void()
