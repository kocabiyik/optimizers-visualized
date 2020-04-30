library(tidyverse)

# 0. generate data ----
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
initial_centroids = X[1:n,]

# Set K
K = n;

# 1.1 functions: find closest centroids ----
find_closest_centroids <- function(X, centroids) {
  
  # placeholder
  idx = rep(0, nrow(X));
  
  for (i in 1:length(idx)) {
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
  return(idx)
}

centroids <- find_closest_centroids(X, initial_centroids)

# 2. visualize ----
df <- tibble(x1=X[,1],
           x2=X[,2],
           centroid = as.factor(centroids)
           )

df_centroids = tibble(centroid_x1 = initial_centroids[,1],
                      centroid_x2 = initial_centroids[,2],
                      centroid = as.factor(1:n)) %>% 
  mutate(img = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/whatsapp/238/magnet_1f9f2.png")

df %>% left_join(df_centroids) %>% 
  ggplot(aes(x1,x2))+
  geom_point(aes(color = centroid))+
  geom_segment(aes(x=x1,
                   xend=centroid_x1,
                   y=x2,
                   yend=centroid_x2,
                   color = centroid
                   ),
               alpha = 0.3)+theme_void()+
  geom_image(data = df_centroids, aes(image=img, x=centroid_x1, y=centroid_x2), size = 0.025)+
  coord_fixed()+
  theme(legend.position = "none")
