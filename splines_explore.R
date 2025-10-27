library(splines)

T <- 4
p <- 3
delta_obs <- 0.005
delta_knots <- 0.05
t_total <- seq(0,T,delta_obs)
tau_total <- seq(delta_knots,T-delta_knots,delta_knots)
B_total <- bs(t_total, knots = tau_total, 
              degree = p, Boundary.knots = c(0,T))
K <- length(tau_total)
ncol(B_total) == K+p

#1
t1 <- seq(0,1,delta_obs)
tau1 <- seq(delta_knots,1-delta_knots,delta_knots)
B1 <- bs(t1, knots = tau1, degree = p, Boundary.knots = c(0,1))
K1 <- length(tau1)
ncol(B1) == K1+p

for(j in 1:K1){
  plot(t1,B1[,j],main = j)
  lines(t1,B_total[1:length(t1),j])
  Sys.sleep(0.5)
}

#2
new_start <- tau1[K1-p+1]
t2 <- seq(new_start, 2, delta_obs)
tau2 <- seq(new_start+delta_knots,2-delta_knots,delta_knots)
B2 <- bs(t2, knots = tau2, degree = p, Boundary.knots = c(new_start,2))
K2 <- length(tau2)

k1 <- which(round(t_total-new_start,4)==0)
k2 <- k1+length(t2)-1
for(j in p:(K2)){
  plot(t2,B2[,j],main = K1+j-p+1)
  lines(t2,B_total[k1:k2,K1+j-p+1])
  Sys.sleep(0.5)
}

#3
new_start <- tau2[K2-p+1]
t3 <- seq(new_start, 3, delta_obs)
tau3 <- seq(new_start+delta_knots,3-delta_knots,delta_knots)
B3 <- bs(t3, knots = tau3, degree = p, Boundary.knots = c(new_start,3))
K3 <- length(tau3)

k1 <- which(round(t_total-new_start,4)==0)
k2 <- k1+length(t3)-1
for(j in p:(K3)){
  plot(t3,B3[,j],main = K1+K2+j-2*(p-1))
  lines(t3,B_total[k1:k2,K1+K2+j-2*(p-1)])
  Sys.sleep(0.5)
}

#4
new_start <- tau3[K3-p+1]
t4 <- seq(new_start, 4, delta_obs)
tau4 <- seq(new_start+delta_knots,4-delta_knots,delta_knots)
B4 <- bs(t4, knots = tau4, degree = p, Boundary.knots = c(new_start,4))
K4 <- length(tau4)

k1 <- which(round(t_total-new_start,4)==0)
k2 <- k1+length(t4)-1
for(j in p:(K4)){
  plot(t3,B3[,j],main = K1+K2+K3+j-3*(p-1))
  lines(t3,B_total[k1:k2,K1+K2+K3+j-3*(p-1)])
  Sys.sleep(0.5)
}
