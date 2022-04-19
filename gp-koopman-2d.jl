using Revise
using ToeplitzMatrices
include("easygpkoopman.jl")

# 2-dim
dt = 0.1
f(x; a=0.1, b=0.2) = [x[2], -b*x[1] + a*x[2] - x[2]^3]
full_f(x) = x + dt*f(x)
    
x0 =[2.0, 0.0] 
n = 1000
ndim = length(x0)
Xm = zeros(ndim, n)       # Trajectory in the state space

xs = -2.1:0.2:2.1
ys = -1.1:0.2:1.1
using LinearAlgebra
using Plots
df(x, y) = normalize(f([x,y]))/10.
xxs = [x for x in xs for y in ys]
yys = [y for x in xs for y in ys]
plt1 = quiver(xxs, yys, quiver=df)

# Ym = zeros(1, n)
xk = x0
Xm[:,1] = x0
for i=2:n
    global xk
    Xm[:,i] = full_f(xk) 
    xk = Xm[:,i]
end
scatter!(Xm[1,:], Xm[2,:], color=:black)

# Define observable priors and find the Hankel matrices
f1(x) = x[1]
f2(x) = x[2]
m = 50 
order = 5 

K1, Ur1, gps1 = GPKoopman.gp_koopman_estimate(f1, Xm, m, order)
K2, Ur2, gps2 = GPKoopman.gp_koopman_estimate(f2, Xm, m, order)

# Try it out at xt
bidx = 0 
xt = [-1.3, 0.01]
N = 10
plt = scatter([xt[1]], [xt[2]], label="")
for i=1:N
    global bidx, xt
    bidx += 1
    μ1, σ1 = GPKoopman.estimate_posterior_f_at_point(xt, Ur1, gps1, K=K1[1:order, 1:order], N=bidx)
    μ2, σ2 = GPKoopman.estimate_posterior_f_at_point(xt, Ur2, gps2, K=K2[1:order, 1:order], N=bidx)
    xt = full_f(xt)
    scatter!([μ1], [μ2], color=:red, label="") 
    covellipse!([μ1, μ2], [σ1 0; 0 σ2], color=:red, label="", fillalpha=0.1)
    scatter!([xt[1]], [xt[2]], color=:green, label="")
end

# xtp_true = full_f(xt)
# μ1, σ1 = GPKoopman.estimate_posterior_f_at_point(xt, Ur1, gps1, K=K1[1:order, 1:order], N=1)
# μ2, σ2 = GPKoopman.estimate_posterior_f_at_point(xt, Ur2, gps2, K=K2[1:order, 1:order], N=1)

# xp_hat = [μ1, μ2]
# xpp_true = full_f(xtp_true)
# μ12, σ12 = GPKoopman.estimate_posterior_f_at_point(xt, Ur1, gps1, K=K1[1:order, 1:order])
# μ22, σ22 = GPKoopman.estimate_posterior_f_at_point(xt, Ur2, gps2, K=K2[1:order, 1:order])

# using StatsPlots
# plt = scatter([μ1, μ12], [μ2, μ22], color=:red, label="GP Estimate (σ)")
# covellipse!([μ1, μ2], [σ1 0; 0 σ2], color=:red, label="", fillalpha=0.1)
# # covellipse!([μ12, μ22], [2*σ12 0; 0 2*σ22], color=:red, label="", fillalpha=0.1)
# scatter!([xt[1], xtp_true[1], xpp_true[1]], [xt[2], xtp_true[2], xpp_true[2]], color=:black, label="True")

plt
# plt1

# 
