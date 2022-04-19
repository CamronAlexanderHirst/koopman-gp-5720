using Revise
using ToeplitzMatrices
include("easygpkoopman.jl")

# 1-dim
f1d(x;) = 0.9*x + 1.5cos.(x)
    
x0 = 0.5
n = 30
ndim = length(x0)
Xm = zeros(ndim, n)       # Trajectory in the state space

# Ym = zeros(1, n)
xk = x0
Xm[1] = x0
for i=2:n
    global xk
    Xm[i] = f1d.(xk) 
    xk = Xm[i]
end
    # plt1 = scatter(Xm[1:end-1], Xm[2:end], color=:black, label="")
plt1 = scatter(Xm[1:end], color=:black, label="")
plt1

# # Define observable priors and find the Hankel matrices
f1(x) = x
m = 11 
order = m 

K1, Ur1, gps1 = GPKoopman.gp_koopman_estimate(f1, Xm, m, order, )

# Try it out at xt
bidx = 0 
xt = [0.52]
N = 15
plt = scatter([bidx], xt, label="")
for i=1:N
    global bidx, xt
    bidx += 1
    μ1, σ1 = GPKoopman.estimate_posterior_f_at_point(xt, Ur1, gps1, K=K1[1:order, 1:order], N=bidx)
    xt = f1d.(xt)
    scatter!([bidx], [μ1], color=:red, label="") 
    plot!([bidx, bidx], [μ1 - 3σ1, μ1 + 3σ1], color=:red, label="")
    scatter!([bidx], xt, color=:green, label="")
end
plt
# plt1