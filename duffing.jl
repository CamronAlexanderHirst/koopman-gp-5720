include("easykoopman.jl")

z(x) = EasyKoopman.example_nd_lift(x)

# Consider first a 1-dimensional function: 
f(x; a=0.1, b=0.2) = [x[2], -b*x[1] + a*x[2] - x[2]^3]
dim = 2 
N = 100                # number of pts
scale = 1.5
range = scale*[-1, 1]

# Generate data
# using Distributions
y(x) = f(x) #+ rand(Normal(0, 0.1), 2)
xt = 2*scale*(rand(N, dim).-0.5)
yt = y.(eachrow(xt))

koopman_est = EasyKoopman.estimate_koopman_operator(z, xt, yt)

x_true = xt
z_true = z.(eachrow(x_true))
lifted_dim = length(EasyKoopman.example_lift(1))
z1_prop = []
z1_prop = push!([koopman_est*z_true[i]' for i=1:size(x_true,1)])
f_true = f.(eachrow(x_true))

# RMSE Calculation at Datapoints
error = 0.
for i=1:size(x_true,1)
    global error
    error += (z1_prop[i][1] - f_true[i][1])^2 + (z1_prop[i][1+lifted_dim] - f_true[i][2])^2  
end
@show error = sqrt(error)

# Open loop prediction
x0 = [0.; 0.2]
H = 2 
z0 = z(x0)'
zk = z0
fk = x0
xs = [x0[1]]
ys = [x0[2]]
fxt = [x0[1]]
fyt = [x0[2]]
for i=1:H
    global zk, fk
    zkp = koopman_est*zk
    push!(xs, zkp[1])
    push!(ys, zkp[1+lifted_dim])
    fkp = f(fk)
    push!(fxt, fkp[1])
    push!(fyt, fkp[2])
    # zk=z(zkp[1:5:end])'
    zk = zkp
    fk=fkp
end 

# Plot for fun
using Plots
plt = scatter(xt[:,1], xt[:,2],  label="Datapoints", dpi=600)
[plot!([xt[i,1], yt[i][1]], [xt[i,2], yt[i][2]], color=:black, label="", linealpha=0.1) for i=1:N]
plot!(xs, ys, label="Koopman")
plot!(fxt, fyt, label="True")
plt