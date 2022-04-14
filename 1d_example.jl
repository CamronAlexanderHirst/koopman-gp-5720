include("easykoopman.jl")


z(x) = EasyKoopman.example_lift(x)

# Consider first a 1-dimensional function: 
f(x) = x + 0.5sin(x) + (abs(x)+0.1)^(1/3) - 0.01x^3 
N = 100                 # number of pts
scale = 20      
range = scale*[-1, 1]

using Distributions
y(x) = f(x) + rand(Normal(0, 0.1))
xt = 2*scale*(rand(N).-0.5)
yt = y.(xt)

koopman_est = EasyKoopman.estimate_koopman_operator(z, xt, yt)

using Plots

plt = scatter(xt, yt, label="", dpi=600)
x_true = range[1]:0.1:range[2]

z_true = z.(x_true)
z_prop = vcat([koopman_est[1,:]'*z_true[i]' for i=1:length(x_true)]...)

plot!(x_true, f.(x_true), color=:black, label="true")
plot!(x_true, z_prop[:,1], width=3, color=:purple, label="Koopman", legend=:topleft, linealpha=0.3)
xlims!(minimum(xt), maximum(xt))
ylims!(-3, 15)
savefig(plt, "gps_vs_koopman-plot.png")
plt