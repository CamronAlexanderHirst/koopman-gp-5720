module GPKoopman

using LinearAlgebra
using ToeplitzMatrices
using GaussianProcesses
using Random

"""
    create_Hankel_matrix

Create a Hankel matrix using the given observable function with parameter m.
"""
function create_Hankel_matrix(f, X, m; σ=0.01)
    r,n = size(X)
    if r == 1
        Y = f.(X) + σ*randn(1, n)
    else
        Y = f.(eachcol(X)) + σ*randn(n)
    end
    H = Array(Hankel(Y[1:m], Y[m:n]))
    return H, Y
end

function find_Koopman_approximation(H, order)
    out = svd(H)
    U = out.U
    Λ = out.S
    Vt = out.Vt
    K_est = pinv(U[1:end-1, 1:end])*U[2:end, 1:end]
    Z_lift = diagm(Λ)*Vt
    return K_est, Z_lift, U
end

function find_posterior_f(U, Z, order, m, X; μ0=MeanZero(), k0=SE(log(0.65), 0.), K=I)
    Ur = U[1,1:order]
    n = size(X,2)
    Xp = X[:, 1:n-m-1]
    Zp = (K*Z)[1:order, 1:n-m-1]

    # GP conditioning
    GPs = []
    for i=1:order
        gp = GP(Xp, Zp[i,:], μ0, k0, log(0.1))
        # optimize!(gp)
        push!(GPs, gp)
    end 
    return Ur, GPs
end

function estimate_posterior_f_at_point(x::Vector, Ur, gps; K=I, N=0)
    # Assume 2D
    if length(x)[1] == 1
        nothing
        xr = x
    else
        xr = reshape(x, length(x), 1)
    end
    μ_vec = zeros(length(Ur))
    σ_vec = zeros(length(Ur))
    μt = 0.
    σt = 0.
    for i=1:length(Ur)
        μ, σ = predict_f(gps[i], xr)
        μ_vec[i] = μ[1]
        σ_vec[i] = σ[1]^2
    end

    μ_vec = (K^N)*μ_vec
    # TODO: This is not correct way to calculate variance
    σ_vec = N*σ_vec
    # for i=1:N
    #     σ_vec += K*(σ_vec)*K' 
    # end
    μt = Ur'*μ_vec
    σt = sum(σ_vec)

    return μt, sqrt(σt) 
end

function gp_koopman_estimate(f, X, m, order)
    H, Y = GPKoopman.create_Hankel_matrix(f, X, m) 
    K, Z, U = GPKoopman.find_Koopman_approximation(H, order)
    Ur, gps = GPKoopman.find_posterior_f(U, Z, order, m, X)
    return K, Ur, gps
end

end # end module