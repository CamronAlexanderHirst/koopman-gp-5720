module EasyKoopman

using LinearAlgebra
using StaticArrays

function example_lift(x)
    return [x x.^2 x.^3 x.^4 sin.(x) 1.]
end

function example_nd_lift(x)
    ndims = length(x)
    nlift = length(example_lift(x[1]))
    xlift = zeros(1,nlift*ndims)
    offset = 1
    for i=1:ndims
        xlift[offset:offset+nlift-1] = example_lift(x[i])
        offset += nlift
    end
    xlift = xlift[1:end-1]
    return xlift'
end

# function example_lift_2d

function estimate_koopman_operator(lifting_function, input, output::Vector)
    lifted_dim = length(lifting_function(input[1,:])) 
    N = size(input,1)
    A = zeros(lifted_dim, lifted_dim)
    B = zeros(lifted_dim, lifted_dim)
    for i=1:N
        in_lift = lifting_function(input[i,:]) 
        A += lifting_function(output[i])'*in_lift
        B += in_lift'*in_lift
    end
    koopman_op = A*inv(B)
    return koopman_op
end

end