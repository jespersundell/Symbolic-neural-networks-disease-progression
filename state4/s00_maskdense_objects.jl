#####################################################################
using Lux, Random, Zygote

struct DenseWithMask <: Lux.AbstractLuxLayer
    init_weight
    init_bias
    init_W_mask
    init_b_mask
    activation
end
function DenseWithMask(weight::AbstractArray, bias::AbstractArray, W_mask::AbstractArray, b_mask::AbstractArray, activation::Function)
    return DenseWithMask(() -> copy(weight), () -> copy(bias), () -> copy(W_mask),() -> copy(b_mask), activation)
end

# weight and bias are parameters
Lux.initialparameters(::AbstractRNG, layer::DenseWithMask) = (weight=layer.init_weight(), bias=layer.init_bias(),)
# W_mask and b_mask are states
Lux.initialstates(::AbstractRNG, layer::DenseWithMask) = (W_mask=layer.init_W_mask(), b_mask=layer.init_b_mask(),)
(l::DenseWithMask)(x, ps, st) = l.activation.((st.W_mask .* ps.weight * x) .+ (st.b_mask.*ps.bias)) , st

rng = Random.default_rng()

function DenseMaskLayer(in_dims, out_dims, activation = identity)
    layer = DenseWithMask(randn(rng, out_dims, in_dims), randn(rng, out_dims), ones(out_dims, in_dims), ones(out_dims), activation)
    return layer
end
####################################################################################