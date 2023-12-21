import LinearAlgebra as la
import FastExpm as fe
import SparseArrays as sa
import Expokit as expo

function propagation_julia(x::Array{ComplexF64, 4})
    dimension = size(x)
    propagator = zeros(ComplexF64, dimension)
    for n in range(1, dimension[1])
        y = x[n, :, :, :]
        for m in range(1, dimension[2])
            z = la.exp(x[n, m, :, :])
            propagator[n, m, :, :] = z
        end
    end
    return propagator
end

function propagation_julia_fast_expm(x::Array{ComplexF64, 4})
    dimension = size(x)
    propagator = zeros(ComplexF64, dimension)
    for n in range(1, dimension[1])
        y = x[n, :, :, :]
        for m in range(1, dimension[2])
            e = sa.sparse(x[n, m, :, :])
            z = fe.fastExpm(e; threshold=0.1)
            propagator[n, m, :, :] = z
        end
    end
    return propagator
end


function expmv_from_expokit(t::Number, H::Array{ComplexF64, 4}, rho::Array{ComplexF64, 3})
    dimension = size(rho)
    rho_prop = zeros(ComplexF64, dimension)
    for n in range(1, dimension[1])
        y = rho[n, :, :]
        for m in range(1, dimension[2])
            each_rho = rho[n, m, :]
            each_H = H[n, m, :, :]
            rho_prop[n, m, :] = expo.expmv(t, each_H, each_rho)
        end
    end
    return rho_prop
end
