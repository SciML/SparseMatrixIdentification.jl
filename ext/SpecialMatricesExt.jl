module SpecialMatricesExt

using SparseMatrixIdentification
using SpecialMatrices

function __init__()
    SparseMatrixIdentification._specialmatrices_loaded[] = true
end

function SparseMatrixIdentification.try_special_matrices(Ad, n, m)
    # Check for Hilbert matrix
    if SparseMatrixIdentification.is_hilbert(Ad)
        return Hilbert(n)
    end

    # Check for Strang matrix
    if SparseMatrixIdentification.is_strang(Ad)
        return Strang(n)
    end

    # Check for Vandermonde matrix
    if SparseMatrixIdentification.is_vandermonde(Ad)
        x = Ad[:, 2]  # Extract base values from second column
        return Vandermonde(x)
    end

    # Check for Cauchy matrix
    if SparseMatrixIdentification.is_cauchy(Ad)
        # Extract x and y vectors for Cauchy matrix construction
        y1 = 1.0 / Ad[1, 1]
        x = zeros(Float64, n)
        y = zeros(Float64, m)
        y[1] = y1
        for i in 2:n
            x[i] = 1.0 / Ad[i, 1] - y[1]
        end
        for j in 2:m
            y[j] = 1.0 / Ad[1, j] - x[1]
        end
        return Cauchy(x, y)
    end

    return nothing
end

end
