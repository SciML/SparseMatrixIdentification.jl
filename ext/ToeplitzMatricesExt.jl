module ToeplitzMatricesExt

using SparseMatrixIdentification
using ToeplitzMatrices

function __init__()
    SparseMatrixIdentification._toeplitzmatrices_loaded[] = true
end

function SparseMatrixIdentification.try_toeplitz(A)
    if SparseMatrixIdentification.is_toeplitz(A)
        first_row = A[1, :]
        first_col = A[:, 1]
        return Toeplitz(first_col, first_row)
    end
    return nothing
end

end
