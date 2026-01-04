module BandedMatricesExt

using SparseMatrixIdentification
using BandedMatrices

function __init__()
    return SparseMatrixIdentification._bandedmatrices_loaded[] = true
end

function SparseMatrixIdentification.try_banded(A, threshold)
    if SparseMatrixIdentification.is_banded(A, threshold)
        return BandedMatrix(A)
    end
    return nothing
end

end
