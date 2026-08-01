module BandedMatricesExt

using SparseMatrixIdentification: SparseMatrixIdentification
using BandedMatrices: BandedMatrix

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
