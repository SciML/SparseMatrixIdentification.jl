module FastAlmostBandedMatricesExt

using SparseMatrixIdentification
using FastAlmostBandedMatrices
using BandedMatrices

function __init__()
    SparseMatrixIdentification._fastalmostbandedmatrices_loaded[] = true
end

function SparseMatrixIdentification.try_almostbanded(Ad, n)
    # Try different bandwidths
    for bw in 1:min(5, n ÷ 2)
        is_ab, bandwidth, rank = SparseMatrixIdentification.is_almost_banded(Ad, bw)
        if is_ab
            # Create AlmostBandedMatrix from FastAlmostBandedMatrices
            # Extract banded part
            banded_part = zeros(eltype(Ad), n, n)
            for j in 1:n
                for i in max(1, j - bw):min(n, j + bw)
                    banded_part[i, j] = Ad[i, j]
                end
            end
            B = BandedMatrix(banded_part, (bw, bw))
            # For now, return the banded part as the structure is detected
            # Full AlmostBandedMatrix construction requires more complex setup
            return AlmostBandedMatrix(B, zeros(eltype(Ad), n, rank))
        end
    end
    return nothing
end

end
