module BlockBandedMatricesExt

using SparseMatrixIdentification
using BlockBandedMatrices
using BandedMatrices

function __init__()
    SparseMatrixIdentification._blockbandedmatrices_loaded[] = true
end

function SparseMatrixIdentification.try_blockbanded(Ad, n)
    blocksize = SparseMatrixIdentification.detect_block_size(Ad)
    if blocksize > 0
        nblocks = n ÷ blocksize
        # Create BlockBandedMatrix with detected block structure
        return BlockBandedMatrix{eltype(Ad)}(Ad, fill(blocksize, nblocks),
            fill(blocksize, nblocks), (1, 1))
    end
    return nothing
end

end
