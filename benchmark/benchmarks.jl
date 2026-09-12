using SparseMatrixIdentification, BenchmarkTools
using StableRNGs, SparseArrays

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

n = 2000
A_sparse = sprand(rng, n, n, 0.005)
# Banded sparse matrix
A_band = spdiagm(-1 => rand(rng, n - 1), 0 => rand(rng, n), 2 => rand(rng, n - 2))
# Block-banded
blk = 50
A_blk = spdiagm(
    -blk => rand(rng, n - blk), 0 => rand(rng, n), blk => rand(rng, n - blk)
)

# =============================================================================
# Structure detection
# =============================================================================

SUITE["detect"] = BenchmarkGroup()

SUITE["detect"]["getstructure_sparse"] = @benchmarkable getstructure($A_sparse)
SUITE["detect"]["getstructure_banded"] = @benchmarkable getstructure($A_band)
SUITE["detect"]["getstructure_blockbanded"] = @benchmarkable getstructure($A_blk)
SUITE["detect"]["sparsestructure"] = @benchmarkable sparsestructure(
    $A_sparse, 0.5
)
SUITE["detect"]["sparsestructure_dense"] = @benchmarkable sparsestructure(
    $(sparse(rand(rng, 500, 500))), 0.5
)
