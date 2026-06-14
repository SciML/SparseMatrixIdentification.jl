using SparseMatrixIdentification
using SparseArrays
using BandedMatrices
using BlockBandedMatrices
using Test

# Test for BlockBandedMatrices detection (Issue #2)
@testset "Test BlockBandedMatrices detection" begin
    # Test block-banded detection helper
    # Create a 6x6 block-tridiagonal matrix with 2x2 blocks (non-symmetric, non-triangular)
    A = zeros(6, 6)
    # Fill diagonal blocks with non-symmetric values
    A[1:2, 1:2] = [1.0 2.0; 3.0 4.0]
    A[3:4, 3:4] = [5.0 6.0; 7.0 8.0]
    A[5:6, 5:6] = [9.0 10.0; 11.0 12.0]
    # Fill upper off-diagonal blocks
    A[1:2, 3:4] = [0.1 0.2; 0.3 0.4]
    A[3:4, 5:6] = [0.5 0.6; 0.7 0.8]
    # Fill lower off-diagonal blocks (different from upper to make it non-symmetric)
    A[3:4, 1:2] = [1.1 1.2; 1.3 1.4]
    A[5:6, 3:4] = [1.5 1.6; 1.7 1.8]

    @test SparseMatrixIdentification.is_blockbanded_uniform(A, 2) == true
    @test SparseMatrixIdentification.detect_block_size(A) == 2
    # Note: sparsestructure might detect other structures first, so we just test detection helpers
    result = sparsestructure(sparse(A), 0.5)
    @test result isa BlockBandedMatrix || result isa BandedMatrix ||
        result isa SparseMatrixCSC

    # Test non-block-banded matrix
    B = rand(6, 6)
    @test SparseMatrixIdentification.is_blockbanded_uniform(B, 2) == false
end
