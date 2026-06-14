using SparseMatrixIdentification
using LinearAlgebra
using SparseArrays
using Test

# Test for `compute_sparsity` function
@testset "Test compute_sparsity" begin
    # Test 1: Identity matrix
    A = Matrix(I, 3, 3)
    A = SparseMatrixCSC(A)
    @test SparseMatrixIdentification.compute_sparsity(A) == 1 - 1 / 3

    # Test 2: Sparse matrix with a few non-zero elements
    A = [0 0 0; 0 5 0; 0 0 0]
    A = SparseMatrixCSC(A)
    @test SparseMatrixIdentification.compute_sparsity(A) == 1 - 1 / 9  # sparsity = 1 non-zero element / 9 total elements

    # Test 3: Full random matrix (sparsity = 0)
    A = [1 2 3; 4 5 6; 7 8 9]
    A = SparseMatrixCSC(A)
    @test SparseMatrixIdentification.compute_sparsity(A) == 0.0  # no sparsity
end
