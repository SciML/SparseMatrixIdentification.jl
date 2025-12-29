using AllocCheck
using SparseMatrixIdentification
using SparseArrays
using LinearAlgebra

@testset "AllocCheck - Zero Allocations" begin
    # Test check_diagonal - critical inner function
    A = [1 2 3; 4 1 2; 5 4 1]
    @test (@allocated SparseMatrixIdentification.check_diagonal(A, 1, 1)) == 0

    # Test is_toeplitz
    @test (@allocated SparseMatrixIdentification.is_toeplitz(A)) == 0

    # Test is_banded
    @test (@allocated SparseMatrixIdentification.is_banded(A, 0.5)) == 0

    # Test compute_bandedness
    @test (@allocated SparseMatrixIdentification.compute_bandedness(A, 1)) == 0

    # Test compute_sparsity
    sparse_A = sparse(A)
    @test (@allocated SparseMatrixIdentification.compute_sparsity(sparse_A)) == 0
end
