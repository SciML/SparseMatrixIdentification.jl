using SparseMatrixIdentification
using LinearAlgebra
using Test

# Test for `compute_bandedness` function
@testset "Test compute_bandedness" begin
    # Test 1: Identity Matrix
    A = Matrix(I, 3, 3)
    bandwidth = 1
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0

    # Test 2: Band matrix (with zeros outside the band)
    A = [1 2 0; 3 4 5; 0 6 7]
    bandwidth = 1
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0

    # Test 3: Sparse random matrix
    A = [1 0 0; 0 2 0; 0 0 3]
    bandwidth = 0
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0

    # Test 4: Full random matrix (bandwidth = 1, not banded)
    A = [1 2 3; 4 5 6; 7 8 9]
    bandwidth = 1
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0
end
