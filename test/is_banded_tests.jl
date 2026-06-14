using SparseMatrixIdentification
using LinearAlgebra
using Test

# Test for `is_banded` function
@testset "Test is_banded" begin
    # Test 1: matrix with filled elements within the band
    A = [1 2 0; 3 4 5; 0 6 7]
    @test SparseMatrixIdentification.is_banded(A, 1 / 3) == false

    # Test 2: Identity matrix (banded)
    A = Matrix(I, 3, 3)
    @test SparseMatrixIdentification.is_banded(A, 1 / 3) == true

    # Test 3: Full random matrix (non-banded)
    A = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.is_banded(A, 1 / 3) == false
end
