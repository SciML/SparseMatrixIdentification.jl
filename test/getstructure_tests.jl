using SparseMatrixIdentification
using LinearAlgebra
using Test

# Test for `getstructure` function
@testset "Test getstructure" begin
    # Test 1: Band matrix
    A = [1 2 0; 3 4 5; 0 6 7]
    @test getstructure(A) == (100.0, 0.2222222222222222)

    # Test 2: Identity matrix
    A = Matrix(I, 3, 3)
    @test getstructure(A) == (100.0, 0.6666666666666667)
end
