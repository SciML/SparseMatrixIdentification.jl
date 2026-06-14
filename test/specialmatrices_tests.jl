using SparseMatrixIdentification
using SparseArrays
using SpecialMatrices
using Test

# Test for SpecialMatrices detection (Issue #3)
@testset "Test SpecialMatrices detection" begin
    # Test Hilbert matrix detection
    H = Matrix(Hilbert(4))
    @test SparseMatrixIdentification.is_hilbert(H) == true
    @test sparsestructure(sparse(H), 0.5) isa Hilbert

    # Test non-Hilbert matrix
    @test SparseMatrixIdentification.is_hilbert([1 2; 3 4]) == false

    # Test Strang matrix detection
    S = Matrix(Strang(5))
    @test SparseMatrixIdentification.is_strang(S) == true
    @test sparsestructure(sparse(S), 0.5) isa Strang

    # Test non-Strang matrix
    @test SparseMatrixIdentification.is_strang([1 2; 3 4]) == false

    # Test Vandermonde matrix detection
    x = [1.0, 2.0, 3.0, 4.0]
    V = Matrix(Vandermonde(x))
    @test SparseMatrixIdentification.is_vandermonde(V) == true
    @test sparsestructure(sparse(V), 0.5) isa Vandermonde

    # Test non-Vandermonde matrix
    @test SparseMatrixIdentification.is_vandermonde([1 2; 3 4]) == false

    # Test Cauchy matrix detection
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    C = Matrix(Cauchy(x, y))
    @test SparseMatrixIdentification.is_cauchy(C) == true
    @test sparsestructure(sparse(C), 0.5) isa Cauchy

    # Test non-Cauchy matrix
    @test SparseMatrixIdentification.is_cauchy([1 2; 3 4]) == false
end
