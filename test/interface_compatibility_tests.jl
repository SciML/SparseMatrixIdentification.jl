using SparseMatrixIdentification
using LinearAlgebra
using SparseArrays
using ToeplitzMatrices
using JLArrays
using Test

# Interface compatibility tests
@testset "Interface Compatibility" begin
    @testset "BigFloat support" begin
        # Test that functions work with BigFloat element types
        A_bf = BigFloat[1 2 3; 4 1 2; 5 4 1]
        @test SparseMatrixIdentification.is_toeplitz(A_bf) == true
        @test SparseMatrixIdentification.check_diagonal(A_bf, 1, 1) == true
        @test SparseMatrixIdentification.is_banded(A_bf, 1 / 3) == false
        @test SparseMatrixIdentification.compute_bandedness(A_bf, 1) == 100.0

        # Test getstructure with BigFloat
        A_bf2 = BigFloat[1 2 0; 3 4 5; 0 6 7]
        result = getstructure(A_bf2)
        @test result[1] == 100.0
        @test result[2] isa AbstractFloat

        # Test sparsestructure with BigFloat
        sparse_bf = SparseMatrixCSC(A_bf)
        result_sparse = sparsestructure(sparse_bf, 1 / 3)
        @test result_sparse isa Toeplitz
    end

    @testset "JLArray error messages" begin
        # JLArrays are GPU-like arrays that don't support fast scalar indexing
        # Verify that we get clear error messages instead of cryptic GPU errors
        A_jl = JLArray([1.0 2.0 3.0; 4.0 1.0 5.0; 6.0 7.0 1.0])

        @test_throws ArgumentError SparseMatrixIdentification.check_diagonal(A_jl, 1, 1)
        @test_throws ArgumentError SparseMatrixIdentification.is_toeplitz(A_jl)
        @test_throws ArgumentError SparseMatrixIdentification.compute_bandedness(A_jl, 1)
        @test_throws ArgumentError SparseMatrixIdentification.is_banded(A_jl, 1 / 3)
        @test_throws ArgumentError getstructure(A_jl)

        # Check that the error message is informative
        try
            SparseMatrixIdentification.is_toeplitz(A_jl)
            @test false  # Should not reach here
        catch e
            @test e isa ArgumentError
            @test occursin("fast scalar indexing", e.msg)
            @test occursin("GPU arrays", e.msg)
        end
    end

    @testset "AbstractMatrix type genericity" begin
        # Test that getstructure accepts AbstractMatrix, not just Matrix
        A_dense = [1 2 0; 3 4 5; 0 6 7]
        A_transpose = transpose([1 3 0; 2 4 6; 0 5 7])

        # Both should work via AbstractMatrix dispatch
        result1 = getstructure(A_dense)
        result2 = getstructure(A_transpose)
        @test result1 == result2
    end
end
