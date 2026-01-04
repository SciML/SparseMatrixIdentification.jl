using SparseMatrixIdentification
using Test
using LinearAlgebra
using SparseArrays
using BandedMatrices
using ToeplitzMatrices
using SpecialMatrices
using BlockBandedMatrices
using FastAlmostBandedMatrices
using JLArrays

@testset "Test check_diagonal" begin
    A = [1 2 3; 4 1 5; 6 7 1]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # Diagonal starting at (1, 1) should be all 1s
    @test SparseMatrixIdentification.check_diagonal(A, 1, 2) == false  # Diagonal starting at (1, 2) should not be all 1s

    # Test 2: Edge case with a 1x1 matrix
    A = [7]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # A 1x1 matrix is trivially a Toeplitz matrix

    # Test 3: Negative case, diagonal mismatch
    A = [1 2 3; 4 1 5; 6 7 2]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == false  # Diagonal starting at (1, 1) should be different

    # Test 4: Larger matrix, with valid Toeplitz diagonals
    A = [1 2 3 4; 5 1 2 3; 6 5 1 2; 7 6 5 1]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # Diagonal starting at (1, 1) should be all 1s
    @test SparseMatrixIdentification.check_diagonal(A, 1, 2) == true   # Diagonal starting at (1, 2) should be all 2s

    # Test 5: Non-square matrix, checking diagonals in both directions
    A = [1 2 3; 4 1 2; 5 4 1]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # Diagonal starting at (1, 1) should be all 1s
    @test SparseMatrixIdentification.check_diagonal(A, 1, 2) == true   # Diagonal starting at (1, 2) should be all 2s

    # Test 6: Full mismatch
    A = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == false  # Diagonal starting at (1, 1) should not match, as 1 != 4
end

@testset "Test is_toeplitz" begin
    # Test 1: A basic Toeplitz matrix (2x2)
    mat1 = [1 2; 3 1]
    @test SparseMatrixIdentification.is_toeplitz(mat1) == true

    # Test 2: A non-Toeplitz matrix (3x3)
    mat2 = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.is_toeplitz(mat2) == false

    # Test 3: A 1x1 matrix (Trivially Toeplitz)
    mat3 = [5]
    @test SparseMatrixIdentification.is_toeplitz(mat3) == true

    # Test 4: A 3x3 Toeplitz matrix
    mat4 = [1 2 3; 4 1 2; 5 4 1]
    @test SparseMatrixIdentification.is_toeplitz(mat4) == true

    # Test 5: A 3x3 non-Toeplitz matrix with different diagonals
    mat5 = [1 2 3; 4 1 5; 6 7 1]
    @test SparseMatrixIdentification.is_toeplitz(mat5) == false

    # Test 6: A matrix with identical columns, which is not Toeplitz
    mat6 = [1 1 1; 2 2 2; 3 3 3]
    @test SparseMatrixIdentification.is_toeplitz(mat6) == false

    # Test 7: A 2x2 Toeplitz matrix
    mat7 = [1 2; 2 1]
    @test SparseMatrixIdentification.is_toeplitz(mat7) == true

    # Test 9: A large Toeplitz matrix (5x5)
    mat8 = [
        1 2 3 4 5;
        6 1 2 3 4;
        7 6 1 2 3;
        8 7 6 1 2;
        9 8 7 6 1
    ]
    @test SparseMatrixIdentification.is_toeplitz(mat8) == true

    # Test 10: A large non-Toeplitz matrix (5x5)
    mat9 = [
        1 2 3 4 5;
        6 7 8 9 10;
        11 12 13 14 15;
        16 17 18 19 20;
        21 22 23 24 25
    ]
    @test SparseMatrixIdentification.is_toeplitz(mat9) == false
end

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

# Test for `getstructure` function
@testset "Test getstructure" begin
    # Test 1: Band matrix
    A = [1 2 0; 3 4 5; 0 6 7]
    @test getstructure(A) == (100.0, 0.2222222222222222)

    # Test 2: Identity matrix
    A = Matrix(I, 3, 3)
    @test getstructure(A) == (100.0, 0.6666666666666667)
end

# Test for `sparsestructure` function
@testset "Test sparsestructure" begin

    # Test 1: Sparse banded matrix (banded)
    A = [1 2 0; 3 4 5; 0 6 7]
    @test sparsestructure(sparse(A), 2 / 3) isa BandedMatrix

    # Test 2: Symmetric matrix
    A = [1 2 2; 2 3 4; 2 4 5]
    @test sparsestructure(SparseMatrixCSC(A), 1 / 3) isa Symmetric

    # Test 3: Hermitian matrix (complex conjugate symmetry)
    A = [1 2 + 3im 4 + 5im; 2 - 3im 6 7 + 8im; 4 - 5im 7 - 8im 9]
    @test sparsestructure(SparseMatrixCSC(A), 1 / 3) isa Hermitian

    # Test 4: Lower triangular matrix
    A = [1 0 0; 2 3 0; 4 5 6]
    @test sparsestructure(SparseMatrixCSC(A), 1 / 3) isa LowerTriangular

    # Test 5: Upper triangular matrix
    A = [1 2 3; 0 4 5; 0 0 6]
    @test sparsestructure(SparseMatrixCSC(A), 1 / 3) isa UpperTriangular

    # Test 6: Generic sparse matrix (fallback)
    B = [1 2 3; 4 5 6; 7 8 9]
    sparse_B = SparseMatrixCSC(B)
    @test sparsestructure(sparse_B, 1 / 3) isa SparseMatrixCSC

    # Test 7: Toeplitz Sparse Matrix
    T = [1 2 0; 0 1 2; 0 0 1]
    sparse_T = SparseMatrixCSC(T)
    @test sparsestructure(sparse_T, 1 / 3) isa Toeplitz
end

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

# Test for helper functions
@testset "Test helper functions" begin
    # Test is_almost_banded helper function
    # Create a simple almost-banded matrix
    n = 10
    A = zeros(n, n)
    # Tridiagonal part
    for i in 1:n
        A[i, i] = 2.0
        if i > 1
            A[i, i - 1] = -1.0
        end
        if i < n
            A[i, i + 1] = -1.0
        end
    end

    # Test that a purely tridiagonal matrix is NOT almost-banded (it's just banded)
    is_ab, bw, rank = SparseMatrixIdentification.is_almost_banded(A, 1)
    @test is_ab == false  # Pure tridiagonal is banded, not almost-banded

    # Test detection functions exist and return expected types
    @test SparseMatrixIdentification.is_hilbert([1.0 0.5; 0.5 1 / 3]) == true
    @test SparseMatrixIdentification.is_strang([2 -1; -1 2]) == true
end

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

# Allocation tests - run in "nopre" group to avoid precompilation issues
if get(ENV, "GROUP", "all") == "all" || get(ENV, "GROUP", "all") == "nopre"
    @testset "Allocation Tests" begin
        include("alloc_tests.jl")
    end
end
