using SparseMatrixIdentification
using LinearAlgebra
using Test

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
