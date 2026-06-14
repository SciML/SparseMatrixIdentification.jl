using SparseMatrixIdentification
using Test

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
