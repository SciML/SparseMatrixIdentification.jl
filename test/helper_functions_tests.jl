using SparseMatrixIdentification
using Test

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
