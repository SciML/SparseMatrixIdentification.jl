module SparseMatrixIdentification
using LinearAlgebra
using SparseArrays
using BandedMatrices
using ToeplitzMatrices

# check the diagonal of a given matrix, helper for is_toeplitz
function check_diagonal(A, i, j)
    N = size(A, 1)
    M = size(A, 2)

    @inbounds num = A[i, j]
    i += 1
    j += 1

    @inbounds while i <= N && j <= M
        if A[i, j] != num
            return false
        end
        i += 1
        j += 1
    end
    return true
end

# check if toeplitz matrix
function is_toeplitz(mat)
    N = size(mat, 1)

    if N == 1
        return true
    end

    M = size(mat, 2)

    for j in 1:M
        if !check_diagonal(mat, 1, j)
            return false
        end
    end

    for i in 2:N
        if !check_diagonal(mat, i, 1)
            return false
        end
    end

    return true
end

# compute the percentage banded for a matrix given a bandwidth
function compute_bandedness(A, bandwidth)
    if bandwidth == 0
        return 100.0
    end

    n = size(A, 1)
    total_band_positions = 0
    non_zero_in_band = 0
    @inbounds for r in 1:n
        for c in 1:n
            if abs(r - c) < bandwidth
                total_band_positions += 1  # This position belongs to the band
                if A[r, c] != 0
                    non_zero_in_band += 1  # This element is non-zero in the band
                end
            end
        end
    end

    percentage_filled = non_zero_in_band / total_band_positions * 100
    return percentage_filled
end

function is_banded(A, threshold)
    n = size(A, 1)  # assuming A is square
    bandwidth = n * threshold
    # Count the number of non-zero entries outside the band
    @inbounds for r in 1:n
        for c in 1:n
            if abs(r - c) >= bandwidth && A[r, c] != 0
                return false
            end
        end
    end

    # If there are any non-zero entries outside the band, it's not banded
    return true
end

# compute the sparsity for a given matrix
function compute_sparsity(A)
    n = size(A, 1)
    percentage_sparsity = length(nonzeros(A)) / n^2
    return 1 - percentage_sparsity
end

export getstructure

"""
    getstructure(A::AbstractMatrix)

Compute structure metrics for a matrix.

Returns a tuple `(percentage_banded, percentage_sparsity)` where:
- `percentage_banded`: The percentage of positions within bandwidth=1 that contain non-zero elements
- `percentage_sparsity`: The proportion of zero elements in the matrix (1 - density)

# Arguments
- `A::AbstractMatrix`: The input matrix to analyze

# Returns
- `Tuple{Float64, Float64}`: A tuple of (bandedness percentage, sparsity percentage)

# Examples
```julia
julia> A = [1 2 0; 3 4 5; 0 6 7]
julia> getstructure(A)
(100.0, 0.2222222222222222)
```
"""
function getstructure(A::AbstractMatrix)::Any
    percentage_banded = compute_bandedness(A, 1)
    percentage_sparsity = compute_sparsity(SparseMatrixCSC(A))

    return (percentage_banded, percentage_sparsity)
end

export sparsestructure

"""
    sparsestructure(A::SparseMatrixCSC, threshold)

Identify the structure of a sparse matrix and return an optimized matrix type.

Analyzes the input sparse matrix and returns the most appropriate specialized matrix type
based on detected structural properties. The function checks for properties in the following
order: Toeplitz, Symmetric, Hermitian, Lower Triangular, Upper Triangular, Banded, and
falls back to SparseMatrixCSC if no special structure is detected.

# Arguments
- `A::SparseMatrixCSC`: The sparse matrix to analyze
- `threshold`: Bandwidth threshold as a fraction of matrix size for banded detection

# Returns
One of the following matrix types based on detected structure:
- `Toeplitz`: If the matrix has constant diagonals
- `Symmetric`: If the matrix is symmetric
- `Hermitian`: If the matrix is Hermitian (complex conjugate symmetric)
- `LowerTriangular`: If all elements above the diagonal are zero
- `UpperTriangular`: If all elements below the diagonal are zero
- `BandedMatrix`: If non-zeros are confined within a band around the diagonal
- `SparseMatrixCSC`: If no special structure is detected

# Examples
```julia
julia> using SparseArrays
julia> A = sparse([1 2 2; 2 3 4; 2 4 5])
julia> sparsestructure(A, 0.5)  # Returns Symmetric

julia> L = sparse([1 0 0; 2 3 0; 4 5 6])
julia> sparsestructure(L, 0.5)  # Returns LowerTriangular
```
"""
function sparsestructure(A::SparseMatrixCSC, threshold)
    sym = issymmetric(A)
    herm = ishermitian(A)
    banded = is_banded(A, threshold)
    posdef = isposdef(A)
    lower_triangular = istril(A)
    upper_triangular = istriu(A)
    toeplitz = is_toeplitz(A)

    n = size(A, 1)

    if toeplitz
        first_row = A[1, :]
        first_col = A[:, 1]
        return Toeplitz(first_col, first_row)
    end

    if sym
        return Symmetric(A)
    end

    if herm
        return Hermitian(A)
    end

    if lower_triangular
        return LowerTriangular(A)
    end

    if upper_triangular
        return UpperTriangular(A)
    end

    if banded
        return BandedMatrix(A)
    end

    return SparseMatrixCSC(A)
end

end
