module SparseMatrixIdentification
using LinearAlgebra
using SparseArrays

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

# Check if matrix is a Hilbert matrix: A[i,j] = 1/(i+j-1)
function is_hilbert(A; rtol = 1e-10)
    n, m = size(A)
    if n != m
        return false
    end
    @inbounds for j in 1:n
        for i in 1:n
            expected = 1 / (i + j - 1)
            if !isapprox(A[i, j], expected; rtol = rtol)
                return false
            end
        end
    end
    return true
end

# Check if matrix is a Strang matrix: tridiagonal Toeplitz with [2, -1, -1] pattern
function is_strang(A; rtol = 1e-10)
    n, m = size(A)
    if n != m || n < 2
        return false
    end
    @inbounds for j in 1:n
        for i in 1:n
            if i == j
                if !isapprox(A[i, j], 2; rtol = rtol)
                    return false
                end
            elseif abs(i - j) == 1
                if !isapprox(A[i, j], -1; rtol = rtol)
                    return false
                end
            else
                if !isapprox(A[i, j], 0; atol = rtol)
                    return false
                end
            end
        end
    end
    return true
end

# Check if matrix is a Vandermonde matrix: V[i,j] = x[i]^(j-1)
function is_vandermonde(A; rtol = 1e-10)
    n, m = size(A)
    if n < 2 || m < 2
        return false
    end
    # First column should be all ones (x^0 = 1)
    @inbounds for i in 1:n
        if !isapprox(A[i, 1], 1; rtol = rtol)
            return false
        end
    end
    # Extract the base values from the second column
    x = A[:, 2]
    # Check that each column j has values x[i]^(j-1)
    @inbounds for j in 3:m
        for i in 1:n
            expected = x[i]^(j - 1)
            if !isapprox(A[i, j], expected; rtol = rtol)
                return false
            end
        end
    end
    return true
end

# Check if matrix is a Cauchy matrix: A[i,j] = 1/(x[i] + y[j])
# Only works for real matrices
function is_cauchy(A; rtol = 1e-10)
    n, m = size(A)
    if n < 2 || m < 2
        return false
    end

    # Only check real matrices
    if eltype(A) <: Complex
        return false
    end

    # Try to extract x and y from first row and first column
    # A[1,j] = 1/(x[1] + y[j]) and A[i,1] = 1/(x[i] + y[1])
    # So: x[1] + y[j] = 1/A[1,j] => y[j] = 1/A[1,j] - x[1]
    # And: x[i] + y[1] = 1/A[i,1] => x[i] = 1/A[i,1] - y[1]
    # From A[1,1]: x[1] + y[1] = 1/A[1,1]
    # Let's set x[1] = 0, then y[1] = 1/A[1,1]

    @inbounds begin
        if A[1, 1] == 0
            return false
        end
        y1 = 1.0 / A[1, 1]
        x = zeros(Float64, n)
        y = zeros(Float64, m)
        y[1] = y1

        # Extract x values from first column
        for i in 2:n
            if A[i, 1] == 0
                return false
            end
            x[i] = 1.0 / A[i, 1] - y[1]
        end

        # Extract y values from first row
        for j in 2:m
            if A[1, j] == 0
                return false
            end
            y[j] = 1.0 / A[1, j] - x[1]
        end

        # Verify all entries
        for j in 1:m
            for i in 1:n
                denom = x[i] + y[j]
                if denom == 0
                    return false
                end
                expected = 1.0 / denom
                if !isapprox(Float64(A[i, j]), expected; rtol = rtol)
                    return false
                end
            end
        end
    end
    return true
end

# Check if matrix has block-banded structure with given block size
function is_blockbanded_uniform(A, blocksize; threshold = 0.0)
    n, m = size(A)
    if n != m || n % blocksize != 0
        return false
    end
    nblocks = n ÷ blocksize

    # Check that entries outside the block-tridiagonal region are zero
    @inbounds for bj in 1:nblocks
        for bi in 1:nblocks
            if abs(bi - bj) > 1  # Outside block-tridiagonal
                # Check if block is all zeros
                for j in ((bj - 1) * blocksize + 1):(bj * blocksize)
                    for i in ((bi - 1) * blocksize + 1):(bi * blocksize)
                        if A[i, j] != 0
                            return false
                        end
                    end
                end
            end
        end
    end
    return true
end

# Detect block size for block-banded matrix (tries common block sizes)
function detect_block_size(A)
    n = size(A, 1)
    # Try block sizes that evenly divide n
    # Only consider block sizes where we have at least 3 blocks (to be meaningful)
    for bs in [2, 3, 4, 5, 6, 8, 10]
        nblocks = n ÷ bs
        if n % bs == 0 && nblocks >= 3 && is_blockbanded_uniform(A, bs)
            return bs
        end
    end
    return 0  # No block structure detected
end

# Check if matrix is almost banded (banded + low-rank fill)
function is_almost_banded(A, bandwidth; rank_threshold = 2)
    n = size(A, 1)
    # Only check for matrices that are large enough to have meaningful structure
    if n < 6
        return false, 0, 0
    end

    # Only check real matrices for now
    if eltype(A) <: Complex
        return false, 0, 0
    end

    # Extract the part outside the band
    outside_band = zeros(Float64, n, n)
    has_nonzero = false
    @inbounds for j in 1:n
        for i in 1:n
            if abs(i - j) > bandwidth
                outside_band[i, j] = Float64(A[i, j])
                if A[i, j] != 0
                    has_nonzero = true
                end
            end
        end
    end

    # Check if the outside-band part has non-zeros
    if !has_nonzero
        return false, 0, 0  # It's just banded, not almost banded
    end

    # Compute SVD to check rank
    try
        S = svdvals(outside_band)
        # Count significant singular values
        tol = maximum(S) * 1e-6
        rank = count(s -> s > tol, S)
        if rank > 0 && rank <= rank_threshold
            return true, bandwidth, rank
        end
    catch
        return false, 0, 0
    end
    return false, 0, 0
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

# Extension point functions - extensions define these using multiple dispatch
# The base module does NOT define fallback methods; instead sparsestructure
# checks if methods exist before calling them.

# Placeholder types for dispatch - extensions will add methods for these
abstract type SpecialMatricesExtTrait end
abstract type ToeplitzMatricesExtTrait end
abstract type BandedMatricesExtTrait end
abstract type BlockBandedMatricesExtTrait end
abstract type FastAlmostBandedMatricesExtTrait end

# Extension flags - set to true when extensions are loaded
const _specialmatrices_loaded = Ref(false)
const _toeplitzmatrices_loaded = Ref(false)
const _bandedmatrices_loaded = Ref(false)
const _blockbandedmatrices_loaded = Ref(false)
const _fastalmostbandedmatrices_loaded = Ref(false)

"""
    try_special_matrices(Ad, n, m)

Extension point for SpecialMatrices.jl. Returns nothing when SpecialMatrices is not loaded.
When SpecialMatrices is loaded, returns Hilbert, Strang, Vandermonde, or Cauchy if detected.
"""
function try_special_matrices end

"""
    try_toeplitz(A)

Extension point for ToeplitzMatrices.jl. Returns nothing when ToeplitzMatrices is not loaded.
When ToeplitzMatrices is loaded, returns Toeplitz if detected.
"""
function try_toeplitz end

"""
    try_blockbanded(Ad, n)

Extension point for BlockBandedMatrices.jl. Returns nothing when BlockBandedMatrices is not loaded.
When BlockBandedMatrices is loaded, returns BlockBandedMatrix if detected.
"""
function try_blockbanded end

"""
    try_almostbanded(Ad, n)

Extension point for FastAlmostBandedMatrices.jl. Returns nothing when FastAlmostBandedMatrices is not loaded.
When FastAlmostBandedMatrices is loaded, returns AlmostBandedMatrix if detected.
"""
function try_almostbanded end

"""
    try_banded(A, threshold)

Extension point for BandedMatrices.jl. Returns nothing when BandedMatrices is not loaded.
When BandedMatrices is loaded, returns BandedMatrix if detected.
"""
function try_banded end

"""
    sparsestructure(A::SparseMatrixCSC, threshold)

Identify the structure of a sparse matrix and return an optimized matrix type.

Analyzes the input sparse matrix and returns the most appropriate specialized matrix type
based on detected structural properties. The function checks for properties in the following
order: Hilbert, Strang, Vandermonde, Cauchy (from SpecialMatrices), Toeplitz, Symmetric,
Hermitian, Lower Triangular, Upper Triangular, BlockBandedMatrix, AlmostBandedMatrix,
BandedMatrix, and falls back to SparseMatrixCSC if no special structure is detected.

Note: Specialized matrix types (Hilbert, Strang, Vandermonde, Cauchy, Toeplitz,
BlockBandedMatrix, AlmostBandedMatrix, BandedMatrix) are only returned if the
corresponding package is loaded. Load the relevant package to enable detection:
- `using SpecialMatrices` for Hilbert, Strang, Vandermonde, Cauchy
- `using ToeplitzMatrices` for Toeplitz
- `using BandedMatrices` for BandedMatrix
- `using BlockBandedMatrices` for BlockBandedMatrix
- `using FastAlmostBandedMatrices` for AlmostBandedMatrix

# Arguments
- `A::SparseMatrixCSC`: The sparse matrix to analyze
- `threshold`: Bandwidth threshold as a fraction of matrix size for banded detection

# Returns
One of the following matrix types based on detected structure:
- `Hilbert`: If A[i,j] = 1/(i+j-1) (requires SpecialMatrices)
- `Strang`: If the matrix is tridiagonal Toeplitz with pattern [2, -1, -1] (requires SpecialMatrices)
- `Vandermonde`: If columns are powers of a base vector (requires SpecialMatrices)
- `Cauchy`: If A[i,j] = 1/(x[i] + y[j]) (requires SpecialMatrices)
- `Toeplitz`: If the matrix has constant diagonals (requires ToeplitzMatrices)
- `Symmetric`: If the matrix is symmetric
- `Hermitian`: If the matrix is Hermitian (complex conjugate symmetric)
- `LowerTriangular`: If all elements above the diagonal are zero
- `UpperTriangular`: If all elements below the diagonal are zero
- `BlockBandedMatrix`: If the matrix has block-banded structure (requires BlockBandedMatrices)
- `AlmostBandedMatrix`: If the matrix is banded plus low-rank fill (requires FastAlmostBandedMatrices)
- `BandedMatrix`: If non-zeros are confined within a band around the diagonal (requires BandedMatrices)
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
    n = size(A, 1)
    m = size(A, 2)

    # Convert to dense for structure detection (needed for special matrices)
    Ad = Matrix(A)

    # Check for SpecialMatrices types first (requires SpecialMatrices extension)
    if _specialmatrices_loaded[]
        result = try_special_matrices(Ad, n, m)
        if result !== nothing
            return result
        end
    end

    # Check for Toeplitz (requires ToeplitzMatrices extension)
    if _toeplitzmatrices_loaded[]
        result = try_toeplitz(A)
        if result !== nothing
            return result
        end
    end

    # Check standard LinearAlgebra properties
    sym = issymmetric(A)
    herm = ishermitian(A)
    lower_triangular = istril(A)
    upper_triangular = istriu(A)

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

    # Check for BlockBandedMatrix (requires BlockBandedMatrices extension)
    if _blockbandedmatrices_loaded[]
        result = try_blockbanded(Ad, n)
        if result !== nothing
            return result
        end
    end

    # Check for AlmostBandedMatrix (requires FastAlmostBandedMatrices extension)
    if _fastalmostbandedmatrices_loaded[]
        result = try_almostbanded(Ad, n)
        if result !== nothing
            return result
        end
    end

    # Check for regular banded (requires BandedMatrices extension)
    if _bandedmatrices_loaded[]
        result = try_banded(A, threshold)
        if result !== nothing
            return result
        end
    end

    return SparseMatrixCSC(A)
end

end
