# SparseMatrixIdentification.jl

SparseMatrixIdentification.jl is a Julia package for identifying and classifying the structure of sparse matrices. It automatically detects matrix properties such as symmetry, bandedness, triangularity, and Toeplitz structure, and returns an optimized matrix type for efficient computation.

## Installation

To install SparseMatrixIdentification.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("SparseMatrixIdentification")
```

## Quick Start

```julia
using SparseMatrixIdentification
using SparseArrays

# Create a sparse matrix
A = sparse([1 2 0; 3 4 5; 0 6 7])

# Get structure metrics (bandedness, sparsity)
bandedness, sparsity = getstructure(A)

# Get the optimal specialized matrix type
optimized = sparsestructure(A, 0.5)
```

## Features

SparseMatrixIdentification.jl provides two main functions:

### `getstructure`

Returns a tuple of `(percentage_banded, percentage_sparsity)` for a given matrix:

- **percentage_banded**: How much of the band (bandwidth=1) is filled with non-zero elements
- **percentage_sparsity**: The proportion of zero elements in the matrix

```julia
A = [1 2 0; 3 4 5; 0 6 7]
bandedness, sparsity = getstructure(A)
# Returns (100.0, 0.222...)
```

### `sparsestructure`

Analyzes a sparse matrix and returns the most appropriate specialized matrix type based on its structure. The function checks for the following properties in order:

1. **Toeplitz** - Constant diagonals (returns `Toeplitz`)
2. **Symmetric** - `A == A'` (returns `Symmetric`)
3. **Hermitian** - `A == A'` for complex matrices (returns `Hermitian`)
4. **Lower Triangular** - All elements above diagonal are zero (returns `LowerTriangular`)
5. **Upper Triangular** - All elements below diagonal are zero (returns `UpperTriangular`)
6. **Banded** - Non-zeros confined to a band around diagonal (returns `BandedMatrix`)
7. **Generic Sparse** - Falls back to `SparseMatrixCSC`

```julia
using SparseArrays

# Symmetric matrix
A = sparse([1 2 2; 2 3 4; 2 4 5])
result = sparsestructure(A, 0.5)  # Returns Symmetric

# Toeplitz matrix
T = sparse([1 2 0; 0 1 2; 0 0 1])
result = sparsestructure(T, 0.5)  # Returns Toeplitz

# Lower triangular matrix
L = sparse([1 0 0; 2 3 0; 4 5 6])
result = sparsestructure(L, 0.5)  # Returns LowerTriangular
```

The `threshold` parameter controls the bandwidth threshold for banded matrix detection, expressed as a fraction of the matrix size.

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and target Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete manifest of all direct and transitive dependencies can be found in the manifest file of the documentation.</summary>
```

```@raw html
<a href="./assets/Manifest.toml">Manifest file</a>
```

```@raw html
</details>
```
