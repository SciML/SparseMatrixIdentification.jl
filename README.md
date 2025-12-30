# SparseMatrixIdentification.jl

[![Build Status](https://github.com/SciML/SparseMatrixIdentification.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SciML/SparseMatrixIdentification.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SparseMatrixIdentification.jl is a Julia package for automatically identifying and classifying the structure of sparse matrices. It detects matrix properties such as symmetry, bandedness, triangularity, Toeplitz structure, and special matrix types (Hilbert, Vandermonde, Cauchy, etc.), returning an optimized matrix type for efficient computation.

## Installation

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

### `getstructure(A)`

Computes structure metrics for a matrix and returns a tuple `(percentage_banded, percentage_sparsity)`:

- **percentage_banded**: The percentage of positions within bandwidth=1 that contain non-zero elements
- **percentage_sparsity**: The proportion of zero elements in the matrix (1 - density)

```julia
julia> A = [1 2 0; 3 4 5; 0 6 7]
julia> getstructure(A)
(100.0, 0.2222222222222222)
```

### `sparsestructure(A, threshold)`

Analyzes a sparse matrix and returns the most appropriate specialized matrix type based on its structure. The `threshold` parameter controls the bandwidth threshold for banded matrix detection (as a fraction of matrix size).

#### Supported Matrix Types

The function checks for the following structures in order of priority:

| Structure | Return Type | Description |
|-----------|-------------|-------------|
| Hilbert | `Hilbert` | A[i,j] = 1/(i+j-1) |
| Strang | `Strang` | Tridiagonal Toeplitz with [2, -1, -1] pattern |
| Vandermonde | `Vandermonde` | V[i,j] = x[i]^(j-1) |
| Cauchy | `Cauchy` | A[i,j] = 1/(x[i] + y[j]) |
| Toeplitz | `Toeplitz` | Constant diagonals |
| Symmetric | `Symmetric` | A == A' |
| Hermitian | `Hermitian` | Complex conjugate symmetric |
| Lower Triangular | `LowerTriangular` | All elements above diagonal are zero |
| Upper Triangular | `UpperTriangular` | All elements below diagonal are zero |
| Block Banded | `BlockBandedMatrix` | Block-tridiagonal structure |
| Almost Banded | `AlmostBandedMatrix` | Banded + low-rank fill |
| Banded | `BandedMatrix` | Non-zeros confined to a band |
| Generic | `SparseMatrixCSC` | Fallback for unstructured matrices |

## Examples

```julia
using SparseMatrixIdentification
using SparseArrays
using SpecialMatrices

# Symmetric matrix detection
A = sparse([1 2 2; 2 3 4; 2 4 5])
sparsestructure(A, 0.5)  # Returns Symmetric{...}

# Toeplitz matrix detection
T = sparse([1 2 0; 0 1 2; 0 0 1])
sparsestructure(T, 0.5)  # Returns Toeplitz{...}

# Lower triangular matrix detection
L = sparse([1 0 0; 2 3 0; 4 5 6])
sparsestructure(L, 0.5)  # Returns LowerTriangular{...}

# Banded matrix detection
B = sparse([1 2 0; 3 4 5; 0 6 7])
sparsestructure(B, 2/3)  # Returns BandedMatrix{...}

# Hilbert matrix detection
H = sparse(Matrix(Hilbert(4)))
sparsestructure(H, 0.5)  # Returns Hilbert(4)

# Vandermonde matrix detection
x = [1.0, 2.0, 3.0, 4.0]
V = sparse(Matrix(Vandermonde(x)))
sparsestructure(V, 0.5)  # Returns Vandermonde{...}
```

## Dependencies

SparseMatrixIdentification.jl integrates with several specialized matrix packages in the Julia ecosystem:

- [BandedMatrices.jl](https://github.com/JuliaMatrices/BandedMatrices.jl)
- [BlockBandedMatrices.jl](https://github.com/JuliaMatrices/BlockBandedMatrices.jl)
- [FastAlmostBandedMatrices.jl](https://github.com/SciML/FastAlmostBandedMatrices.jl)
- [ToeplitzMatrices.jl](https://github.com/JuliaMatrices/ToeplitzMatrices.jl)
- [SpecialMatrices.jl](https://github.com/JuliaMatrices/SpecialMatrices.jl)
- [SemiseparableMatrices.jl](https://github.com/JuliaMatrices/SemiseparableMatrices.jl)

## Documentation

For more information, see the [documentation](https://docs.sciml.ai/SparseMatrixIdentification/stable/).

## Contributing

- Please refer to the [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac) for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
