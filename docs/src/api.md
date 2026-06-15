# API Reference

## Exported Functions

```@docs
getstructure
sparsestructure
```

## Extension Point Functions

These functions are extension points that package extensions add methods to.
They return `nothing` unless the corresponding optional package is loaded.

```@docs
SparseMatrixIdentification.try_special_matrices
SparseMatrixIdentification.try_toeplitz
SparseMatrixIdentification.try_banded
SparseMatrixIdentification.try_blockbanded
SparseMatrixIdentification.try_almostbanded
```
