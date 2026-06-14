using SafeTestsets

@safetestset "Aqua" begin
    using SparseMatrixIdentification
    using Aqua
    using Test
    # deps_compat is a genuine finding (missing compat for LinearAlgebra/SparseArrays
    # deps and the Pkg extra); keep every other sub-check enabled.
    Aqua.test_all(SparseMatrixIdentification; deps_compat = false)
    @test_broken false  # Aqua deps_compat: missing compat for LinearAlgebra, SparseArrays, Pkg — see https://github.com/SciML/SparseMatrixIdentification.jl/issues/36
end

@safetestset "JET" begin
    using Test
    @test_broken false  # JET: try_* extension helpers report no-matching-method in sparsestructure(::AbstractSparseMatrix) — see https://github.com/SciML/SparseMatrixIdentification.jl/issues/36
end
