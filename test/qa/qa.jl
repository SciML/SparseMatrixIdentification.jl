using SparseMatrixIdentification
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(SparseMatrixIdentification)
end

@testset "JET" begin
    JET.test_package(SparseMatrixIdentification; target_defined_modules = true)
end
