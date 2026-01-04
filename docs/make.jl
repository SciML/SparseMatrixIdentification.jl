using Documenter
using SparseMatrixIdentification

cp(
    joinpath(@__DIR__, "Manifest.toml"), joinpath(@__DIR__, "src/assets/Manifest.toml");
    force = true
)

makedocs(
    sitename = "SparseMatrixIdentification.jl",
    authors = "Anastasia Dunca",
    modules = [SparseMatrixIdentification],
    clean = true,
    doctest = false,
    linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SparseMatrixIdentification/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/SciML/SparseMatrixIdentification.jl.git";
    push_preview = true
)
