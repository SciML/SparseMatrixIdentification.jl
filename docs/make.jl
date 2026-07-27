using Documenter
using SparseMatrixIdentification

makedocs(
    sitename = "SparseMatrixIdentification.jl",
    authors = "Anastasia Dunca",
    modules = [SparseMatrixIdentification],
    checkdocs = :exports,
    clean = true,
    doctest = true,
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
