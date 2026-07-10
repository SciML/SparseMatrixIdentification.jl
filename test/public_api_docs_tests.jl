using SparseMatrixIdentification
using Test

const SMI = SparseMatrixIdentification
const SRC_FILE = normpath(joinpath(@__DIR__, "..", "src", "SparseMatrixIdentification.jl"))
const DOCS_SRC_DIR = normpath(joinpath(@__DIR__, "..", "docs", "src"))
const EXTENSION_POINT_NAMES = (
    :try_special_matrices,
    :try_toeplitz,
    :try_banded,
    :try_blockbanded,
    :try_almostbanded,
)

function _append_symbol_names!(names, declaration)
    cleaned = first(split(declaration, '#'))
    for token in split(cleaned, [',', ' ', '\t']; keepempty = false)
        name = replace(strip(token), r"\(.*$" => "")
        if occursin(r"^[A-Za-z_]\w*[!?]?$", name)
            push!(names, Symbol(name))
        end
    end
    return names
end

function _declared_public_names()
    public_names = Set(Symbol.(names(SMI; all = false, imported = false)))
    delete!(public_names, nameof(SMI))

    source_lines = split(read(SRC_FILE, String), '\n')
    line_index = 1
    while line_index <= length(source_lines)
        line = strip(source_lines[line_index])
        if startswith(line, "export ") || startswith(line, "public ")
            declaration = replace(line, r"^(export|public)\s+" => "")
            while endswith(strip(first(split(declaration, '#'))), ",") &&
                    line_index < length(source_lines)
                line_index += 1
                declaration *= " " * strip(source_lines[line_index])
            end
            _append_symbol_names!(public_names, declaration)
        elseif occursin(r"^([A-Za-z_]\w*\.)?@public\s+", line)
            _append_symbol_names!(public_names, replace(line, r"^([A-Za-z_]\w*\.)?@public\s+" => ""))
        end
        line_index += 1
    end

    return public_names
end

function _has_docstring(name)
    doc = Docs.doc(Docs.Binding(SMI, name))
    rendered = sprint(show, MIME("text/plain"), doc)
    return !startswith(rendered, "No documentation found.")
end

function _append_docs_entry!(names, line)
    isempty(line) && return names
    startswith(line, "#") && return names

    entry = first(split(line))
    entry = replace(entry, r"\(.*$" => "")
    entry = last(split(entry, '.'))
    if occursin(r"^[A-Za-z_]\w*[!?]?$", entry)
        push!(names, Symbol(entry))
    end
    return names
end

function _documented_reference_names()
    documented_names = Set{Symbol}()
    has_module_autodocs = false

    for (root, _, files) in walkdir(DOCS_SRC_DIR)
        for file in files
            endswith(file, ".md") || continue

            in_docs_block = false
            in_autodocs_block = false
            for line in eachline(joinpath(root, file))
                stripped = strip(line)
                if startswith(stripped, "```@docs")
                    in_docs_block = true
                    in_autodocs_block = false
                elseif startswith(stripped, "```@autodocs")
                    in_docs_block = false
                    in_autodocs_block = true
                elseif startswith(stripped, "```")
                    in_docs_block = false
                    in_autodocs_block = false
                elseif in_docs_block
                    _append_docs_entry!(documented_names, stripped)
                elseif in_autodocs_block && occursin("SparseMatrixIdentification", stripped)
                    has_module_autodocs = true
                end
            end
        end
    end

    if has_module_autodocs
        union!(documented_names, _declared_public_names())
    end

    return documented_names
end

@testset "Public API documentation coverage" begin
    public_names = sort!(collect(_declared_public_names()); by = string)
    documented_names = _documented_reference_names()

    @test [name for name in public_names if !_has_docstring(name)] == Symbol[]
    @test setdiff(Set(public_names), documented_names) == Set{Symbol}()
end

@testset "Documented extension point coverage" begin
    documented_names = _documented_reference_names()

    @test [name for name in EXTENSION_POINT_NAMES if !_has_docstring(name)] == Symbol[]
    @test setdiff(Set(EXTENSION_POINT_NAMES), documented_names) == Set{Symbol}()
end
