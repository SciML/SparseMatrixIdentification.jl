using SciMLTesting, SparseMatrixIdentification, Test
using JET

function _run_qa()
    return run_qa(
        SparseMatrixIdentification;
        explicit_imports = true,
        ei_kwargs = (;
            # `fast_scalar_indexing` is not public (unexported, not declared public) in
            # ArrayInterface; it is the only non-public name this package accesses.
            all_qualified_accesses_are_public = (; ignore = (:fast_scalar_indexing,)),
        ),
        api_docs_kwargs = (; rendered = true),
    )
end

function _qa_julia_cmd()
    julia_args = String[]
    skip_arg = false
    for arg in Base.julia_cmd().exec
        if skip_arg
            skip_arg = false
        elseif arg == "--project"
            skip_arg = true
        elseif !startswith(arg, "--project=")
            push!(julia_args, arg)
        end
    end
    return Cmd(julia_args)
end

if isdefined(SciMLTesting, :run_api_docs)
    _run_qa()
else
    if get(ENV, "SPARSEMATRIXIDENTIFICATION_QA_SUBPROCESS", "") == "1"
        error("The QA environment must resolve SciMLTesting with run_api_docs.")
    end

    script = "using Pkg; Pkg.instantiate(); include($(repr(@__FILE__)))"
    cmd = `$(_qa_julia_cmd()) --startup-file=no --project=$(@__DIR__) -e $script`
    withenv("SPARSEMATRIXIDENTIFICATION_QA_SUBPROCESS" => "1") do
        run(cmd)
        @test true
    end
end
