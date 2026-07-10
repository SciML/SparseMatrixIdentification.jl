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

if isdefined(SciMLTesting, :run_api_docs)
    _run_qa()
else
    if get(ENV, "SPARSEMATRIXIDENTIFICATION_QA_SUBPROCESS", "") == "1"
        error("The QA environment must resolve SciMLTesting with run_api_docs.")
    end

    script = "using Pkg; Pkg.instantiate(); include($(repr(@__FILE__)))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) -e $script`
    withenv("SPARSEMATRIXIDENTIFICATION_QA_SUBPROCESS" => "1") do
        @test success(cmd)
    end
end
