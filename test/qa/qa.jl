using SciMLTesting, SparseMatrixIdentification, Test
using JET

run_qa(
    SparseMatrixIdentification;
    explicit_imports = true,
    ei_kwargs = (;
        # `fast_scalar_indexing` is not public (unexported, not declared public) in
        # ArrayInterface; it is the only non-public name this package accesses.
        all_qualified_accesses_are_public = (; ignore = (:fast_scalar_indexing,)),
    ),
)
