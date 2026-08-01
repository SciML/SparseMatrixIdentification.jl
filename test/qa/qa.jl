using SciMLTesting, SparseMatrixIdentification
using JET

# ExplicitImports only sees an extension module once its trigger packages are loaded
# (`Base.get_extension` returns `nothing` otherwise), so load every weakdep here to put
# the extensions under QA. The `SemiseparableMatrices` weakdep has no `[extensions]`
# entry, so there is no extension module for it to scan.
using BandedMatrices, BlockBandedMatrices, FastAlmostBandedMatrices
using SpecialMatrices, ToeplitzMatrices

run_qa(
    SparseMatrixIdentification;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            # Owner: SparseMatrixIdentification itself. Every one of these is the
            # package's own internals reached from its own extension modules, which is
            # the only way an extension can hook into its parent: the `try_*` stubs are
            # the methods each extension adds, the `_*_loaded` `Ref`s are the flags the
            # extension's `__init__` flips, and the `is_*`/`detect_*` predicates are the
            # structure detectors the extension dispatches on. None of them are declared
            # `public`, and there is no public spelling to use instead.
            ignore = (
                :_bandedmatrices_loaded, :_blockbandedmatrices_loaded,
                :_fastalmostbandedmatrices_loaded, :_specialmatrices_loaded,
                :_toeplitzmatrices_loaded,
                :try_almostbanded, :try_banded, :try_blockbanded,
                :try_special_matrices, :try_toeplitz,
                :detect_block_size, :is_almost_banded, :is_banded, :is_cauchy,
                :is_hilbert, :is_strang, :is_toeplitz, :is_vandermonde,
            ),
        ),
    ),
)
