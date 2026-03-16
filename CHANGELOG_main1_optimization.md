# `main_1.py` Optimization Log

Date: 2026-03-16

Scope:
- Keep the core algorithm unchanged.
- Improve runtime and memory use by restructuring matrix computations.

What changed:
- Replaced column-by-column B-spline basis construction with a single batched `BSpline` evaluation.
- Replaced block-wise VCM design assembly loops with broadcasted tensor construction.
- Vectorized group soft-thresholding and block norm computation inside FISTA.
- Moved fold-invariant work in cross-validation out of the lambda loop.
- Added warm starts along the lambda path in CV.
- Avoided repeated spectral norm estimation of the same Gram matrix.
- Reworked the main solver path to use implicit block gradient computation from `B` and `X`, instead of repeatedly materializing the full design matrix `Phi`.
- Removed a few redundant post-fit prediction/design computations.

Theory status:
- Objective function unchanged.
- Penalty unchanged.
- FISTA update rule unchanged.
- CV selection logic unchanged.
- O/C/N partition logic unchanged.

Validation performed:
- B-spline design matrix matched the original implementation exactly in a direct numerical check.
- Group soft-thresholding matched the original implementation exactly in a direct numerical check.
- Explicit-`Phi` solver and implicit-block solver matched to numerical precision:
  - coefficient max absolute difference about `3.6e-9`
  - prediction max absolute difference about `3.5e-10`
- `python -m py_compile main_1.py` passed after the changes.

Observed runtime impact in local tests:
- Earlier optimized version:
  - `P=80`, `n_per_segment=300`
  - stage 1 about `41s`
  - stage 2 about `41s`
- Current version:
  - same setting
  - stage 1 about `8.6s`
  - stage 2 about `11.5s`

Notes:
- `main_1.py` uses `knot_step`, not `n_inner_per_unit`.
- For server runs, the config was set to the safer nearby choice `n_inner = 15`, i.e.:
  - `knot_step = 1 / 16 = 0.0625`
  - this is exactly representable in binary floating point and avoids the user's concern about `1 / 17`
