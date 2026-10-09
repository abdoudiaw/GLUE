# EIRENE training dump

Project author and maintainer: Abdou Diaw

## Purpose

Create paired training events at the native B2.5--EIRENE seam without changing the coupled solution.

Each event contains:

- BRAEIR: the plasma background passed from B2.5 to EIRENE;
- EIRBRA: raw `SNI`, `SMO`, `SEE`, `SEI`, reaction-channel decompositions, strata and correction metadata returned by EIRENE;
- WNEUTRALS: B2-grid neutral state, neutral fluxes, radiation, wall, pumping, sputtering and resolved-surface tallies consumed by B2.5;
- CESTIM: triangular-grid particle, energy and velocity estimators read by B2.5's repeated-call averaging path;
- call indices, stratum labels, `tflux`, `flux_scale`, EIRENE call controls and code versions.

The return state is saved immediately after `eirene_eirsrt`, before B2.5 mapping, scaling and linearization. WNEUTRALS quantities already mapped to B2 units inside EIRENE retain those interface units.

## Training boundary

The surrogate replaces only the `eirene_eirsrt` result. It receives the saved
BRAEIR fields and call controls and returns the dynamic EIRBRA, WNEUTRALS and
applicable CESTIM state. The existing B2.5 code then performs all normal
post-return mapping, scaling, corrections, accumulation, averaging and
semi-implicit linearization.

Initially retain every dynamic return field consumed by B2.5 as a candidate
target. Exact pass-through copies and fixed configuration/mapping arrays do
not need to be learned; the adapter can restore them deterministically.
Downstream B2.5 coefficients and `balance.nc` quantities are validation
observables, not outputs of the raw-interface model.

## Coupling path

The coupled executable exchanges most data through shared Fortran modules, not through `fort.*` files. EIRENE's `eirmod_infcop` calls `eirene_wneutrals_fill` for each stratum and `eirene_wneutrals_save` for totals and surface properties. These routines map triangular-grid estimators and surface tallies into `eirmod_wneutrals`; B2.5 reads those arrays after `eirene_eirsrt` returns. B2.5 also reads `eirmod_eirbra` and averages selected `eirmod_cestim` arrays when repeated EIRENE calls are enabled.

## Code change

- `b2mod_eirene.F` reads `eirene_training_dump` and calls the writer after EIRENE.
- `b2mod_eirene_training_dump.F90` performs rank-0 NetCDF output only.
- `b2input.xml` registers the new switch with default `0`.

No source value is modified and no surrogate is called.

Enable in `b2mn.dat`:

```text
eirene_training_dump=1
```

Files are named:

```text
eirene_training_v2_b2call_########_single_call_####.nc
eirene_training_v2_b2call_########_average_used_by_b2_0000.nc
```

The second form is written only when `eirene_averaging_multiple_calls.gt.1`.

## Schema-v2 inventory

The current writer declares 175 array variables plus 23 unique global
metadata attributes:

| Prefix | Maximum count | Meaning |
|---|---:|---|
| `braeir_*` | 39 | B2 plasma-background input fields |
| `b2_*` | 3 | Per-stratum `tflux`, `flux_scale`, and `crcstra` controls |
| `eirbra_*` | 39 | Raw sources, reaction channels, and stratum/source metadata |
| `wneutrals_*` | 79 | Neutral-grid, radiation, wall, pumping, sputtering, inventory, and surface-return state |
| `cestim_*` | 15 | Triangle-grid particle, energy, and velocity-moment estimators |

Some species- or surface-dependent variables are conditional on the
corresponding EIRENE allocation. Therefore the authoritative inventory for a
specific configuration is its NetCDF header:

```bash
ncdump -h eirene_training_v2_b2call_00000000_single_call_0001.nc
```

Before defining the model schema, classify every returned variable as
`learned_dynamic`, `fixed_configuration`, `exact_pass_through`,
`derived_reduction`, or `validation_only`. Until equivalence is demonstrated,
retain every dynamic variable that B2.5 reads as a candidate learned target.

## Validation

```bash
python validate_eirene_training_dump.py RUN_DIRECTORY
```

The validator checks the schema, shapes, finite values and the averaging identity for the EIRBRA and WNEUTRALS arrays averaged by B2.5.

Single-case acceptance requires:

1. a clean coupled build;
2. identical B2.5 results from matched logger-off and logger-on restarts;
3. the expected number of event files and no non-finite values;
4. exact recovery of the averaged EIRBRA and WNEUTRALS arrays when repeated calls are used.

Only after these checks pass should the 761 short restarts be launched.

## Contract boundary

Schema v2 covers the BRAEIR input and the call-dependent EIRBRA, WNEUTRALS and CESTIM state consumed by B2.5. Static EIRENE configuration and persistent `fort.13`/`fort.15` restart state are separate from the per-call return contract.

The active model repository is `https://github.com/abdoudiaw/solstice` (pull
request 1 at the 2026-10-01 audit). `ORNL-Fusion/solstice` is reference-only
for this project.

## Short explanation for review

This is an observation-only diagnostic at the existing coupling seam. It records the BRAEIR input and the EIRBRA/WNEUTRALS/CESTIM return state consumed by B2.5. The default run path is unchanged because the switch is off. The data will first be validated against one coupled run, then used to recover training events from short restarts of the existing ensemble.
