# Mutant Information

- Baseline system: A (A-like topology, one arrestin).
- Compared mutants: F670G, I669G, R668G.
- RUN aggregation: matching RUN ids across A and each mutant (RUN1-RUN5).
- Frame cutoff: first 100 frames removed per RUN.
- Helicity residues: 665,666,667,668,669,670.

## Quick Summary

| Mutant | Runs | A bound mGlu | Mutant bound mGlu | A community edges | Mutant community edges |
|---|---|---|---|---:|---:|
| F670G | RUN1,RUN2,RUN3,RUN4,RUN5 | R | R | 159515 | 155925 |
| I669G | RUN1,RUN2,RUN3,RUN4,RUN5 | R | R | 159515 | 132929 |
| R668G | RUN1,RUN2,RUN3,RUN4,RUN5 | R | R | 159515 | 131963 |

## Files

- `summary_mut_vs_A_combined.json`: full global run summary.
- `mutant_metadata.json`: compact machine-readable mutant metadata.
- Per-mutant details are in sibling folders (`F670G`, `I669G`, `R668G`).
