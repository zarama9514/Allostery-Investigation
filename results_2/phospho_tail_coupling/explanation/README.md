This folder contains an interpretation of the phospho-tail coupling results relative to the mGlu3-betaarr1 paper.

Paper anchor
- The article reports that pS857, pS859, and pT860 on the mGlu3 C-tail engage the betaarr1 N-domain.
- In the dynamic data, the strongest support for the paper is the combination of low SASA, high tail-contact occupancy, and high salt-bridge occupancy for these same three sites.

What to read first
- `phospho_key_site_summary.png`: one-page heatmap view of SASA, best contact occupancy, and best salt-bridge occupancy.
- `A_minus_mutants_signed_delta_sasa.png`: signed `A - mutant` SASA shifts for pS857, pS859, pT860, and TOTAL.
- `agreement_summary.png`: compact agree / partially agree / disagree table by system.
- `agreement_summary.csv`: the same classification in a machine-readable table.
- `key_site_metrics.csv`: exact values used in this explanation.

Interpretation

A vs B
- `A:T_to_A` and `B:T_to_A` are both strongly consistent with the paper for pS857/pS859/pT860. In `A`, best contact occupancies are 91.6/97.6/98.7% and best salt occupancies are 84.8/64.3/77.6%. In `B:T_to_A`, the same values are 99.4/95.6/99.4% and 97.8/65.8/76.2%.
- `B:T_to_A` is slightly more buried than `A` for pS857 and pS859 by SASA (54.8 -> 44.0 A^2 and 41.0 -> 38.2 A^2), which fits a maintained phospho-tail engagement.
- `B:L_to_C` does not behave like an equally strong second copy. Its SASA is much higher for pS857/pS859/pT860 (65.0/73.2/85.7 A^2), and its best contact occupancies collapse to 54.0/7.5/66.0%. This does not cleanly support a fully symmetric, equally locked 2:2 arrestin engagement throughout the trajectory.

A vs F670G
- `F670G` is only a mild perturbation overall: TOTAL mean SASA changes from 308.0 to 306.7 A^2.
- The mutation weakens the article-like pattern mainly at pS859 and pT860. pS859 SASA rises from 41.0 to 48.5 A^2, pT860 rises from 57.8 to 64.6 A^2, and best contact occupancies drop from 97.6/98.7% to 64.8/85.2%.
- pS857 stays close to `A` in SASA and still keeps a strong salt bridge (71.0%), so `F670G` looks partially consistent with the paper but less stable than `A` in the 859/860 region.

A vs I669G
- `I669G` is the least consistent with the article-like engaged state. TOTAL mean SASA increases from 308.0 to 336.0 A^2.
- The clearest signal is pS857: SASA jumps from 54.8 to 84.7 A^2, while the best pS857 contact lifetime drops from 244.0 to 84.6 ps and the best pS857 salt lifetime drops from 225.9 to 69.4 ps.
- pS859 and pT860 also remain more exposed or less stably wired than in `A`, so `I669G` is the strongest dynamic argument for weakened phospho-tail engagement with arrestin N-domain.

A vs R668G
- `R668G` shows the strongest burial for pS857 and pS859: SASA falls from 54.8/41.0 to 28.3/26.3 A^2, and best contact occupancies reach 94.2/99.4% with salt occupancies 87.6/67.2%.
- That is more consistent with a locked article-like pS857/pS859 engagement than `A` itself. The caveat is pT860: its SASA is slightly higher than in `A` (57.8 -> 68.1 A^2), even though its contact occupancy remains maximal (100.0%).
- So `R668G` supports the paper strongly for pS857/pS859 and remains broadly supportive for pT860, but the 860 site is not buried more deeply than in `A`.

Bottom line
- Best agreement with the paper: `A:T_to_A`, `B:T_to_A`, and `R668G:T_to_A`.
- Partial agreement with mild weakening: `F670G:T_to_A`.
- Clear weakening relative to the paper-like engaged state: `I669G:T_to_A`.
- Internal tension with a perfectly symmetric 2:2 interpretation: `B:L_to_C`, because the second tail is much more exposed and much less persistently engaged than `B:T_to_A`.
