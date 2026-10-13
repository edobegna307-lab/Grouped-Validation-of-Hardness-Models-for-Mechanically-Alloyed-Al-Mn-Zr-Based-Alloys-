# Supplementary Materials

Data and code for: **"Grouped Validation of Hardness Models for Mechanically Alloyed Al–Mn–Zr-Based Alloys after Hot-Press Consolidation"** (Eshetu, M.A.; Novikov, A.S.) — retitled and retargeted to *Science of Sintering* in the fifth revision round (see below); previously titled "Limits of Alloy-Family Transferability..." and targeted at Computational Materials Science / Materials & Design.

## Contents

| File | Description |
|---|---|
| `Supplementary_Code_and_Data_Package.zip` | A single self-contained archive of the full reproducibility package: `Supplementary_Code_S1_analysis.py`, `Supplementary_Data_S1_clean_dataset.csv`, a dedicated `README.md` (run instructions, a section-by-section map to the manuscript, the full data dictionary, and seed/reproducibility notes), `requirements.txt` (exact pinned, verified-working package versions), `run_output_reference.log` (a full console transcript from a verified run, for diffing your own output against), **`Final_Reproducibility_Record.md` and `Data_Correction_Log.md` (new in the thirteenth round)**, and the eight `fig_S1`…`fig_S8` figures the script regenerates. Unzip and follow its own README; this is the easiest way to get everything needed to reproduce the analysis in one download. |
| `Supplementary_Data_S1_clean_dataset.csv` | The 42-specimen experimental dataset (3 alloys × 14 hot-press/anneal conditions) used for all statistical modelling in the manuscript. **Updated in the twelfth round**: `grain_size` and `density` rounded to uniform decimal precision (see twelfth-round notes below); all other columns and all 42 rows unchanged. Also included inside the zip above. |
| `Supplementary_Code_S1_analysis.py` | Complete, self-contained Python script that reproduces every statistic, table and figure underlying the manuscript from the raw dataset above, including leave-one-out cross-validation, grouped (condition- and alloy-family-out) cross-validation with per-family observed-vs-predicted breakdown, a grouped (LOAO) no-skill baseline, case-resampling intervals, paired dataset-specific comparisons (all six pairwise tests), baseline model comparisons with documented hyperparameters, residual diagnostics with an influence-sensitivity refit, SHAP/permutation attribution for the primary 8-feature M3, a supplementary 9-feature M3 sensitivity check that adds back the lattice parameter, the two 7-feature M3 descriptor-redundancy variants, the pooled grain-size-vs-hardness correlation, repeated 5-fold CV with mean/SD/median/IQR, the Hall–Petch-like heterogeneity test, and a surrogate-ranking robustness check (top-5 candidates and a 10-seed refit). **Re-verified in the twelfth round** against the corrected dataset by two independent end-to-end runs (identical output both times). Also included inside the zip above, alongside its own detailed README. |
| `Manuscript_Science_of_Sintering_Al-Mn-Zr-RE_Hardness_CORRECTED.docx` | **The current main paper** (updated in the twelfth round for the dataset correction above — see that section for exactly which numbers changed; still exactly 10 pages). Supersedes `Manuscript_Science_of_Sintering_Al-Mn-Zr-RE_Hardness.docx` below, which is kept only for revision history. Condensed to a ~125-word abstract to comply with *Science of Sintering*'s own length limits. Contains Title/Authors/Abstract/Keywords, a condensed Introduction/Methods/Results/Discussion/Conclusion, 4 core tables (Table I–IV; no figures are embedded in the main text — all figures were relocated to the Supplementary Information as Figs. S1–S8 to meet the 10-page limit), full back matter (CRediT, Funding, Ethics, Data availability, Declaration of competing interest, Acknowledgments), and the 19-entry reference list. Incorporates two rounds of detailed expert reviewer comments (precision edits to statistical/methodological wording throughout Methods, Results and Discussion) verified individually against the text before being applied. |
| `Manuscript_Science_of_Sintering_Al-Mn-Zr-RE_Hardness.docx` | **Superseded** — the ninth/tenth-round version, kept for revision history only. Use the `_CORRECTED` file above for anything current. |
| `Supplementary_Information_Al-Mn-Zr-RE_Hardness.docx` | A separate, not-page-limited document holding everything moved out of the main paper to make it fit 10 pages: the full process matrix and parameter ranges, the complete 42-row specimen-level dataset (Table S1), the baseline model comparison, grouped no-skill baseline, LOAO per-family breakdown, all six pairwise model comparisons, the paired post-treatment hardness ratio by family (**now Table S16, new in the thirteenth round**), the surrogate-ranking robustness checks, the exploratory UTS analysis and the literature comparison, eight figures (S1–S8), and the extended validation/leakage-control and itemised limitations detail. Tables S1–S21 are now fully populated and sequential (S16 filled; S19 remains legitimately narrative-only), and the two previously duplicate trailing section headings are now S22/S23. **Updated in the twelfth round**: the dataset table, grain-size/density summary tables, the M3 descriptor-redundancy table (S17), the M1 coefficient table (S18), the LOCO/LOAO table (S20), the LOOCV table (S21), the per-specimen SHAP table, and Figures S2/S3/S6/S7 (plus the S4–S5 composite) were all updated and re-embedded for the corrected dataset; verified still renders cleanly with no distortion or overflow. |
| `Chem_Proc_Proceeding_Paper_Al-Mn-Zr-RE_Hardness.docx` | A condensed conference proceedings manuscript, consolidating several of the same outputs into fewer combined panels for space; has not been updated to match the latest revision of the full manuscript. |
| `Graphical_Abstract_Al-Mn-Zr-RE_Hardness.png` / `.tiff` | **New in the tenth revision round.** A four-panel graphical abstract (13 × 5 in at 300 dpi, 3900 × 1500 px) walking through the experimental campaign, the three models compared, the LOOCV/LOCO/LOAO validation design with the headline R² values, and the key finding. Matches the current manuscript title, model definitions (M3 = 8 predictors) and wording exactly. |
| `Cover_Letter_Science_of_Sintering.docx` | **New in the tenth revision round.** A submission cover letter to the Editor-in-Chief of *Science of Sintering*: fit/novelty statement, reproducibility statement, originality and no-dual-submission declaration, and a placeholder block for the three reviewer suggestions the journal's guidelines require (names/affiliations/emails must be supplied by the authors — these cannot be fabricated). |
| `Declaration_of_Interest.docx` | **New.** Standalone signed-statement version of the competing-interest declaration (same content as the manuscript's back matter, plus the AI-use disclosure), for submission systems that require it as a separate file. |
| `Author_Statement.docx` | **New.** Standalone originality/no-dual-submission/authorship-order statement with signature lines for both authors, plus the CRediT contribution statement. |
| `Ethics_Statement.docx` | **New.** Standalone "not applicable" ethics/consent statement (no human or animal subjects) and the funding statement, for submission systems that require these as a separate upload. |
| `Highlights.docx` | **New.** Five standalone highlight bullets (each ≤ 85 characters, the common submission-system limit), rewritten from the Highlights section cut from the main text in the ninth round to save page space. |
## Requirements

- Python 3.10+
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`

Install with:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn shap
```

## Reproducing the results

Both files must be in the same working directory (the script reads the CSV by its exact filename above).

```bash
python Supplementary_Code_S1_analysis.py
```

This prints every reported statistic to the console and saves 8 PNG figures to the working directory. Runtime is a few minutes on a standard laptop CPU; no GPU is required. A fixed random seed (42) is used throughout except where the script deliberately varies it (the 20-seed repeated 5-fold CV, Section 3.4, and the 10-seed surrogate-ranking robustness check, Section 3.11). Last verified (fourth-round revision, corrected grain-size/lattice-parameter/density data): exit code 0, all 8 figures regenerated, all printed values — including the updated M3 (R² = 0.717), the 9-feature lattice-parameter sensitivity check (R² = 0.692), the grouped (LOAO) no-skill baseline (R² = −0.885, unchanged), and the updated family-specific Hall–Petch-like regressions and repeated-CV statistics — checked against the manuscript and matching exactly, in a clean sandbox with only the corrected CSV and the analysis script present.

## Dataset column reference

| Column | Description |
|---|---|
| `milling_time` | Ball-milling time (h); constant at 20 h for all specimens |
| `hotpress_temp` | Hot-press consolidation temperature (°C): 350, 400, or 450 |
| `anneal_temp` | Post-hot-press anneal temperature (°C): 350, 400, or 450 |
| `anneal_time` | Post-hot-press anneal time (h): 2 or 4 |
| `alloy` | Alloy identity: `Al-Mn-Zr`, `Al-Mn-Zr-Ce`, or `Al-Mn-ZR-Y` |
| `microhardness` | As-processed Vickers micro-hardness (HV) — primary modelling target |
| `hardness450` | Hardness after a common 450 °C thermal-stability anneal (HV) |
| `UTS` | Ultimate tensile strength (MPa), from tensile testing; see manuscript Section 2.2 for the tensile-vs-compressive terminology note (this is not equivalent to the compressive/yield strength values reported in some closely related literature) and the caveat that exact test temperature and strain rate are not recorded in this column |
| `grain_size` | XRD-derived (Williamson–Hall) crystallite size (nm) — see manuscript Section 2.2 on why this is not equated with metallographic grain size |
| `aT` | Crystallographic lattice parameter (nm) |
| `density` | Bulk density of the sintered compact (g/cm³), Archimedes method |

## Known limitations of this dataset (see manuscript Section 4.4, Limitations, for full discussion)

- Each alloy was milled as a single powder batch, then divided into 14 independently hot-pressed and annealed specimens under different conditions. So specimens are independent at the pressing/annealing level, but **batch identity is perfectly confounded with alloy identity** (all 14 specimens of one alloy share one parent powder batch) and the two cannot be statistically separated with this design. Furnace-run identifiers are not resolved to individual specimens either, so a possible furnace-run effect is also entangled with, and cannot be separated from, the alloy-family/batch effect. Consequently, leave-one-alloy-family-out (LOAO) validation in the manuscript is described as a **combined alloy-family/batch test with an unresolved furnace-run component**, not a test of composition transferability alone.
- The 14 realised (hot-press T, anneal T, anneal time) combinations are a *selected, non-randomised process matrix* — not a complete factorial design and not a randomised, replicated design of experiments in the formal statistical sense — drawn from the full 3×3×2 = 18-combination process space; the four combinations not produced (350/350/4, 350/450/4, 400/350/4, 400/450/4 °C/°C/h) are listed in manuscript Table 1.
- Individual hardness-indent values, the number of indents excluded per specimen under the ±10% screening rule, per-specimen standard deviations, the tensile test temperature/strain rate, yield strength (σ₀.₂), and plastic deformation (ψ%) are not included in this dataset, though some were measured in the originating characterisation programme.
- Hot pressing and the post-consolidation anneal used two distinct thermal environments: hot pressing under a high-purity argon atmosphere with argon cooling to room temperature, followed by annealing of all 42 specimens in the same Nabertherm N30/65A muffle furnace under ambient air with forced-air convection, air-cooled to room temperature. The hot-press heating rate was **not directly measured** in this campaign; ≈30 °C·min⁻¹ is cited in the manuscript only as a contextual literature value from a closely related process, not a confirmed parameter of this study. The post-anneal heating rate, the number of separate furnace runs, the loading arrangement within each run, and the exact chronological processing sequence were not recorded and could not be reconstructed from laboratory records. Reported compositions are nominal charge compositions; bulk consolidated composition was not independently verified by chemical analysis.

These are stated explicitly so that anyone reusing this dataset is aware of what it does and does not support.

## Citation

If you use this dataset or code, please cite the manuscript above. A citable archive (Zenodo) with a version-locked DOI will be linked here upon manuscript submission/acceptance.

## License

Specify a license here (e.g., MIT for code, CC-BY-4.0 for data) before making the repository public, if not already set at the repository level.
