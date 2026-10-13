Supplementary Materials

Data and code for: "Grouped Validation of Hardness Models for Mechanically Alloyed Al–Mn–Zr-Based Alloys after Hot-Press Consolidation" Mintesenot Admasu Eshetu, Alexander S. Novikov — submitted to Science of Sintering.

The study compares a Hall–Petch-like linear model (M1), a process/formulation model (M2, OLS and Random Forest) and a microstructure-enriched Random Forest (M3) for as-processed micro-hardness of 42 mechanically alloyed, hot-press-consolidated Al–Mn–Zr, Al–Mn–Zr–Ce and Al–Mn–Zr–Y specimens, under three validation schemes: leave-one-out (LOOCV), leave-one-condition-out (LOCO) and leave-one-alloy-family-out (LOAO).

Contents
File	Description
Manuscript_Science_of_Sintering_Al-Mn-Zr-RE_Hardness_CORRECTED.docx	Main manuscript (10 pages).
Supplementary_Information_Al-Mn-Zr-RE_Hardness.docx	Supplementary Information: Tables S1–S21 (specimen-level dataset, summary statistics, model hyperparameters, cross-validation results, SHAP values, baselines, sensitivity checks) and Figures S1–S8.
Supplementary_Data_S1_clean_dataset.csv	Supplementary Data S1: the audited 42-specimen dataset (one row per specimen).
Supplementary_Code_S1_analysis.py	Supplementary Code S1: one self-contained script that regenerates every statistic, table value and figure in the manuscript and Supplementary Information.
Supplementary_Code_and_Data_Package.zip	The complete reproducibility package: the code and data above, its own README.md (run instructions, section-to-manuscript map, data dictionary, seeds), requirements.txt (pinned versions), run_output_reference.log (reference console output), Final_Reproducibility_Record.md, Data_Correction_Log.md, and the eight regenerated figures fig_S1…fig_S8.
High_Resolution_Figures.zip	High-resolution figure files.
Graphical_Abstract_Al-Mn-Zr-RE_Hardness.png / .tiff / .pdf	Graphical abstract.
Cover_Letter_Science_of_Sintering.docx, Highlights.docx, Declaration_of_Interest.docx, Author_Statement.docx, Ethics_Statement.docx	Standalone submission documents.
Requirements

Python 3.11 (verified with 3.11.15) and the pinned packages in the package's requirements.txt: pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, seaborn 0.13.2, matplotlib 3.10.9, shap 0.51.0.

bash
pip install -r requirements.txt
Reproducing the results

Place the script and the CSV in the same directory (the script reads the CSV by its exact filename), then run:

bash
python3 Supplementary_Code_S1_analysis.py

The script prints every reported statistic to the console, organised into twelve labelled sections, and saves eight PNG figures. Runtime is roughly 8–10 minutes on one CPU core; no GPU is needed. All random seeds are fixed (42), except where the script varies them deliberately (the 20-seed repeated 5-fold CV and the 10-seed surrogate-ranking robustness check). Compare your console output with run_output_reference.log; a fresh run with the pinned environment reproduces it exactly. See the package README.md for the map from each script section to the manuscript and Supplementary Information.

Dataset column reference
Column	Description
alloy	Alloy family: Al-Mn-Zr, Al-Mn-Zr-Ce, or Al-Mn-ZR-Y (the capitalisation of the third string is as recorded; the code matches on it exactly)
milling_time	Ball-milling time (h); 20 h for all specimens
hotpress_temp	Hot-press consolidation temperature (°C): 350, 400 or 450
anneal_temp	Post-consolidation anneal temperature (°C): 350, 400 or 450
anneal_time	Post-consolidation anneal time (h): 2 or 4
microhardness	As-processed Vickers micro-hardness, HV0.05 (HV); the primary modelling target
hardness450	Hardness after a common 450 °C thermal-stability anneal (HV); used only for the post-treatment ratio
grain_size	XRD-derived (Williamson–Hall) coherent-domain size (nm); not equated with metallographic grain size
aT	Al-matrix lattice parameter; used only in a supplementary sensitivity check
density	Bulk density, Archimedes method (g·cm⁻³)
UTS	Ultimate tensile strength (MPa); exploratory only, never a modelling target
Known limitations of the dataset

These are stated so that anyone reusing the data knows what it does and does not support (see also the manuscript's Discussion).

Each alloy was milled as a single parent powder batch and divided into 14 specimens. Specimens are independent at the pressing/annealing level, but alloy-family identity is confounded with parent-batch identity (and partially with furnace-run identity, which was not resolved per specimen). LOAO validation is therefore a combined alloy-family/batch test, not a test of composition transferability alone.
The 14 realised (hot-press T, anneal T, anneal time) combinations are a selected, non-randomised subset of the 18 possible; the four combinations not produced are listed in Supplementary Table S4.
Individual indent values, per-specimen standard deviations, tensile test temperature and strain rate, yield strength and ductility are not part of this dataset.
Hot pressing was done under high-purity argon; all 42 specimens were annealed in the same muffle furnace under ambient air. The heating rates, number of separate furnace runs, loading arrangement and exact processing chronology were not recorded. Compositions are nominal charge compositions; bulk composition was not independently verified by chemical analysis.
Citation

If you use this dataset or code, please cite the manuscript above. A version-locked archive DOI will be added upon deposition.

License

Specify a license (e.g. MIT for code, CC-BY-4.0 for data) before making the repository public, if not already set at the repository level.
