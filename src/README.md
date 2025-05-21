# Scripts and Notebooks

**IMPORTANT: Download [raw data](../data/README.md) and [model files](../model/README.md) before running the scripts!**

## Step 0: Prepare Data

- Run `00_convert.py`, `01_tokenizer.py` and `02_merge.py` sequentially.

## Step 1: Run Models and Collect Router/Model Output

- Run `10_calculate.py` for each model (e.g., `python 10_calculate.py qwen3`).
- Then, run `11_merge.py`, `12_rank.py` and `13_refine.py` sequentially.

## Step 2: Calculate Vocabulary Spcialization

- Run `20_vocab_calc.py`, `21_vocab_calcgen.py` and `22_vocab_refine.py` sequentially.

## Step 3: Calculate Segment Routing Best Performance (SRP)

- Run `30_srp_calc.py`, `31_srp_refine.py` and `32_srp_merge.py` sequentially.
- Run `33_srp_plot.ipynb` to plot global SRP results.
- Run `34_srp_domain.ipynb` to plot domain-wise SRP results.
- Run `35_srp_spec.ipynb` to compare SRP with expert specialization.

## Step 4: Calculate Segment Cache Best Hit Rate (SCH)

- Run `40_sch_calc.py`, `41_sch_refine.py` and `42_sch_merge.py` sequentially.
- Run `43_srp_plot.ipynb` to plot SCH results.

## Additional Scripts and Notebooks

### Case Study

- Run `A0_case_calculate.py` for each model (e.g., `python A0_case_calculate.py qwen3`).
- Then, run `A1_case_merge.py` and `A2_case_refine.py` sequentially.
- Run `A3_case_plot.ipynb` to plot results.

### Base Model vs. Post-trained Model

- Run `B0_post.ipynb` to compare base models with their post-trained (e.g., SFT) variants.

### Position-wise SRP

- Run `C0_srp_pos_calc.py`, `C1_srp_pos_refine.py` and `C2_srp_pos_merge.py` sequentially.
- Run `C3_srp_pos_plot.ipynb` to plot position-wise SRP results.
