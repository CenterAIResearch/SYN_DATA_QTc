# Section 1: Find the best synthetic data generator based on the metrics

- In the colab, in the `Sec_1_Find_Best_SYN_Data_Gen_Method`, `SDV_Synthesize_a_table_(GC_CTGAN_TVAE)_QTc_replacement_sampling.ipynb`
  generate synthetic data of each method and save the folders in the `Section_1/Syn_Data_by_methods` folder.

- In the colab, in the `Sec_1_Find_Best_SYN_Data_Gen_Method`, `Synthcity_Synthesize_a_table_(BN_RTVAE_DDPM)_QTc_replacement_sampling.ipynb` generate synthetic data of each method and save the folders in the `Section_1/Syn_Data_by_methods` folder.

- run the `Section_1/Generate_Metrics.py`
- run the `Section_1/draw_boxplots.py`
- run the `Section_1/draw_table.py`

# Section 2: Conduct machine learning experiments

- Comparison of Models Trained on Original Data vs. Synthetic Data
- Comparison of Models Trained on Original Data vs. Combined Original and Synthetic Data
