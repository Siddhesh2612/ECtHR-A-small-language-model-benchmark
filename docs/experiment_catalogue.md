# Historical experiment catalogue

These publication copies preserve the distinct notebook code versions recovered from the MSc archive. They are research records, not the supported training entry point. No historical notebook has been rerun during curation.

Identical code-cell hashes are grouped even when original outputs differ. Outputs, attachments, execution counts and runtime metadata are removed. Credential literals read `HF_TOKEN` from the environment; remaining credential mentions are redacted. Personal absolute paths become `data/historical/<original filename>`. Local relative Parquet paths remain historical dependencies and are not supplied. Original source files remain unchanged in the private audit.

The manifest records source-relative paths, original file and code hashes, publication hashes and duplicate sources. Original saved outputs are intentionally private; reported metrics are preserved separately in `results/historical/reported_results.csv`.

## Important interpretation limits

- Most historical classification experiments use an explicit eleventh no-violation output. New benchmark-compatible experiments use ten article outputs with no violation derived during evaluation; scores should not be silently pooled.
- Historic article display names are incorrect; numeric IDs were preserved in the audited local preprocessing round-trip. Prompt-based experiments can be affected by descriptions.
- GPT-Neo 125M and RoBERTa training notebooks were not located. A file named GPT2.ipynb loads DialogRPT. T5 checkpoint IDs indicate T5-base rather than FLAN-T5.
- Phi-2 and T5 use test data during checkpoint selection. Their historic scores are not independent test estimates.
- Historical analysis code is preserved, including figure defects; unavailable baseline metrics must not be interpreted as zero.
- Metric rows are source transcriptions. Missing values stay blank; unusual token/budget strings remain as written. The GPT-Neo 2,048-token report-only row is unreconciled and must not be silently corrected.
- Token removal does not revoke credentials. Any still-active credentials from the original archive need rotation through their issuing service.

## Distinct code versions

| Publication notebook | Checkpoint identity / purpose | Source |
| --- | --- | --- |
| [gpt_neo_token_1024_epochs_3_lr_3e5_4d42dc56](../notebooks/historical/gpt_neo/gpt_neo_token_1024_epochs_3_lr_3e5_4d42dc56.ipynb) | EleutherAI/gpt-neo-1.3B | `GPT-Neo/GPT-Neo_token_1024_epochs_3_lr_3e5.ipynb` |
| [gpt_neo_token_1024_steps_1000_lr_3e5_4c8acff4](../notebooks/historical/gpt_neo/gpt_neo_token_1024_steps_1000_lr_3e5_4c8acff4.ipynb) | EleutherAI/gpt-neo-1.3B | `GPT-Neo/GPT-Neo_token_1024_steps_1000_lr_3e5.ipynb` |
| [gptneo_results_0c65e682](../notebooks/historical/analysis/gptneo_results_0c65e682.ipynb) | Analysis / preprocessing | `GPT-Neo/GPTNeo_results.ipynb` |
| [gpt2_c4df95a0](../notebooks/historical/dialogrpt/gpt2_c4df95a0.ipynb) | microsoft/DialogRPT-updown (loaded by GPT2.ipynb) | `GPT2/GPT2.ipynb` |
| [gpt_2_rpt_down_token_1024_epochs_3_lr_3e5_1_99486eaa](../notebooks/historical/dialogrpt/gpt_2_rpt_down_token_1024_epochs_3_lr_3e5_1_99486eaa.ipynb) | microsoft/DialogRPT-updown | `GPT2_microsoft:DialogRPT-updown/GPT-2_Rpt_down_token_1024_epochs_3_lr_3e5 (1).ipynb` |
| [phi2_lr_1e5_epochs_3_token_1024_maxgrad_norm_1_0_1_2b3f9f01](../notebooks/historical/phi2/phi2_lr_1e5_epochs_3_token_1024_maxgrad_norm_1_0_1_2b3f9f01.ipynb) | microsoft/phi-2 | `Phi2/Phi2_lr_1e5_epochs_3_token_1024_maxgrad_norm_1.0 (1).ipynb` |
| [phi2_lr_1e5_epochs_3_token_1024_maxgrad_norm_1_0_ca540d4c](../notebooks/historical/phi2/phi2_lr_1e5_epochs_3_token_1024_maxgrad_norm_1_0_ca540d4c.ipynb) | microsoft/phi-2 | `Phi2/Phi2_lr_1e5_epochs_3_token_1024_maxgrad_norm_1.0.ipynb` |
| [phi2_lr_1e5_epochs_3_token_512_maxgrad_norm_1_0_097eeb4f](../notebooks/historical/phi2/phi2_lr_1e5_epochs_3_token_512_maxgrad_norm_1_0_097eeb4f.ipynb) | microsoft/phi-2 | `Phi2/Phi2_lr_1e5_epochs_3_token_512_maxgrad_norm_1.0.ipynb` |
| [phi_2forclassification_1_e5baa06c](../notebooks/historical/phi2/phi_2forclassification_1_e5baa06c.ipynb) | microsoft/phi-2 | `Phi2/phi-2forclassification (1).ipynb` |
| [t5_lr_1e5_epochs_5_token_512_d15de0e5](../notebooks/historical/t5/t5_lr_1e5_epochs_5_token_512_d15de0e5.ipynb) | google-t5/t5-base | `T5/T5_lr_1e5_epochs_5_token_512.ipynb` |
| [t5_lr_1e6_epochs_5_token_512_51019520](../notebooks/historical/t5/t5_lr_1e6_epochs_5_token_512_51019520.ipynb) | google-t5/t5-base | `T5/T5_lr_1e6_epochs_5_token_512.ipynb` |
| [gemma2withoutquant_1_d65fdc7a](../notebooks/historical/gemma/gemma2withoutquant_1_d65fdc7a.ipynb) | google/gemma-2-2b | `dissertation_code/Gemma_2_2b/Gemma2withoutQuant (1).ipynb` |
| [gemma2withoutquant_b5f04aca](../notebooks/historical/gemma/gemma2withoutquant_b5f04aca.ipynb) | google/gemma-2-2b | `dissertation_code/Gemma_2_2b/Gemma2withoutQuant.ipynb` |
| [newfewshotsgemma2_1_9973a3b7](../notebooks/historical/gemma/newfewshotsgemma2_1_9973a3b7.ipynb) | google/gemma-2-2b | `dissertation_code/Gemma_2_2b/NewFewShotsGemma2 (1).ipynb` |
| [newfewshotsgemma2_dc3da5ba](../notebooks/historical/gemma/newfewshotsgemma2_dc3da5ba.ipynb) | google/gemma-2b-it | `dissertation_code/Gemma_2_2b/NewFewShotsGemma2.ipynb` |
| [gemma2_2b_fewshot_2ca59c44](../notebooks/historical/gemma/gemma2_2b_fewshot_2ca59c44.ipynb) | google/gemma-2-2b | `dissertation_code/Gemma_2_2b/gemma2-2b-fewshot.ipynb` |
| [preprocessing_ecthr_a_0c20500b](../notebooks/historical/preprocessing/preprocessing_ecthr_a_0c20500b.ipynb) | Analysis / preprocessing | `dissertation_code/preprocessing ecthr_a.ipynb` |
| [phi1_lr_1e_6_epochs_3_maxlength_1024_loramodel_r_16_loraalpha_16_a6abd5f9](../notebooks/historical/phi1/phi1_lr_1e_6_epochs_3_maxlength_1024_loramodel_r_16_loraalpha_16_a6abd5f9.ipynb) | microsoft/phi-1 | `phi1/Phi1_lr_1e-6_epochs_3_maxlength_1024_loramodel_r_16_loraalpha_16.ipynb` |
| [phi1_lr_1e_6_steps_1000_maxlength_1024_loramodel_r_16_loraalpha_16_ca7a16bb](../notebooks/historical/phi1/phi1_lr_1e_6_steps_1000_maxlength_1024_loramodel_r_16_loraalpha_16_ca7a16bb.ipynb) | microsoft/phi-1 | `phi1/Phi1_lr_1e-6_steps_1000_maxlength_1024_loramodel_r_16_loraalpha_16.ipynb` |
| [phi1_lr_2e_5_steps_1400_maxlength_1024_loramodel_r_16_loraalpha_16_0c34f8ca](../notebooks/historical/phi1/phi1_lr_2e_5_steps_1400_maxlength_1024_loramodel_r_16_loraalpha_16_0c34f8ca.ipynb) | microsoft/phi-1 | `phi1/Phi1_lr_2e-5_steps_1400_maxlength_1024_loramodel_r_16_loraalpha_16.ipynb` |
| [phi1_lr_3e_5_steps_1000_maxlength_1024_loramodel_r_16_loraalpha_16_082ab971](../notebooks/historical/phi1/phi1_lr_3e_5_steps_1000_maxlength_1024_loramodel_r_16_loraalpha_16_082ab971.ipynb) | microsoft/phi-1 | `phi1/Phi1_lr_3e-5_steps_1000_maxlength_1024_loramodel_r_16_loraalpha_16.ipynb` |
