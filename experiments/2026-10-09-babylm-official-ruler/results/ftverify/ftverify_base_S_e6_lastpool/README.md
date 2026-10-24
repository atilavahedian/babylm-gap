# ftverify base_S_e6 lastpool (executor 3; VM bl-ftverify-base-s-e6-lastpool-1011-101001, gcp-project-a, commit ref-0171)
- Identity: MISMATCH, and only on multirc (predictions.json + results.txt). The other 6 tasks are bit-identical between concurrent (FT_CONC=3) and sequential.
- Wall: concurrent 10:18:09 to 11:17:02 = 3533 s; sequential 11:17:02 to 12:22:44 = 3942 s. Concurrency saves only about 10%.
- Sequential GLUE (eval/), all 7 off majority (mcc > 0): boolq acc .678 mcc .281; mnli .582; mrpc acc .706 mcc .213; multirc acc .669 mcc .308; qqp acc .767 mcc .512; rte acc .626 mcc .249; wsc acc .635 mcc .183.
- The concurrent multirc result was overwritten by the sequential pass. Only its sha survives (ft_sha_*_concurrent.txt).
- Untrained half (score_hf_base_S_e6_untrained_lastpool): cut at VM maxdur 13:10Z; 6/7 tasks synced (multirc missing). See that dir.
