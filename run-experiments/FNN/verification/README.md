# FNN Verification

Verify trained FNN policies with PlaJA.

```bash
cd run-experiments/FNN/verification
./run_subset.sh
./run_socs26.sh
```

The scripts read the selected params file, load `<output_root>/evaluation/results.json`, and verify every evaluated model listed there. Outputs are written under `<output_root>/verification/`.
