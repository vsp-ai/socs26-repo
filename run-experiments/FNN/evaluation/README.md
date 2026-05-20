# FNN Evaluation

Evaluate trained FNN policies with PlaJA.

```bash
cd run-experiments/FNN/evaluation
./run_subset.sh
./run_socs26.sh
```

The scripts read the selected params file, load `<output_root>/training/results.json`, and evaluate every trained model listed there. Outputs are written under `<output_root>/evaluation/`.
