# NN Evaluation

Evaluate the original teacher NN policies with PlaJA.

```bash
cd run-experiments/NN/evaluation
./run_subset.sh
./run_socs26.sh
```

The scripts read the selected params file, generate one evaluation configuration per unique benchmark NNET policy, and run PlaJA inside Docker. Outputs are written under `<output_root>/evaluation/`.
