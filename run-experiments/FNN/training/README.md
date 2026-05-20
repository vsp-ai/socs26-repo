# FNN Training

Train FNN models from the generated datasets.

```bash
cd run-experiments/FNN/training
./run_subset.sh
./run_socs26.sh
```

The scripts read the selected params file, generate the training configurations, and train each configuration sequentially. Outputs are written under `<output_root>/training/`, and trained models are written under `<model_root>/`.

Run data generation before training so each benchmark has the expected `.data` and `.info` files.
