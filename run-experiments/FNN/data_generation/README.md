# FNN Data Generation

Generate PlaJA traces and convert them into FNN `.data` and `.info` files.

```bash
cd run-experiments/FNN/data_generation
./run_subset.sh
./run_socs26.sh
```

The scripts assume the PlaJA codebase is external and available through `plaja_root` in the selected params file. PlaJA is executed inside the dependency Docker image, and the expected binary is `<plaja_root>/build/PlaJA`.

From the repository root, compile PlaJA first if the binary is missing:

```bash
run-experiments/PLAJA-COMPILE/compile_plaja.sh /path/to/plaja
```

Outputs are written under `<output_root>/data_generation/`, and generated datasets are written into each benchmark's `training_dataset` directory.
