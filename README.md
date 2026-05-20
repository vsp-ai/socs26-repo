# Experiments with FNNs

## Experiments requirements:
*  our version of the /fnn implementation that (you can find on this repository)
*  /benchmarks (you can find on this repository)
*  plaja installantion in specific from here (LINK AVAILABLE SOON)

## Disclaimer

Here we adopt the term "FNN" when we refer to the model evaluated in our paper.
However the specific architecure we use is from Wang, 2021, you can find his
originla codebase in [here](https://github.com/12wang3/rrl). Our codebase turns to be very different from his,
so using his codebase directly would require additional implementation effort, but possible.

## Description

We prepare scripts on run-experiments to help with the pipeline we use in our paper.
The steps you must follow are:

1) **generate training** data with /run-experiments/generate-data/run.sh
2) **train FNN** with /run-experiments/FNN/training/run.sh
3) **evaluate FNN** with /run-experiments/FNN/evaluation/run.sh
4) **verify FNN** with /run-experiments/FNN/verification/run.sh
5) **(optional) train NN teacher** with /run-experiments/NN/training/run.sh
6) **evaluate NN teacher** with /run-experiments/NN/evaluate/run.sh
7) **verify NN teacher** with /run-experiments/NN/verification/run.sh

## Considerations

 - training NNs is optional because we already provide trained NNs in /benchmarks
 - each subdirectory in run-experiments has a file *params.json* where you can try different hyperparameters and configurations. We already provide the same setup as in our paper but you still need to point to the plaja and fnn paths.
 - The folder /benchmark-settings include two files were we point to the correct benchmarks files, one
that uses a subset of the benchmarks from our paper, another with the complete set. **IF YOU ARE RUNNING OUR EXPERIMENTS LOCALLY IS HIGHLY ADVISED TO USE THE SMALL SET** since these experiments takes a considerable amount of time to run. For a complete testing, we suggest to run them in a cluster, which require to adapt our run-experiment scripts for its best use.
 - as an example we also provide a few FNN trained models you can check at /fnn-examples

