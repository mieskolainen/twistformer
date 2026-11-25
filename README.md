# twistformer [R&D]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a href="misc/logo.png"><img src="misc/logo.png" alt="logo" width="480"></a>


## Algebraic-geometric models

- `twistgraph`: algorithmically generated thousands of graph features, geometry and physics motivated, including cellular sheafs **[new R&D, MM 2026]**
- `twistformer`: Twistor algebra based transformer **[new R&D, MM 2026]**
- `transformer`: Standard transformer, with optional geometric distance scores ([Transformer paper](https://arxiv.org/abs/1706.03762))
- `triformer`: 3-point attention transformer ([3-point attention paper](https://arxiv.org/abs/1909.00668))
- `gatr`: Clifford-Geometric algebra based transformer (Euclidean, Projective, Conformal, Lorentz) ([GATr paper](https://arxiv.org/abs/2305.18415))
- `pelican`: Lorentz algebra decomposition model ([PELICAN paper](https://arxiv.org/abs/2211.00454))
- Generative model backends: flow matching and continuous or discrete normalizing flows


## Install

From the repository root:

```sh
conda env create -f environment.yml
conda activate twistformer
pip install -r requirements.txt
```

At CERN to avoid collision with system libraries, you might need to install pip packages with

```sh
env -u PIP_CONSTRAINT PIP_CONFIG_FILE=/dev/null python -m pip install -r requirements.txt
```

## Train and evaluate

As a simple test, download the [top-tagging dataset on Zenodo](https://zenodo.org/records/2603256) and extract under `zenodo_2603256/`. Then run sheaf-gram boosted decision trees

```sh
python -m twistformer.cli.train \
  --config configs/toptag/twistgraph_sheaf_gram.yml --output runs/toptag --device cuda

python -m twistformer.cli.evaluate \
  --checkpoint runs/toptag/checkpoints/best.pt --split test --device cuda
```

This runs **under 10 minutes** with a GPU machine, and one should obtain test results:

```
0.98533 +- 0.00015 AUC and 1756 +- 202 QCD rejection at 30% top efficiency
```

Which is comparable to Transformer or GNN based taggers on this dataset within the sample size (cf. scaling laws to be studied).

> All modelcards are fast examples, systematic work and tuning may improve results.


## Twistboard

To monitor the training and evaluation results, run locally:

```sh
python -m twistboard serve --root runs --open
```

For remote training, run the server without `--open` and forward its port locally:

```sh
ssh -N -L 8765:127.0.0.1:8765 user@training-host
```

Alternatively, use a cloud sync like `cernbox` and just run `twistboard` locally.


## ONNX export

```sh
python -m pip install -e '.[onnx]'
python -m twistformer.cli.export --checkpoint runs/toptag/checkpoints/best.pt \
  --output runs/toptag/model.onnx --max-tokens 128
```

## Multi-GPU

Launch one process per GPU. Batch size and worker count are per GPU.

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m twistformer.cli.train --config configs/toptag/twistformer.yml \
  --strategy ddp --device cuda --output runs/toptag-ddp
```

Resume with the same GPU count:

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m twistformer.cli.train --resume runs/toptag-ddp/checkpoints/last.pt --device cuda
```


## References

If you find this work interesting:

```
@misc{twistformer,
  author  = "{Mikael Mieskolainen}",
  title   = "twistformer",
  url     = "https://github.com/mieskolainen/twistformer",
  version = {0.0.2},
  date    = {2026-11},
}
```

(©) 2026 Mikael Mieskolainen (mikael.mieskolainen@cern.ch)

Licensed under the [MIT License](LICENSE).
