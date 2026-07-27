# OAT: Ordered Action Tokenization

<img src="assets/rss-2026-paper-finalist.svg" alt="The Best/Outstanding Paper Finalist at RSS 2026" width="620">

> ### ⚠️ This repository is archived
>
> This repository preserves the implementation of the original RSS 2026 OAT paper for reproducibility, but it is no longer actively maintained. For the full paper, current OAT implementations, and all future development, please use **[praxis-vla](https://github.com/Chaoqi-LIU/praxis-vla)**. For comprehensive and maintained robot-policy evaluation, use **[praxis-eval](https://github.com/Chaoqi-LIU/praxis-eval)**.
>
> **New projects should not build on this repository.**

**Full paper:** [[Paper]](https://arxiv.org/abs/2607.21670) | [[Code]](https://github.com/Chaoqi-LIU/praxis-vla) | [[Webpage]](https://ordered-action-tokenization.github.io/)

**RSS 2026 paper:** [[Paper]](https://arxiv.org/abs/2602.04215) | [[Code]](https://github.com/Chaoqi-LIU/oat)

**Original RSS 2026 authors:**

[Chaoqi Liu](https://chaoqi-liu.com)<sup>1</sup>,
[Xiaoshen Han](https://xshenhan.github.io/)<sup>1</sup>,
[Jiawei Gao](https://gao-jiawei.com/)<sup>1</sup>,
[Yue Zhao](https://zhaoyue-zephyrus.github.io/)<sup>2</sup>,
[Haonan Chen](https://haonan16.github.io/)<sup>1</sup>,
[Yilun Du](https://yilundu.github.io/)<sup>1</sup>

<sup>1</sup> Harvard University<br>
<sup>2</sup> Stanford University

## Known issue: LIBERO reset ordering

> **⚠️ Warning:** The archived [`LiberoEnv.reset()` implementation](https://github.com/Chaoqi-LIU/oat/blob/4bb0c01611c5b1ec8a6268cb376aca2cb519c881/oat/env/libero/env.py#L143) captures the observation before calling `_let_objects_fall()`. As a result, the initial image and robot state returned to the policy are taken before the scene has settled.
>
> The correct order is: reset the environment, let the objects fall, and only then acquire a fresh observation/image for the policy. Please use [praxis-eval](https://github.com/Chaoqi-LIU/praxis-eval) for the maintained and more comprehensive evaluation stack.

## Legacy reproduction

The instructions below are preserved only for reproducing the original RSS implementation. New work should use [praxis-vla](https://github.com/Chaoqi-LIU/praxis-vla).

### Quick start

1. Clone with submodules so that `third_party/LIBERO` is available:

   ```bash
   git clone --recurse-submodules git@github.com:Chaoqi-LIU/oat.git
   # or, after a plain clone:
   git submodule update --init --recursive
   ```

#### Option 1: uv

2. Install `uv` if you do not already have it. Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. Initialize the project and install all dependencies and local editable sources:

   ```bash
   uv sync
   uv pip install -e .
   ```

#### Option 2: (micro)conda/mamba

2. Install `micromamba` if you do not already have it. Follow [micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

3. Initialize the project and install all dependencies and local editable sources:
   ```bash
   micromamba env create -f conda_env.yaml
   ```

> **Note:** We encountered issues running `uv` on our Slurm cluster, so we also provide `conda`/`mamba` as an alternative. The example commands below use `uv`.

### Preparing LIBERO datasets

We provide a prebuilt `libero10` dataset on Hugging Face: [chaoqi-liu/libero10_N500.zarr](https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr/resolve/main/libero10_N500.zarr.zip?download=true). Alternatively, follow the instructions below to build the dataset locally.

1. Download the LIBERO releases (e.g., `libero_spatial`, `libero_object`, `libero_goal`, `libero_100`) into `data/libero/hdf5_datasets/`. (`libero10` is contained in `libero100`)

    ```bash
    uv run third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_[spatial/object/goal/100]
    ```

2. Convert each HDF5 dump into the repo's zarr format:

   ```bash
   uv run scripts/convert_libero_dataset.py --root_dir data/libero --hdf5_dir_name hdf5_datasets
   ```

   The script converts every `*.hdf5` file it finds, saves `task_N{episodes}.zarr` under `data/libero/`, and prompts before overwriting existing exports. Use `-n/--num_sample_demo` to limit how many demos per task if needed.

3. Compose a `libero10` multitask zarr:

   ```bash
   uv run scripts/compose_libero_multitask_dataset.py --multitask_name libero10 --root_dir data/libero
   ```

   This merges `*.zarr` datasets related to `libero10` using `scripts/merge_data.py`, shuffles the episodes, and writes `data/libero/libero10_N{total}.zarr`.

### Train OAT Tokenizer

After you have `data/libero/libero10_N{n}.zarr` ready, train the action tokenizer that OAT policies consume:

```bash
HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10
```

### Train OAT Policy

Once the tokenizer checkpoint exists, train the policy that predicts action tokens and decodes them back into actions:

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=[path/to/oattok.ckpt]
```
Setting `lazy_eval=false` evaluates the policy during training every `training.rollout_every` epochs.

### Evaluate OAT Policy on LIBERO

Evaluate a trained checkpoint using `scripts/eval_policy_sim.py`:

```bash
uv run scripts/eval_policy_sim.py \
  --checkpoint [path/to/oatpolicy.ckpt] \
  --output_dir output/eval/libero10 \
  --num_exp 5  # run 5 times, so we can get stderr
```

The script instantiates the same LIBERO runner and dataset from `oat.config.task.policy.libero.libero10` and dumps per-checkpoint statistics plus optional videos to `output/eval/libero10`.

## Citation

Please cite the full paper for OAT in visuomotor policy learning and the RSS paper for the original action tokenizer. If you use the maintained praxis-vla research platform, please also cite Praxis.

```bibtex
@misc{liu2026oatpolicy,
      title={Ordered Action Tokens for Visuomotor Policy Learning},
      author={Chaoqi Liu and Yue Zhao and Haonan Chen and Xiaoshen Han and Jiawei Gao and Ehsan Adeli and Yilun Du},
      year={2026},
      eprint={2607.21670},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.21670},
}

@misc{liu2026oat,
      title={OAT: Ordered Action Tokenization},
      author={Chaoqi Liu and Xiaoshen Han and Jiawei Gao and Yue Zhao and Haonan Chen and Yilun Du},
      year={2026},
      eprint={2602.04215},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.04215},
}

@misc{liu2026praxis,
      title={Praxis: A Controlled Laboratory for Vision-Language-Action Policy Research},
      author={Liu, Chaoqi},
      year={2026},
      url={https://chaoqi-liu.com/praxis/},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
