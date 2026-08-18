# OAT on NUS Vanda

This directory adapts OAT's Slurm examples to Vanda's PBS Pro scheduler.

## Naming and billing

`bimanual_tokenizer` is the workspace and experiment name. It is not a Vanda
billing project. The supplied starter PBS files explicitly use the `auto_free`
queue because `personal-xuanweiliu` is not provisioned in Allocation Manager.
Once `hpc project` lists a real project, remove `#PBS -q auto_free` and add a
`#PBS -P ...` line containing that exact project identifier.

## Server setup

Run these commands on the Vanda login node after the GitHub fork exists:

```bash
mkdir -p /scratch/$USER/bimanual_tokenizer
cd /scratch/$USER/bimanual_tokenizer
git clone --recurse-submodules https://github.com/Evan-XLiu/oat.git
cd oat
git remote add upstream https://github.com/Chaoqi-LIU/oat.git
git remote -v
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Do not install Python dependencies on the login node. Request a short GPU
session and build the environment on that compute node:

```bash
qsub -I \
  -q auto_free \
  -l select=1:ngpus=1:ncpus=32:mem=240gb:gpu-type=nvidia \
  -l walltime=02:00:00
cd /scratch/$USER/bimanual_tokenizer/oat
bash vanda/setup_env.sh
exit
```

The setup is deliberately stored under `/scratch`; OAT and its CUDA/PyTorch
dependencies can exceed the 40 GB home quota. `/scratch` is not backed up and
files older than 60 days can be purged, so keep source code pushed to GitHub and
archive important checkpoints separately.

## Validate before training

```bash
cd /scratch/$USER/bimanual_tokenizer/oat
qsub vanda/preflight.pbs
qstat -awn1
```

Inspect the resulting `oat_preflight.o<job-id>` file. It must report an NVIDIA
A40, CUDA availability, and a successful OAT import.

## Train the tokenizer

The upstream LIBERO configuration expects this dataset:

```text
data/libero/libero10_N500.zarr
```

Once it exists, submit:

```bash
qsub vanda/train_oattok.pbs
qstat -awn1
```

Training runs in offline Weights & Biases mode by default, avoiding an API-key
prompt in unattended jobs. Set up online W&B separately only when required.
The free tier is best-effort: queue parameters can change, and inefficient or
over-provisioned jobs may be terminated. Use a real project allocation for
long-running development as soon as NUS provisions it.

## Important upstream status

OAT is archived and no longer maintained. Its README recommends `praxis-vla`
for new development and documents a known LIBERO reset-ordering bug. These PBS
files preserve the requested OAT workflow but do not resolve that upstream bug.
