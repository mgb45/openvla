# Adapting OpenVLA for Air-Gapped GB200 Training

**Status:** proposed plan, not yet executed
**Owner:** Michael Burke
**Target hardware:** 4–8 Blackwell GPUs on a GB200 NVL72 partition (Grace/aarch64 hosts, NVLink domain)
**Scope of this document:** what to change in this repo, in what order, with what exit criteria, to go from the stock OpenVLA codebase to a reproducible air-gapped training + representation-analysis pipeline.

---

## 1. Research framing

The compute is a means to an end. The end is:

1. **Now:** representation manifolds of a VLA — what geometry the hidden states occupy, and how a rollout traces a path through that geometry.
2. **Next:** comparing backbones (what kind of visual/linguistic pretraining produces what kind of manifold) and heads (how the action decoder reads that manifold).
3. **Later:** extracting modular, reusable policies and components out of a monolithic VLA.

Two consequences follow, and they drive almost every decision below:

- **The scientific payload is not the final checkpoint.** It is a dense, comparable record of *how representations evolve* — across training steps, across model variants, across embodiments. Regenerating that record means getting back onto an air-gapped machine, which is expensive. So the training loop must emit analysis artifacts *from the very first real run*, not as a retrofit.
- **Comparability is the scarce resource.** Manifold comparisons across checkpoints and across variants are only meaningful if the inputs are byte-identical. A frozen probe set, fixed seeds, and a fixed projection matrix must be established in Phase 0 and never changed. Changing them silently invalidates every cross-run comparison made before the change.

A third consequence, which is the main strategic recommendation of this plan:

> **Do not spend the first compute window reproducing OpenVLA-7B.** A faithful reproduction is roughly 21.5k A100-hours (64 A100s × 14 days). Eight B200s deliver perhaps 35–40 A100-equivalents of realised dense bf16 throughput, so a full reproduction is ~3 weeks of wall clock for a result that already exists and that you can ingest as a checkpoint. Use the window instead for **many small runs with heavy instrumentation**. A 3B-class VLA trained on a Bridge+Fractal mixture for 2–3 days gives you a variant. Three weeks gives you a *sweep*, which is what representation-manifold and modularity work actually needs.

Keep exactly one 7B run as a reference point, ideally initialised from the released `openvla-7b` weights rather than trained from the base VLM.

---

## 2. Constraints, and what each one forces

| Constraint | Forced consequence |
|---|---|
| GPUs are Blackwell (sm_100) | `torch==2.2.0` in `pyproject.toml` cannot run at all. Needs torch ≥ 2.7 on CUDA ≥ 12.8. This cascades into `transformers`, `timm`, `tokenizers`. **This is the single largest porting risk.** |
| Hosts are Grace (aarch64) | Every wheel must exist for `linux/aarch64`. TF, `tensorflow-graphics`, `flash-attn`, `mujoco` are the ones that bite. |
| Image built by GitHub Actions, run air-gapped | **Everything network-dependent happens at build time.** The image is the only thing that crosses with an internet-connected build behind it. Anything the code fetches lazily at runtime (HF hub, TFDS checksums, W&B) is a runtime failure. |
| No internet at runtime | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=offline`, `TFDS_DATA_DIR` local. Model weights and datasets are ingested separately. |
| Ingest is slow and manual | **Over-provision ingest.** Stage every base VLM, dataset, and asset you might plausibly want in a single ingest. Discovering in week 3 that you need one more 4 GB checkpoint costs days. |
| Egress is slow and manual | Artifacts must be small, self-describing, and log-spaced. A naive `save_interval=2500` on a 7B model produces ~30 GB per checkpoint and >1 TB per run. Unmanageable. |
| 20 TB storage | Enough for a resized OXE Magic Soup mixture with room to spare, *not* enough for raw OXE including DROID. Resize before ingest. |
| 4–8 GPUs, 192 GB HBM each | Memory is a non-issue (a 7.5B FSDP full-shard run needs ~15 GB/GPU for params+grads+Adam). Throughput and batch-size scaling are the issues. The `expected_world_size` assertions in `vla-scripts/train.py:98` are hardcoded to 8/64 and must be re-specified. |

### Risk register, ranked

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Dependency stack upgrade (torch 2.2 → 2.7+, transformers 4.40 → current) breaks `prismatic/extern/hf/modeling_prismatic.py` | Blocks everything | Pin to a **known-good NGC base image** so torch/CUDA/flash-attn are pre-solved; upgrade `transformers` minimally and add a numerics regression test (Phase 1) that compares logits against a stored reference. |
| R2 | A required wheel has no aarch64 build | Blocks build | Resolve entirely in CI. Build natively on `ubuntu-24.04-arm`, never under QEMU. Make `tensorflow-graphics` a lazy/optional import and drop DROID from initial mixtures. |
| R3 | Runtime code makes a network call nobody anticipated | Wasted cluster slot | Phase 0 runs the container with networking disabled (`--network none`) in CI itself, as a gate. |
| R4 | Analysis artifacts are wrong/incomparable and this is only discovered after egress | Loses a whole run | Freeze probe set + seeds + projection matrix in Phase 0; validate the full capture→pack→unpack→load path on a 10-minute run *before* any long run. |
| R5 | Checkpoint volume exceeds what egress can carry | Loses the run's science | Log-spaced checkpoints, bf16 for analysis copies, fp32 only for the single resumable latest. |
| R6 | Job dies at hour 40 with no resume | Loses days | Resume path is tested in Phase 0, not assumed. `train.py` already supports `--pretrained_checkpoint` + `--resume_step/--resume_epoch`; verify it round-trips optimizer state. |

---

## 3. Target architecture

```
 ┌─ internet side ────────────────────────────────────────────┐
 │  GitHub repo ──► GitHub Actions (ubuntu-24.04-arm)         │
 │                    └─ docker build --platform linux/arm64  │
 │                    └─ smoke test: --network none           │
 │                    └─ docker save | zstd ──► image.tar.zst │
 │                                                             │
 │  Workstation ──► fetch OXE via prepare_open_x.sh (256px)   │
 │              ──► fetch base VLM / openvla-7b weights       │
 │              ──► build probe set + JL projection matrix     │
 │              ──► sha256 manifest                            │
 └──────────────────────────┬──────────────────────────────────┘
                            │  ingest (manual, slow, batched)
 ┌──────────────────────────▼──────────────────────────────────┐
 │  air-gapped GB200 partition                                 │
 │    /data/oxe/            RLDS trees (TFDS layout)           │
 │    /data/weights/        HF + prismatic checkpoints         │
 │    /data/probes/         frozen probe set + projection       │
 │    /runs/<run_id>/       checkpoints, jsonl logs, captures   │
 └──────────────────────────┬──────────────────────────────────┘
                            │  egress (manual, slow, batched)
 ┌──────────────────────────▼──────────────────────────────────┐
 │  analysis side: manifold geometry, cross-run comparison     │
 └─────────────────────────────────────────────────────────────┘
```

### Container base — decision

**Use `nvcr.io/nvidia/pytorch:<YY.MM>-py3` as the base.** It is multi-arch (linux/arm64 published), ships a torch build with sm_100 kernels, and ships flash-attn / Transformer Engine / cuDNN already compiled. The alternative — `nvidia/cuda:12.8-devel` plus `pip install torch --index-url .../cu128` plus a source build of flash-attn — requires compiling flash-attn on a **2-vCPU** arm64 GitHub runner, which is not viable (private-repo arm64 runners get 2 vCPUs; public get 4).

Consequences of that choice, all of which must be handled explicitly in the Dockerfile:

- The NGC image already has torch. **Do not let `pip install -e .` pull a different one.** Strip the torch/torchvision/torchaudio pins from `pyproject.toml` and install with a `constraints.txt` that pins them to the versions already present.
- Install **`tensorflow-cpu`**, never GPU TF. The RLDS pipeline is CPU-only and already calls `tf.config.set_visible_devices([], "GPU")` at `prismatic/vla/datasets/rlds/dataset.py:35`. A GPU TF build will fight NGC's cuDNN/cuBLAS for symbol versions.
- On aarch64, `tensorflow` resolves to the AWS-maintained `tensorflow-cpu-aws` build. Confirm the exact version that has an aarch64 wheel for the container's Python before pinning; TF 2.15 is not guaranteed to be it.
- Keep **`attn_implementation="sdpa"` as the default and flash-attn as an opt-in**. PyTorch SDPA with the cuDNN backend is correct on Blackwell; treat flash-attn as a throughput optimisation to enable after Phase 1 proves numerics, not as a dependency.

### Image transfer

Two viable forms; confirm which the cluster wants before Phase 0:

- `docker save | zstd -T0 -19` → `image.tar.zst`, loaded with `docker load` / `podman load`.
- `enroot import dockerd://<image>` → `.sqsh`, if the partition is Slurm + Pyxis (common for NVL72 deployments). This is usually the better option: no daemon needed, and the squashfs is directly mountable.

**Open question for the cluster admin — resolve before writing the workflow:** container runtime (Docker / Podman / Enroot+Pyxis), scheduler (Slurm / bare `torchrun`), and whether the ingest path has a size cap per transfer.

---

## 4. Phased plan

Each phase has an explicit exit criterion. Do not start the next phase until the previous one's criterion is met — the whole point is to spend the first days de-risking plumbing rather than discovering plumbing failures at hour 40 of a training run.

### Phase 0 — Plumbing (target: 3 days, ≤1 GPU-hour of cluster time)

Prove the loop *image → ingest → run → artifact → egress* closes, using a model small enough that nothing about it is interesting.

Work items:

- `docker/Dockerfile` + `docker/constraints.txt` (see §5.1).
- `.github/workflows/build-image.yml` (see §5.2), including the `--network none` smoke test.
- `scripts/airgap/build_manifest.py` — walks the staging tree, emits `manifest.json` with sha256 per file, total bytes, and a human-readable ingest checklist.
- `scripts/airgap/pack_run.py` — takes `/runs/<run_id>/`, produces a single compressed egress bundle with a manifest and drops anything not on an allowlist.
- Run `vla-scripts/train.py` against `DummyDataset` (`prismatic/vla/datasets/datasets.py:180`) on 1 GPU for 50 steps, save a checkpoint, kill it, resume from the checkpoint, save again.

**Exit criteria:**
1. Image builds natively on arm64 in CI in under 30 minutes.
2. Container starts with no network and reaches step 1 of training on the real cluster.
3. A checkpoint is written, a run is resumed from it, and losses continue smoothly rather than spiking.
4. An egress bundle is produced, moved off, and unpacked on the analysis side with all checksums matching.

### Phase 1 — Data and numerics validation (target: 2 days, ~50 GPU-hours)

Prove the RLDS pipeline works offline on aarch64 and that Blackwell numerics match a known reference.

Work items:

- Ingest **Tier 1** data (§6) and the `openvla-7b` HF checkpoint.
- New VLA configs for world sizes 4 and 8 (§5.3).
- Numerics regression test: a stored set of (image, instruction) → logits reference computed on a known-good x86/Ampere machine outside the airlock, ingested as a small `.npz`. Assert max abs deviation on the action-token logits is within bf16 tolerance.
- LoRA fine-tune of `openvla-7b` on `bridge_orig` via `vla-scripts/finetune.py` for ~5k steps.

**Exit criteria:**
1. Action-token accuracy and L1 loss track the curve shape reported for OpenVLA fine-tuning; no divergence, no NaNs.
2. Numerics regression test passes.
3. Measured throughput (samples/sec/GPU) recorded — this is the number every later time budget depends on.

### Phase 2 — Instrumented reference run (target: 5–7 days, ~1000 GPU-hours)

The first run whose output is a scientific asset.

Work items:

- `prismatic/analysis/` package (§7) — probe set loader, `RepresentationRecorder`, streaming geometry statistics, capture scheduler.
- Wire capture into `run_vla_training` in `prismatic/training/strategies/base_strategy.py:245`.
- Log-spaced checkpointing + bf16 analysis copies (§8).
- Train the workhorse variant: 3B-class base VLM, Tier 1 mixture, 8 GPUs.

**Exit criteria:**
1. A converged (or usefully-far-along) VLA checkpoint.
2. Per-layer geometry statistics logged at every capture step for the whole run.
3. Raw probe activations at ~10 log-spaced steps, egressed and loadable.
4. Total egress volume under whatever the transfer budget turns out to be.

### Phase 3 — Backbone sweep (target: 2–3 weeks)

Now the question becomes scientific rather than infrastructural: *does the choice of visual pretraining change the shape of the manifold, or only its coordinates?*

The registry contains exactly one clean comparison, and it is worth using precisely rather than approximately. These four base VLMs are all 224px, all single-stage, and all built on **the same LLM (Vicuña v1.5 7B)**, so a difference between them is attributable to visual pretraining alone:

| Base VLM | Visual pretraining |
|---|---|
| `in1k-224px+7b` | supervised ImageNet-21K+1K |
| `dinov2-224px+7b` | self-supervised, no language |
| `clip-224px+7b` | language-aligned, contrastive |
| `siglip-224px+7b` | language-aligned, sigmoid loss |

**`prism-dinosiglip-224px+7b` does not belong in this set.** It is the OpenVLA base and the obvious thing to reach for, but it uses Llama-2 7B rather than Vicuña, so including it as a fifth arm confounds vision backbone with language model. Keep it as a reference point, reported separately. The same applies to `phi-2+3b`, which is both a different LLM and 336px — a useful cheap workhorse, but its own axis.

Every one of these must be in the Phase-0/1 ingest. They are 13.5 GB each, 54 GB for the set — trivial next to the data, and impossible to add later without another airlock crossing. This is the single most important reason to over-provision ingest early.

### Phase 4 — Head modularity (target: later)

Cut the seam now, use it later. See §9.

---

## 5. Code changes, file by file

### 5.1 `docker/Dockerfile` (new)

Sketch — versions to be pinned against whatever NGC tag CI resolves:

```dockerfile
ARG NGC_TAG=25.10-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/openvla
COPY docker/constraints.txt /tmp/constraints.txt

# CPU-only TF: the RLDS pipeline never touches the GPU.
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      "tensorflow-cpu" "tensorflow_datasets" "dlimp @ git+https://github.com/moojink/dlimp_openvla"

COPY . /opt/openvla
RUN pip install --no-cache-dir -c /tmp/constraints.txt -e .

# Fail the build, not the cluster job, if anything is missing.
RUN python -c "import torch, tensorflow as tf, transformers, timm, draccus; \
               print(torch.__version__, torch.version.cuda, tf.__version__)" \
 && python -c "from prismatic.vla.datasets import RLDSDataset; print('rlds ok')"
```

`docker/constraints.txt` pins torch/torchvision/torchaudio to the versions already inside the NGC image, so `pip install -e .` cannot replace them.

### 5.2 `.github/workflows/build-image.yml` (new)

```yaml
on: { push: { branches: [main] }, workflow_dispatch: }
jobs:
  build:
    runs-on: ubuntu-24.04-arm          # native arm64; never QEMU
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: docker build --platform linux/arm64 -f docker/Dockerfile -t openvla:${{ github.sha }} .
      - name: Offline smoke test
        run: docker run --rm --network none openvla:${{ github.sha }} python -c "import prismatic; print('ok')"
      - name: Export
        run: docker save openvla:${{ github.sha }} | zstd -T0 -19 -o openvla-${{ github.sha }}.tar.zst
      - uses: actions/upload-artifact@v4
        with: { name: image, path: openvla-*.tar.zst }
```

The `--network none` step is the gate that catches lazy HF-hub or TFDS-checksum fetches before they cost a cluster slot.

### 5.3 `pyproject.toml`

- Remove `torch==2.2.0`, `torchvision==0.17.0`, `torchaudio==2.2.0` pins (supplied by the base image).
- Change `tensorflow==2.15.0` → `tensorflow-cpu` with a version resolved from aarch64 wheel availability.
- Move `tensorflow_graphics` to an optional extra. It is imported lazily at three of four sites already; the fourth, `prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py:6`, is module-level. Make it lazy and DROID becomes optional — which it should be anyway, since DROID is excluded from the initial data tiers.
- Relax `transformers==4.40.1` to a tested range; the numerics regression test in Phase 1 is what makes this safe.
- Remove `wandb` from the required set, or keep it but never initialise it online.

### 5.4 `prismatic/conf/vla.py`

Add configs for the actual hardware. The existing OXE configs assume 64 GPUs at `global_batch_size=2048`. On 8 GPUs, matching that global batch means per-device 256 via gradient accumulation:

```python
@dataclass
class Exp_DinoSigLIP_224px_Bridge_RT1_8GPU(VLAConfig):
    vla_id: str = "prism-dinosiglip-224px+mx-bridge_rt_1+n1"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"
    data_mix: str = "bridge_rt_1"
    expected_world_size: int = 8
    global_batch_size: int = 1024      # accum = 1024 / (8 * 64) = 2
    per_device_batch_size: int = 64    # 192 GB HBM; tune upward in Phase 1
    ...
```

Per-device batch size is a **Phase 1 measurement**, not a guess. With 192 GB and gradient checkpointing there is a lot of headroom above the stock 32; find the knee empirically and record it.

### 5.5 `vla-scripts/train.py`

- `trackers` default → `("jsonl",)`. W&B online is a guaranteed hang behind the airlock.
- `hf_token` must become optional — the current unconditional `cfg.hf_token.read_text()` will raise when the file does not exist.
- Soften the `expected_world_size` assertion at line 98 into a warning plus an explicit `--allow_world_size_mismatch` override, so a 4-GPU debug run of an 8-GPU config does not require editing a dataclass.
- Add `--capture` options for the analysis layer (§7).

### 5.6 `prismatic/training/strategies/base_strategy.py`

- `run_vla_training` (line 245) is where the capture hook goes: at scheduled global steps, pause, run the frozen probe set through the model under `torch.no_grad()`, write the capture, resume.
- Note `num_workers=0` on the VLA dataloader (line 264) is deliberate — tf.data owns the parallelism. On Grace (72 cores/socket) the tf.data thread pool sizing is worth a Phase-1 sweep; it is the likeliest input-bound bottleneck.

### 5.7 `prismatic/models/backbones/llm/base_llm.py` — skip base LLM weight download

On the training path (`load(..., load_for_training=True)`), `HFLLMBackbone.__init__` runs `llm_cls.from_pretrained(hf_hub_path)` and downloads the full base LLM. Those weights are then *immediately overwritten*: `PrismaticVLM.from_pretrained` calls `llm_backbone.load_state_dict(...)` with the prismatic checkpoint's own `llm_backbone` entry. The download is pure waste.

The existing `inference_mode` flag already selects `_from_config` instead, but flipping it is wrong — it also sets `use_cache=True` and skips `enable_input_require_grads()`, both of which training needs. Add a separate `init_llm_weights: bool = True` that controls only the `from_pretrained` versus `_from_config` branch.

Payoff: ~13 GB less ingest per LLM backbone, and — more importantly — the gated `meta-llama/Llama-2-7b-hf` repo reduces to a config and a tokenizer, about 2 MB. Note that vision backbone weights are *not* redundant in the same way: `from_pretrained` loads them only `if "vision_backbone" in model_state_dict`, so timm weights must still be staged.

### 5.8 New: `prismatic/analysis/`, `scripts/airgap/`

See §7 and §8. The ingest tooling in `scripts/airgap/` is written and documented in [`ingest-runbook.md`](ingest-runbook.md).

---

## 6. Data plan

Ingest **resized** data only. `prepare_open_x.sh` from `kpertsch/rlds_dataset_mod` resizes to 256×256 with JPEG encoding and fixes several known channel-order bugs; run it on the internet side, ingest the output. Raw OXE will not fit sensibly in 20 TB once DROID is involved; resized Magic Soup will, with room for checkpoints and captures.

| Tier | Mixture | Raw size | Purpose |
|---|---|---|---|
| **T0** | `openvla/modified_libero_rlds` | **10 GB** | Phase 0 smoke. Also the only route to *closed-loop rollouts* — see below. |
| **T1** | `bridge_rt_1` (Bridge V2 + Fractal) | **243 GB** | Phase 1–2 workhorse. Two large, well-understood, visually distinct embodiments — enough diversity for cross-embodiment manifold questions, small enough to iterate. |
| **T2** | `oxe_magic_soup` | **2.19 TB** | Phase 3. 22 datasets; the breadth needed for "does the manifold factor by embodiment?" |
| **T2′** | `oxe_magic_soup`, subsampled | **~1.1 TB** | The same mixture with `kuka` at 15% and `language_table` at 10% of shards. |
| **T3** | `oxe_magic_soup_plus_minus` (adds DROID) | very large | Only if T2 proves cheap. Drags in `tensorflow_graphics`. Defer. |

Sizes measured against `gs://gresearch/robotics` on 2026-08-19, scoped to each dataset's `0.1.0` version.

Two datasets dominate the transfer and are both heavily downweighted in training, so they cost far more to move than they contribute:

| dataset | raw size | share of T2 bytes | sampling weight in the mixture |
|---|---|---|---|
| `kuka` | 839 GB | 38% | 0.83 |
| `language_table` | 429 GB | 20% | 0.10 |

Subsampling those two roughly halves T2. That is a scientific decision as well as a logistical one — it changes what the model sees — so it needs an explicit call, and the fraction used has to be recorded in the manifest.

**Version scoping is not optional.** Several datasets keep multiple sibling prefixes in the bucket: `language_table` has `0.0.1`, `0.1.0`, `captions` and `long_horizon`, and a naive recursive copy pulls 1189 GB instead of 429 GB. Others (`cmu_stretch`) additionally keep a stray un-versioned copy of their shards at the dataset root, which TFDS ignores but a recursive copy would carry in and which breaks the resulting tree.

**There are two different Bridge datasets.** The mixtures here use `bridge_orig` — the original BridgeData V2 release, served over HTTP from Berkeley — not `bridge`, the 416 GB Open-X re-export in the same bucket. They have different action conventions. Substituting one for the other trains without error and learns the wrong thing.

**Rollouts.** The research goal is explicitly about *how rollouts move through representation space*, which needs closed-loop execution, not just offline forward passes on logged data. Offline, there is no real robot and no internet. So LIBERO must be ingested in Phase 0: the `libero_*` mixtures are already registered in `prismatic/vla/datasets/rlds/oxe/mixtures.py`, and `experiments/robot/libero/run_libero_eval.py` already exists. Risks to check **at build time in CI**, not on the cluster: MuJoCo and robosuite aarch64 wheels, headless EGL rendering inside the container, and the LIBERO asset bundle (task BDDL files and object meshes) which is a separate download and must be on the ingest manifest.

---

### 6.1 Ingest tooling

Two scripts, plus a runbook written for engineers on the staging side who know nothing about this project: [`docs/ingest-runbook.md`](ingest-runbook.md).

- **`scripts/airgap/oxe_fetch.py`** — standard library only. No gcloud SDK, no gsutil, no credentials: anonymous HTTPS against the GCS JSON API, which returns per-object size and MD5, so the manifest carries real checksums. Three subcommands: `plan` resolves and prices a transfer without downloading anything, `fetch` downloads resumably, `verify` re-checks against the manifest. Reads mixture definitions out of `mixtures.py` directly, so it cannot drift from what training expects.
- **`scripts/airgap/hf_fetch.py`** — runs *inside the training image*. This matters: most of what this codebase downloads is implicit. Vision backbones come from `timm.create_model(..., pretrained=True)`, which resolves a model name to a repo and revision through logic that changes between timm versions. A hand-written URL list is silently wrong the next time anything is upgraded. So the script resolves by **execution** — it points `HF_HOME` at the staging directory and runs the same calls training runs. Its `verify` re-opens everything with `HF_HUB_OFFLINE=1`, which is the only check that proves the cache is sufficient rather than merely that the download succeeded.

Both emit a manifest. Those two files are the only record of what was actually moved, and — for a subsampled mixture — the only record of what the model was trained on.

## 7. Representation instrumentation

This is the part that turns a training run into a dataset for the actual research. Design it before Phase 2, freeze it, never change it.

### 7.1 Frozen probe set

Built **outside** the airlock, ingested as a single file, versioned by hash:

- ~512 (episode, timestep) pairs, stratified across datasets in the mixture and tagged with `dataset_name` so manifolds can be coloured by embodiment.
- Drawn from held-out validation splits only.
- Stored as pre-transformed tensors — raw pixels plus tokenised instruction — so that *no part of the data pipeline* sits between the probe set and the model. A change to an image transform must not silently change what the probe measures.
- Shipped alongside a fixed random projection matrix (seeded Johnson–Lindenstrauss, `d_model → 512`) used for the cheap high-volume captures.

`probe_set_v1.npz` + `projection_v1.npz`, both hashed into every capture's manifest. If either ever changes, the version number increments and old captures are never mixed with new ones.

### 7.2 Two-track capture

Full activations for everything, at every checkpoint, is too much data. The split:

**Track A — cheap streaming statistics, logged every capture step to jsonl.** Computed over a much larger sample than Track B, costing almost nothing:

- per-layer mean and variance of the residual stream
- eigenspectrum of the per-layer covariance (streaming/low-rank), and from it participation ratio and effective rank
- intrinsic dimension estimates (two-NN or similar) per layer
- per-layer between-embodiment vs within-embodiment scatter — the direct measurement of whether embodiments occupy separable regions
- cosine similarity of layer-`l` representations to their own value at step 0 — how much each layer actually moves during training

**Track B — raw activations, at ~10 log-spaced steps.** Hooks capture:

- vision backbone output and projector output
- residual stream at every *k*-th LLM decoder layer
- at a fixed set of token positions: last image patch, last prompt token, and each of the 7 action-token positions

Sizing for a 7B model: 512 probes × 16 layers × 9 positions × 4096 dims × 2 bytes (fp16) ≈ **600 MB per capture**, ~6 GB per run at 10 captures. That is egressable. The JL-projected variant is ~75 MB and can be captured far more often.

Log spacing (0, 500, 1k, 2k, 4k, 8k, 16k, 32k, 64k, final) matters: representation structure forms early and then refines, so linear spacing wastes almost all its samples on the boring end.

### 7.3 Rollout traces

For the LIBERO evaluations, record the *same* hook set at every timestep of a rollout, plus the action taken and task success. A rollout trace is then literally a path through the representation space that Track B characterises, which is the core object the research question is about. These are small — a few hundred timesteps per episode — so capture generously.

### 7.4 Implementation shape

```
prismatic/analysis/
  probes.py        # ProbeSet loader; hash validation against manifest
  recorder.py      # RepresentationRecorder: forward hooks, position selection, fp16 pack
  geometry.py      # streaming covariance, effective rank, participation ratio, intrinsic dim
  schedule.py      # log-spaced capture step generator
  rollout.py       # per-timestep capture for closed-loop eval
```

Keep it strictly non-invasive: hooks registered and removed around the capture, no change to the training math, and a hard assertion that model outputs are bit-identical with capture on and off. An instrumentation layer that perturbs training is worse than no instrumentation.

---

## 8. Checkpoint and egress policy

The stock `save_interval=2500` with FSDP `FULL_STATE_DICT` writes ~30 GB of fp32 per checkpoint for a 7B model. Over a 100k-step run that is >1 TB — more than the egress process will tolerate.

Policy:

- **One** rolling fp32 checkpoint for resume, overwritten in place. Never egressed.
- **Log-spaced bf16 analysis checkpoints** (~10 per run, ~15 GB each for 7B, ~6 GB for a 3B-class model). These are the ones that leave.
- Everything else — jsonl logs, geometry statistics, captures, config, dataset statistics — is small and always egressed.

Every egress bundle carries a `manifest.json` recording: git SHA, container image digest, resolved config, probe set hash, projection hash, dataset statistics hash, step numbers, and sha256 per file. A bundle that cannot reconstruct its own provenance is not worth the transfer slot.

---

## 9. Modularity seam (cheap now, essential later)

The eventual goal — extracting reusable modules from a monolithic VLA — is much easier if the architecture has explicit seams before training rather than after. Two cuts are cheap to make now:

1. **Pluggable action head.** Currently action prediction is fused into the LM head via `prismatic/vla/action_tokenizer.py` (actions discretised into vocabulary bins). Introduce an `ActionHead` interface — `forward(hidden_states, ...) -> action_logits_or_continuous` — with the existing tokeniser behaviour as `DiscreteTokenHead`, the default and bit-identical implementation. This is a refactor with no behaviour change, and it is what makes the "compare different heads" work in Phase 4 a config change rather than a fork.

2. **Named taps.** Give every hook point a stable string name, recorded in the capture manifest. Module extraction later means asking "what does the model compute between tap X and tap Y" — that question is only answerable if taps are named consistently across every run in the sweep.

Do not build the modularity machinery now. Do make sure Phase 2 and 3 runs are recorded in a way that does not have to be redone when you do.

---

## 10. Immediate next actions

1. Confirm with the cluster admin: container runtime, scheduler, per-transfer size cap, GPU count actually allocated (4 vs 8 changes the batch-size configs).
2. Write `docker/Dockerfile` + `constraints.txt`, and get the arm64 CI build green — including the `--network none` smoke test. Nothing else can be validated until this exists.
3. Resolve the Hugging Face token question now, not later. `meta-llama/Llama-2-7b-hf` is gated; a read token from an account that has accepted the licence unblocks it. Worth knowing: the entire Phase 3 sweep runs on Vicuña, which is **not** gated, so a token failure costs only the Llama-2 reference run.
4. Hand `docs/ingest-runbook.md` and `scripts/airgap/` to the staging-side engineers and agree the transfer budget — T1 at 243 GB is the recommended first crossing. Start it immediately; it is days of wall clock and the long pole.
5. Stage weights with `--profile all` (~170 GB). Over-provisioning here is nearly free relative to the data and removes an entire future airlock crossing.
6. Build the probe set and freeze it.
