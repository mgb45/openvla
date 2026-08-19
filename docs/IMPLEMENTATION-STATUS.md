# OpenVLA Air-Gapped GB200 Training — Implementation Status

**Last updated:** 2026-08-19  
**Target:** Phase 0 (plumbing validation) in 3 days  
**Audience:** You and your engineers. This is the checklist and roadmap, not the architecture essay.

---

## What's Done ✅

### Research & Data
- [x] Confirmed research direction: representation manifolds, backbone comparisons, modular policies
- [x] Measured Open-X Embodiment bucket (anonymous access, no auth needed)
- [x] Built transfer budget: T1 **243 GB**, full Magic Soup **2.19 TB** (subsampled **~1.1 TB**)
- [x] Found and documented two bucket traps (version scoping, bridge_orig vs bridge)

### Scripts for Staging-Side Engineers
- [x] `scripts/airgap/oxe_fetch.py` — download RLDS datasets (stdlib-only, no gcloud SDK needed)
  - `plan` subcommand: price a transfer without downloading
  - `fetch` subcommand: resumable download with shard subsampling
  - `verify` subcommand: MD5 check against manifest
  - End-to-end tested against live bucket
- [x] `scripts/airgap/hf_fetch.py` — stage model weights (runs inside training image)
  - Profiles: `core` (32 GB), `sweep` (60 GB), `libero` (74 GB), `all` (170 GB)
  - Resolves by execution (not hand-written URLs) so it survives upgrades
  - `verify` re-opens with networking disabled to prove offline sufficiency
  - CLI and syntax verified; full test awaits image build
- [x] `docs/ingest-runbook.md` — handoff document for engineers on the staging side
  - No knowledge of the research project assumed
  - Step-by-step walkthrough with estimated times and measured sizes
  - Troubleshooting section

### Documentation
- [x] `docs/airgap-vla-plan.md` — full architecture and phased plan (420 lines)
- [x] Published as artifact with measured data: https://claude.ai/code/artifact/934690da-bbce-4efe-9ffd-259739de669e

---

## What's To Do 🔧

### Immediate (this week, Phase 0 gate)

#### 1. Container & CI — **CRITICAL PATH**
- [ ] Write `docker/Dockerfile` (NGC base, offline env vars, TF-CPU, constraints.txt)
- [ ] Write `docker/constraints.txt` (torch/torchvision/torchaudio pinned to NGC versions)
- [ ] Write `.github/workflows/build-image.yml` (arm64 native, `--network none` smoke test)
- [ ] Get arm64 CI build green in under 30 min — **this gates everything else**
- [ ] Verify `hf_fetch.py` runs inside the image

#### 2. Talk to cluster admin
- [ ] Confirm: container runtime (Docker/Podman/Enroot), scheduler (Slurm/bare torchrun), per-transfer cap, 4 vs 8 GPU allocation
- [ ] Get the NGC image tag you want to base on
- [ ] Ask whether they need `.sqsh` (enroot) or `.tar.zst` (docker load)

#### 3. Start the data ingest (it's days of wall clock — start now)
```bash
# On staging side, with internet:
python3 scripts/airgap/oxe_fetch.py plan --mixture bridge_rt_1 --out /staging/oxe
python3 scripts/airgap/oxe_fetch.py fetch --out /staging/oxe --jobs 8
python3 scripts/airgap/oxe_fetch.py verify --out /staging/oxe --deep
```
**Measured time:** ~24 hours for T1 (243 GB) with 8 parallel jobs on a decent link.

#### 4. Stage the weights
```bash
# Inside the training image (once built):
export HF_TOKEN="<huggingface token>"  # see note below
docker run --rm -v /staging/weights:/weights openvla:<tag> \
  python scripts/airgap/hf_fetch.py fetch --out /weights --profile all
docker run --rm -v /staging/weights:/weights openvla:<tag> \
  python scripts/airgap/hf_fetch.py verify --out /weights
```
**Measured size:** all profiles ~170 GB. Stage `all`; over-provisioning here is cheap.

**Hugging Face token note:** `meta-llama/Llama-2-7b-hf` is gated. If you can't get a token:
- Phase 3 backbone sweep still runs (Vicuña is ungated)
- Only the Llama-2 reference run is blocked
- Report this upfront so it doesn't surprise you later

#### 5. Code: Unblock dependencies
- [ ] `pyproject.toml`: remove `torch==2.2.0` pins, change `tensorflow==2.15.0` → `tensorflow-cpu`, move `tensorflow_graphics` to optional
- [ ] `prismatic/models/backbones/llm/base_llm.py`: add `init_llm_weights: bool = True` flag to skip downloading base LLM (13 GB per backbone)
- [ ] `vla-scripts/train.py`: 
  - Make `hf_token` optional
  - Soften `expected_world_size` assertion → warning + `--allow_world_size_mismatch` flag
  - Default `trackers=("jsonl",)` not W&B

#### 6. Code: Phase 0 minimal configs
- [ ] Add 4-GPU and 8-GPU configs to `prismatic/conf/vla.py` (adjust per-device batch size empirically in Phase 1)
- [ ] Create `scripts/airgap/test_config.yaml` for DummyDataset smoke test

#### 7. Probe set (frozen forever after this)
- [ ] Build a stable set of ~512 (episode, timestep) pairs from held-out validation splits
- [ ] Compute a seeded Johnson-Lindenstrauss projection matrix (d_model → 512)
- [ ] Hash both and record in manifest
- [ ] Ingest both into Phase 0 transfer

### Phase 1 (2 days, ~50 GPU-hours)
- [ ] Ingest Tier 1 data (bridge_rt_1 + probe set + projection matrix already in Phase 0 ingest)
- [ ] Build logits regression test against a reference checkpoint (x86/Ampere baseline)
- [ ] Run `finetune.py` on `bridge_orig` for ~5k steps
- [ ] Measure throughput (samples/sec/GPU) — every later time budget depends on this
- [ ] **Gate:** logits match published curve, numerics regression passes, no NaNs

### Phase 2 (5–7 days, ~1000 GPU-hours)
- [ ] Implement `prismatic/analysis/` package
  - `probes.py`: load frozen probe set, validate hash
  - `recorder.py`: forward hooks, streaming geometry computation
  - `geometry.py`: effective rank, participation ratio, intrinsic dim, embodiment scatter
  - `schedule.py`: log-spaced capture steps
- [ ] Wire capture into `run_vla_training` 
- [ ] Log-spaced checkpointing: fp32 rolling (never egressed), bf16 analysis copies (egressed)
- [ ] Train 3B-class workhorse on Tier 1 mixture, 8 GPUs
- [ ] **Gate:** converged checkpoint, geometry stats for whole run, raw activations at 10 steps, egress under budget

### Phase 3 (2–3 weeks)
- [ ] Train sweep: `in1k-224px+7b`, `dinov2-224px+7b`, `clip-224px+7b`, `siglip-224px+7b` (all Vicuña, 224px, single-stage)
- [ ] Compare manifold geometry across vision backbones
- [ ] All four prismatic base VLMs must already be in weights ingest (54 GB total) — this is why Phase 0 over-provisioning matters

### Phase 4 (later)
- [ ] Pluggable action head interface (bit-identical default)
- [ ] Named taps for modularity extraction
- [ ] Compare action head variants

---

## Risk Checklist

| Risk | Severity | Mitigation | Status |
|---|---|---|---|
| torch 2.2 → 2.7+ upgrade breaks code | **high** | NGC base image pre-solves; add logits regression test in Phase 1 | 📋 pending image |
| aarch64 wheel missing | high | Native CI build on arm64; make TF-graphics lazy | ✅ script design |
| Runtime network call | high | CI runs with `--network none` as gate | ✅ workflow design |
| Analysis artifacts wrong/incomparable | high | Freeze probe set + seeds + projection before Phase 2 | ✅ documented |
| Checkpoint volume explodes | medium | Log-spaced bf16 copies + fp32 rolling | ✅ policy defined |
| Resume doesn't work | medium | Test in Phase 0, not Phase 2 | ✅ plan phase 0 |

---

## Concrete Next Steps (This Week)

**Today/tomorrow:**
1. [ ] Pick an NGC image tag and ask admin for it
2. [ ] Ask admin for: runtime, scheduler, per-transfer cap, GPU count
3. [ ] Create `docker/Dockerfile` skeleton

**This week:**
1. [ ] Get CI green on arm64
2. [ ] Hand staging team the runbook + `oxe_fetch.py` and start T1 download (24h)
3. [ ] `hf_fetch.py` inside image with `--profile all` (several hours)

**By end of week:**
1. [ ] Both manifests verified on the air-gapped side
2. [ ] Phase 0 code edits done (5 files, mostly small)
3. [ ] DummyDataset smoke test: image → ingest → train 50 steps → resume → egress → unpack

---

## Transfer Budgets (Measured)

| Scenario | Data | Weights | Total |
|---|---|---|---|
| **Phase 0 test** | 10 GB (LIBERO) | 32 GB (core) | **42 GB** |
| **Phase 1–2** | 243 GB (T1) | 60 GB (sweep) | **303 GB** |
| **Phase 3** | 1.1 TB (magic soup trimmed) | 60 GB (sweep) | **1.2 TB** |
| **All in** | 2.19 TB (full magic soup) | 170 GB (all) | **2.4 TB** |

Starting assumption: 20 TB available. Recommend: T1 + all weights first (~340 GB), then T2′ in second ingest if Phase 2 succeeds.

---

## Files Needing Changes

Core codebase (~8 files, mostly small):
- `docker/Dockerfile` (new)
- `docker/constraints.txt` (new)
- `.github/workflows/build-image.yml` (new)
- `pyproject.toml` (5 lines)
- `prismatic/conf/vla.py` (new configs)
- `prismatic/models/backbones/llm/base_llm.py` (1 parameter)
- `vla-scripts/train.py` (4 edits)
- `prismatic/training/strategies/base_strategy.py` (add capture hook)

Analysis layer (~6 new files):
- `prismatic/analysis/probes.py`
- `prismatic/analysis/recorder.py`
- `prismatic/analysis/geometry.py`
- `prismatic/analysis/schedule.py`
- `prismatic/analysis/rollout.py`
- `prismatic/analysis/__init__.py`

Already written:
- `scripts/airgap/oxe_fetch.py` ✅
- `scripts/airgap/hf_fetch.py` ✅
- `docs/ingest-runbook.md` ✅
- `docs/airgap-vla-plan.md` ✅

---

## Key Numbers to Remember

| Item | Value | Why it matters |
|---|---|---|
| T1 transfer | 243 GB, ~24h | Recommended first ingest |
| Full Magic Soup | 2.19 TB (1.1 TB trimmed) | Phase 3 scope |
| Weights (all) | 170 GB | One ingest, never revisit |
| Throughput | unknown (Phase 1 measurement) | Gates all time budgets |
| Probe set | 512 examples, frozen | Every run must use same one |
| Analysis checkpoints | 10 per run | Log-spaced, egressable |
| Resume interval | 2500 steps | Rolling fp32 checkpoint |

