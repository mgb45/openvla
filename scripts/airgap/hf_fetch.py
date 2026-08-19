#!/usr/bin/env python3
"""
hf_fetch.py

Stages every model weight OpenVLA needs into a single directory for air-gapped transfer.

Why this is not just a list of URLs: most of what this codebase downloads is *implicit*.
The vision backbones come from `timm.create_model(..., pretrained=True)`, which resolves a
timm model name to an HF repo and revision through logic that changes between timm versions.
Hand-enumerating those URLs produces a list that is silently wrong the next time anything is
upgraded. So this script resolves by *execution*: it points HF_HOME at the staging directory
and runs the same calls training runs, letting the libraries populate the cache themselves.

That means this script must run INSIDE the training image, so the resolution matches exactly:

    docker run --rm -v /staging:/staging openvla:<tag> \
        python scripts/airgap/hf_fetch.py fetch --out /staging/weights

The resulting directory is a complete HF cache. On the air-gapped side, point HF_HOME at it
and set HF_HUB_OFFLINE=1; nothing will reach for the network.

  fetch    Download everything for the selected profile.
  verify   Re-open the cache offline and confirm every artifact resolves without network.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- What to stage ------------------------------------------------------------------
#
# Prismatic base VLMs, keyed by the model_id used in prismatic/models/registry.py. Each is
# two files in TRI-ML/prismatic-vlms: a config and a ~30 GB checkpoint.
PRISMATIC_VLMS = {
    # The OpenVLA base VLM, plus the reference model.
    "core": ["prism-dinosiglip-224px+7b"],
    # The controlled vision-backbone sweep. These four are the only clean comparison in the
    # registry: all 224px, all single-stage, and all on the SAME LLM (Vicuna v1.5 7B), so a
    # difference between them is attributable to visual pretraining alone.
    #   in1k    -- supervised ImageNet-21K+1K
    #   dinov2  -- self-supervised, no language
    #   clip    -- language-aligned, contrastive
    #   siglip  -- language-aligned, sigmoid loss
    # NOTE: prism-dinosiglip-224px+7b is deliberately NOT in this list. It uses Llama-2 7B
    # rather than Vicuna, so including it would confound vision backbone with LLM. Stage it
    # via the 'core' profile and treat it as a reference point, not a sweep member.
    "sweep": ["in1k-224px+7b", "dinov2-224px+7b", "clip-224px+7b", "siglip-224px+7b"],
    "libero": [],
    "all": [
        "prism-dinosiglip-224px+7b",
        "in1k-224px+7b",
        "dinov2-224px+7b",
        "clip-224px+7b",
        "siglip-224px+7b",
        "phi-2+3b",  # CLIP @336 + Phi-2; a cheap workhorse, on its own axis (different LLM)
    ],
}

# HF-format model repos, snapshotted whole. All ungated.
LIBERO_FT = [
    f"openvla/openvla-7b-finetuned-libero-{suite}" for suite in ("spatial", "object", "goal", "10")
]
HF_REPOS = {
    "core": ["openvla/openvla-7b"],
    "sweep": [],
    # Reference LIBERO policies. These matter more than they look: they are the only way to
    # get known-good closed-loop rollouts on the far side, which is what the representation
    # work needs before any locally-trained policy is good enough to roll out.
    "libero": LIBERO_FT,
    "all": ["openvla/openvla-7b"] + LIBERO_FT,
}

# HF *dataset* repos (repo_type="dataset").
HF_DATASETS = {
    "core": [],
    "sweep": [],
    "libero": ["openvla/modified_libero_rlds"],   # ~10 GB, RLDS format
    "all": ["openvla/modified_libero_rlds"],
}

# timm vision backbones, as (model_name, img_size). Mirrors the dicts in
# prismatic/models/backbones/vision/*.py -- keep in sync if you add a backbone.
DINOV2 = ("vit_large_patch14_reg4_dinov2.lvd142m", 224)
SIGLIP = ("vit_so400m_patch14_siglip_224", 224)
CLIP224 = ("vit_large_patch14_clip_224.openai", 224)
CLIP336 = ("vit_large_patch14_clip_336.openai", 336)
IN1K = ("vit_large_patch16_224.augreg_in21k_ft_in1k", 224)

TIMM_BACKBONES = {
    "core": [DINOV2, SIGLIP],
    "sweep": [IN1K, DINOV2, CLIP224, SIGLIP],
    "libero": [DINOV2, SIGLIP],
    "all": [IN1K, DINOV2, CLIP224, CLIP336, SIGLIP],
}

# LLM backbones. `--full-llm-weights` is off by default because the training path in this
# repo loads the base LLM and then immediately overwrites it: PrismaticVLM.from_pretrained
# calls llm_backbone.load_state_dict() over the top. Only the config and tokenizer are
# genuinely needed -- roughly 2 MB instead of 13 GB per backbone.
#
# Skipping the weights requires the `init_llm_weights=False` patch described in
# docs/airgap-vla-plan.md. Without it, training reaches for HF at runtime and fails offline.
#
# Worth noting for risk: the controlled sweep needs only Vicuna, which is NOT gated. If an
# HF token for meta-llama cannot be arranged, Phase 3 still runs in full; only the Llama-2
# reference run is affected, and the ungated openvla/openvla-7b covers the finetune path.
LLM_REPOS = {
    "core": ["meta-llama/Llama-2-7b-hf"],
    "sweep": ["lmsys/vicuna-7b-v1.5"],
    "libero": ["meta-llama/Llama-2-7b-hf"],
    "all": ["meta-llama/Llama-2-7b-hf", "lmsys/vicuna-7b-v1.5", "microsoft/phi-2"],
}

GATED = {"meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def cmd_fetch(args) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(out)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    profile = args.profile
    report, failures = [], []

    def record(kind, name, start_size):
        size = tree_size(out) - start_size
        report.append({"kind": kind, "name": name, "bytes": max(0, size)})
        print(f"    -> {human(max(0, size))}", flush=True)

    # 1. Prismatic base VLM checkpoints -------------------------------------------------
    for model_id in PRISMATIC_VLMS[profile]:
        print(f"\n[prismatic-vlm] {model_id}", flush=True)
        before = tree_size(out)
        try:
            for fn in (f"{model_id}/config.json", f"{model_id}/checkpoints/latest-checkpoint.pt"):
                hf_hub_download(repo_id="TRI-ML/prismatic-vlms", filename=fn, token=token)
            record("prismatic-vlm", model_id, before)
        except Exception as e:  # noqa: BLE001
            print(f"    !! {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(("prismatic-vlm", model_id, str(e)))

    # 2. HF-format repos ----------------------------------------------------------------
    for repo in HF_REPOS[profile]:
        print(f"\n[hf-repo] {repo}", flush=True)
        before = tree_size(out)
        try:
            snapshot_download(repo_id=repo, token=token)
            record("hf-repo", repo, before)
        except Exception as e:  # noqa: BLE001
            print(f"    !! {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(("hf-repo", repo, str(e)))

    for repo in HF_DATASETS[profile]:
        print(f"\n[hf-dataset] {repo}", flush=True)
        before = tree_size(out)
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", token=token)
            record("hf-dataset", repo, before)
        except Exception as e:  # noqa: BLE001
            print(f"    !! {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(("hf-dataset", repo, str(e)))

    # 3. timm vision backbones ----------------------------------------------------------
    import timm

    for name, img_size in TIMM_BACKBONES[profile]:
        print(f"\n[timm] {name} @ {img_size}px", flush=True)
        before = tree_size(out)
        try:
            timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
            record("timm", name, before)
        except Exception as e:  # noqa: BLE001
            print(f"    !! {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(("timm", name, str(e)))

    # 4. LLM configs + tokenizers (and optionally weights) ------------------------------
    from transformers import AutoConfig, AutoTokenizer

    for repo in LLM_REPOS[profile]:
        print(f"\n[llm] {repo}{' (full weights)' if args.full_llm_weights else ' (config + tokenizer only)'}", flush=True)
        before = tree_size(out)
        try:
            AutoConfig.from_pretrained(repo, token=token)
            AutoTokenizer.from_pretrained(repo, token=token)
            if args.full_llm_weights:
                snapshot_download(repo_id=repo, token=token, ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5"])
            record("llm", repo, before)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            hint = (
                f"'{repo}' is a gated repository. A Hugging Face account must accept its licence, "
                "then export HF_TOKEN=<that account's read token> before running this script."
                if repo in GATED
                else str(e)
            )
            print(f"    !! {hint}", file=sys.stderr)
            failures.append(("llm", repo, hint))
        except Exception as e:  # noqa: BLE001
            print(f"    !! {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(("llm", repo, str(e)))

    manifest = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile,
        "full_llm_weights": args.full_llm_weights,
        "bytes_total": tree_size(out),
        "artifacts": report,
        "failures": failures,
    }
    (out / "weights_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Staged {human(manifest['bytes_total'])} into {out}")
    print(f"Manifest: {out / 'weights_manifest.json'}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S) -- these will break training offline:")
        for kind, name, err in failures:
            print(f"  [{kind}] {name}\n      {err}")
        sys.exit(1)


def cmd_verify(args) -> None:
    """Re-resolve everything with the network hard-disabled. This is the real test: it
    proves the staged cache is sufficient, rather than that the download succeeded."""
    out = Path(args.out).resolve()
    os.environ["HF_HOME"] = str(out)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    manifest = json.loads((out / "weights_manifest.json").read_text())
    profile = manifest["profile"]
    from huggingface_hub import hf_hub_download

    import timm
    from transformers import AutoConfig, AutoTokenizer

    bad = []
    for model_id in PRISMATIC_VLMS[profile]:
        try:
            for fn in (f"{model_id}/config.json", f"{model_id}/checkpoints/latest-checkpoint.pt"):
                hf_hub_download(repo_id="TRI-ML/prismatic-vlms", filename=fn)
        except Exception as e:  # noqa: BLE001
            bad.append(f"prismatic-vlm {model_id}: {e}")
    for name, img_size in TIMM_BACKBONES[profile]:
        try:
            timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
        except Exception as e:  # noqa: BLE001
            bad.append(f"timm {name}: {e}")
    for repo in LLM_REPOS[profile]:
        try:
            AutoConfig.from_pretrained(repo)
            AutoTokenizer.from_pretrained(repo)
        except Exception as e:  # noqa: BLE001
            bad.append(f"llm {repo}: {e}")

    if bad:
        print(f"{len(bad)} artifact(s) do NOT resolve offline:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)
    print(f"All artifacts for profile '{profile}' resolve offline. Cache is transfer-ready.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="staging directory; becomes HF_HOME on the far side")
        if name == "fetch":
            p.add_argument("--profile", choices=sorted(PRISMATIC_VLMS), default="core")
            p.add_argument(
                "--full-llm-weights",
                action="store_true",
                help="also stage base LLM weights (~13 GB each). Only needed if the "
                "init_llm_weights patch has not been applied -- see the docstring.",
            )
    args = ap.parse_args()
    {"fetch": cmd_fetch, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
