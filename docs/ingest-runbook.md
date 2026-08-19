# Ingest Runbook

**Audience:** engineers with internet access on the staging side, preparing data for transfer into the air-gapped environment. No knowledge of the research project is assumed.
**Companion document:** [`airgap-vla-plan.md`](airgap-vla-plan.md) explains why.

---

## 0. What this is

The training environment has no internet. Everything the training code would normally download at runtime has to be staged on a disk out here and carried in. Two categories:

1. **Robot datasets** — public Open-X Embodiment data, hundreds of gigabytes to a couple of terabytes depending on choices made in step 2.
2. **Model weights** — pretrained vision and language model checkpoints, tens to hundreds of gigabytes.

Everything else — the code, its Python dependencies, CUDA, the whole software stack — is baked into the Docker image and does not need staging.

Two scripts do the work. Both are resumable: if a transfer dies, re-run the same command and it picks up where it stopped.

| Script | Runs on | Needs |
|---|---|---|
| `scripts/airgap/oxe_fetch.py` | bare host | `python3` ≥ 3.8, `wget`, outbound HTTPS. **No** pip installs, no cloud SDK, no credentials. |
| `scripts/airgap/hf_fetch.py` | inside the training image | `docker`, outbound HTTPS, and possibly a Hugging Face token (see §5). |

Respects the standard `https_proxy` / `HTTPS_PROXY` environment variables.

---

## 1. Before you start

- [ ] A disk with room for the plan you choose in §2, **plus 10%**.
- [ ] The training image, loaded locally (`docker load < openvla-<sha>.tar.zst`) — needed for step 5.
- [ ] Confirm with the project owner whether a Hugging Face token is required (§5). This is the one item that can block you, and it is easier to resolve before you start than halfway through.

Pick a staging root and use it consistently:

```bash
export STAGING=/mnt/staging/openvla
mkdir -p "$STAGING"/{oxe,weights}
```

---

## 2. Decide the transfer budget first

`plan` resolves exactly what would be downloaded and prints the total. **It downloads nothing.** Always run it, read the number, and agree it with the project owner before starting a fetch.

```bash
python3 scripts/airgap/oxe_fetch.py plan --mixture oxe_magic_soup --out "$STAGING/oxe"
```

Measured totals, August 2026:

| Plan | Size | What it is |
|---|---|---|
| `--mixture bridge_rt_1` | **243 GB** | Bridge V2 + Fractal. The recommended first transfer — enough to train and validate everything. |
| `--mixture oxe_magic_soup` | **2.19 TB** | The full mixture, everything at full size. |
| `oxe_magic_soup` with the trims below | **~1.1 TB** | The same mixture, half the bytes. |

Two datasets dominate and are both heavily downweighted during training, so they cost far more to move than they contribute:

| dataset | full size | share of bytes | sampling weight |
|---|---|---|---|
| `kuka` | 839 GB | 38% | 0.83 |
| `language_table` | 429 GB | 20% | 0.10 |

Subsampling them is the single biggest lever available:

```bash
python3 scripts/airgap/oxe_fetch.py plan \
  --mixture oxe_magic_soup \
  --fraction kuka=0.15 \
  --fraction language_table=0.10 \
  --out "$STAGING/oxe"
```

`--fraction` keeps an evenly spaced subset of the record shards — evenly spaced rather than a prefix, because shards are written in episode order and a prefix would bias toward whatever was collected first. The script rewrites each dataset's `dataset_info.json` afterwards so the result is still a valid, readable dataset.

> **This is a scientific decision, not just a logistics one.** Subsampling changes what the model sees. Get sign-off from the project owner on the fractions before fetching; record whatever you used, since the manifest is the only record of it.

---

## 3. Fetch the datasets

```bash
python3 scripts/airgap/oxe_fetch.py fetch --out "$STAGING/oxe" --jobs 8
```

Runs for hours to days. Safe to interrupt and re-run — completed files are skipped by size. Raise `--jobs` if the link is fast and underused; lower it if you are saturating something shared.

Then verify:

```bash
python3 scripts/airgap/oxe_fetch.py verify --out "$STAGING/oxe" --deep
```

`--deep` re-computes an MD5 for every file and checks it against the value the storage bucket published. It is slow and it is the point of the exercise — a corrupted shard produces a training run that fails days later, or worse, does not fail at all.

### One trap worth knowing about

There are two different Bridge datasets. The mixtures in this project use **`bridge_orig`**, the original BridgeData V2 release, served over plain HTTP from Berkeley — *not* `bridge`, the Open-X re-export sitting in the same bucket. They have different action conventions. Substituting one for the other trains without any error and learns the wrong thing.

`oxe_fetch.py` handles this: it pulls `bridge_orig` over HTTP with `wget` and names the directory correctly. If `wget` is missing it prints the exact manual command instead. Do not "fix" a missing `bridge_orig` by copying `bridge` into place.

---

## 4. Fetch the model weights

This one runs *inside* the training image, so that the libraries resolve exactly the same files training will later ask for. Enumerating the URLs by hand does not work reliably — the vision backbones are resolved by library logic, not by fixed paths.

```bash
docker run --rm \
  -v "$STAGING/weights":/weights \
  -e HF_TOKEN="$HF_TOKEN" \
  openvla:<tag> \
  python scripts/airgap/hf_fetch.py fetch --out /weights --profile core
```

Profiles, smallest first. Ask the project owner which one applies:

| Profile | Contents | Size |
|---|---|---|
| `core` | The OpenVLA base model (13.6 GB), the released OpenVLA-7B (15.1 GB), two vision backbones. The minimum to train anything. | **~32 GB** |
| `sweep` | Four vision-backbone variants at 13.5 GB each, for the comparison experiments. | **~60 GB** |
| `libero` | Four reference policies (15.1 GB each) plus 10.2 GB of simulator datasets, for closed-loop evaluation. | **~74 GB** |
| `all` | Everything above, plus a smaller Phi-2 variant. | **~170 GB** |

Measured August 2026. Small next to the datasets — if in doubt, stage `all`. A second trip out here for one missing 13 GB checkpoint costs far more than moving it now.

Then confirm the staged cache is actually self-sufficient:

```bash
docker run --rm -v "$STAGING/weights":/weights openvla:<tag> \
  python scripts/airgap/hf_fetch.py verify --out /weights
```

`verify` re-opens every artifact with networking disabled at the library level. It is the only check that proves the cache will work on the far side; a successful download does not.

---

## 5. The Hugging Face token question

One model — `meta-llama/Llama-2-7b-hf` — is gated. Downloading it requires a Hugging Face account that has accepted Meta's licence, and a read token from that account passed as `HF_TOKEN`.

If you cannot get a token, say so early rather than working around it. It is not fatal:

- The `sweep` profile needs only ungated models and works without any token.
- The `core` profile needs the gated repo for a configuration file and a tokeniser — about 2 MB, not the 13 GB of weights.

The project owner can resolve this by accepting the licence on their own account and providing a read-only token. Treat that token as a credential: it is scoped to their account, and it should not be committed, logged, or left in shell history.

---

## 6. Handover

Final layout:

```
$STAGING/
  oxe/
    oxe_manifest.json         <- what was fetched, with sizes and checksums
    bridge_orig/ fractal20220817_data/ ...
  weights/
    weights_manifest.json     <- what was staged, and anything that failed
    hub/ ...                  <- becomes HF_HOME inside the airlock
```

Before the transfer:

- [ ] `oxe_fetch.py verify --deep` passed.
- [ ] `hf_fetch.py verify` passed with no failures listed.
- [ ] Both manifest files are present — they are how the far side confirms the transfer arrived intact, and the only record of which subsampling fractions were used.
- [ ] The two `*_manifest.json` files sent to the project owner separately, so a mismatch can be diagnosed without reading the whole disk.

After the transfer, on the air-gapped side, re-run both `verify` commands against the delivered copy. Everything needed for that is already on the disk; neither check touches the network.

---

## 7. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `plan` fails with an unknown mixture | Typo — the error lists every valid mixture name. |
| Fetch stops partway | Re-run the same `fetch` command. Completed files are skipped. |
| `verify` reports `SIZE` or `MD5` on a few files | Delete those files and re-run `fetch`. |
| `verify` reports `MISSING` for many files | The `plan` was re-run with different options after fetching, so the manifest no longer matches the disk. Re-run `plan` with the original options, or re-fetch. |
| `GatedRepoError` | See §5. |
| Downloads are very slow | Raise `--jobs`. If that does not help, the bottleneck is the link or the proxy, not the script. |
| Disk fills mid-fetch | Re-run `plan` to get the true total, free space, then re-run `fetch`. Nothing already downloaded is lost. |
