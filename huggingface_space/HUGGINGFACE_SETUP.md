# Hugging Face Spaces Deployment Guide

Complete step-by-step instructions for deploying the induction circuit dashboard
to Hugging Face Spaces. The dashboard runs on **CPU Basic** (free tier) via
ONNX Runtime — no GPU required.

---

## Prerequisites

- A Hugging Face account (free): https://huggingface.co/join
- `git` and `git-lfs` installed locally
- The GitHub repo cloned and experiments complete (checkpoints exist)

---

## Step 1 — Create the Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `induction-circuit-stability` (or your choice)
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) — sufficient for ONNX inference
   - **Visibility**: Public (required for paper submission) or Private
3. Click **Create Space**

You will get a URL like:
`https://huggingface.co/spaces/Mattral/induction-circuit-stability`

---

## Step 2 — Clone the Space repository

```bash
# Install git-lfs (required for ONNX files > 10 MB)
git lfs install

# Clone the Space repo (replace with your username and space name)
git clone https://huggingface.co/spaces/Mattral/induction-circuit-stability
cd induction-circuit-stability
```

---

## Step 3 — Copy the Space files

From the GitHub repository root, copy the `huggingface_space/` contents:

```bash
# From inside your cloned Space repo:
cp /path/to/mech-interp-induction/huggingface_space/app.py .
cp /path/to/mech-interp-induction/huggingface_space/requirements.txt .
cp /path/to/mech-interp-induction/huggingface_space/README.md .
```

Your Space repo should now contain:
```
induction-circuit-stability/
├── README.md          ← Hugging Face Space card (must stay at root)
├── app.py             ← Self-contained Gradio app
├── requirements.txt   ← Space-specific dependencies
└── onnx_models/       ← YOU WILL ADD THIS IN STEP 4
    ├── model_pre.onnx
    └── model_post.onnx
```

---

## Step 4 — Export ONNX models and upload them

ONNX files are large (> 10 MB) so they are tracked by git-lfs automatically.
Run the export from the GitHub repository root (requires completed checkpoints):

```bash
# In the GitHub repo directory (not the Space repo):
python src/viz/dashboard/onnx_export.py \
    --pre  checkpoints/code_seed42/step_000000.pt \
    --post checkpoints/code_seed42/step_$(ls checkpoints/code_seed42/ | tail -1 | grep -o '[0-9]*') .pt \
    --output /tmp/onnx_models/ \
    --seq-len 64
```

Then copy into the Space repo and track with git-lfs:

```bash
# Back in the Space repo:
mkdir -p onnx_models
cp /tmp/onnx_models/model_pre.onnx  onnx_models/
cp /tmp/onnx_models/model_post.onnx onnx_models/

# git-lfs tracks these automatically due to file size;
# verify with:
git lfs status
```

---

## Step 5 — Edit README.md to update your Space URL

In `README.md`, update the GitHub link:
```markdown
- GitHub: [Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning](https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning)
```

Also update the GitHub repo's `README.md`:
```markdown
## Interactive Dashboard

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Mattral/induction-circuit-stability)
```

---

## Step 6 — Commit and push

```bash
git add README.md app.py requirements.txt onnx_models/
git commit -m "Deploy induction circuit dashboard (ONNX, CPU Basic)"
git push
```

Hugging Face will build the Space automatically. Watch the build log at:
`https://huggingface.co/spaces/Mattral/induction-circuit-stability`

Cold-start time depends on ONNX model size; target is **< 3 seconds** after models are cached.

---

## Step 7 — Measure load time and record it

Open the Space in an incognito browser window (cold cache). Time from page-load to
the interface being interactive. Record this in `PAPER_CHECKLIST.md`:

```markdown
- [x] Dashboard loads in < 3 seconds on standard laptop CPU (measured: X.Xs)
```

If load time exceeds 3 seconds, optimise by:
1. Reducing ONNX sequence length: re-export with `--seq-len 32`
2. Using `ort.SessionOptions` with `inter_op_num_threads=1` (already set in app.py)
3. Enabling ONNX constant-folding during export (already set: `do_constant_folding=True`)

---

## Differences Between GitHub Repo and HF Space

| Aspect | GitHub Repo | HF Space |
|--------|-------------|----------|
| Structure | Full project (src/, notebooks/, paper/, tests/) | Only app.py + requirements.txt + onnx_models/ |
| Imports | `from src.circuits import ...` | All code in app.py (self-contained) |
| ONNX models | Generated locally, in .gitignore | Committed to Space repo via git-lfs |
| Tokenizer | Loaded from local cache | Downloaded from HF Hub on first run |
| Hardware | GPU for training, CPU for dashboard | CPU Basic (free tier) |
| Python deps | Full requirements.txt (14 packages) | Space requirements.txt (6 packages) |
| Secrets | None needed | None needed (all models local to Space) |

---

## Troubleshooting

**"ONNX model not found" error in dashboard**
→ You forgot Step 4. The Space app expects `onnx_models/model_pre.onnx` and
  `onnx_models/model_post.onnx` to exist in the Space repo root.

**git-lfs quota exceeded (free tier = 1 GB/month)**
→ Use `huggingface_hub` Python API to upload models directly:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="onnx_models/model_pre.onnx",
    path_in_repo="onnx_models/model_pre.onnx",
    repo_id="Mattral/induction-circuit-stability",
    repo_type="space",
)
```

**Build fails with dependency error**
→ Check that `gradio==4.37.2` is in requirements.txt.
  HF Spaces pins Gradio; mismatches cause build failures.

**Load time > 3 seconds**
→ The ONNX model is being loaded from cold cache. On HF Spaces, models are
  cached after the first request. Subsequent requests will be < 500 ms.
  If cold-start consistently exceeds 3 s, re-export with smaller seq_len.
