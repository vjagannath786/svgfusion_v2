# SVGFusion v2 — Codebase Reference

> **Purpose of this document**: Authoritative reference for AI agents and developers navigating this codebase. It describes the full system architecture, every active class and function, the data flow, and which file version is canonical for each component.

SVGFusion v2 is an advanced pipeline for generating high-quality Scalable Vector Graphics (SVG) from text prompts. It introduces a two-stage VAE + Diffusion Transformer architecture with progressive DINOv2-conditioned encoding and CLIP-conditioned generation.

Based on Research Paper: https://ximinng.github.io/SVGFusionProject/

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Version Map — Which File Is Active](#3-version-map--which-file-is-active)
4. [Data Pipeline](#4-data-pipeline)
5. [Module: svgutils/](#5-module-svgutils)
6. [Module: models/](#6-module-models)
7. [Module: evaluation/](#7-module-evaluation)
8. [Training Pipeline (Step-by-Step)](#8-training-pipeline-step-by-step)
9. [Inference / Generation Pipeline](#9-inference--generation-pipeline)
10. [Tensor Format Reference](#10-tensor-format-reference)
11. [Key Hyperparameters](#11-key-hyperparameters)
12. [Dependencies](#12-dependencies)
13. [Appendix: Evolution of Versions](#appendix-evolution-of-versions)

---

## 1. System Overview

SVGFusion v2 is a **text-to-SVG generative model** built on a two-stage pipeline:

```
Stage 1 — VP-VAE (Vector-Primitive VAE)
  SVG file → parse → tensor → [Encoder] → latent z → [Decoder] → reconstructed SVG tensor

Stage 2 — VS-DiT (Vector-SVG Diffusion Transformer)
  Text prompt → CLIP encoder → context sequence
  Gaussian noise → [VS-DiT denoiser conditioned on CLIP context] → clean latent z
  latent z → [VP-VAE Decoder] → SVG tensor → SVG file
```

**Visual conditioning during VAE training**: the encoder takes both the SVG tensor sequence and cumulative DINOv2 (ViT-S) visual embeddings of progressively drawn elements as pixel-level context.

**Text conditioning during DiT training and inference**: CLIP (ViT-L/14) provides the sequence context (`last_hidden_state`, dim=768) and pooled embedding for the cross-attention blocks in VS-DiT.

---

## 2. Repository Layout

```
svgfusion_v2/
│
├── svgutils/                         # Data processing & dataset utilities
│   ├── __init__.py                   # Public API (exports canonical classes)
│   ├── svgparsing.py                 # SVG file → structured Python dict
│   ├── svgtensor.py                  # Parsed dict → normalized hybrid tensor  [ACTIVE]
│   ├── tensorsvg.py                  # Tensor → SVG string reconstruction      [ACTIVE]
│   ├── datasetpreparation_v5.py      # DINOv2-aligned progressive dataset       [ACTIVE]
│   ├── dataset_preparation_v2.py     # Older dataset prep (CLS tokens only)    [SUPERSEDED]
│   ├── prepare_latents.py            # Extracts z-latents from trained VAE      [ACTIVE]
│   └── prepare_latents_v1.py         # Older latent extraction script           [SUPERSEDED]
│
├── models/                           # Neural network definitions & training
│   ├── __init__.py                   # Public API (imports canonical versions)
│   ├── vpvae_accelerate_ce_v2.py     # VP-VAE (per-token latent, binned CE)     [ACTIVE]
│   ├── vpvae_accelerate_hybrid.py    # VP-VAE (pooled latent, hybrid loss)      [SUPERSEDED]
│   ├── vpvae_accelerate_ce.py        # VP-VAE (pooled latent, CE bins)          [SUPERSEDED]
│   ├── vpvae_accelerate.py           # VP-VAE (pure continuous MSE)             [SUPERSEDED]
│   ├── vp_dit_v1.py                  # VS-DiT model + diffusion utils           [ACTIVE]
│   ├── vp_dit.py                     # VS-DiT (older, extra samplers)           [SUPERSEDED]
│   ├── vp_dit_training_v1.py         # VS-DiT training script                   [ACTIVE]
│   ├── vp_dit_training.py            # Older DiT training script                [SUPERSEDED]
│   └── vp_dit_training_subset.py     # DiT training on keyword subset           [SUPERSEDED]
│
├── evaluation/
│   ├── evaluate_hybrid.py            # VAE reconstruction evaluation            [ACTIVE]
│   ├── evaluate_ce.py                # CE-VAE reconstruction evaluation         [SUPERSEDED]
│   ├── evaluation_ce_v1.py           # Older CE evaluation                      [SUPERSEDED]
│   ├── generate_samples.py           # End-to-end text→SVG generation           [ACTIVE]
│   ├── generate_samples_v3.py        # Previous generation script               [SUPERSEDED]
│   ├── generate_samples_v2.py        # Previous generation script               [SUPERSEDED]
│   └── generate_samples_v1.py        # First generation script                  [SUPERSEDED]
│
├── install_dependencies.sh
├── LICENSE
└── README.md                         # This file
```

---

## 3. Version Map — Which File Is Active

| Component | Active File | Superseded Files |
|---|---|---|
| SVG Parser | `svgutils/svgparsing.py` | — |
| SVG → Tensor | `svgutils/svgtensor.py` (`SVGToTensor_Normalized`) | `SVGToTensor_Normalized_v0` (same file, older class) |
| Tensor → SVG | `svgutils/tensorsvg.py` (`TensorToSVGHybrid`) | — |
| Dataset Prep | `svgutils/datasetpreparation_v5.py` | `dataset_preparation_v2.py` |
| Latent Extraction | `svgutils/prepare_latents.py` | `prepare_latents_v1.py` |
| VP-VAE Model | `models/vpvae_accelerate_ce_v2.py` | `vpvae_accelerate_hybrid.py`, `vpvae_accelerate_ce.py`, `vpvae_accelerate.py` |
| VS-DiT Model | `models/vp_dit_v1.py` | `vp_dit.py` |
| DiT Training | `models/vp_dit_training_v1.py` | `vp_dit_training.py`, `vp_dit_training_subset.py` |
| VAE Evaluation | `evaluation/evaluate_hybrid.py` | `evaluate_ce.py`, `evaluation_ce_v1.py` |
| SVG Generation | `evaluation/generate_samples.py` | `generate_samples_v1/v2/v3.py` |

**`models/__init__.py` imports from**:
- `vpvae_accelerate_ce_v2.py` → `VPVAE`, `VPVAEEncoder`, `VPVAEDecoder`, `MultiHeadAttention`, `TransformerBlock`, `apply_rope`, `set_seed`
- `vp_dit_v1.py` → `VS_DiT`, `VS_DiT_Block`, `TimestepEmbedder`, `MLP`, `ddim_sample`, `get_linear_noise_schedule`, `precompute_diffusion_parameters`, `noise_latent`
- `vp_dit_training_v1.py` → `zDataset`

**`svgutils/__init__.py` imports from**:
- `svgparsing.py` → `SVGParser`
- `svgtensor.py` → `SVGToTensor_Normalized`
- `datasetpreparation_v5.py` → `DynamicProgressiveSVGDataset`, `load_dino_model_components`
- `tensorsvg.py` → `tensor_to_svg_file_hybrid_wrapper`

---

## 4. Data Pipeline

```
Raw .svg files
      │
      ▼
SVGParser.parse_svg()                   ← svgparsing.py
      │  Returns: {viewport, elements: [{type, commands, style}, ...]}
      ▼
SVGToTensor_Normalized.create_tensor_for_element()   ← svgtensor.py
      │  Returns: Tensor [N_rows, 14]  (hybrid format, see §10)
      ▼
preprocess_svg_for_dynamic_storage()    ← datasetpreparation_v5.py
      │  For each element k:
      │    1. Rasterize SVG[0..k] via cairosvg → PIL Image
      │    2. DINOv2-small → patch token sequence [257, 384]
      │    Stores: full_svg_matrix_content, aligned_pixel_patch_tokens,
      │            actual_element_row_counts  per file as .pt
      ▼
DynamicProgressiveSVGDataset[i]         ← datasetpreparation_v5.py
      │  Returns: (svg_matrix [L,14], pixel_embeds [L,257,384], padding_mask [L])
      │  Adds BOS/EOS tokens, pads to max_seq_len=1024
      ▼
VPVAE.forward()                         ← vpvae_accelerate_ce_v2.py (training)
      │  Encoder → z [B, L, latent_dim=128]
      │  Decoder → element_logits, command_logits, param_bin_logits_list
      ▼
prepare_latents.py (post-training)
      │  Encodes all SVGs → z tensors → zdataset_vpvae.pt
      ▼
zDataset / vp_dit_training_v1.py
      │  (z, CLIP text tokens) → VS-DiT training
      ▼
generate_samples.py (inference)
      │  Text → CLIP → noise → DDIM → z_hat → VAE Decoder → tensor → SVG
```

---

## 5. Module: `svgutils/`

### 5.1 `svgparsing.py`

#### `class SVGConstraints`

Static constraint registry for valid SVG content.

| Attribute | Type | Description |
|---|---|---|
| `VALID_ELEMENTS` | `set` | `{'svg','g','path','clipPath','rect','circle','ellipse','title'}` |
| `VALID_ATTRIBUTES` | `dict[str, set]` | Per-element allowed attribute names |
| `VALID_PATH_COMMANDS` | `set` | `{M,m,L,l,H,h,V,v,C,c,S,s,Q,q,T,t,A,a,Z,z}` |
| `VALID_TRANSFORMS` | `set` | `{translate,scale,rotate,skewX,skewY,matrix}` |
| `max_svg_size` | `int` | 10000 (max file length in chars) |

Methods: `validate_element(name)`, `validate_attributes(name, attrs)`, `validate_path_commands(cmds)`.

#### `class SVGParser`

Parses `.svg` files into a structured Python dict.

| Method | Signature | Returns |
|---|---|---|
| `parse_svg` | `(svg_file: str) -> Dict` | `{viewport, elements: list}` |
| `parse_path_element` | `(element) -> Dict` | `{commands: list, style: dict}` |
| `parse_circle_element` | `(element) -> Dict` | `{commands: {cx,cy,r,...}, style: dict}` |
| `parse_rect_element` | `(element) -> Dict` | `{commands: {x,y,w,h,rx,ry,...}, style: dict}` |
| `parse_ellipse_element` | `(element) -> Dict` | `{commands: {cx,cy,rx,ry,...}, style: dict}` |
| `parse_path_commands` | `(path_data: str) -> List[Dict]` | `[{command, values, original}, ...]` |
| `parse_style_attributes` | `(element) -> dict` | Parsed style key→value dict |

`viewport` dict keys: `width`, `height`, `viewBox` (`{min_x, min_y, width, height}`).

Elements with `style` tags raise `ValueError` to reject CSS-heavy files.

---

### 5.2 `svgtensor.py`

Two classes exist in this file. **`SVGToTensor_Normalized` is the active one** (used by `__init__.py`). `SVGToTensor_Normalized_v0` is the earlier version with a different quantization strategy.

#### `class SVGToTensor_Normalized` ← **ACTIVE**

Converts parsed SVG elements to a **hybrid quantized tensor** where discrete IDs (element type, command type) and binned continuous values (geometry, color) coexist in integer columns.

**Constructor constants**:

| Attribute | Value | Description |
|---|---|---|
| `num_geom_params` | 8 | Geometry columns per row |
| `num_fill_style_params` | 3 | Style columns: R, G, B (fill only) |
| `output_matrix_cols` | 14 | Total = 3 header cols + 8 geom + 3 style |
| `num_bins` | 256 | Quantization resolution |
| `COORD_MIN / COORD_MAX` | −128.0 / 127.0 | Coordinate normalization range |
| `RADIUS_MIN / RADIUS_MAX` | 0.0 / 128.0 | Radius / dimension normalization |
| `COLOR_MIN / COLOR_MAX` | 0.0 / 255.0 | RGB normalization |

**Vocabularies**:

```python
ELEMENT_TYPES = {'<BOS>':0, 'rect':1, 'circle':2, 'ellipse':3, 'path':4, '<EOS>':5, '<PAD>':6}

PATH_COMMAND_TYPES = {
    'NO_CMD':0, 'm':1, 'l':2, 'h':3, 'v':4, 'c':5,
    's':6, 'q':7, 't':8, 'a':9, 'z':10,
    'CIRCLE':11, 'ELLIPSE':12, 'RECT':13
}
```

**Key methods**:

| Method | Description |
|---|---|
| `create_tensor_for_element(element_data)` | Converts one parsed element dict to a 2D tensor `[N_rows, 14]`. For paths: one row per path command + one style row. For shapes: one DEF row + one style row. Returns `None` for unknown types. |
| `_normalize(value, min_val, max_val)` | Quantizes a float to bin index `[0, 255]` using `floor(value_shifted / range * 256)`. Returns `int`. |
| `_get_fill_style_params_normalized(style_data)` | Returns fill RGB as bin indices tensor `[3]`. |

---

### 5.3 `tensorsvg.py`

#### `class TensorToSVGHybrid`

Inverse converter: tensor rows → SVG element strings.

**Constructor**: `__init__(svg_tensor_converter_instance)` — takes a `SVGToTensor_Normalized` instance for its vocabulary maps and range constants.

| Method | Description |
|---|---|
| `reconstruct_svg_elements(tensor_data_hybrid, actual_len=None)` | Main method. Iterates tensor rows, accumulates path `d` segments, emits `<path>`, `<rect>`, `<circle>`, `<ellipse>` strings. Returns `list[str]`. |
| `_denormalize(bin_index, min_val, max_val)` | Converts bin index → approximate float via midpoint: `(bin + 0.5) * range / num_bins + min`. |
| `_format_geo_params_for_path_d(cmd_id, geo_params_norm)` | Formats geometry tensor columns into SVG path `d` attribute fragment string for the given command. |
| `_format_style_attributes(style_params_norm)` | Formats style columns into `fill="#rrggbb"` attribute string. |

**Top-level function** exported by `__init__.py`:

```python
tensor_to_svg_file_hybrid_wrapper(
    tensor_data,          # [L, 14] tensor
    output_path,          # Path to write .svg
    svg_tensor_converter, # SVGToTensor_Normalized instance
    actual_len=None,      # Trim to this length before processing
    viewbox_size=512      # Output viewBox dimension
)
```

---

### 5.4 `datasetpreparation_v5.py` ← **ACTIVE**

Key distinction from `dataset_preparation_v2.py`: uses **full DINOv2 patch token sequences** `[257, 384]` instead of only CLS tokens `[1, 384]`, and saves each SVG as a separate `.pt` file instead of a single combined list.

#### `load_dino_model_components() -> (model, processor, device, embed_dim, seq_length)`

Loads `facebook/dinov2-small`. Returns the model, HuggingFace processor, device, hidden dim (384), and patch sequence length (257 including CLS).

#### `preprocess_svg_for_dynamic_storage(...)`

Processes a single `.svg` file into a `.pt` file. Skips if output already exists (resumable).

**Output `.pt` structure per file**:
```python
{
    'filename': str,
    'full_svg_matrix_content': Tensor[N_cmds, 14],        # raw unpadded SVG tensor
    'actual_element_row_counts': Tensor[N_elements],      # rows per element
    'aligned_pixel_patch_tokens': Tensor[N_elements, 257, 384]  # DINOv2 per stage
}
```

#### `class DynamicProgressiveSVGDataset(Dataset)`

PyTorch dataset. At `__getitem__` constructs a training sample by selecting a random "stage" (prefix of k elements) from the preprocessed data, then padding to `max_seq_len`.

`__getitem__` returns: `(svg_matrix [L,14], pixel_embeds [L,257,384], padding_mask [L])`.

---

### 5.5 `prepare_latents.py` ← **ACTIVE**

Standalone script to encode all training SVGs through a trained `VPVAE` and extract z-latents.

**Output**: `zdataset_vpvae.pt` — list of `{'filename': str, 'z': Tensor[L_svg, latent_dim]}` dicts.

**`class zDataset(Dataset)`**: wraps the z-list; `__getitem__` returns `{'z', 'text', 'filename'}`. Text is looked up from `svg_captions.json` keyed by filename stem.

---

## 6. Module: `models/`

### 6.1 `vpvae_accelerate_ce_v2.py` ← **ACTIVE VP-VAE**

The current production VP-VAE. Key differences from earlier versions:
- **Per-token sequential latent**: encoder outputs `mu, log_var` of shape `[B, L_svg, latent_dim]` (not a single pooled vector), making the latent a sequence for the DiT to process.
- **Binned CE reconstruction**: decoder outputs class logits for element type, command type, and each of the N continuous geometry/style parameters (quantized into 256 bins) rather than raw MSE regression.

#### Utility functions

| Function | Signature | Description |
|---|---|---|
| `apply_rope(x)` | `(Tensor[B,L,D]) -> Tensor[B,L,D]` | Rotary Position Embeddings. Handles odd `D` by padding then trimming. |
| `set_seed(seed, accelerator=None)` | `(int, Accelerator)` | Sets all RNG seeds. |

#### `class MultiHeadAttention(nn.Module)`

Custom multi-head attention supporting cross-attention (`kdim ≠ d_model`).

`forward(query, key, value, key_padding_mask=None, attn_mask=None)` — `key_padding_mask` is `[B, S_key]` bool.

#### `class TransformerBlock(nn.Module)`

Standard pre-norm transformer block using `nn.MultiheadAttention` (PyTorch built-in).

`forward(x, padding_mask=None, attn_mask=None)` → `x` same shape.

#### `class VPVAEEncoder(nn.Module)` ← per-token latent version

**Architecture**:
```
svg_matrix_hybrid [B, L, 14]
  → element_type_embedding [B, L, 64]   (col 0, learned Embedding)
  + command_type_embedding [B, L, 64]   (col 1, learned Embedding)
  + shared_param_embedding (cols 2..13, each bin index → 64, shared Embedding)
  → concat → [B, L, 64+64+N*64]
  → apply_rope
  → svg_projection Linear → [B, L, d_model=512]

pixel_embedding [B, L, 384] → pixel_projection Linear → [B, L, 512]

cross_attention(query=svg_proj, key=pixel_proj, value=pixel_proj)
  + residual → cross_attn_norm
  → N=4 × TransformerBlock (self-attention)
  → fc_mu    → mu      [B, L, 128]
  → fc_var   → log_var [B, L, 128]
```

`forward(svg_matrix_hybrid, pixel_embedding, svg_padding_mask=None, pixel_padding_mask=None)` → `(mu, log_var)` each `[B, L, latent_dim]`.

#### `class VPVAEDecoder(nn.Module)` ← binned CE head version

**Architecture**:
```
z [B, L, latent_dim=128]
  → fc_latent Linear → [B, L, d_model=512]
  → apply_rope
  → N=4 × TransformerBlock
  → decoder_norm LayerNorm
  → element_type_head Linear → element_logits  [B, L, 7]
  → command_type_head Linear → command_logits  [B, L, 14]
  → param_heads ModuleList (one Linear per param) → [B, L, 256] each
```

`forward(z, target_len)` → `(element_type_logits, command_type_logits, param_logits_list)`.

#### `class VPVAE(nn.Module)` ← **main model**

```python
VPVAE(
    num_element_types,          # 7
    num_command_types,          # 14
    element_embed_dim,          # 64
    command_embed_dim,          # 64
    num_other_continuous_svg_features,  # 11 or 12
    pixel_feature_dim,          # 384
    encoder_d_model,            # 512
    decoder_d_model,            # 512
    encoder_layers,             # 4
    decoder_layers,             # 4
    num_heads,                  # 8
    latent_dim,                 # 128
    max_seq_len,                # 1024
    element_padding_idx=0,
    command_padding_idx=0,
    num_bins=256,
    svg_param_embed_dim=64
)
```

`reparameterize(mu, logvar)` — samples `z = mu + eps * exp(0.5 * logvar)` (element-wise).

`forward(svg_matrix_hybrid, pixel_embedding, svg_padding_mask, pixel_padding_mask)` → `(element_type_logits, command_type_logits, param_logits_list, mu, logvar)`.

**Loss function**:
```python
vp_vae_hybrid_loss(
    element_logits, command_logits, param_logits_list,
    target_svg_matrix_hybrid,
    mu, logvar,
    element_pad_idx=0, command_pad_idx=0,
    svg_padding_mask=None,
    kl_weight=0.1,
    ce_elem_loss_weight=1.0,
    ce_cmd_loss_weight=1.0,
    ce_param_loss_weight=1.0
)
# Returns: (total_loss, ce_elem, ce_cmd, avg_param_ce, kl_loss)
```

KL annealing: `get_kl_weight(step, total_steps, max_kl_weight, anneal_portion, schedule)`.

---

### 6.2 `vp_dit_v1.py` ← **ACTIVE VS-DiT**

Defines the diffusion transformer and all diffusion utilities.

#### Utility functions

| Function | Description |
|---|---|
| `apply_rope(x)` | RoPE for `[B,L,D]` tensors; handles odd `D`. |
| `modulate(x, shift, scale)` | AdaLN: `x * (1 + scale) + shift`. |

#### `class TimestepEmbedder(nn.Module)`

Maps scalar timestep `t` → embedding vector `[B, hidden_size]`.

- `timestep_embedding(t, dim, max_period=10000)` — sinusoidal encoding (static method)
- `forward(t)` → `[B, hidden_size]` via 2-layer SiLU MLP

#### `class MLP(nn.Module)`

2-layer feed-forward: `fc1 → act1 → fc2 → act2` (GELU default). Used inside `VS_DiT_Block`.

#### `class MultiHeadAttention(nn.Module)`

Custom MHA with cross-attention support via `kdim`/`vdim` params. Same interface as the VAE version.

#### `class SelfAttention(nn.Module)`

Optimized self-attention with fused `qkv = Linear(dim, 3*dim)`. Optional `fp32_attention` flag.

| Init Param | Default | Description |
|---|---|---|
| `dim` | — | Input/output dimension |
| `num_heads` | 8 | Number of heads |
| `qkv_bias` | `True` | Bias in QKV projection |
| `fp32_attention` | `False` | Compute attention in FP32 |

#### `class VS_DiT_Block(nn.Module)`

Single DiT block with AdaLN-Zero conditioning.

**Constructor params**:
| Param | Description |
|---|---|
| `hidden_dim` | Internal block dimension (384) |
| `context_dim` | Cross-attention key/value dimension (768 for CLIP) |
| `num_heads` | Attention heads (6) |
| `mlp_ratio` | MLP hidden expansion (default 4.0, training uses 8.0) |
| `dropout` | Dropout probability |

**Submodules**: `norm1, norm2, norm3` (affine-free LayerNorm), `self_attn` (nn.MHA), `cross_attn` (nn.MHA with kdim=hidden_dim), `mlp` (MLP).

**Forward**:
```python
forward(
    x,                        # [B, L, hidden_dim] latent sequence
    gamma_1, beta_1, alpha_1, # AdaLN params [B, hidden_dim] from timestep
    gamma_2, beta_2, alpha_2,
    context_seq,              # [B, S, hidden_dim] projected CLIP sequence
    context_padding_mask=None # [B, S] bool
) -> x [B, L, hidden_dim]
```

Flow: self-attn (AdaLN) → cross-attn (latent queries, CLIP keys/values) → MLP (AdaLN).

#### `class VS_DiT(nn.Module)` ← **main DiT model**

```python
VS_DiT(
    latent_dim=128,   # Input feature dim per latent token
    hidden_dim=384,   # Internal DiT dimension
    context_dim=768,  # CLIP last_hidden_state dim
    num_blocks=12,
    num_heads=6,
    mlp_ratio=8.0,
    dropout=0.1
)
```

**Submodules**:

| Name | Type | Description |
|---|---|---|
| `t_embedder` | `TimestepEmbedder` | Timestep → `[B, hidden_dim]` |
| `proj_in` | `Linear(latent_dim, hidden_dim)` | Project latent tokens into DiT space |
| `context_proj` | `Linear(context_dim, hidden_dim)` | Project CLIP sequence into DiT space |
| `modulation_mlp` | `Sequential(SiLU, Linear(hidden_dim, 6*hidden_dim))` | Compute 6 AdaLN parameters from timestep embedding |
| `blocks` | `ModuleList[VS_DiT_Block × num_blocks]` | Transformer blocks |
| `final_norm` | `LayerNorm` | Pre-output norm |
| `final_modulation_mlp` | `Sequential(SiLU, Linear(hidden_dim, 2*hidden_dim))` | Final AdaLN shift/scale |
| `final_proj` | `Linear(hidden_dim, latent_dim)` | Project back to latent space |

**Forward**:
```python
forward(
    z,                    # [B, L, latent_dim] noisy latent sequence
    t,                    # [B] timestep indices
    context_seq,          # [B, S, context_dim] CLIP last_hidden_state
    pooled_context=None,  # [B, context_dim] (reserved, not used in blocks)
    context_padding_mask=None  # [B, S]
) -> predicted_noise [B, L, latent_dim]
```

Weight initialization: `initialize_weights_v0/v1/v2/v3` methods defined; `v2` is the most complete (zero-init for output layers, xavier for others).

#### Diffusion Utility Functions

| Function | Signature | Description |
|---|---|---|
| `get_linear_noise_schedule(timesteps)` | `(int) -> Tensor[T]` | Returns `betas` linearly from `0.0001` to `0.02`. |
| `precompute_diffusion_parameters(betas, device)` | `(Tensor, device) -> dict` | Returns dict: `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, etc. |
| `noise_latent(z0, t, diff_params, device)` | `(Tensor, Tensor, dict, device) -> (z_noisy, epsilon)` | Forward diffusion: `z_t = sqrt_abar * z0 + sqrt_1m_abar * eps`. |
| `ddim_sample(model, shape, context_seq, pooled_context, context_padding_mask, diff_params, device, num_steps, eta, cfg_scale, uncond_context, ...)` | — | DDIM reverse sampling with CFG. `shape = [B, L, latent_dim]`. Returns `z_0_hat`. |
| `ddim_sample_fixed(...)` | — | Variant of DDIM with deterministic fixed schedule. |

**Note**: `ddim_sample_improved` and `dpm_solver_2m_20_steps` are in the older `vp_dit.py` and are imported by `generate_samples.py`.

---

### 6.3 `vp_dit_training_v1.py` ← **ACTIVE DiT Training Script**

#### `class zDataset(Dataset)`

| Method | Description |
|---|---|
| `__init__(z_data_list, captions_dict)` | Accepts list of `{'z': tensor, 'filename': str}` dicts plus captions lookup. Computes global mean/std. |
| `__getitem__(idx)` | Returns `{'z': Tensor, 'text': str, 'filename': str}`. Text from `svg_captions.json`. |

**Training loop uses**:
- CLIP `ViT-L/14` (`CLIPTextModel` + `CLIPTokenizer`) for text encoding
- `VS_DiT` with AdaLN + cross-attention conditioning on CLIP `last_hidden_state`
- Classifier-free guidance: 10% random text dropout during training
- `cosine_warmup_scheduler` (custom `LambdaLR`)
- WandB logging

---

## 7. Module: `evaluation/`

### `evaluate_hybrid.py` ← **ACTIVE VAE Evaluation**

Loads a trained `VPVAE`, runs encoder → decoder on held-out precomputed SVG data, reconstructs SVG files via `tensor_to_svg_file_hybrid_wrapper`, and computes reconstruction loss metrics.

Config that must match the trained model: `num_element_types=7`, `num_command_types=14`, `element_embed_dim=64`, `command_embed_dim=64`, `latent_dim=128`, `encoder/decoder_d_model=512`, `num_heads=8`.

### `generate_samples.py` ← **ACTIVE End-to-End Generation**

End-to-end text → SVG generation. Key constants:

```python
LATENT_SEQ_LEN           = 256    # Sequence length of VAE latent
LATENT_FEATURE_DIM       = 128    # Feature dim per latent token
VSDIT_HIDDEN_DIM         = 384
VSDIT_CONTEXT_DIM        = 768    # CLIP ViT-L/14
VSDIT_NUM_BLOCKS         = 12
VSDIT_NUM_HEADS          = 6
DIFFUSION_NUM_TIMESTEPS  = 1000
SAMPLING_NUM_STEPS_DDIM  = 100
SAMPLING_CFG_SCALE       = 3.0
```

---

## 8. Training Pipeline (Step-by-Step)

### Step 1: Prepare Dataset

Run `datasetpreparation_v5.py` (edit `SVG_INPUT_DIR` and `OUTPUT_DIR` inside the script):
```bash
python svgutils/datasetpreparation_v5.py
```
Produces per-file `.pt` files consumed by `DynamicProgressiveSVGDataset`.

### Step 2: Train VP-VAE

```bash
accelerate launch models/vpvae_accelerate_ce_v2.py
```

Key config inside `main()`:
- `total_steps`: 5000+
- `batch_size_per_device`: 32
- `latent_dim`: 128, `max_seq_len_train`: 1024
- `kl_weight_max`: 0.005 (with cosine annealing)

Saves best checkpoint as `.pt` state dict.

### Step 3: Extract Latents

```bash
python svgutils/prepare_latents.py
# Edit MODEL_PATH to point to trained VAE checkpoint
# Produces: zdataset_vpvae.pt
```

### Step 4: Prepare Captions

Requires `svg_captions.json` at `./svg_captions.json` — a dict mapping `{filename_stem: text_description}`. This is an **external input** not generated by this repo.

### Step 5: Train VS-DiT

```bash
python models/vp_dit_training_v1.py
# Requires: zdataset_vpvae.pt, svg_captions.json, CLIP model checkpoint
```

Key config inside `if __name__ == "__main__":`:
- `latent_dim`: 128, `hidden_dim`: 384, `context_dim`: 768
- `num_blocks`: 12, `num_heads`: 6
- `total_steps`: 3000+

### Step 6: Generate SVGs

```bash
python evaluation/generate_samples.py
# Requires: trained VAE + DiT checkpoint paths, CLIP model
# Outputs to: ./generated_svgs_vsdit_seqcond_sequential_latent/
```

---

## 9. Inference / Generation Pipeline

```
Input: text prompt (str)
  │
  ├─ CLIPTokenizer.tokenize(prompt) → token_ids [1, 77]
  ├─ CLIPTextModel(token_ids)
  │     → last_hidden_state [1, 77, 768]   (sequence context)
  │     → pooled_output     [1, 768]       (reserved)
  │
  ├─ z_T ~ N(0, I)  shape [1, LATENT_SEQ_LEN=256, LATENT_FEATURE_DIM=128]
  │
  ├─ DDIM loop (100 steps, CFG scale=3.0):
  │    for t in reversed(timesteps):
  │      ε_cond   = VS_DiT(z_t, t, last_hidden_state, ...)
  │      ε_uncond = VS_DiT(z_t, t, null_context, ...)
  │      ε        = ε_uncond + 3.0 * (ε_cond - ε_uncond)
  │      z_{t-1}  = DDIM update step
  │
  ├─ z_0 [1, 256, 128]
  │     → VPVAE.decoder(z_0, target_len=1024)
  │     → element_logits [1, 1024, 7]
  │     → command_logits [1, 1024, 14]
  │     → param_logits   [1, 1024, 256] × N_params
  │
  ├─ argmax over each logit head → discrete tensor [1, 1024, 14]
  │
  └─ tensor_to_svg_file_hybrid_wrapper(tensor, output_path, ...) → output.svg
```

---

## 10. Tensor Format Reference

Each row in the SVG tensor (`[N_rows, 14]`) represents one rendering command:

| Col Index | Name | Type | Range | Description |
|---|---|---|---|---|
| 0 | `element_type` | int 0–6 | `ELEMENT_TYPES` | 0=BOS, 1=rect, 2=circle, 3=ellipse, 4=path, 5=EOS, 6=PAD |
| 1 | `cmd_seq_idx` | int | 0–100 | Index of this command within its parent element |
| 2 | `cmd_type` | int 0–13 | `PATH_COMMAND_TYPES` | 0=NO_CMD, 1=m, 2=l, 3=h, 4=v, 5=c, 6=s, 7=q, 8=t, 9=a, 10=z, 11=CIRCLE, 12=ELLIPSE, 13=RECT |
| 3 | `geom_0` | int 0–255 | bin index | x0 or cx |
| 4 | `geom_1` | int 0–255 | bin index | y0 or cy |
| 5 | `geom_2` | int 0–255 | bin index | x1 or rx |
| 6 | `geom_3` | int 0–255 | bin index | y1 or ry |
| 7 | `geom_4` | int 0–255 | bin index | x2 or width |
| 8 | `geom_5` | int 0–255 | bin index | y2 or height |
| 9 | `geom_6` | int 0–255 | bin index | x3 / end x |
| 10 | `geom_7` | int 0–255 | bin index | y3 / end y |
| 11 | `fill_R` | int 0–255 | bin index | Fill color Red |
| 12 | `fill_G` | int 0–255 | bin index | Fill color Green |
| 13 | `fill_B` | int 0–255 | bin index | Fill color Blue |

**Quantization formula**: `bin = clamp(floor((v - min) / (max - min) * 256), 0, 255)`.

**De-quantization formula**: `v ≈ (bin + 0.5) * (max - min) / 256 + min`.

**Special tokens**: BOS row prepended, EOS row appended, remainder padded to `max_seq_len=1024` with PAD rows.

---

## 11. Key Hyperparameters

### VP-VAE (current config in `vpvae_accelerate_ce_v2.py`)

| Param | Value |
|---|---|
| `latent_dim` | 128 |
| `encoder_d_model` / `decoder_d_model` | 512 |
| `encoder_layers` / `decoder_layers` | 4 |
| `num_heads` | 8 |
| `element_embed_dim` / `command_embed_dim` | 64 |
| `svg_param_embed_dim` | 64 |
| `num_bins` | 256 |
| `max_seq_len` | 1024 |
| `pixel_feature_dim` | 384 (DINOv2-small CLS) |
| `kl_weight_max` | 0.005 |
| `total_steps` | 5000 |
| `learning_rate` | 3e-4 |
| `weight_decay` | 0.1 |

### VS-DiT (current config in `vp_dit_training_v1.py`)

| Param | Value |
|---|---|
| `latent_dim` | 128 |
| `hidden_dim` | 384 (S-size configuration) |
| `context_dim` | 768 (CLIP ViT-L/14) |
| `num_blocks` | 12 |
| `num_heads` | 6 |
| `mlp_ratio` | 8.0 |
| `noise_steps` | 1000 |
| `beta_start` / `beta_end` | 0.0001 / 0.02 |
| `cfg_dropout_prob` | 0.1 |
| `total_steps` | 3000+ |
| `learning_rate` | 3e-4 |
| `weight_decay` | 0.1 |

---

## 12. Dependencies

| Library | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `transformers` | CLIP (text encoder), DINOv2 (vision encoder) |
| `accelerate` | Multi-GPU/distributed training for VAE |
| `wandb` | Experiment tracking and artifact logging |
| `cairosvg` | SVG → PNG rasterization for DINOv2 visual conditioning |
| `Pillow` | Image loading and processing |
| `tqdm` | Progress bars |
| `numpy` | Numerical utilities |
| `matplotlib` | Visualization |
| `scikit-learn` | t-SNE for latent space visualization |

Install: `bash install_dependencies.sh`

---

## Appendix: Evolution of Versions

### VP-VAE Lineage

| Version | File | Encoder Output | Decoder Output | Key Change |
|---|---|---|---|---|
| v0 | `vpvae_accelerate.py` | `mu, logvar` scalar `[B, latent_dim]` | `[B, L, num_features]` via Tanh | Pure continuous, MSE loss only |
| v1 | `vpvae_accelerate_hybrid.py` | `mu, logvar` scalar `[B, latent_dim]` | element logits + command logits + continuous params (Tanh) | Hybrid: CE for discrete, MSE for continuous |
| v2 | `vpvae_accelerate_ce.py` | `mu, logvar` scalar `[B, latent_dim]` | element logits + command logits + per-param bin logits | All-CE binned reconstruction |
| **v3 (active)** | `vpvae_accelerate_ce_v2.py` | `mu, logvar` **per-token** `[B, L, latent_dim]` | element logits + command logits + per-param bin logits | Per-token sequential latent; enables DiT to model sequence |

### VS-DiT Lineage

| Version | File | Key Change |
|---|---|---|
| v0 | `vp_dit.py` | Scalar latent `[B, latent_dim]`; includes `ddim_sample_improved` and `dpm_solver_2m_20_steps` |
| **v1 (active)** | `vp_dit_v1.py` | Sequential latent `[B, L, latent_dim]`; cleaner AdaLN; `SelfAttention` with fused QKV added |

### Dataset Preparation Lineage

| Version | File | Key Change |
|---|---|---|
| v1 | `dataset_preparation_v2.py` | Stores single-file precomputed list; uses CLS token only `[1, 384]` |
| **v2 (active)** | `datasetpreparation_v5.py` | Per-file `.pt` output (resumable); uses full patch sequence `[257, 384]` |
