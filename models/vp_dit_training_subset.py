
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import math
import random
import wandb
import os
import sys
import json
import traceback
from pathlib import Path

# Adjust sys.path for your project structure
#sys.path.append("/home/svgfusion_v2/models/") # Example path
sys.path.append(".") # Add current directory as well

# --- Import Model and Utils from vp_dit.py ---
try:
    # --- MODIFIED: Ensure importing the correct classes for sequence conditioning ---
    from vp_dit import VS_DiT, TimestepEmbedder, MLP, VS_DiT_Block # Included these for self-containment
    from vp_dit import get_linear_noise_schedule, precompute_diffusion_parameters, noise_latent, ddim_sample
    print("Successfully imported VS_DiT model components and diffusion utilities.")
except ImportError as e:
    print(f"Error importing from vp_dit.py: {e}")
    print("Please ensure vp_dit.py is in the correct path and its classes are defined.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     traceback.print_exc()
     sys.exit(1)

# --- Set Seed Utility (Unchanged) ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)
        except AttributeError: pass

# --- Load Captions (Unchanged) ---
caption_file = '/workspace/captions/svg_captions.json'
try:
    with open(caption_file, 'r') as f:
        svg_captions = json.load(f)
    print(f"Loaded captions from {caption_file}")
except FileNotFoundError:
    print(f"Error: Caption file not found at {caption_file}"); sys.exit(1)
except Exception as e:
     print(f"Error loading captions: {e}"); sys.exit(1)

class zDataset(Dataset):
    def __init__(self, z_file_list_path, mean_path='./z_mean_banana_overfit.pt', std_path='./z_std_banana_overfit.pt', filter_keywords=None):
        # z_file_list_path points to the list of dicts: [{'path': str, 'text': str}, ...]
        print(f"Loading latent file list from: {z_file_list_path}")
        self.z_data_info = torch.load(z_file_list_path)
        print(f"Found {len(self.z_data_info)} total latent samples (all stages).")

        # --- FILTERING LOGIC ---
        if filter_keywords:
            print(f"Filtering dataset for keywords: {filter_keywords}")
            # Filter for "circle" samples
            # self.z_data_info = [
            #     item for item in self.z_data_info 
            #     if filter_keywords in os.path.basename(item['path']).lower() and 'circled-m' not in os.path.basename(item['path']).lower()
            #     and 'hollow' not in os.path.basename(item['path']).lower()
            # ]
            self.z_data_info = [
                item for item in self.z_data_info 
                if any(keyword in os.path.basename(item['path']).lower() for keyword in filter_keywords) 
                and 'circled-m' not in os.path.basename(item['path']).lower()
                and 'hollow' not in os.path.basename(item['path']).lower()
            ]
            
            print(f"Filtered dataset to {len(self.z_data_info)} samples containing 'circle,square, diamond'.")
            # Debug: Print first few filenames
            for i, item in enumerate(self.z_data_info[:5]):
                print(f"  - {os.path.basename(item['path'])}")
            print("----------------------")
            
            if len(self.z_data_info) == 0:
                print("WARNING: Filtered dataset is empty! Check your keywords.")
                sys.exit(1)
            
            # --- DATASET AUGMENTATION (REPETITION) ---
            # The dataset is tiny (~9 samples). We repeat it 100x to ensure
            # each batch has a good mix of colors and gradients are stable.
            print(f"Augmenting dataset by repeating it 100 times...")
            self.z_data_info = self.z_data_info * 10
            print(f"New dataset size: {len(self.z_data_info)}")

        # --- Load pre-calculated mean and std ---
        # if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        #     print(f"ERROR: Statistics files not found ('{mean_path}', '{std_path}').")
        #     print("Please run the `compute_z_stats.py` script first on your new latent dataset.")
        #     sys.exit(1)
            
        # self.z_mean = torch.load(mean_path)
        # self.z_std = torch.load(std_path)
        
        # --- DYNAMIC STATS CALCULATION (Crucial for Overfitting) ---
        print("Calculating stats from filtered dataset (Dynamic) - VALID TOKENS ONLY...")
        all_valid_z = []
        for item in self.z_data_info:
            if '/workspace' in item['path']:
                pass
            else:
                item['path'] = '/workspace/zdataset-patch-tokens/z_latents_data/' + item['path']
            
            z = torch.load(item['path']) # [N, D]
            # Identify valid rows (non-zero)
            # Assuming padding is exactly zero or very close
            is_valid = z.abs().sum(dim=-1) > 1e-6
            valid_z = z[is_valid] # [N_valid, D]
            all_valid_z.append(valid_z)
        
        if all_valid_z:
            all_z_tensor = torch.cat(all_valid_z, dim=0) # [Total_Valid_N, D]
            self.z_mean = all_z_tensor.mean(dim=0)
            self.z_std = all_z_tensor.std(dim=0)
        else:
            # Fallback if no valid tokens found (should not happen)
            print("WARNING: No valid tokens found for stats calculation!")
            self.z_mean = torch.zeros(32)
            self.z_std = torch.ones(32)
        
        # Avoid division by zero
        self.z_std[self.z_std < 1e-6] = 1.0
        
        print(f"Computed Dynamic Stats (Valid Only). Mean: {self.z_mean.mean().item():.4f}, Std: {self.z_std.mean().item():.4f}")
        
        # --- SAVE STATS FOR SAMPLING ---
        print(f"Saving dynamic stats to {mean_path} and {std_path}...")
        torch.save(self.z_mean, mean_path)
        torch.save(self.z_std, std_path)
        print("Stats saved.")

    def __len__(self):
        return len(self.z_data_info)

    def __getitem__(self, idx):
        item_info = self.z_data_info[idx]
        
        # Load the latent tensor on-demand from its individual file
        # This is very memory efficient.
        if '/workspace' in item_info['path']:
            pass
        else:
            item_info['path'] = '/workspace/zdataset-patch-tokens/z_latents_data/' + item_info['path']
        
        z = torch.load(item_info['path'])
        
        # We return the raw z. Normalization will happen in the training loop.
        
        # --- PROMPT STANDARDIZATION ---
        # Override the complex training prompts with simple ones based on filename
        filename = os.path.basename(item_info['path'])
        text = item_info['text']
        
        if "circle" in filename:
            # Extract color from filename (e.g., "orange-circle.pt_step_0.pt")
            if "orange" in filename: text = "a orange circle"
            elif "blue" in filename: text = "a blue circle"
            elif "red" in filename: text = "a red circle"
            elif "green" in filename: text = "a green circle"
            elif "yellow" in filename: text = "a yellow circle"
            elif "purple" in filename: text = "a purple circle"
            elif "black" in filename: text = "a black circle"
            elif "white" in filename: text = "a white circle"
            elif "brown" in filename: text = "a brown circle"
            # Keep original text for unknown colors or if not matching above
        elif "square" in filename:
            if "orange" in filename: text = "a orange square"
            elif "blue" in filename: text = "a blue square"
            elif "red" in filename: text = "a red square"
            elif "green" in filename: text = "a green square"
            elif "yellow" in filename: text = "a yellow square"
            elif "purple" in filename: text = "a purple square"
            elif "black" in filename: text = "a black square"
            elif "white" in filename: text = "a white square"
            elif "brown" in filename: text = "a brown square"
        elif "diamond" in filename:
            if "orange" in filename: text = "a orange diamond"
            elif "blue" in filename: text = "a blue diamond"
            elif "red" in filename: text = "a red diamond"
            elif "green" in filename: text = "a green diamond"
            elif "yellow" in filename: text = "a yellow diamond"
            elif "purple" in filename: text = "a purple diamond"
            elif "black" in filename: text = "a black diamond"
            elif "white" in filename: text = "a white diamond"
            elif "brown" in filename: text = "a brown diamond"
        elif "banana" in filename:
            text = "A ripe yellow banana with the peel pulled back to reveal the fruit. The banana has a green stem and a small brown tip, designed in a simple flat style."
        
        return {'z': z, 'text': text, 'filename': filename}


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, initial_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = min(
            float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
            1.0
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay
        return decayed_lr / initial_lr
    return LambdaLR(optimizer, lr_lambda)

# --- Collate Function for DataLoader ---
# This needs to be a top-level function if num_workers > 0
def collate_fn_zdataset(batch):
    # 'batch' is a list of dicts: [{'z': tensor, 'text': str, 'filename': str}, ...]
    
    # Pad 'z' tensors to the max sequence length in the batch
    # Find max sequence length in this batch
    max_seq_len_batch = max(item['z'].shape[0] for item in batch)
    latent_dim = batch[0]['z'].shape[1] # latent_dim is constant
    
    padded_z_tensors = []
    text_prompts = []
    latent_masks = []
    filenames = []

    for item in batch:
        z_tensor = item['z']
        pad_len = max_seq_len_batch - z_tensor.shape[0]
        # Pad with zeros (or a specific padding value if your model distinguishes it)
        # Assuming zeros correspond to padding in the latent space
        padded_z = F.pad(z_tensor, (0, 0, 0, pad_len), "constant", 0)
        padded_z_tensors.append(padded_z)
        
        # Create mask: 1 for valid, 0 for padding
        # Since VAE zeroes out padding, we detect non-zero rows in the original z_tensor
        # We also handle any additional padding added by the collate_fn (pad_len)
        
        # Check for non-zero rows in the input tensor
        is_valid_row = z_tensor.abs().sum(dim=-1) > 1e-6 # [Seq_Len]
        
        # Append False for any extra padding added by collate
        if pad_len > 0:
            mask = torch.cat([is_valid_row, torch.zeros(pad_len, dtype=torch.bool)])
        else:
            mask = is_valid_row
            
        latent_masks.append(mask)
        
        text_prompts.append(item['text'])
        filenames.append(item['filename'])

    return {
        'z': torch.stack(padded_z_tensors), # Shape: [B, max_seq_len_batch, latent_dim]
        'latent_mask': torch.stack(latent_masks), # Shape: [B, max_seq_len_batch]
        'text': text_prompts, # List of strings
        'filename': filenames # List of strings
    }


# =============================================================================
# Training Configuration & Setup
# =============================================================================
if __name__ == "__main__":

    # --- WandB Initialization ---
    # The config is defined first, then wandb.init is called
    # This allows config to be easily accessible and potentially modified before init
    initial_config = {
        # Training Params
        "learning_rate": 3e-4,
        "total_steps": 10000,
        "batch_size": 8,
        "warmup_steps": 300,
        "lr_decay_min": 1e-6,
        "weight_decay": 1e-4,
        "log_interval": 20,
        "eval_interval": 100,
        "val_split": 0.1,
        "seed": 42,
        "cfg_dropout_prob": 0.1, # Probability to drop out conditional context during training
        "cfg_scale_eval": 7.0,   # Guidance scale for DDIM sampling during evaluation (set to 0 for unconditional)
        "grad_clip_norm": 2.0,

        # Model Params (These will be dynamically inferred/verified)
        "latent_dim": None, # Will be inferred from zDataset
        "num_svg_tokens": None, # Will be inferred (max_seq_len_train from VAE)
        "hidden_dim": 384,        # Increased from 256 to 512
        "context_dim": 768,       # CLIP ViT-B/32 last_hidden_state is 768
        "num_blocks": 12,         # Increased from 1 to 4
        "num_heads": 6,           # Increased from 2 to 8 (512/8 = 64 dim per head)
        "mlp_ratio": 4.0,         # MLP expansion ratio (d_ff = hidden_dim * mlp_ratio)
        "dropout": 0.2,           # Disable dropout for overfitting
        "cfg_dropout_prob": 0.1, # Disable CFG dropout for overfitting

        # Diffusion Params
        "noise_steps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "ddim_eta": 0.0, # 0.0 for deterministic DDIM
        "fp32_attention": True, # Force attention in FP32 to prevent NaN/Inf

        # Data/Paths
        "z_dataset_path": "./z-list-pt/z_latents_file_list.pt",
        "clip_model_path": "/workspace/clip-model-large-vit/clip-vit-large-patch14/", # HuggingFace CLIP model name
        "output_model_dir": "saved_models_vsdit_square_overfit", # New directory for square overfit
        
        # --- SUBSET FILTERING ---
        "filter_keywords": ["circle", "square", "banana"] # Target SPECIFIC single square
    }

    # --- Setup Device ---
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load zDataset ---
    print(f"Loading zDataset using file list from '{initial_config['z_dataset_path']}'...")
    try:
        full_zdataset = zDataset(
            z_file_list_path=initial_config['z_dataset_path'],
            filter_keywords=initial_config.get('filter_keywords') # Pass keywords here
        )
        print(f"Initialized zDataset with {len(full_zdataset)} items.")
    except Exception as e:
        print(f"Error loading/initializing zDataset: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Infer / Verify Model Dimensions from Loaded Data ---
    if full_zdataset:
        sample_z = full_zdataset[0]['z'] # A sample z tensor: [N, latent_dim]
        initial_config['num_svg_tokens'] = sample_z.shape[0] # N
        initial_config['latent_dim'] = sample_z.shape[1]     # latent_dim
        print(f"Inferred num_svg_tokens (N): {initial_config['num_svg_tokens']}, latent_dim: {initial_config['latent_dim']}")
    else:
        print("Error: zDataset is empty. Cannot infer model dimensions."); sys.exit(1)

    # --- Load CLIP Model and Tokenizer for Training ---
    print(f"Loading CLIP model from {initial_config['clip_model_path']}...")
    try:
        clip_model = CLIPTextModel.from_pretrained(initial_config['clip_model_path']).to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained(initial_config['clip_model_path'])
        clip_model.eval()
        # Verify config context_dim matches CLIP LAST_HIDDEN_STATE dimension
        clip_output_dim = clip_model.config.hidden_size
        if initial_config['context_dim'] != clip_output_dim:
            print(f"FATAL ERROR: Config context_dim ({initial_config['context_dim']}) != CLIP last_hidden_state dim ({clip_output_dim}).")
            sys.exit(1)
        else:
            print(f"CLIP last_hidden_state dim ({clip_output_dim}) matches config.context_dim.")
    except Exception as e:
        print(f"Error loading CLIP: {e}"); traceback.print_exc(); sys.exit(1)
    
    # Now that all dimensions are inferred, initialize wandb config
    run = wandb.init(
        project="vp-dit-training-subset", # Changed project name
        config=initial_config
    )
    config = wandb.config # Access config through wandb.config for consistency
    run_name = run.name if run else "local-run"
    set_seed(config.seed)

    # Log source code to WandB
    try:
        code_artifact = wandb.Artifact(name=f'source-code-{run_name}', type='code',
                                       description='Source code for VP-DIT training run',
                                       metadata=dict(config))
        # Add relevant script files
        code_artifact.add_file('./models/vp_dit_v1.py')
        code_artifact.add_file('./models/vp_dit_training_subset.py') # Log this file
        # Add other relevant scripts if needed (e.g., dataset_preparation_dynamic.py, prepare_latents.py)
        run.log_artifact(code_artifact)
        print("Logged source code as WandB artifact.")
    except Exception as e:
        print(f"Warning: Failed to log code artifact: {e}")

    # --- Train/Validation Split ---
    #val_size = int(config.val_split * len(full_zdataset))
    
    # Handle single-sample overfitting case
    if len(full_zdataset) < 3:
        print(f"Dataset too small ({len(full_zdataset)}) for split. Using full dataset for BOTH train and val (Overfitting Mode).")
        train_dataset = full_zdataset
        val_dataset = full_zdataset
    else:
        #val_size = config.val_split # Keep user's hardcoded value for larger datasets
        val_size = int(config.val_split * len(full_zdataset))
        #val_size = 3
        # if val_size >= len(full_zdataset):
        #      val_size = int(0.2 * len(full_zdataset)) # Fallback if 3 is too big
        
        train_size = len(full_zdataset) - val_size
        train_dataset, val_dataset = random_split(full_zdataset, [train_size, val_size])
        
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset) if val_dataset else 0}")

    # --- Create DataLoaders ---
    # Using num_workers=0 to avoid multiprocessing pickle errors.
    # If your environment/imports are fully pickle-compatible, you can try num_workers > 0.
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn_zdataset)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn_zdataset) if val_dataset else None

    # --- Initialize VS_DiT Model ---
    print("Initializing VS-DiT model (Sequence Conditioning)...")
    vpdit_model = VS_DiT(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        context_dim=config.context_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        fp32_attention=config.fp32_attention
    )
    print(f"VS_DiT Model Parameters: {sum(p.numel() for p in vpdit_model.parameters() if p.requires_grad) / 1e6:.2f} M")

    #vpdit_model.initialize_weights() # Apply custom weight initialization

    # --- Setup Diffusion Parameters ---
    print("Setting up diffusion parameters...")
    betas = get_linear_noise_schedule(config.noise_steps)
    diff_params = precompute_diffusion_parameters(betas, device) # Precompute on device

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(vpdit_model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay, betas=(0.9, 0.999))
    print("Using custom cosine_warmup_scheduler.")
    scheduler = cosine_warmup_scheduler(
                    optimizer,
                    warmup_steps=config.warmup_steps, # Use original warmup steps
                    total_steps=config.total_steps,
                    initial_lr=config.learning_rate,
                    min_lr=config.lr_decay_min
                )

    # --- Watch model with WandB ---
    if run: wandb.watch(vpdit_model, log="gradients", log_freq=config.log_interval * 50)

    # --- Training Loop Function ---
    def train_loop(model, config, train_loader, val_loader, optimizer, scheduler, clip_model, clip_tokenizer, target_device):
        print("\nStarting VS-DiT Training (Sequence Conditioning with Empty String Uncond)...")
        
        model.to(target_device); model.train()
        criterion = nn.MSELoss()
        best_eval_loss = float('inf')
        os.makedirs(config.output_model_dir, exist_ok=True)
        current_run_name = wandb.run.name if wandb.run else "local_run"
        best_model_path = os.path.join(config.output_model_dir, f"vsdit_clip_seqcond_{current_run_name}_best.pth")

        global_step = 0
        pbar = tqdm(total=config.total_steps, desc="Training Steps")
        running_train_loss = 0.0
        train_iterator = iter(train_loader)

        # --- Pre-calculate empty string embedding ONCE for efficiency ---
        with torch.no_grad():
            clip_model.eval()
            # Tokenize a batch of empty strings to get consistent sequence length for unconditional
            # The max_length should be determined by what CLIP tokenizer produces for typical prompts.
            # Using a sensible default or deriving from a sample prompt is good.
            # Here, we'll try to determine it dynamically from the first batch of conditional contexts.
            
            # For now, let's make a reasonable assumption for CLIP-ViT-B/32
            max_text_seq_len_for_uncond = 77 # Common CLIP max_length
            
            empty_text_inputs = clip_tokenizer(
                [""] * config.batch_size, # Batch of empty strings
                padding='max_length',
                max_length=max_text_seq_len_for_uncond,
                truncation=True,
                return_tensors="pt"
            ).to(target_device)
            empty_outputs = clip_model(**empty_text_inputs)
            uncond_context_template = empty_outputs.last_hidden_state # [B, S_uncond, D_ctx]
            uncond_mask_template = ~(empty_text_inputs.attention_mask.bool()) # [B, S_uncond]
        # --- End Pre-calculation ---


        while global_step < config.total_steps:
            model.train()
            try: batch = next(train_iterator)
            except StopIteration: train_iterator = iter(train_loader); batch = next(train_iterator)
            except Exception as e: 
                print(f"Dataloader error at step {global_step}: {e}"); 
                traceback.print_exc()
                global_step += 1 # Increment step to avoid infinite loop on persistent error
                pbar.update(1)
                continue

            ### applying normalization to z
            # 1. Load the raw data from the batch
            z0_raw = batch['z'].to(target_device)
            
            # Handle Subset wrapper from random_split
            if hasattr(train_loader.dataset, 'dataset'):
                z_mean = train_loader.dataset.dataset.z_mean.to(target_device)
                z_std = train_loader.dataset.dataset.z_std.to(target_device)
            else:
                z_mean = train_loader.dataset.z_mean.to(target_device)
                z_std = train_loader.dataset.z_std.to(target_device)
            
            z0_batch = (z0_raw - z_mean) / z_std
            
            #latent_mask = batch['latent_mask'].to(target_device) # True = Valid, False = Pad
            #z0_batch[~latent_mask] = 0.0
            
            # DEBUG: Verify z0_batch stats
            if global_step == 0:
                print(f"DEBUG: z0_batch shape: {z0_batch.shape}")
                print(f"DEBUG: z0_batch mean: {z0_batch.mean().item():.4f}, std: {z0_batch.std().item():.4f}")
                print(f"DEBUG: z0_batch norm: {z0_batch.norm().item():.4f}")

            text_prompts = batch['text'] # List of strings
            current_batch_size = z0_batch.shape[0]
            
            # DATA AUGMENTATION: Add small noise to create diversity
            # Since we only have 9 unique samples repeated 100x, adding noise creates
            # 900 truly different training examples instead of 9 memorized patterns.
            # This helps the model learn robust color->latent mappings.
            aug_noise = torch.randn_like(z0_batch) * 0.05  # 5% noise
            #z0_batch[latent_mask] = z0_batch[latent_mask] + aug_noise[latent_mask] 
            z0_batch = z0_batch + aug_noise

            optimizer.zero_grad()

            # 1. Get Conditional Text Embeddings (last_hidden_state) and Mask
            with torch.no_grad():
                clip_model.eval() # Ensure CLIP model is in eval mode
                text_inputs = clip_tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                text_outputs = clip_model(**text_inputs)
                clip_outputs = clip_model(**text_inputs)
                context_final = clip_outputs.last_hidden_state
                pooled_context = clip_outputs.pooler_output # [B, D_clip]
                mask_final = ~(text_inputs.attention_mask.bool())

            # 3. Diffusion Process
            # Use uniform timestep sampling for balanced learning
            t = torch.randint(0, config.noise_steps, (current_batch_size,), device=target_device).long()
            zt_batch, noise_batch = noise_latent(z0_batch, t, diff_params, target_device)




            # Get latent mask (True = Valid)
            latent_mask = batch['latent_mask'].to(target_device) # [B, N]
            latent_padding_mask = ~latent_mask # (True = Padding)

            # 4. Forward pass with mixed context
            # CRITICAL FIX: Disable Attention Masking
            # We want the model to attend to the entire sequence (including padding)
            # because during sampling, we don't know the length and don't mask.
            # This ensures training matches sampling.
            predicted_noise = model(zt_batch, t, context_final, pooled_context, mask_final, latent_mask=None)

            # 5. Calculate loss breakdown
            loss_unreduced = F.mse_loss(predicted_noise, noise_batch, reduction='none') # [B, N, D]
            
            # Create masks
            latent_mask_expanded = latent_mask.unsqueeze(-1).expand_as(loss_unreduced)
            padding_mask_expanded = ~latent_mask_expanded
            
            # Calculate separate losses
            loss_valid = (loss_unreduced * latent_mask_expanded.float()).sum() / (latent_mask_expanded.sum() + 1e-6)
            loss_padding = (loss_unreduced * padding_mask_expanded.float()).sum() / (padding_mask_expanded.sum() + 1e-6)
            
            # Weighted Loss (Optional: can tune this)
            # For now, just sum them or use mean, but LOG them separately
            # loss = loss_unreduced.mean()
            
            # CRITICAL FIX: Balanced Loss
            # We treat "learning the shape" and "learning the padding" as equal tasks
            loss = loss_valid + loss_padding
            #loss = loss_valid

            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss Total={loss.item():.6f} | Valid={loss_valid.item():.6f} | Pad={loss_padding.item():.6f}")

            if torch.isnan(loss).any() or torch.isinf(loss).any(): 
                print(f"NaN/Inf loss at step {global_step}. Skipping."); 
                optimizer.zero_grad(); 
                global_step += 1
                pbar.update(1)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            # LR Logic
            current_lr = optimizer.param_groups[0]['lr'] # Get current LR before optimizer step for logging

            optimizer.step()
            scheduler.step() # Scheduler updates after optimizer step
            current_lr = scheduler.get_last_lr()[0] # Get the new LR after scheduler step

            # Logging
            running_train_loss += loss.item()
            if global_step % config.log_interval == 0:
                avg_train_loss_interval = running_train_loss / config.log_interval
                
                # Get text conditioning scale value
                text_scale = model.text_conditioning_scale.item()
                
                print(f"Step {global_step}/{config.total_steps} | Train Loss: {avg_train_loss_interval:.4f} | "
                      f"Valid Loss: {loss_valid.item():.4f} | Pad Loss: {loss_padding.item():.4f} | "
                      f"LR: {current_lr:.2e} | Text Scale: {text_scale:.3f}")
                if run:
                    wandb.log({
                        "train/loss": avg_train_loss_interval,
                        "train/loss_valid": loss_valid.item(),
                        "train/loss_padding": loss_padding.item(),
                        "train/lr": current_lr,
                        "train/text_conditioning_scale": text_scale
                    }, step=global_step)
                running_train_loss = 0.0

            # Evaluation
            if (global_step % config.eval_interval == 0 and global_step > 0) or (global_step == config.total_steps - 1):
                if val_loader is None: 
                    print(f"Skipping eval step {global_step} - No validation data.")
                else:
                    print(f"\n--- Evaluating at step {global_step} ---")
                    model.eval(); total_eval_loss = 0.0; num_eval_batches = 0
                    with torch.no_grad():
                        for eval_batch in tqdm(val_loader, desc=f"Step {global_step} Eval", leave=False):
                            z0_eval = eval_batch['z'].to(target_device)
                            text_eval_prompts = eval_batch['text']
                            eval_batch_size = z0_eval.shape[0]

                            eval_text_inputs = clip_tokenizer(text_eval_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                            clip_eval_outputs = clip_model(**eval_text_inputs)
                            eval_context_seq = clip_eval_outputs.last_hidden_state
                            eval_pooled_context = clip_eval_outputs.pooler_output # [B, D_clip]
                            eval_mask = ~(eval_text_inputs.attention_mask.bool())
                            
                            # 3. Sample Timesteps (Uniform)
                            t_eval = torch.randint(0, config.noise_steps, (eval_batch_size,), device=target_device).long()
                            
                            # 4. Forward pass
                            zt_eval, noise_eval = noise_latent(z0_eval, t_eval, diff_params, target_device)
                            
                            # Get latent mask (True = Valid)
                            latent_mask_eval = eval_batch['latent_mask'].to(target_device) # [B, N]
                            latent_padding_mask_eval = ~latent_mask_eval
                            
                            # Pass mask to model
                            # CRITICAL FIX: Disable Attention Masking in Eval too!
                            # Match Training behavior (where we pass None)
                            predicted_noise_eval = model(zt_eval, t_eval, eval_context_seq, eval_pooled_context, eval_mask, latent_mask=None)
                            
                            # Calculate Masked Eval Loss
                            loss_unreduced_eval = F.mse_loss(predicted_noise_eval, noise_eval, reduction='none')
                            latent_mask_expanded_eval = latent_mask_eval.unsqueeze(-1).expand_as(loss_unreduced_eval)
                            
                            # Zero out loss for padding
                            loss_masked_eval = loss_unreduced_eval * latent_mask_expanded_eval.float()
                            
                            # Average over valid elements
                            num_valid_elements_eval = latent_mask_expanded_eval.sum()
                            
                            if num_valid_elements_eval > 0:
                                eval_loss = loss_masked_eval.sum() / num_valid_elements_eval
                            else:
                                eval_loss = torch.tensor(0.0, device=target_device)

                            if not torch.isnan(eval_loss).any() and not torch.isinf(eval_loss).any():
                                total_eval_loss += eval_loss.item(); num_eval_batches += 1
                    avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else float('inf')
                    print(f"Avg Eval Loss: {avg_eval_loss:.4f}")
                    if run: wandb.log({"eval/avg_loss": avg_eval_loss}, step=global_step)

                    # --- Save Best Model ---
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        torch.save(model.state_dict(), best_model_path)
                        print(f"✨ New best model saved with Eval Loss: {best_eval_loss:.4f} to {best_model_path} ✨")
                        if run:
                            try:
                                artifact = wandb.Artifact(f'model-{run_name}-best', type='model')
                                artifact.add_file(best_model_path)
                                run.log_artifact(artifact)
                                print("Logged best model as WandB artifact.")
                            except Exception as e:
                                print(f"Warning: Failed to log best model artifact: {e}")

            model.train() # Ensure model is back in train mode
            global_step += 1
            pbar.update(1)
            if global_step >= config.total_steps: break # Break if total steps reached

        pbar.close()
        print("Training finished!"); print(f"Best eval loss: {best_eval_loss:.4f}")
        if os.path.exists(best_model_path): print(f"Best model saved at: {best_model_path}")
        else: print("No best model was saved.")
        if run: wandb.finish()


    # --- Run Training ---
    train_loop(
        model=vpdit_model, config=config, train_loader=train_dataloader,
        val_loader=val_dataloader, optimizer=optimizer, scheduler=scheduler,
        clip_model=clip_model, clip_tokenizer=clip_tokenizer, target_device=device
    )
