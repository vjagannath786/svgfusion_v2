# --- START OF FILE vp_dit.py --- # Renamed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR # Using transformers version
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import math
import random
import wandb
import os
import sys
sys.path.append(".")

# --- Import Model and Utils from test_vsdit_hidden_seqcond.py ---
# Make sure the correct file with sequence conditioning model is imported
try:
    # --- MODIFIED: Import from the sequence conditioning version ---
    from vp_dit import VS_DiT, TimestepEmbedder, MLP, VS_DiT_Block
    from vp_dit import get_linear_noise_schedule, precompute_diffusion_parameters, noise_latent, ddim_sample
    print("Successfully imported from test_vsdit_hidden_seqcond.py")
except ImportError as e:
    print(f"Error importing from test_vsdit_hidden_seqcond.py: {e}")
    print("Please ensure test_vsdit_hidden_seqcond.py is in the correct path.")
    exit()
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     exit()

# --- Set Seed Utility (Unchanged) ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)
        except AttributeError: pass

import json

# --- Load Captions (Unchanged) ---
caption_file = './svg_captions.json'
try:
    with open(caption_file, 'r') as f:
        svg_captions = json.load(f)
    print(f"Loaded captions from {caption_file}")
except FileNotFoundError:
    print(f"Error: Caption file not found at {caption_file}"); exit()
except Exception as e:
     print(f"Error loading captions: {e}"); exit()




# --- zDataset Definition (Unchanged, still returns text) ---
class zDataset(Dataset):
    def __init__(self, z_data_list, captions_dict):
        self.z_data = z_data_list
        all_z = torch.stack([torch.tensor(item['z'], dtype=torch.float32) for item in z_data_list])
        self.z_mean = all_z.mean(0, keepdim=True)
        self.z_std = all_z.std(0, keepdim=True)
        self.z_std = torch.where(self.z_std < 1e-8, torch.tensor(1e-8), self.z_std)
        print(f"zDataset stats: Mean shape {self.z_mean.shape}, Std shape {self.z_std.shape}")

    def __len__(self): return len(self.z_data)

    def __getitem__(self, idx):
        item = self.z_data[idx]
        filename_key = item['filename'].split(".svg")[0]
        text = svg_captions.get(filename_key, filename_key)
        z = torch.tensor(item['z'], dtype=torch.float32)
        #if 'purple' in text:
        #    print(filename_key)
        #z_normalized = (z - self.z_mean) / self.z_std
        return {'z': z, 'text': f"{filename_key}-{text}", 'filename': filename_key} # Return normalized z



def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, initial_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))  # Warmup
        # Ensure progress never exceeds 1.0 and spans full decay phase
        progress = min(
            float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
            1.0  # Hard clamp to prevent overshooting
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine decay
        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay  # No division by initial_lr
        return decayed_lr / initial_lr  # Normalize to [0, 1] for LambdaLR
    return LambdaLR(optimizer, lr_lambda)
# =============================================================================
# Training Configuration & Setup
# =============================================================================
if __name__ == "__main__":

    # --- WandB Initialization ---
    run = wandb.init(
        project="vp-dit-training", # New project name
        config={
            # Training Params
            "learning_rate": 2e-4, # Adjusted as per paper text
            "total_steps": 10000,  # Increased training steps
            "batch_size": 32,     # Adjusted batch size
            "warmup_steps": 200, # Adjusted warmup
            "lr_decay_min": 2e-6,# Adjusted as per paper text
            "weight_decay": 0.1,   # Adjusted as per paper text
            "log_interval": 10,
            "eval_interval": 100, # Evaluate less frequently for longer runs
            "val_split": 0.05,
            "seed": 42,
            "cfg_dropout_prob": 0.1,
            "cfg_scale_eval": 0.0,
            "grad_clip_norm": 2.0, # Adjusted as per paper text

            # Model Params (Using test_vsdit_hidden_seqcond)
            "latent_dim": 128,
            "hidden_dim": 384,        # Internal DiT dimension (like VP-DiT S)
            # --- MODIFIED: context_dim must match CLIP LAST_HIDDEN_STATE dim ---
            "context_dim": 768,       # CLIP ViT-B/32 last_hidden_state is 768
            "num_blocks": 12,         # Like VP-DiT S
            "num_heads": 6,           # hidden_dim=384 divisible by 6
            "mlp_ratio": 8.0,         # Using deeper MLP ratio from discussion
            "dropout": 0.1,

            # Diffusion Params
            "noise_steps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "ddim_eta": 0.0,

            # Data/Paths
            "z_dataset_path": "./datasets/zdataset_vpvae_ce.pt",
            "clip_model_path": "/Users/varun_jagannath/Documents/D/test python/clip-vit-large-patch14", # Using Large CLIP
            "output_model_dir": "saved_models_vsdit_clip_seqcond" # New directory
        }
    )
    config = wandb.config
    run_name = run.name if run else "local-run"
    set_seed(config.seed)


    try:
        code_artifact = wandb.Artifact(name=f'source-code-{run_name}', type='code',
                                       description='Source code for VP-DIT training run on last hidden state',
                                       metadata=dict(config))
        # Add this script specifically
        code_artifact.add_file('vp_dit.py') # Make sure filename matches
        code_artifact.add_file('vp_dit_training.py') # Ensure this is the correct file
        #code_artifact.add_file('generate_vsdit_svg_last.py')
        # Add other relevant scripts if needed (e.g., evaluation script)
        # if os.path.exists('evaluate_vp_vae_recon_discrete_v3.py'):
        #     code_artifact.add_file('evaluate_vp_vae_recon_discrete_v3.py')
        run.log_artifact(code_artifact)
        print("Logged source code as WandB artifact.")
    except Exception as e:
        print(f"Warning: Failed to log code artifact: {e}")





    # --- Setup Device ---
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load zDataset ---
    print(f"Loading zDataset from '{config.z_dataset_path}'...")
    try:
        loaded_z_data_list = torch.load(config.z_dataset_path)
        if isinstance(loaded_z_data_list, list):
            full_zdataset = zDataset(loaded_z_data_list, svg_captions)
        elif isinstance(loaded_z_data_list, zDataset):
             print("Warning: Loaded existing zDataset object.")
             full_zdataset = loaded_z_data_list
        else: raise ValueError("Unexpected dataset format.")
        print(f"Initialized zDataset with {len(full_zdataset)} items.")
    except FileNotFoundError: print(f"Error: zDataset file not found: {config.z_dataset_path}"); exit()
    except Exception as e: print(f"Error loading/initializing zDataset: {e}"); exit()



    # subset_keywords = [
    #     "basketball", "bird","umbrella","1st-place-medal","anchor",
    #     "blue-circle", "blue-square", "blue-heart", "brown-circle",
    #     "brown-heart", "brown-square", "banana","crescent-moon","dango","downcast-face",
    #     "egg","drop-of-blood","droplet","face-without-mouth","large-orange-diamond",
    #     "large-blue-diamond","lemon","locked","melon","megaphone","neutral-face"
    # ]

    #subset_keywords = ["square","circle","diamond","droplet"]


    # --- 3. Filter the Raw Data ---
    # subset_data_list = []
    # filenames_in_subset = set() # Keep track to avoid duplicates if keywords overlap
    # for item in tqdm(loaded_z_data_list):
    #     #print(item)
    #     filename_key = item['filename']
    #     # Check if any keyword is in the filename
    #     if any(keyword in filename_key for keyword in subset_keywords):
    #          if filename_key not in filenames_in_subset:
    #               subset_data_list.append(item)
    #               filenames_in_subset.add(filename_key)

    # print(f"Created subset with {len(subset_data_list)} items.")

    # if not subset_data_list:
    #     print("Error: Subset is empty! Check keywords or dataset content.")
    #     exit()

    # subset_dataset = zDataset(subset_data_list, svg_captions)

    # full_zdataset = subset_dataset

    # print("done creating subset dataset")





    # --- Train/Validation Split ---
    val_size = int(config.val_split * len(full_zdataset))
    #val_size = 1
    train_size = len(full_zdataset) - val_size
    #train_size = 1
    if val_size == 0 and len(full_zdataset) > 0:
        print("Warning: Validation split is 0."); train_dataset = full_zdataset; val_dataset = None
    elif val_size > 0: train_dataset, val_dataset = random_split(full_zdataset, [train_size, val_size])
    else: print("Error: No data loaded."); exit()
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset) if val_dataset else 0}")

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False) if val_dataset else None

    # --- Load CLIP ---
    print(f"Loading CLIP model from {config.clip_model_path}...")
    try:
        clip_model = CLIPTextModel.from_pretrained(config.clip_model_path).to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_path)
        clip_model.eval()
        # --- MODIFIED: Verify config context_dim matches CLIP LAST_HIDDEN_STATE dimension ---
        clip_output_dim = clip_model.config.hidden_size # Use hidden_size for last_hidden_state
        if config.context_dim != clip_output_dim:
            print(f"FATAL ERROR: Config context_dim ({config.context_dim}) != CLIP last_hidden_state dim ({clip_output_dim}).")
            exit()
        else: print(f"CLIP last_hidden_state dim ({clip_output_dim}) matches config.context_dim.")
    except Exception as e: print(f"Error loading CLIP: {e}"); exit()

    # --- Initialize VS_DiT Model (from test_vsdit_hidden_seqcond) ---
    print("Initializing VS-DiT model (Sequence Conditioning)...")
    # --- MODIFIED: Ensure using the correct VS_DiT class import ---
    vpdit_model = VS_DiT(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        context_dim=config.context_dim, # Should match CLIP last_hidden_state dim
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout
    )
    print(f"VS_DiT Model Parameters: {sum(p.numel() for p in vpdit_model.parameters() if p.requires_grad) / 1e6:.2f} M")

    vpdit_model.initialize_weights()

    # --- Setup Diffusion Parameters ---
    print("Setting up diffusion parameters...")
    betas_cpu = get_linear_noise_schedule(config.noise_steps)
    # diff_params created inside train_loop

    # --- Optimizer & Scheduler ---
    # Using AdamW params from paper
    optimizer = optim.AdamW(vpdit_model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay, betas=(0.9, 0.95))
    print("Using transformers scheduler with warmup and cosine decay.")
    # Using transformers scheduler which handles warmup internally
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.warmup_steps,
    #     num_training_steps=config.total_steps # Needs total steps
    # )
    steps_after_warmup = config["total_steps"] - config["warmup_steps"]
    #scheduler = CosineAnnealingLR(optimizer, T_max=steps_after_warmup, eta_min=config.lr_decay_min)
    scheduler = cosine_warmup_scheduler(
                    optimizer,
                    warmup_steps=config["warmup_steps"] ,
                    total_steps=config["total_steps"] ,
                    initial_lr=config["learning_rate"],
                    min_lr=config["lr_decay_min"]
                )
    # LR decay min is handled by the scheduler's cosine curve ending near 0

    # --- Watch model with WandB ---
    if run: wandb.watch(vpdit_model, log="gradients", log_freq=config.log_interval * 50) # Log less often

    # --- Training Function (MODIFIED for Sequence Context) ---
    def train_loop_v0(model, config, train_loader, val_loader, optimizer, scheduler, clip_model, clip_tokenizer, target_device):
        print("\nStarting VS-DiT Training (Sequence Conditioning)...")
        betas_cpu = get_linear_noise_schedule(config.noise_steps)
        diff_params = precompute_diffusion_parameters(betas_cpu, target_device)
        model.to(target_device); model.train()
        criterion = nn.MSELoss()
        best_eval_loss = float('inf')
        os.makedirs(config.output_model_dir, exist_ok=True)
        best_model_path = os.path.join(config.output_model_dir, f"vsdit_clip_seqcond_{run_name}_best.pth") # New name

        global_step = 0
        pbar = tqdm(total=config.total_steps, desc="Training Steps")
        running_train_loss = 0.0
        train_iterator = iter(train_loader)

        while global_step < config.total_steps:
            model.train()
            try: batch = next(train_iterator)
            except StopIteration: train_iterator = iter(train_loader); batch = next(train_iterator)
            except Exception as e: print(f"Dataloader error: {e}"); continue

            z0_batch = batch['z'].to(target_device)
            text_prompts = batch['text']
            optimizer.zero_grad()

            # --- MODIFIED: Get Text Embeddings (last_hidden_state) and Mask ---
            with torch.no_grad():
                clip_model.eval()
                # Tokenize text
                text_inputs = clip_tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                # Get CLIP outputs
                text_outputs = clip_model(**text_inputs)
                # <<< USE LAST HIDDEN STATE >>>
                context_seq = text_outputs.last_hidden_state # Shape: [B, S, context_dim]
                # <<< GET PADDING MASK >>> (True where padded)
                context_padding_mask = ~(text_inputs.attention_mask.bool()) # Invert attention mask

            t = torch.randint(0, config.noise_steps, (z0_batch.shape[0],), device=target_device).long()
            zt_batch, noise_batch = noise_latent(z0_batch, t, diff_params, target_device)

            # --- MODIFIED: Classifier-Free Guidance for Sequence ---
            # context_final = context_seq.clone() # Use clone to avoid modifying original
            # mask_final = context_padding_mask.clone() if context_padding_mask is not None else None
            # for i in range(z0_batch.shape[0]):
            #     if random.random() < config.cfg_dropout_prob:
            #         context_final[i] = torch.zeros_like(context_final[i])
            #         if mask_final is not None:
            #             # Mask should be all False (valid) for zero embedding
            #             mask_final[i] = torch.zeros_like(mask_final[i], dtype=torch.bool)


            # Get empty string embedding for unconditional
            with torch.no_grad():

                empty_text = clip_tokenizer(
                    [""] * z0_batch.shape[0],  # Batch of empty strings
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(target_device)
                
                empty_outputs = clip_model(**empty_text)
                uncond_context = empty_outputs.last_hidden_state
                uncond_mask = ~(empty_text.attention_mask.bool())

                # Create mask for CFG dropout
                cfg_mask = (torch.rand(z0_batch.shape[0], device=target_device) > config.cfg_dropout_prob)
                
                # Reshape masks for proper broadcasting
                cfg_mask_3d = cfg_mask.reshape(-1, 1, 1).expand(-1, context_seq.size(1), context_seq.size(2))
                cfg_mask_2d = cfg_mask.reshape(-1, 1).expand(-1, context_seq.size(1))

                # Ensure uncond_context matches the shape of context_seq
                if uncond_context.shape != context_seq.shape:
                    uncond_context = uncond_context.expand_as(context_seq)

                # Apply torch.where
                context_final = torch.where(cfg_mask_3d, context_seq, uncond_context)

                # Mix conditional and unconditional using the mask
                #context_final = torch.where(cfg_mask_3d, context_seq, uncond_context)
                mask_final = torch.where(cfg_mask_2d, context_padding_mask, uncond_mask)

            # --- MODIFIED: Forward pass with sequence context and mask ---
            predicted_noise = model(zt_batch, t, context_final, mask_final)

            loss = criterion(predicted_noise, noise_batch)
            if torch.isnan(loss).any(): print(f"NaN loss at step {global_step}. Skipping."); optimizer.zero_grad(); continue

            loss.backward()
            # Use grad clip norm from config
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            # === LR Logic ===
            current_lr = 0
            if global_step < config["warmup_steps"]:
                lr_scale = float(global_step + 1) / float(config["warmup_steps"])
                current_lr = config["learning_rate"] * lr_scale
                for param_group in optimizer.param_groups: param_group['lr'] = current_lr
            elif global_step == config["warmup_steps"]: print(f"Warmup finished at step {global_step}. Starting cosine scheduler.")
            
            
            optimizer.step()
            if global_step >= config["warmup_steps"]:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]


            #scheduler.step() # Step scheduler after optimizer

            # --- Logging ---
            running_train_loss += loss.item()
            if global_step % config.log_interval == 0:
                 avg_train_loss_interval = running_train_loss / config.log_interval
                 #current_lr = scheduler.get_last_lr()[0]
                 pbar.set_postfix({"Step": global_step, "Loss": f"{avg_train_loss_interval:.4f}", "LR": f"{current_lr:.2e}"})
                 if run:
                      wandb.log({ "step": global_step, "train/loss": loss.item(),
                                  "train/avg_loss_interval": avg_train_loss_interval,
                                  "learning_rate": current_lr }, step=global_step)
                 running_train_loss = 0.0

            # --- Evaluation ---
            if (global_step % config.eval_interval == 0 and global_step > 0) or (global_step == config.total_steps - 1):
                if val_loader is None: print(f"Skipping eval step {global_step} - No validation data.")
                else:
                    print(f"\n--- Evaluating at step {global_step} ---")
                    model.eval(); total_eval_loss = 0.0; num_eval_batches = 0
                    with torch.no_grad():
                        for eval_batch in tqdm(val_loader, desc=f"Step {global_step} Eval", leave=False):
                            z0_eval = eval_batch['z'].to(target_device)
                            text_eval_prompts = eval_batch['text']

                            # --- MODIFIED: Get eval context (last_hidden_state) and mask ---
                            eval_text_inputs = clip_tokenizer(text_eval_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                            eval_context_seq = clip_model(**eval_text_inputs).last_hidden_state
                            eval_mask = ~(eval_text_inputs.attention_mask.bool())

                            t_eval = torch.randint(0, config.noise_steps, (z0_eval.shape[0],), device=target_device).long()
                            zt_eval, noise_eval = noise_latent(z0_eval, t_eval, diff_params, target_device)
                            # --- MODIFIED: Eval forward pass ---
                            predicted_noise_eval = model(zt_eval, t_eval, eval_context_seq, eval_mask)
                            eval_loss = criterion(predicted_noise_eval, noise_eval)

                            if not torch.isnan(eval_loss).any():
                                total_eval_loss += eval_loss.item(); num_eval_batches += 1

                    avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else float('inf')
                    print(f"Avg Eval Loss: {avg_eval_loss:.4f}")
                    if run: wandb.log({"eval/avg_loss": avg_eval_loss}, step=global_step)

                    # --- Save Best Model ---
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        torch.save(model.state_dict(), best_model_path)
                        print(f"✨ New best model saved with Eval Loss: {best_eval_loss:.4f} to {best_model_path} ✨")
                        # if run:
                        #     artifact = wandb.Artifact(f'model-{run_name}-best', type='model')
                        #     artifact.add_file(best_model_path)
                        #     run.log_artifact(artifact)

                    # --- Run DDIM Sampling ---
                    # print("Running DDIM sampling (example)...")
                    # num_samples = min(4, config.batch_size)
                    # sample_prompts = ["a smiling emoji face", "a red heart shape", "an arrow pointing right", "a simple house outline"][:num_samples]
                    # # --- MODIFIED: Get sequence context/mask for sampling ---
                    # sample_text_inputs = clip_tokenizer(sample_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                    # sample_context_seq = clip_model(**sample_text_inputs).last_hidden_state
                    # sample_mask = ~(sample_text_inputs.attention_mask.bool())

                    # # --- MODIFIED: Call ddim_sample with sequence context ---
                    # generated_z0 = ddim_sample(
                    #     model=model, shape=(num_samples, config.latent_dim),
                    #     context_seq=sample_context_seq, context_padding_mask=sample_mask,
                    #     diff_params=diff_params, num_timesteps=config.noise_steps,
                    #     target_device=target_device, cfg_scale=config.cfg_scale_eval,
                    #     eta=config.ddim_eta
                    # )
                    # print(f"Generated z0 shape: {generated_z0.shape}")
                    # print(f"Generated z0 mean: {generated_z0.mean().item():.4f}, std: {generated_z0.std().item():.4f}")
                    # print("--- Evaluation complete ---")

            model.train() # Ensure model is back in train mode
            global_step += 1
            pbar.update(1)
            if global_step >= config.total_steps: break

        pbar.close()
        print("Training finished!"); print(f"Best eval loss: {best_eval_loss:.4f}")
        if os.path.exists(best_model_path): print(f"Best model saved at: {best_model_path}")
        else: print("No best model was saved.")
        if run: wandb.finish()

    def train_loop(model, config, train_loader, val_loader, optimizer, scheduler, clip_model, clip_tokenizer, target_device):
        print("\nStarting VS-DiT Training (Sequence Conditioning with Empty String Uncond)...")
        betas_cpu = get_linear_noise_schedule(config.noise_steps)
        diff_params = precompute_diffusion_parameters(betas_cpu, target_device)

        model.to(target_device); model.train()
        criterion = nn.MSELoss()
        best_eval_loss = float('inf')
        os.makedirs(config.output_model_dir, exist_ok=True)
        # Make sure run_name is defined if wandb is used, or provide a default
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
            # Use a placeholder text to determine max_length based on tokenizer's usual output for typical prompts
            # This ensures uncond_context_seq has a compatible shape for broadcasting/stacking later
            placeholder_texts = ["a"] * config.batch_size # Use batch_size for placeholder
            max_len_for_uncond = clip_tokenizer(placeholder_texts, padding='max_length', truncation=True, return_tensors="pt").input_ids.shape[1]

            empty_text_inputs = clip_tokenizer(
                [""] * config.batch_size, # Batch of empty strings
                padding='max_length',
                max_length=max_len_for_uncond, # Pad to a consistent length
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
            except Exception as e: print(f"Dataloader error: {e}"); continue

            z0_batch = batch['z'].to(target_device) # Normalized z
            text_prompts = batch['text'] # List of strings
            current_batch_size = z0_batch.shape[0] # Get actual batch size, might be smaller for last batch
            optimizer.zero_grad()

            # 1. Get Conditional Text Embeddings (last_hidden_state) and Mask
            with torch.no_grad():
                clip_model.eval()
                text_inputs = clip_tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                text_outputs = clip_model(**text_inputs)
                cond_context_seq = text_outputs.last_hidden_state # Shape: [B, S_cond, D_ctx]
                cond_padding_mask = ~(text_inputs.attention_mask.bool()) # Shape: [B, S_cond]

            # 2. Prepare context for DiT (Classifier-Free Guidance dropout)
            context_final_list = []
            mask_final_list = []

            # Get the pre-calculated unconditional context and mask for the current batch size
            current_uncond_context = uncond_context_template[:current_batch_size, :cond_context_seq.shape[1], :] # Slice to match cond_context_seq length
            current_uncond_mask = uncond_mask_template[:current_batch_size, :cond_context_seq.shape[1]]

            for i in range(current_batch_size):
                if random.random() < config.cfg_dropout_prob:
                    #print("I have gone to unconditional")
                    # Use pre-calculated empty string embedding and its mask
                    context_final_list.append(current_uncond_context[i])
                    mask_final_list.append(current_uncond_mask[i])
                else:
                    # Use conditional embedding and its mask
                    context_final_list.append(cond_context_seq[i])
                    mask_final_list.append(cond_padding_mask[i])

            context_final = torch.stack(context_final_list)
            mask_final = torch.stack(mask_final_list)

            # 3. Diffusion Process
            t = torch.randint(0, config.noise_steps, (current_batch_size,), device=target_device).long()
            zt_batch, noise_batch = noise_latent(z0_batch, t, diff_params, target_device)

            # 4. Forward pass with potentially mixed conditional/unconditional context
            predicted_noise = model(zt_batch, t, context_final, mask_final)

            # 5. Calculate loss
            loss = criterion(predicted_noise, noise_batch)
            if torch.isnan(loss).any(): print(f"NaN loss at step {global_step}. Skipping."); optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            # LR Logic
            current_lr = optimizer.param_groups[0]['lr'] # Get current LR before optimizer step for logging
            if global_step < config["warmup_steps"]:
                lr_scale = float(global_step + 1) / float(config["warmup_steps"])
                for param_group in optimizer.param_groups: param_group['lr'] = config["learning_rate"] * lr_scale
            elif global_step == config["warmup_steps"]:
                print(f"Warmup finished at step {global_step}. Optimizer LR: {optimizer.param_groups[0]['lr']:.2e}. Starting cosine scheduler.")
                # Ensure scheduler starts from the correct LR after warmup
                for param_group in optimizer.param_groups: param_group['lr'] = config["learning_rate"]
                # Initialize scheduler here if not using get_cosine_schedule_with_warmup
                # scheduler = CosineAnnealingLR(optimizer, T_max=config["total_steps"] - config["warmup_steps"], eta_min=config.lr_decay_min)


            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0] # Get the new LR after scheduler step

            # if global_step >= config["warmup_steps"]:
            #     scheduler.step()
            #     # After scheduler.step(), get the new LR for logging
            #     current_lr = scheduler.get_last_lr()[0]
            # else: # During warmup, actual LR is what we set
            #     current_lr = optimizer.param_groups[0]['lr']


            # Logging
            running_train_loss += loss.item()
            if global_step % config.log_interval == 0:
                avg_train_loss_interval = running_train_loss / config.log_interval
                pbar.set_postfix({"Step": global_step, "Loss": f"{avg_train_loss_interval:.4f}", "LR": f"{current_lr:.2e}"})
                if run:
                    wandb.log({ "step": global_step, "train/loss": loss.item(),
                                "train/avg_loss_interval": avg_train_loss_interval,
                                "learning_rate": current_lr }, step=global_step)
                running_train_loss = 0.0

            # Evaluation
            if (global_step % config.eval_interval == 0 and global_step > 0) or (global_step == config.total_steps - 1):
                if val_loader is None: print(f"Skipping eval step {global_step} - No validation data.")
                else:
                    # ... (Evaluation logic as before, ensuring it uses LAST_HIDDEN_STATE and PADDING_MASK for eval_context) ...
                    print(f"\n--- Evaluating at step {global_step} ---")
                    model.eval(); total_eval_loss = 0.0; num_eval_batches = 0
                    with torch.no_grad():
                        for eval_batch in tqdm(val_loader, desc=f"Step {global_step} Eval", leave=False):
                            z0_eval = eval_batch['z'].to(target_device)
                            text_eval_prompts = eval_batch['text']
                            eval_batch_size = z0_eval.shape[0]

                            eval_text_inputs = clip_tokenizer(text_eval_prompts, padding=True, truncation=True, return_tensors="pt").to(target_device)
                            eval_context_seq = clip_model(**eval_text_inputs).last_hidden_state
                            eval_mask = ~(eval_text_inputs.attention_mask.bool())

                            t_eval = torch.randint(0, config.noise_steps, (eval_batch_size,), device=target_device).long()
                            zt_eval, noise_eval = noise_latent(z0_eval, t_eval, diff_params, target_device)
                            predicted_noise_eval = model(zt_eval, t_eval, eval_context_seq, eval_mask)
                            eval_loss = criterion(predicted_noise_eval, noise_eval)

                            if not torch.isnan(eval_loss).any():
                                total_eval_loss += eval_loss.item(); num_eval_batches += 1
                    avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else float('inf')
                    print(f"Avg Eval Loss: {avg_eval_loss:.4f}")
                    if run: wandb.log({"eval/avg_loss": avg_eval_loss}, step=global_step)

                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        torch.save(model.state_dict(), best_model_path)
                        print(f"✨ New best model saved with Eval Loss: {best_eval_loss:.4f} to {best_model_path} ✨")
                        # ... (wandb artifact saving if active) ...

                    # --- Optional: DDIM Sampling during eval (ensure it uses the latest ddim_sample) ---
                    # print("Running DDIM sampling (example)...")
                    # ... (sampling logic as before, ensure ddim_sample is called correctly)

            model.train()
            global_step += 1
            pbar.update(1)
            if global_step >= config.total_steps: break

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

# --- END OF FILE vp_dit.py ---