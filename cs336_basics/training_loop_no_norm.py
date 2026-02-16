import torch 
import torch.nn as nn
import numpy as np
import os
import typing
import argparse
import time
import logging

from cs336_basics.implementation import (
    AdamW, 
    cross_entropy, 
    learning_rate_schedule, 
    gradient_clipping,
    data_loading,
    save_checkpoint,
    load_checkpoint
)

from cs336_basics.implementation_no_norm import TransformerLM

from cs336_basics.tokenizer_implementation import Tokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def train(args):
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load Data 
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found at {args.train_data}")

    # Handle .txt files by tokenizing on-the-fly
    if args.train_data.endswith(".txt"):
        logging.info(f"Loading tokenizer from {args.tokenizer_vocab} and {args.tokenizer_merges}...")
        
        # Initialize Tokenizer with special tokens if needed
        special_tokens = ["<|endoftext|>"]
        tokenizer = Tokenizer.from_files(
            vocab_filepath=args.tokenizer_vocab,
            merges_filepath=args.tokenizer_merges,
            special_tokens=special_tokens
        )
        
        logging.info("Tokenizing training text...")
        with open(args.train_data, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode and convert to uint16 numpy array
        encoded_ids = tokenizer.encode(text)
        train_data = np.array(encoded_ids, dtype=np.uint16)
        logging.info(f"Tokenized {len(train_data)} tokens.")

        # Handle validation data same way
        val_data = None
        if args.val_data and os.path.exists(args.val_data):
            logging.info("Tokenizing validation text...")
            with open(args.val_data, 'r', encoding='utf-8') as f:
                val_text = f.read()
            val_ids = tokenizer.encode(val_text)
            val_data = np.array(val_ids, dtype=np.uint16)
            
    else:
        # Standard binary memory map for pre-processed .bin files
        train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
        val_data = None
        if args.val_data and os.path.exists(args.val_data):
            val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    # Initialize Model 
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        max_seq_len=args.context_length, 
        device=device
    ).to(device)

    # Initialize Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=args.max_lr, 
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logging.info(f"Resumed from iteration {start_iter}")

    # Training
    model.train()
    
    for i in range(start_iter, args.max_iters):
        t0 = time.time()
        
        # Update Learning Rate
        lr = learning_rate_schedule(
            step=i, 
            max_learning_rate=args.max_lr, 
            minimum_learning_rate=args.min_lr, 
            warmup_iterations=args.warmup_iters, 
            cos_anneal_iterations=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get Batch
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, device)

        # Forward & Backward
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        # Gradient Clipping & Step
        grad_norm = gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        dt = time.time() - t0

        # Logging
        if i % args.log_interval == 0:
            logging.info(f"Iter {i}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iter": i, "train/loss": loss.item(), "train/lr": lr, 
                    "train/grad_norm": grad_norm, "train/step_time_ms": dt*1000
                })

        # Validation
        if val_data is not None and i % args.eval_interval == 0 and i > 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = data_loading(val_data, args.batch_size, args.context_length, device)
                val_logits = model(val_inputs)
                val_loss = cross_entropy(val_logits, val_targets)
                logging.info(f"Iter {i}: val_loss {val_loss.item():.4f}")
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({"iter": i, "val/loss": val_loss.item()})
            model.train()

        # Save Checkpoint
        if i % args.save_interval == 0 and i > 0:
            save_path = os.path.join(args.save_dir, f"ckpt_{i}.pt")
            save_checkpoint(model, optimizer, i, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to .txt or .bin training data")
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--tokenizer_vocab", type=str, default=None, help="Path to vocab.json (required for .txt data)")
    parser.add_argument("--tokenizer_merges", type=str, default=None, help="Path to merges.json (required for .txt data)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)

    # Model config (Default to Tiny Stories)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer config
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=18000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_iters", type=int, default=20000)

    # Logistics
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check dependencies for .txt training
    if args.train_data.endswith(".txt") and (not args.tokenizer_vocab or not args.tokenizer_merges):
        parser.error("--tokenizer_vocab and --tokenizer_merges are required when training from .txt file")

    train(args)