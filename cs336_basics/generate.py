import torch
import argparse
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from cs336_basics.implementation import TransformerLM, decoding
from cs336_basics.tokenizer_implementation import Tokenizer

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"]
    )
    eos_id = tokenizer.special_tokens["<|endoftext|>"]

    # Initialize Model Architecture 
    print("Initializing model architecture...")
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

    # Load the Trained Weights
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")
        
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("Model weights loaded successfully.")

    # Prepare Prompt
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Run Generation
    print(f"\nGenerating text (Temp={args.temperature}, Top-P={args.top_p})...")
    output_indices = decoding(
        model=model,
        prompt_tokens=prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Decode and Print
    output_text = tokenizer.decode(output_indices[0].tolist())
    
    print("\n" + "#"*50)
    print("GENERATED OUTPUT:")
    print("#"*50)
    print(output_text)
    print("#"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Checkpoint and Config
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="data/merges.json")
    
    # Model Architecture Defaults
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Generation Params
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    generate(args)