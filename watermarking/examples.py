"""
Example usage of the watermarking library.

This module demonstrates how to use the different watermarking schemes:
1. Binary watermarking
2. Token-level watermarking
3. Independent hash watermarking

It also includes functions for detecting watermarks in generated text.
"""

import os
import argparse
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prc import KeyGen

# Import our watermarking modules
from watermarking import (
    BinaryWatermarkModel,
    TokenWatermarkModel,
    IndependentHashModel,
    detect_binary_text_watermark,
    detect_token_watermark,
    detect_independent_hash_watermark
)

def setup_keys(n, r, experiment_id=None):
    """
    Set up PRC keys for watermarking.
    
    Args:
        n: The length of the PRC codeword
        r: The dimension of the parity-check matrix
        experiment_id: Optional ID for the experiment
        
    Returns:
        encoding_key, decoding_key: The PRC keys for encoding and decoding
    """
    # Create keys directory if it doesn't exist
    os.makedirs('keys', exist_ok=True)
    
    # If experiment_id is provided, use it to load/save keys
    key_path = f'keys/keys_{n}_{r}.pkl'
    if experiment_id is not None:
        key_path = f'keys/keys_{n}_{r}_{experiment_id}.pkl'
    
    # Check if keys already exist
    if os.path.exists(key_path):
        print(f"Loading keys from {key_path}")
        with open(key_path, 'rb') as f:
            keys = pickle.load(f)
        encoding_key, decoding_key = keys
    else:
        print(f"Generating new keys with n={n}, r={r}")
        encoding_key, decoding_key = KeyGen(n, r)
        
        # Save keys
        with open(key_path, 'wb') as f:
            pickle.dump((encoding_key, decoding_key), f)
    
    return encoding_key, decoding_key

def binary_watermarking_example(args):
    """
    Example of using binary watermarking.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Binary Watermarking Example ===")
    
    # Load model and tokenizer
    print(f"Loading model {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Set up keys
    encoding_key, decoding_key = setup_keys(args.n, args.r, experiment_id=args.experiment_id)
    
    # Create binary watermarking model
    binary_model = BinaryWatermarkModel(
        original_model=model,
        encoding_key=encoding_key,
        decoding_key=decoding_key,
        n=args.n,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        temperature=args.temperature
    )
    
    # Generate watermarked text
    print(f"Generating watermarked text with prompt: {args.prompt}")
    start_time = time.time()
    output_text_binary = binary_model.watermarked_generate(
        args.prompt, 
        args.num_tokens,
        greedy=args.greedy
    )
    generation_time = time.time() - start_time
    
    print(f"Generated text (binary watermarking): {output_text_binary}")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Detect watermark
    print("Detecting watermark...")
    output_tokens = tokenizer.encode(output_text_binary)
    threshold, hamming_weight, result = detect_binary_text_watermark(binary_model, output_tokens)
    
    print(f"Watermark detected: {result}")
    print(f"Hamming weight: {hamming_weight}")
    print(f"Threshold: {threshold}")

def token_watermarking_example(args):
    """
    Example of using token-level watermarking.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Token Watermarking Example ===")
    
    # Load model and tokenizer
    print(f"Loading model {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Set up keys
    encoding_key, decoding_key = setup_keys(args.n, args.r, experiment_id=args.experiment_id)
    
    # Create token watermarking model
    token_model = TokenWatermarkModel(
        original_model=model,
        encoding_key=encoding_key,
        decoding_key=decoding_key,
        n=args.n,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        temperature=args.temperature
    )
    
    # Generate watermarked text
    print(f"Generating watermarked text with prompt: {args.prompt}")
    start_time = time.time()
    output_tokens, output_text = token_model.watermarked_generate_by_token(
        args.prompt, 
        args.num_tokens,
        top_p=args.top_p,
        greedy=args.greedy
    )
    generation_time = time.time() - start_time
    
    print(f"Generated text (token watermarking): {output_text}")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Detect watermark
    print("Detecting watermark...")
    threshold, hamming_weight, result = detect_token_watermark(token_model, output_tokens)
    
    print(f"Watermark detected: {result}")
    print(f"Hamming weight: {hamming_weight}")
    print(f"Threshold: {threshold}")

def independent_hash_watermarking_example(args):
    """
    Example of using independent hash watermarking.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Independent Hash Watermarking Example ===")
    
    # Load model and tokenizer
    print(f"Loading model {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Set up keys
    encoding_key, decoding_key = setup_keys(args.n, args.r, experiment_id=args.experiment_id)
    
    # Create independent hash watermarking model
    hash_model = IndependentHashModel(
        original_model=model,
        encoding_key=encoding_key,
        decoding_key=decoding_key,
        n=args.n,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        temperature=args.temperature
    )
    
    # Generate watermarked text
    print(f"Generating watermarked text with prompt: {args.prompt}")
    start_time = time.time()
    output_tokens, output_text, _, _, _, position_hash_functions, _, _ = hash_model.watermarked_generate_by_token(
        args.prompt, 
        args.num_tokens,
        top_p=args.top_p,
        greedy=args.greedy
    )
    generation_time = time.time() - start_time
    
    print(f"Generated text (independent hash watermarking): {output_text}")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Detect watermark
    print("Detecting watermark...")
    threshold, hamming_weight, result = detect_independent_hash_watermark(
        hash_model, 
        output_tokens,
        position_hash_functions
    )
    
    print(f"Watermark detected: {result}")
    print(f"Hamming weight: {hamming_weight}")
    print(f"Threshold: {threshold}")

def main():
    parser = argparse.ArgumentParser(description="Examples of different watermarking schemes")
    parser.add_argument("--model_id", type=str, default="facebook/opt-1.3b", help="Model ID from HuggingFace")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for text generation")
    parser.add_argument("--num_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--n", type=int, default=128, help="Length of the PRC codeword")
    parser.add_argument("--r", type=int, default=32, help="Dimension of the parity-check matrix")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--example", type=str, default="all", 
                        choices=["all", "binary", "token", "independent_hash"],
                        help="Which example to run")
    parser.add_argument("--experiment_id", type=str, default=None, help="Experiment ID for key management")
    
    args = parser.parse_args()
    
    if args.example == "all" or args.example == "binary":
        binary_watermarking_example(args)
    
    if args.example == "all" or args.example == "token":
        token_watermarking_example(args)
    
    if args.example == "all" or args.example == "independent_hash":
        independent_hash_watermarking_example(args)

if __name__ == "__main__":
    main() 