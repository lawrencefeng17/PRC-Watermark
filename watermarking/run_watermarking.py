"""
Script to run different watermarking methods.

This script allows running and evaluating different watermarking approaches:
1. Binary watermarking (bit-level)
2. Token watermarking (token-level)
3. Independent hash watermarking (position-specific hashing)

Results for each method are saved to a timestamped directory with detailed logs,
generated text, and visualization plots.

Usage:
    python -m watermarking.run_watermarking --prompt "Your prompt" --num_tokens 100 --methods token
"""

import os
import argparse
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import shutil
import datetime
import sys
import math
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prc import KeyGen

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import our watermarking modules directly
from watermarking.binary_watermarking import BinaryWatermarkModel
from watermarking.token_watermarking import TokenWatermarkModel
from watermarking.independent_token_watermarking import IndependentHashModel
from watermarking.detection import (
    detect_binary_text_watermark,
    detect_token_watermark,
    detect_independent_hash_watermark,
    regenerate_independent_hash_functions
)

def setup(exp_id, n, message_length, fpr, prc_t):
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, message_length, false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
        with open(f'keys/{exp_id}.pkl', 'wb') as f:  # Save the keys to a file
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:  # Load the keys from a file
            encoding_key, decoding_key = pickle.load(f)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{exp_id}.pkl')
    return encoding_key, decoding_key

def main():
    parser = argparse.ArgumentParser(description="Run different watermarking methods")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID from HuggingFace")
    parser.add_argument("--prompt", type=str, default="Write a thrilling story about a murder investigation in an old mansion.", help="Prompt for text generation")
    parser.add_argument("--num_tokens", type=int, default=2**11, help="Number of tokens to generate")
    parser.add_argument("--n", type=int, default=2**11, help="Length of the PRC codeword")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.00, help="Top-p sampling parameter")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--methods", type=str, default="all", 
                        choices=["all", "binary", "token", "independent_hash"],
                        help="Which method to run")
    parser.add_argument("--experiment_id", type=str, default=None, help="Experiment ID for key management")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Directory to save the results")
    parser.add_argument("--fpr", type=float, default=0.00001, help="False positive rate for PRC code")
    parser.add_argument("--prc_t", type=int, default=3, help="Sparsity of the parity-check matrix")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose logging and plots")
    parser.add_argument("--new", action="store_true", help="Force generation of new text even if cached version exists")
    
    args = parser.parse_args()
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup experiment ID
    method_name = args.methods if args.methods != "all" else "combined"
    exp_id = f'watermark_{method_name}_n_{args.n}_t_{args.prc_t}_temp_{args.temperature}_tokens_{args.num_tokens}_top_p_{args.top_p}_greedy_{args.greedy}'
    
    # Create experiment directory with timestamp
    experiment_dir = os.path.join(args.output_dir, f"{exp_id}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    plots_dir = os.path.join(experiment_dir, "plots")
    tokens_dir = os.path.join(experiment_dir, "tokens")
    text_dir = os.path.join(experiment_dir, "text")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(experiment_dir, "experiment_log.txt")
    
    def log_message(message):
        """Helper function to log messages both to console and log file"""
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
    
    # Log experiment configuration
    log_message(f"Experiment ID: {exp_id}")
    log_message(f"Timestamp: {timestamp}")
    log_message(f"Parameters: {vars(args)}")
    
    # Load model and tokenizer
    log_message("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_message(f"Model loaded on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Log the prompt
    log_message(f"Prompt: {args.prompt}")
    
    # Get the encoding key
    log_message(f"Setting up PRC keys for {exp_id}")
    message_length = 0
    encoding_key, decoding_key = setup(exp_id, args.n, message_length, args.fpr, args.prc_t)
    log_message(f"PRC keys set up")
    
    # Save a copy of the key to the experiment directory
    key_path = os.path.join(experiment_dir, f"{exp_id}_key.pkl")
    with open(key_path, 'wb') as f:
        pickle.dump((encoding_key, decoding_key), f)
    log_message(f"Saved key to {key_path}")
    
    # Initialize dictionary to store results
    results = {
        "config": vars(args),
        "methods": {}
    }
    
    # Custom function to save plots to the plots directory
    def save_plot_to_dir(plot_path):
        # If the plot exists in the root directory (old location)
        if os.path.exists(plot_path):
            # Get the filename
            filename = os.path.basename(plot_path)
            # Create new path in plots directory
            new_path = os.path.join(plots_dir, filename)
            # Move the file
            shutil.move(plot_path, new_path)
            return new_path
        else:
            log_message(f"Warning: Plot file {plot_path} not found, skipping")
        return None
    
    # Record experiment start time
    start_time = datetime.datetime.now()
    log_message(f"Generation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test Binary Watermarking
    if args.methods == "all" or args.methods == "binary":
        log_message("\n=== Running Binary Watermarking ===")
        
        output_tokens_path = os.path.join(tokens_dir, "binary_tokens.pkl")

        if not args.debug:
            if not os.path.exists(f'encoding.pkl'):
                token_counts = {token_id: 0 for token_id in range(model.config.vocab_size)}
                with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
                    example_corpus = f.readlines()
                for sentence in example_corpus:
                    input_ids = tokenizer.encode(sentence)
                    for token_id in input_ids:
                        token_counts[token_id] += 1

                encoding = huffman_encode(token_counts)
                # save encoding to file
                with open(f'encoding.pkl', 'wb') as f:
                    pickle.dump(encoding, f)
            else:
                with open(f'encoding.pkl', 'rb') as f:
                    encoding = pickle.load(f)
            log_message(f"Encoding loaded")

            # Save a copy of the encoding to the experiment directory
            encoding_path = os.path.join(experiment_dir, "encoding.pkl")
            shutil.copy('encoding.pkl', encoding_path)
            log_message(f"Saved encoding copy to {encoding_path}")

            decoding = {code: token_id for token_id, code in encoding.items()}
        else:
            # Generate truly random unique encodings for each token
            log_message("Generating random encoding")
            vocab_size = model.config.vocab_size
            code_length = math.ceil(math.log2(vocab_size)) + 3  # Add a few extra bits to reduce collisions
            
            # Generate unique random codes for each token
            encoding = {}
            used_codes = set()
            
            for i in range(vocab_size):
                # Keep generating random codes until we find a unique one
                while True:
                    # Generate a random binary code of the specified length
                    random_code = ''.join(str(np.random.randint(0, 2)) for _ in range(code_length))
                    if random_code not in used_codes:
                        encoding[i] = random_code
                        used_codes.add(random_code)
                        break
            
            decoding = {code: i for i, code in encoding.items()}
            
            # Save the random encoding
            encoding_path = os.path.join(experiment_dir, "random_encoding.pkl")
            with open(encoding_path, 'wb') as f:
                pickle.dump(encoding, f)
            log_message(f"Saved random encoding to {encoding_path}")

        if not os.path.exists(output_tokens_path) or args.new:
            log_message("Creating binary watermarking model")
            binary_model = BinaryWatermarkModel(
                original_model=model,
                encoding_key=encoding_key,
                decoding_key=decoding_key,
                n=args.n,
                tokenizer=tokenizer,
                encoding=encoding,
                decoding=decoding,
                temperature=args.temperature
            )
            
            # Generate watermarked text
            log_message(f"Generating watermarked text (binary method)")
            start_time_binary = time.time()
            output_tokens, output_text, binary_tokens = binary_model.watermarked_generate(
                args.prompt, 
                args.n,  # Generate n bits
                debug=args.debug
            )
            generation_time = time.time() - start_time_binary
            
            # Save the tokens
            with open(output_tokens_path, 'wb') as f:
                pickle.dump(output_tokens, f)
            
            # Save the generated text
            output_text_path = os.path.join(text_dir, "binary_output.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
                
            # Move any generated plots to the plots directory
            save_plot_to_dir("hat_p_i_distribution.png")
            save_plot_to_dir("entropy_distribution.png")
            
            log_message(f"Binary generation completed in {generation_time:.2f} seconds")
            log_message(f"Generated {len(output_tokens)} tokens")
            log_message(f"Output text saved to {output_text_path}")
            
            # Detect watermark
            log_message("Detecting watermark...")
            threshold, hamming_weight, result = detect_binary_text_watermark(binary_model, output_tokens)
            
            log_message(f"Threshold: {threshold}, Hamming weight: {hamming_weight}")
            log_message(f"Detection result: {result}")
            
            # Save detection results
            detection_path = os.path.join(experiment_dir, "binary_detection_results.json")
            detection_results = {
                "threshold": float(threshold),
                "hamming_weight": float(hamming_weight),
                "detection_result": bool(result),
                "generation_time": generation_time,
                "method": "binary"
            }
            with open(detection_path, 'w') as f:
                json.dump(detection_results, f, indent=2)
                
            # Save to overall results
            results["methods"]["binary"] = detection_results
            
        else:
            log_message(f"Using existing binary output from {output_tokens_path}")
            with open(output_tokens_path, 'rb') as f:
                output_tokens = pickle.load(f)
                output_text = tokenizer.decode(output_tokens)
                
            # Load detection results if they exist
            detection_path = os.path.join(experiment_dir, "binary_detection_results.json")
            if os.path.exists(detection_path):
                with open(detection_path, 'r') as f:
                    results["methods"]["binary"] = json.load(f)
                    log_message("Loaded existing detection results")
            else:
                log_message("Warning: Detection results not found")
    
    # Test Token Watermarking
    if args.methods == "all" or args.methods == "token":
        log_message("\n=== Running Token Watermarking ===")
        
        output_tokens_path = os.path.join(tokens_dir, "token_tokens.pkl")
        
        if not os.path.exists(output_tokens_path) or args.new:
            log_message("Creating token watermarking model")
            token_model = TokenWatermarkModel(
                original_model=model,
                encoding_key=encoding_key,
                decoding_key=decoding_key,
                n=args.n,
                tokenizer=tokenizer,
                vocab_size=model.config.vocab_size,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Generate watermarked text
            log_message(f"Generating watermarked text (token method)")
            start_time_token = time.time()
            token_output_tokens, token_output_text, pushforward_probs, prc_bits, hashed_tokens, rejection_count, entropies = token_model.watermarked_generate_by_token(
                args.prompt, 
                args.num_tokens,
                top_p=args.top_p,
                greedy=args.greedy,
                debug=args.debug
            )
            generation_time = time.time() - start_time_token
            
            # Save the tokens
            with open(output_tokens_path, 'wb') as f:
                pickle.dump(token_output_tokens, f)
            
            # Save the generated text
            output_text_path = os.path.join(text_dir, "token_output.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(token_output_text)
                
            # Move any generated plots to the plots directory
            save_plot_to_dir("rejections_vs_index.png")
            save_plot_to_dir("rejection_rate_vs_entropy.png")
            save_plot_to_dir("bucket_0_distribution.png")
            save_plot_to_dir("bucket_1_distribution.png")
            
            # Calculate rejection rate
            rejection_rate = rejection_count / args.num_tokens
            
            # Calculate match rate between PRC bits and hashed tokens
            match_count = sum(1 for i in range(min(len(prc_bits), len(hashed_tokens))) 
                            if prc_bits[i] == hashed_tokens[i])
            match_rate = match_count / min(len(prc_bits), len(hashed_tokens))
            
            log_message(f"Token generation completed in {generation_time:.2f} seconds")
            log_message(f"Generated {len(token_output_tokens)} tokens")
            log_message(f"Rejection rate: {rejection_rate:.4f}")
            log_message(f"Match rate: {match_rate:.4f}")
            log_message(f"Output text saved to {output_text_path}")
            
            # Create additional plot for entropy comparison
            if args.debug and entropies is not None and hasattr(token_model, 'rejections'):
                rejections_array = np.array(token_model.rejections, dtype=float)
                
                if len(rejections_array) > 0:
                    accepted_entropy = np.mean([e for i, e in enumerate(entropies) if not token_model.rejections[i]])
                    rejected_entropy = np.mean([e for i, e in enumerate(entropies) if token_model.rejections[i]])
                    
                    plt.figure(figsize=(8, 6))
                    plt.bar(['Accepted', 'Rejected'], [accepted_entropy, rejected_entropy])
                    plt.title("Average Entropy: Accepted vs Rejected Tokens")
                    plt.ylabel("Average Entropy")
                    plt.savefig(os.path.join(plots_dir, "token_entropy_comparison.png"))
                    plt.close()
                    
                    log_message(f"Average entropy for accepted tokens: {accepted_entropy:.4f}")
                    log_message(f"Average entropy for rejected tokens: {rejected_entropy:.4f}")
            
            # Detect watermark
            log_message("Detecting watermark...")
            threshold, hamming_weight, result = detect_token_watermark(token_model, token_output_tokens)
            
            log_message(f"Threshold: {threshold}, Hamming weight: {hamming_weight}")
            log_message(f"Detection result: {result}")
            
            # Save detection results
            detection_path = os.path.join(experiment_dir, "token_detection_results.json")
            detection_results = {
                "threshold": float(threshold),
                "hamming_weight": float(hamming_weight),
                "detection_result": bool(result),
                "generation_time": generation_time,
                "rejection_rate": rejection_rate,
                "match_rate": match_rate,
                "method": "token"
            }
            with open(detection_path, 'w') as f:
                json.dump(detection_results, f, indent=2)
                
            # Save to overall results
            results["methods"]["token"] = detection_results
            
        else:
            log_message(f"Using existing token output from {output_tokens_path}")
            with open(output_tokens_path, 'rb') as f:
                token_output_tokens = pickle.load(f)
                token_output_text = tokenizer.decode(token_output_tokens)
                
            # Load detection results if they exist
            detection_path = os.path.join(experiment_dir, "token_detection_results.json")
            if os.path.exists(detection_path):
                with open(detection_path, 'r') as f:
                    results["methods"]["token"] = json.load(f)
                    log_message("Loaded existing detection results")
            else:
                log_message("Warning: Detection results not found")
    
    # Test Independent Hash Watermarking
    if args.methods == "all" or args.methods == "independent_hash":
        log_message("\n=== Running Independent Hash Watermarking ===")
        
        output_tokens_path = os.path.join(tokens_dir, "independent_hash_tokens.pkl")
        
        if not os.path.exists(output_tokens_path) or args.new:
            log_message("Creating independent hash watermarking model")
            hash_model = IndependentHashModel(
                original_model=model,
                encoding_key=encoding_key,
                decoding_key=decoding_key,
                n=args.n,
                tokenizer=tokenizer,
                vocab_size=model.config.vocab_size,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Generate watermarked text
            log_message(f"Generating watermarked text (independent hash method)")
            start_time_hash = time.time()
            hash_output_tokens, hash_output_text, pushforward_probs, prc_bits, hashed_tokens, position_hash_tensors, rejection_count, entropies = hash_model.watermarked_generate_by_token(
                args.prompt, 
                args.num_tokens,
                top_p=args.top_p,
                greedy=args.greedy,
                debug=args.debug
            )
            generation_time = time.time() - start_time_hash
            
            # Save the tokens and hash functions
            with open(output_tokens_path, 'wb') as f:
                # Move tensors to CPU before pickling
                position_hash_tensors_cpu = {pos: tensor.cpu() for pos, tensor in position_hash_tensors.items()}
                # Also save the num_precomputed_hashes value for proper modulo calculation during detection
                pickle_data = {
                    'tokens': hash_output_tokens,
                    'hash_tensors': position_hash_tensors_cpu,
                    'num_precomputed_hashes': hash_model.num_precomputed_hashes if hasattr(hash_model, 'num_precomputed_hashes') else len(position_hash_tensors_cpu)
                }
                pickle.dump(pickle_data, f)
            
            # Save the generated text
            output_text_path = os.path.join(text_dir, "independent_hash_output.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(hash_output_text)
                
            # Move any generated plots to the plots directory
            save_plot_to_dir("rejections_vs_index.png")
            save_plot_to_dir("rejection_rate_vs_entropy.png")
            save_plot_to_dir("bucket_0_distribution.png")
            save_plot_to_dir("bucket_1_distribution.png")
            
            # Calculate rejection rate
            rejection_rate = rejection_count / args.num_tokens
            
            # Calculate match rate between PRC bits and hashed tokens
            match_count = sum(1 for i in range(min(len(prc_bits), len(hashed_tokens))) 
                            if prc_bits[i] == hashed_tokens[i])
            match_rate = match_count / min(len(prc_bits), len(hashed_tokens))
            
            log_message(f"Independent hash generation completed in {generation_time:.2f} seconds")
            log_message(f"Generated {len(hash_output_tokens)} tokens")
            log_message(f"Rejection rate: {rejection_rate:.4f}")
            log_message(f"Match rate: {match_rate:.4f}")
            log_message(f"Output text saved to {output_text_path}")
            
            # Create additional plot for entropy comparison
            if args.debug and entropies is not None and hasattr(hash_model, 'rejections'):
                rejections_array = np.array(hash_model.rejections, dtype=float)
                
                if len(rejections_array) > 0:
                    accepted_entropy = np.mean([e for i, e in enumerate(entropies) if not hash_model.rejections[i]])
                    rejected_entropy = np.mean([e for i, e in enumerate(entropies) if hash_model.rejections[i]])
                    
                    plt.figure(figsize=(8, 6))
                    plt.bar(['Accepted', 'Rejected'], [accepted_entropy, rejected_entropy])
                    plt.title("Average Entropy: Accepted vs Rejected Tokens")
                    plt.ylabel("Average Entropy")
                    plt.savefig(os.path.join(plots_dir, "independent_hash_entropy_comparison.png"))
                    plt.close()
                    
                    log_message(f"Average entropy for accepted tokens: {accepted_entropy:.4f}")
                    log_message(f"Average entropy for rejected tokens: {rejected_entropy:.4f}")
            
            # Detect watermark
            log_message("Detecting watermark...")
            threshold, hamming_weight, result = detect_independent_hash_watermark(
                hash_model, 
                hash_output_tokens,
                position_hash_tensors
            )
            
            log_message(f"Threshold: {threshold}, Hamming weight: {hamming_weight}")
            log_message(f"Detection result: {result}")
            
            # Save detection results
            detection_path = os.path.join(experiment_dir, "independent_hash_detection_results.json")
            detection_results = {
                "threshold": float(threshold),
                "hamming_weight": float(hamming_weight),
                "detection_result": bool(result),
                "generation_time": generation_time,
                "rejection_rate": rejection_rate,
                "match_rate": match_rate,
                "method": "independent_hash"
            }
            with open(detection_path, 'w') as f:
                json.dump(detection_results, f, indent=2)
                
            # Save to overall results
            results["methods"]["independent_hash"] = detection_results
            
        else:
            log_message(f"Using existing independent hash output from {output_tokens_path}")
            with open(output_tokens_path, 'rb') as f:
                hash_output_tokens, position_hash_tensors = pickle.load(f)
                hash_output_text = tokenizer.decode(hash_output_tokens)
                # Move tensors back to the device if needed
                position_hash_tensors = {pos: tensor.to(device) for pos, tensor in position_hash_tensors.items()}
                
            # Load detection results if they exist
            detection_path = os.path.join(experiment_dir, "independent_hash_detection_results.json")
            if os.path.exists(detection_path):
                with open(detection_path, 'r') as f:
                    results["methods"]["independent_hash"] = json.load(f)
                    log_message("Loaded existing detection results")
            else:
                log_message("Warning: Detection results not found")
    
    # Record experiment end time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    log_message(f"Experiment completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Total duration: {duration}")
    
    # Save experiment summary
    summary_path = os.path.join(experiment_dir, "experiment_summary.json")
    summary = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "duration_seconds": duration.total_seconds(),
        "parameters": vars(args),
        "results": results,
        "model_id": args.model_id,
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        "prompt": args.prompt
    }
    
    with open(summary_path, 'w') as f:
        # Convert numpy and torch types to standard Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, (np.ndarray, list)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        serializable_summary = convert_to_serializable(summary)
        json.dump(serializable_summary, f, indent=2)
    
    # Print summary
    log_message("\n=== Experiment Summary ===")
    for method, data in results["methods"].items():
        log_message(f"\n{method.upper()} Watermarking:")
        log_message(f"  Watermark detected: {data['detection_result']}")
        log_message(f"  Hamming weight: {data['hamming_weight']} (threshold: {data['threshold']})")
        if 'rejection_rate' in data:
            log_message(f"  Rejection rate: {data['rejection_rate']:.4f}")
        if 'match_rate' in data:
            log_message(f"  Match rate: {data['match_rate']:.4f}")
    
    log_message(f"\nResults saved to: {experiment_dir}")
    
if __name__ == "__main__":
    main() 