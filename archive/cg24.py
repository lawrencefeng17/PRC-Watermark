"""
Implementation of Watermarking Schemes described in "Pseudorandom Error-Correcting Codes," Christ & Gunn 2024.

See page 50 for the scheme.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import math
import pickle
import json
from collections import defaultdict
import argparse
import os
import datetime
import shutil
from tqdm import tqdm

from src.prc import KeyGen
from huffman import huffman_encode

from binarized import BinarizedModel

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset

# --- PRC Key Generation ---
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

def detect_hamming_text_per_token(binarized_model, watermarked_text):
    """
    Detects if the provided text is watermarked, using the token-level watermarking scheme in binarized.py.
    
    This function maps each token back to its bucket (0 or 1) using the hash function,
    and then checks if the resulting sequence matches the PRC codeword pattern.
    """
    hash_function = binarized_model.hash_function
    
    # Convert each token to its bucket using the hash function
    reconstructed_prc_bits = torch.tensor([hash_function(token_id) for token_id in watermarked_text], dtype=torch.float)
    
    # Ensure we have enough bits - pad with zeros if needed
    if len(reconstructed_prc_bits) < binarized_model.n:
        reconstructed_prc_bits = torch.cat([
            reconstructed_prc_bits, 
            torch.zeros(binarized_model.n - len(reconstructed_prc_bits))
        ])
    # If we have too many bits, truncate
    elif len(reconstructed_prc_bits) > binarized_model.n:
        print(f"Truncating {len(reconstructed_prc_bits)} bits to {binarized_model.n} bits")
        reconstructed_prc_bits = reconstructed_prc_bits[:binarized_model.n]
        
    # Apply parity check matrix to get detection result
    parity_check_matrix = binarized_model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # compute Px
    Px = (parity_check_matrix @ reconstructed_prc_bits) % 2
    
    # compute Pz, where z is the one-time pad
    z = binarized_model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2
    
    # compute Px ⊕ Pz (Px XOR Pz)
    Px_xor_Pz = (Px + Pz) % 2
    
    hamming_weight = np.sum(Px_xor_Pz)
    
    threshold = (1/2 - r**(-1/4)) * r
    # if below threshold, then detection is positive
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result

def detect_hamming_binary(binarized_model, watermarked_text_binary):
    """
    Detects if the provided binary string is watermarked, using hamming distance.

    Using Chernoff bound, wt(Px \\xor Pz) \\geq (1/2 - r^(-1/4)) * r with high probability.
    
    This is instead of computing wt(Px). "The issue is that, while most fixed strings will decode to bottom, a small fraction of strings will decode to 1 regardless of P. [e.g. 0 decodes to 1 because wt(0) = 0]." To address this, we include a one-time pad in the public key.

    Described on Page 14 of CG24 and in Section 5 of CG24.
    """
    if len(watermarked_text_binary) < binarized_model.n:
        watermarked_text_binary = torch.cat([watermarked_text_binary, torch.zeros(binarized_model.n - len(watermarked_text_binary))])

    # wt(Px) < (1/2 - r^(-1/4)) * r, output 1, where P is the parity check matrix
    parity_check_matrix = binarized_model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # compute Px
    Px = (parity_check_matrix @ watermarked_text_binary) % 2

    # compute Pz, where z is the one-time pad
    z = binarized_model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2

    # compute Px \\xor Pz
    Px_xor_Pz = (Px + Pz) % 2
    
    hamming_weight = np.sum(Px_xor_Pz)
    
    threshold = (1/2 - r**(-1/4)) * r
    # if below threshold, then detection
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result

def detect_hamming_text(binarized_model, watermarked_text):
    """
    Detects if the provided text is watermarked, using hamming distance.
    """
    # convert watermarked_text to binary string using encoding
    watermarked_text_binary = ''.join([binarized_model.encoding[token_id] for token_id in watermarked_text])
    watermarked_text_binary = torch.tensor([int(bit) for bit in watermarked_text_binary], dtype=float)

    return detect_hamming_binary(binarized_model, watermarked_text_binary)

def convert_watermarked_text_to_binary(binarized_model, watermarked_text):
    watermarked_text_binary = ''.join([binarized_model.encoding[token_id] for token_id in watermarked_text])
    watermarked_text_binary = torch.tensor([int(bit) for bit in watermarked_text_binary], dtype=float)
    watermarked_text_binary = 2 * watermarked_text_binary - 1
    return watermarked_text_binary

def convert_watermarked_by_token_to_binary(binarized_model, watermarked_text):
    watermarked_text_binary = torch.tensor([binarized_model.hash_function(token_id) for token_id in watermarked_text], dtype=float)
    watermarked_text_binary = 2 * watermarked_text_binary - 1
    return watermarked_text_binary

def corrupt_watermarked_text(binarized_model, watermarked_text, corruption_rate=0.1):
    """
    Randomly flips bits in the binary representation of watermarked text.
    
    Args:
        binarized_model: The binarized model containing encoding information.
        watermarked_text: List of token IDs to be corrupted.
        corruption_rate: Probability of flipping each bit (between 0 and 1).
        
    Returns:
        Corrupted binary tensor with values in {-1, 1}.
    """
    # Convert the watermarked text to binary format
    watermarked_text_binary = convert_watermarked_text_to_binary(binarized_model, watermarked_text)
    
    # Create a mask of bits to flip based on the corruption rate
    num_bits = len(watermarked_text_binary)
    flip_mask = torch.rand(num_bits) < corruption_rate
    
    # Flip the selected bits (multiply by -1 since values are {-1, 1})
    watermarked_text_binary[flip_mask] *= -1
    
    return watermarked_text_binary

def story_prompts():
    """
    From MarkMyWords benchmark. 
    """
    story_prompt = "Write a {}story about {}."
    
    t1 = ["", "funny ", "sad ", "dramatic ", "suspenseful ", "thrilling "]
    t2 = [
        "a man on a quest to find the Holy Grail.",
        "two college friends falling in love.",
        "a policeman saving a building held hostage by group of terrorists.",
        "the struggle of publishing an academic paper.",
        "a murder investigation in an old mansion.",
        "a young prodigy that becomes orphaned.",
        "a middle-aged woman that discovers a ghost and befriends it.",
        "a long journey to Japan that is interrupted by a disaster.",
        "a poor child that comes into an unexpected fortune.",
        "three strangers that win a getaway vacation together.",
        "a retired astronaut that joins a risky interstellar rescue mission.",
        "an AI that begins to question its own existence.",
        "a small coastal town plagued by inexplicable supernatural occurrences.",
        "a reclusive writer that receives mysterious, prophetic letters in the mail.",
        "a linguist tasked with deciphering an ancient language that holds the secrets of a lost civilization.",
        "an antique restorer that finds an enchanted mirror showing glimpses of different timelines.",
        ]


    story_topics = [(i, j) for i in t1 for j in t2][:100]
    story_prompts = [story_prompt.format(i, j) for i, j in story_topics]
    return story_prompts

def main():
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--prompt', type=str, default='Write a thrilling story about a murder investigation in an old mansion.')
    parser.add_argument('--test_num', type=int, default=10)
    # parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    # parser.add_argument('--dataset_id', type=str, default='databricks/databricks-dolly-15k')
    parser.add_argument('--inf_steps', type=int, default=50)
    parser.add_argument('--nowm', type=int, default=0)
    parser.add_argument('--fpr', type=float, default=0.00001)
    parser.add_argument('--prc_t', type=int, default=3)
    parser.add_argument('--n', type=int, default=2**14)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--message_length', type=int, default=0)    
    parser.add_argument('--new', action='store_true')
    # by default, we use the token-level watermarking scheme
    parser.add_argument('--bit', action='store_true', default=False)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--greedy', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup experiment ID
    hf_cache_dir = '/home/lawrence/.cache/huggingface/hub'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_num = args.test_num
    model_id = args.model_id
    nowm = args.nowm
    fpr = args.fpr
    prc_t = args.prc_t
    n = args.n
    debug = args.debug
    temperature = args.temperature
    message_length = args.message_length
    top_p = args.top_p
    greedy = args.greedy
    exp_id = f'binarize_num_{test_num}_steps_{args.inf_steps}_t_{prc_t}_fpr_{fpr}_nowm_{nowm}_n_{n}_temperature_{temperature}_message_length_{message_length}_top_p_{top_p}_greedy_{greedy}'
    
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

    log_message("Loading model...")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    log_message(f"Model loaded on {device}")

    # Get the prompt
    prompt = args.prompt
    log_message(f"Prompt: {prompt}")

    # Get the encoding key
    log_message(f"Setting up PRC keys for {exp_id}")
    encoding_key, decoding_key = setup(exp_id, n, message_length, fpr, prc_t)
    log_message(f"PRC keys set up")

    # Save a copy of the key to the experiment directory
    key_path = os.path.join(experiment_dir, f"{exp_id}_key.pkl")
    with open(key_path, 'wb') as f:
        pickle.dump((encoding_key, decoding_key), f)
    log_message(f"Saved key to {key_path}")

    if not debug:
        if not os.path.exists(f'encoding.pkl'):
            token_counts = {token_id: 0 for token_id in range(vocab_size)}
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
        vocab_size = len(tokenizer)  # Or from config
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

    # Binarize the model
    binarized_model = BinarizedModel(
        original_model=model,
        encoding_key=encoding_key,
        decoding_key=decoding_key,
        n=n,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        encoding=encoding,
        decoding=decoding,
        temperature=temperature,
        top_p=top_p,
    )
    log_message(f"Binarized model loaded")
    
    # test parity check matrix on codeword
    P = binarized_model.decoding_key[1]
    Px = (P @ binarized_model.prc_codeword) % 2

    # compute Pz, where z is the one-time pad
    z = binarized_model.decoding_key[2]
    Pz = (P @ z) % 2

    # compute Px \\xor Pz
    Px_xor_Pz = (Px + Pz) % 2

    hamming_weight_codeword = np.sum(Px_xor_Pz)
    log_message(f"Hamming weight of codeword: {hamming_weight_codeword}")
    r = P.shape[0]
    threshold = (1/2 - r**(-1/4)) * r
    log_message(f"Threshold: {threshold}")
    log_message(f"For a random codeword, the expected hamming weight is {r/2}")

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
            print(f"Warning: Plot file {plot_path} not found, skipping")
        return None

    # Record experiment start time
    start_time = datetime.datetime.now()
    log_message(f"Generation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.bit:
        # generate watermarked text
        output_tokens_path = os.path.join(tokens_dir, f"output_tokens.pkl")
        
        if not os.path.exists(output_tokens_path) or args.new:
            log_message("Generating watermarked text by bit")
            output_tokens, output_text = binarized_model.watermarked_generate(prompt, num_bits=n, debug=debug)
            
            # Save the tokens
            with open(output_tokens_path, 'wb') as f:
                pickle.dump(output_tokens, f)
            
            # Save the generated text
            output_text_path = os.path.join(text_dir, "output_text.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            
            # Move any generated plots to the plots directory
            save_plot_to_dir("hat_p_i_distribution.png")
            save_plot_to_dir("entropy_distribution.png")
        else:
            with open(output_tokens_path, 'rb') as f:
                output_tokens = pickle.load(f)
                output_text = ''.join([tokenizer.decode([token_id]) for token_id in output_tokens])
            
            # Save the text if it doesn't exist
            output_text_path = os.path.join(text_dir, "output_text.txt")
            if not os.path.exists(output_text_path):
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                    
        log_message(f"Output text saved to {output_text_path}")
        print(output_text)

        # detect watermark
        threshold, hamming_weight, result = detect_hamming_text(binarized_model, output_tokens)
        log_message(f"Threshold: {threshold}, Hamming weight: {hamming_weight}, Result: {result}")
        
        # Save detection results
        detection_path = os.path.join(experiment_dir, "detection_results.json")
        detection_results = {
            "threshold": float(threshold),
            "hamming_weight": float(hamming_weight),
            "detection_result": bool(result),
            "method": "bit-level"
        }
        with open(detection_path, 'w') as f:
            json.dump(detection_results, f, indent=2)
    else:
        # generate watermarked text per token
        output_tokens_path = os.path.join(tokens_dir, f"output_tokens.pkl")
        
        if not os.path.exists(output_tokens_path) or args.new:
            log_message(f"Generating watermarked text per token, greedy={greedy}")
            output_tokens, output_text, _, _, _, _, entropies = binarized_model.watermarked_generate_by_token(prompt, num_tokens=n, greedy=greedy, debug=debug)
            
            # Save the tokens
            with open(output_tokens_path, 'wb') as f:
                pickle.dump(output_tokens, f)
            
            # Save the generated text
            output_text_path = os.path.join(text_dir, "output_text.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            
            # Move any generated plots to the plots directory
            save_plot_to_dir("rejections_vs_index.png")
            save_plot_to_dir("rejection_rate_vs_entropy.png")
            save_plot_to_dir("bucket_0_distribution.png")
            save_plot_to_dir("bucket_1_distribution.png")
        else:
            with open(output_tokens_path, 'rb') as f:
                output_tokens = pickle.load(f)
                output_text = ''.join([tokenizer.decode([token_id]) for token_id in output_tokens])
            
            # Save the text if it doesn't exist
            output_text_path = os.path.join(text_dir, "output_text.txt")
            if not os.path.exists(output_text_path):
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                    
        log_message(f"Output text saved to {output_text_path}")
        rejection_rate = binarized_model.rejection_count / len(output_tokens)
        log_message(f"Rejection rate: {rejection_rate}")
        average_entropy_among_rejected_tokens = np.mean(entropies[binarized_model.rejections])
        log_message(f"Average entropy among rejected tokens: {average_entropy_among_rejected_tokens}")
        average_entropy_among_accepted_tokens = np.mean(entropies[~np.array(binarized_model.rejections)])
        log_message(f"Average entropy among accepted tokens: {average_entropy_among_accepted_tokens}")
        average_entropy = np.mean(entropies)
        log_message(f"Average entropy: {average_entropy}")

        plt.figure(figsize=(10, 5))
        plt.bar(['rejected', 'accepted'], [average_entropy_among_rejected_tokens, average_entropy_among_accepted_tokens])
        plt.savefig(os.path.join(plots_dir, "average_entropy_among_rejected_and_accepted_tokens.png"))
        plt.close()

        # detect watermark
        threshold, hamming_weight, result = detect_hamming_text_per_token(binarized_model, output_tokens)
        log_message(f"Threshold: {threshold}, Hamming weight: {hamming_weight}, Result: {result}")
        
        # Save detection results
        detection_path = os.path.join(experiment_dir, "detection_results.json")
        detection_results = {
            "threshold": float(threshold),
            "hamming_weight": float(hamming_weight),
            "rejection_rate": float(rejection_rate),
            "detection_result": bool(result),
            "method": "token-level"
        }
        with open(detection_path, 'w') as f:
            json.dump(detection_results, f, indent=2)
    
    # Record experiment end time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    log_message(f"Generation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        "detection_results": detection_results,
        "model_id": model_id,
        "device": device,
        "prompt": prompt
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_message(f"Experiment results saved to {experiment_dir}")
    log_message(f"Experiment summary saved to {summary_path}")
    
    if debug:
        breakpoint()


if __name__ == "__main__":
    main()
