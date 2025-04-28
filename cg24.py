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
from tqdm import tqdm

from src.prc import Encode, Encode_No_OTP, Decode, KeyGen
from huffman import huffman_encode, huffman_decode, build_huffman_tree, generate_huffman_codes

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

def main():
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--prompt', type=str, default='Tell me a fantastical story about a wizard.')
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B')
    # parser.add_argument('--dataset_id', type=str, default='databricks/databricks-dolly-15k')
    parser.add_argument('--inf_steps', type=int, default=50)
    parser.add_argument('--nowm', type=int, default=0)
    parser.add_argument('--fpr', type=float, default=0.00001)
    parser.add_argument('--prc_t', type=int, default=3)
    parser.add_argument('--n', type=int, default=2**11)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--message_length', type=int, default=0)    
    parser.add_argument('--new', action='store_true')
    # by default, we use the token-level watermarking scheme
    parser.add_argument('--bit', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

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
    exp_id = f'binarize_num_{test_num}_steps_{args.inf_steps}_t_{prc_t}_fpr_{fpr}_nowm_{nowm}_n_{n}_temperature_{temperature}_message_length_{message_length}'

    print("Loading model...")
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

    print(f"Model loaded on {device}")

    # Load the dataset
    # dataset = load_dataset(args.dataset_id, split='train')
    # print(f"Dataset loaded with {len(dataset)} examples")

    # Get the prompt
    prompt = args.prompt
    print(f"Prompt: {prompt}")

    # Get the encoding key
    print(f"Setting up PRC keys for {exp_id}")
    encoding_key, decoding_key = setup(exp_id, n, message_length, fpr, prc_t)
    print(f"PRC keys set up")

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
        print(f"Encoding loaded")

        decoding = {code: token_id for token_id, code in encoding.items()}
    else:
        # Generate truly random unique encodings for each token
        print("Generating random encoding")
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

    # Binarize the model
    binarized_model = BinarizedModel(
        original_model=model,
        encoding_key=encoding_key,
        decoding_key=decoding_key,
        n=n,
        tokenizer=tokenizer,
        encoding=encoding,
        decoding=decoding,
        temperature=temperature)
    print(f"Binarized model loaded")
    
    # test parity check matrix on codeword
    P = binarized_model.decoding_key[1]
    Px = (P @ binarized_model.prc_codeword) % 2

    # compute Pz, where z is the one-time pad
    z = binarized_model.decoding_key[2]
    Pz = (P @ z) % 2

    # compute Px \\xor Pz
    Px_xor_Pz = (Px + Pz) % 2

    print(f"Px \\xor Pz: {Px_xor_Pz}")
    hamming_weight_codeword = np.sum(Px_xor_Pz)
    print(f"Hamming weight of codeword: {hamming_weight_codeword}")
    r = P.shape[0]
    threshold = (1/2 - r**(-1/4)) * r
    print(f"Threshold: {threshold}")
    print(f"For a random codeword, the expected hamming weight is {r/2}")

    if args.bit:
        # generate watermarked text
        if not os.path.exists(f'output_tokens_{exp_id}.pkl') or args.new:
            print("Generating watermarked text by bit")
            output_tokens, output_text = binarized_model.watermarked_generate(prompt, num_bits=n, debug=debug)
            with open(f'output_tokens_{exp_id}.pkl', 'wb') as f:
                pickle.dump(output_tokens, f)
        else:
            with open(f'output_tokens_{exp_id}.pkl', 'rb') as f:
                output_tokens = pickle.load(f)
                output_text = ''.join([tokenizer.decode([token_id]) for token_id in output_tokens])
        # save output tokens
        print(f"Output text: {output_text}")

        # detect watermark
        threshold, hamming_weight, result = detect_hamming_text(binarized_model, output_tokens)
        print(f"Threshold: {threshold}, Hamming weight: {hamming_weight}, Result: {result}")
    else:
        # generate watermarked text per token
        if not os.path.exists(f'output_tokens_{exp_id}.pkl') or args.new:
            print("Generating watermarked text per token")
            output_tokens, output_text = binarized_model.watermarked_generate_by_token(prompt, num_tokens=n, debug=debug)
            with open(f'output_tokens_{exp_id}.pkl', 'wb') as f:
                pickle.dump(output_tokens, f)
        else:
            with open(f'output_tokens_{exp_id}.pkl', 'rb') as f:
                output_tokens = pickle.load(f)
                output_text = ''.join([tokenizer.decode([token_id]) for token_id in output_tokens])
        print(f"Output text: {output_text}")

        # detect watermark
        threshold, hamming_weight, result = detect_hamming_text_per_token(binarized_model, output_tokens)
        print(f"Threshold: {threshold}, Hamming weight: {hamming_weight}, Result: {result}")
        
    breakpoint()


if __name__ == "__main__":
    main()
