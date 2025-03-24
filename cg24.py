"""
Implementation of Watermarking Schemes described in "Pseudorandom Error-Correcting Codes," Christ & Gunn 2024.

See page 50 for the scheme.
"""
import torch
import numpy as np
import pickle
import json
from collections import defaultdict
import argparse
import os
from tqdm import tqdm

from src.prc import Encode, Decode, KeyGen

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset

# --- Huffman Encoding ---
def build_huffman_tree(frequencies):
    #using a dictionary instead of a heap
    nodes = {symbol: {"freq": freq, "left": None, "right": None, "symbol": symbol} for symbol, freq in frequencies.items()}
    while len(nodes) > 1:
        #find two least frequent nodes
        left_symbol, right_symbol = sorted(nodes, key=lambda symbol: nodes[symbol]["freq"])[:2]
        left_node = nodes.pop(left_symbol)
        right_node = nodes.pop(right_symbol)
        new_node = {"freq": left_node["freq"] + right_node["freq"], "left": left_node, "right": right_node}
        nodes[f"{left_symbol}, {right_symbol}"] = new_node

    [(root_symbol, root_node)] = nodes.items()
    return root_node

def generate_huffman_codes(tree):
    codes = {}
    def traverse(node, current_code=""):
        if node["left"] is None and node["right"] is None:
            # Store the code for the symbol, not the frequency
            codes[node["symbol"]] = current_code
            return
        traverse(node["left"], current_code + "0")
        traverse(node["right"], current_code + "1")
    traverse(tree)
    return codes

def huffman_encode(frequencies):
  tree = build_huffman_tree(frequencies)
  codes = generate_huffman_codes(tree)
  encoding = {token: codes[token] for token, freq in frequencies.items()}
  return encoding

def huffman_decode(encoding, encoded_string):
    decoding = {code: symbol for symbol, code in encoding.items()} # This line was correct
    decoded_sequence = []
    current_code = ""
    for bit in encoded_string:
        current_code += bit
        if current_code in decoding:
            decoded_sequence.append(decoding[current_code])
            current_code = ""
    return decoded_sequence

class BinarizedModel:
    def __init__(self, original_model, encoding_key, tokenizer=None, frequencies=None, encoding=None, decoding=None):
        """
        Args:
            original_model: The original (non-binary) language model.
            encoding_key: The key for the PRC encoding.
            tokenizer: The tokenizer for the model.
            frequencies: A dictionary mapping original tokens to frequencies.
            encoding:  A dictionary mapping original tokens to binary strings (prefix-free).
            decoding: A dictionary mapping binary strings to original tokens.
        """
        self.original_model = original_model
        self.tokenizer = tokenizer
        self.device = next(original_model.parameters()).device
        self.encoding_key = encoding_key
        self.prc_codeword = (Encode(encoding_key) + 1) / 2
        self.prc_index = 0

        assert frequencies is not None or (encoding is not None and decoding is not None)

        if frequencies is not None:
            self.generate_huffman_encoding(frequencies)
        else:
            self.encoding = encoding
            self.decoding = decoding  # Corrected: This should be the *decoding* dict
            
        # Precompute prefix mappings for optimization
        self._precompute_prefix_mappings()

    def _precompute_prefix_mappings(self):
        """
        Precompute mappings from prefixes to possible tokens for faster lookup.
        This builds a dictionary mapping each possible prefix to the set of tokens
        that could follow it.
        """
        # Initialize prefix-to-tokens mapping
        self.prefix_to_tokens = {}
        
        # For each token and its binary code
        for token_id, code in self.encoding.items():
            # Add all prefixes of this code to the mapping
            for i in range(len(code) + 1):
                prefix = code[:i]
                if prefix not in self.prefix_to_tokens:
                    self.prefix_to_tokens[prefix] = set()
                self.prefix_to_tokens[prefix].add(token_id)
                
        # Convert sets to frozen sets for efficiency
        self.prefix_to_tokens = {prefix: frozenset(tokens) for prefix, tokens in self.prefix_to_tokens.items()}
        
        # Create mapping from prefix + bit to new possible tokens
        self.prefix_extension = {}
        for prefix in self.prefix_to_tokens:
            self.prefix_extension[(prefix, '0')] = frozenset(
                token_id for token_id in self.prefix_to_tokens[prefix]
                if len(self.encoding[token_id]) > len(prefix) and self.encoding[token_id][len(prefix):len(prefix)+1] == '0'
            )
            self.prefix_extension[(prefix, '1')] = frozenset(
                token_id for token_id in self.prefix_to_tokens[prefix]
                if len(self.encoding[token_id]) > len(prefix) and self.encoding[token_id][len(prefix):len(prefix)+1] == '1'
            )

    def generate_huffman_encoding(self, frequencies):
        self.encoding = huffman_encode(frequencies)
        self.decoding = {code: token_id for token_id, code in self.encoding.items()}
        
    def predict_binary_probs(self, original_token_probs, prefix):
        """
        Optimized version that uses precomputed prefix mappings.
        
        THIS LOGIC IS MESSED UP. We want to consider tokens whose encoding starts with the prefix then a zero. 
        """
        # Get possible tokens for this prefix
        if prefix in self.prefix_to_tokens:
            possible_tokens = self.prefix_to_tokens[prefix]
        else:
            assert False
            
        # If no possible tokens, return equal probabilities
        if not possible_tokens:
            assert False
        
        # Calculate probability for bit '1'
        prob_of_zero = sum(original_token_probs.get(token_id, 0) for token_id in possible_tokens)
        # consider floating point precision
        prob_of_zero = min(prob_of_zero, 1.0)
        prob_of_one = 1.0 - prob_of_zero
        
        # For efficiency, return both probabilities
        return prob_of_zero, prob_of_one

    def sample_binary_token(self, x_i, hat_p_i):
        """
        Samples a binary token from the biased probabilities.

        x_i is the current index in the PRC codeword.
        hat_p_i is the E[p] which specifies the distribution over the next binary token.
        """
        if hat_p_i <= 0.5:
            # t_i <- Ber(2x_i * hat_p_i)
            return np.random.binomial(1, 2 * x_i * hat_p_i)
        else:
            # t_i <- Ber(1 - 2(1 - x_i)(1 - hat_p_i))
            return np.random.binomial(1, 1 - 2 * (1 - x_i) * (1 - hat_p_i))

    def watermarked_generate(self, prompt, num_tokens):
        """
        Generates text using the watermarked, binarized model.
        Important: num_tokens is now the number of tokens in the *original* vocab.
        Args:
            num_tokens: the number of tokens *from the original vocabulary* we want to generate.
        """
        binary_tokens = []
        output_tokens = []
        output_text = ""

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device) # Encode the prompt!

        for _ in tqdm(range(num_tokens), desc="Generating tokens"):
            with torch.no_grad():
                outputs = self.original_model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]  # Get logits for the last token
                probs = torch.softmax(logits, dim=0)
                original_token_probs = {i: probs[i].item() for i in range(len(probs))}

            prefix = ""
            # loop until s becomes a valid encoding, and do not stop if eos_token is generated
            while True:
                prob_of_zero, prob_of_one = self.predict_binary_probs(original_token_probs, prefix)
                
                # Sample bit using PRC watermarking
                x_i = self.prc_codeword[self.prc_index].item() 
                next_bit = self.sample_binary_token(x_i, prob_of_one)
                binary_tokens.append(next_bit)
                self.prc_index += 1

                # if we've used all the bits in the PRC codeword, reset it
                if self.prc_index == len(self.prc_codeword):
                    self.prc_index = 0
                    self.prc_codeword = (Encode(self.encoding_key) + 1) / 2

                prefix += str(next_bit)

                if len(prefix) % 1000 == 0:
                    print(f"Prefix: {len(prefix)}")

                if prefix in self.decoding:
                    decoded_token_id = self.decoding[prefix] 
                    output_tokens.append(decoded_token_id)
                    decoded_str = self.tokenizer.decode([decoded_token_id])
                    input_ids = torch.cat([input_ids, torch.tensor([[decoded_token_id]]).to(self.device)], dim=-1)
                    output_text += decoded_str
                    break # Exit the inner loop

        return output_tokens, output_text

def setup(vocab_size, exp_id, n, fpr, prc_t):
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
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
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--prompt', type=str, default='Tell me a fantastical story about a wizard.')
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    # parser.add_argument('--dataset_id', type=str, default='databricks/databricks-dolly-15k')
    parser.add_argument('--inf_steps', type=int, default=50)
    parser.add_argument('--nowm', type=int, default=0)
    parser.add_argument('--fpr', type=float, default=0.00001)
    parser.add_argument('--prc_t', type=int, default=3)
    args = parser.parse_args()
    print(args)

    hf_cache_dir = '/home/lawrence/.cache/huggingface/hub'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 4 * 64 * 64  # the length of a PRC codeword (for stable diffusion)
    test_num = args.test_num
    model_id = args.model_id
    nowm = args.nowm
    fpr = args.fpr
    prc_t = args.prc_t
    exp_id = f'binarize_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'

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
    encoding_key, decoding_key = setup(vocab_size, exp_id, n, fpr, prc_t)
    print(f"PRC keys set up")

    if not os.path.exists(f'encoding_{exp_id}.pkl'):
        token_counts = {token_id: 0 for token_id in range(vocab_size)}
        with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
            example_corpus = f.readlines()
        for sentence in example_corpus:
            input_ids = tokenizer.encode(sentence)
            for token_id in input_ids:
                token_counts[token_id] += 1

        encoding = huffman_encode(token_counts)
        # save encoding to file
        with open(f'encoding_{exp_id}.pkl', 'wb') as f:
            pickle.dump(encoding, f)
    else:
        with open(f'encoding_{exp_id}.pkl', 'rb') as f:
            encoding = pickle.load(f)
    print(f"Encoding loaded")

    decoding = {code: token_id for token_id, code in encoding.items()}

    # Binarize the model
    binarized_model = BinarizedModel(model, encoding_key, tokenizer=tokenizer, encoding=encoding, decoding=decoding)

    # Generate text
    output_tokens, output_text = binarized_model.watermarked_generate(prompt, num_tokens=7)
    print(f"Output text: {output_text}")

if __name__ == "__main__":
    main()