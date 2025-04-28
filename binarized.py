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

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset

class BinarizedModel:
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, frequencies=None, encoding=None, decoding=None, temperature=1.0, hash_function=None):
        """
        Args:
            original_model: The original (non-binary) language model.
            encoding_key: The key for the PRC encoding.
            tokenizer: The tokenizer for the model.
            frequencies: A dictionary mapping original tokens to frequencies.
            encoding:  A dictionary mapping original tokens to binary strings (prefix-free).
            decoding: A dictionary mapping binary strings to original tokens.
            temperature: The temperature for the model.
        """
        self.original_model = original_model
        self.tokenizer = tokenizer
        self.device = next(original_model.parameters()).device
        self.encoding_key = encoding_key
        self.decoding_key = decoding_key
        X_pm1 = Encode(encoding_key)
        self.prc_codeword = ((1 - X_pm1) / 2).long()
        self.temperature = temperature
        self.hash_function = hash_function
        self.rejection_count = 0

        assert len(self.prc_codeword) == n
        self.n = n

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
        Optimized version that calculates the probability of the next bit being 0 or 1.
        
        For bit 0: sum probabilities of tokens whose encoding starts with prefix+'0'
        For bit 1: sum probabilities of tokens whose encoding starts with prefix+'1'
        """
        # Get tokens that could follow this prefix with a 0 or 1
        prefix_plus_zero = (prefix, '0')
        prefix_plus_one = (prefix, '1')
        
        # Get tokens that could follow this prefix with a 0
        tokens_with_zero = self.prefix_extension.get(prefix_plus_zero, frozenset())
        # Get tokens that could follow this prefix with a 1
        tokens_with_one = self.prefix_extension.get(prefix_plus_one, frozenset())
        
        # If no possible continuations, return equal probabilities
        if not tokens_with_zero and not tokens_with_one:
            assert False
        
        # Calculate probability for bit '0'
        prob_of_zero = sum(original_token_probs.get(token_id, 0) for token_id in tokens_with_zero)
        # Calculate probability for bit '1'
        prob_of_one = sum(original_token_probs.get(token_id, 0) for token_id in tokens_with_one)
        
        # Normalize to ensure they sum to 1
        total = prob_of_zero + prob_of_one
        if total > 0:
            prob_of_zero /= total
            prob_of_one /= total
        else:
            # If all tokens have zero probability, default to equal probabilities
            prob_of_zero = 0.5
            prob_of_one = 0.5
        
        prob_of_one = min(prob_of_one, 1.0)
        # For efficiency, return both probabilities
        return 1.0 - prob_of_one, prob_of_one

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

    def watermarked_generate(self, prompt, num_bits, debug=True):
        """
        Generates text using the watermarked, binarized model with KV-caching for improved efficiency.
        
        Args:
            prompt: The initial prompt for generation
            num_bits: The number of binary bits to generate
            debug: Whether to generate debug plots and statistics (default True for backward compatibility)
        """
        binary_tokens = []
        output_tokens = []
        output_text = ""

        # Encode the prompt and prepare for generation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Store the hat_p_i values for debugging
        hat_p_i_values = [] if debug else None
        entropies = [] if debug else None

        if debug:
            rejection_count = 0

        # Initial forward pass to get the KV cache for the prompt
        with torch.no_grad():
            # Get the initial output and KV cache
            outputs = self.original_model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]  # Get logits for the last token
            probs = torch.softmax(logits / self.temperature, dim=0)
            original_token_probs = {i: probs[i].item() for i in range(len(probs))}
            if debug:
                entropy = -torch.sum(probs * torch.log2(probs))
                entropies.append(entropy.cpu().item())

        with tqdm(total=num_bits, desc="Generating bits", disable=not debug) as pbar:
            while len(binary_tokens) < num_bits:
                prefix = ""
                # loop until s becomes a valid encoding, and do not stop if eos_token is generated
                while True:
                    prob_of_zero, prob_of_one = self.predict_binary_probs(original_token_probs, prefix)

                    if debug:
                        hat_p_i_values.append(prob_of_one)

                    # Sample bit using PRC watermarking
                    x_i = self.prc_codeword[self.prc_index].item() 
                    next_bit = self.sample_binary_token(x_i, prob_of_one)
                    if next_bit != x_i:
                        rejection_count += 1

                    binary_tokens.append(next_bit)
                    self.prc_index += 1

                    if debug:
                        pbar.update(1)

                    # if we've used all the bits in the PRC codeword, reset it
                    if self.prc_index == len(self.prc_codeword):
                        if debug:
                            print("Generating new PRC codeword")
                        self.prc_index = 0
                        X_pm1 = Encode(self.encoding_key)
                        self.prc_codeword = ((1 - X_pm1) / 2).long()

                    prefix += str(next_bit)

                    if prefix in self.decoding:
                        decoded_token_id = self.decoding[prefix] 
                        output_tokens.append(decoded_token_id)
                        decoded_str = self.tokenizer.decode([decoded_token_id])
                        output_text += decoded_str
                        
                        # Create a tensor with just the new token for the forward pass
                        next_token_tensor = torch.tensor([[decoded_token_id]], device=self.device)
                        
                        # Forward pass with KV cache
                        with torch.no_grad():
                            outputs = self.original_model(
                                input_ids=next_token_tensor,
                                past_key_values=past_key_values,
                                use_cache=True
                            )
                            # Update KV cache for next iteration
                            past_key_values = outputs.past_key_values
                            
                            # Get logits for the next token prediction
                            logits = outputs.logits[0, -1, :]
                            probs = torch.softmax(logits, dim=0)
                            original_token_probs = {i: probs[i].item() for i in range(len(probs))}
                            if debug:   
                                entropy = -torch.sum(probs * torch.log2(probs))
                                entropies.append(entropy.cpu().item())
                            
                        break # Exit the inner loop

                    if len(binary_tokens) >= num_bits:
                        break
                
                if len(binary_tokens) >= num_bits:
                    break

        if debug:
            print(f"Generated {len(hat_p_i_values)} binary bits.")
            # Plot histogram or print summary statistics
            plt.figure(figsize=(8, 6))  # Create a new figure for the first plot
            plt.hist(hat_p_i_values, bins=20, range=(0, 1))
            plt.title("Distribution of hat_p_i values encountered")
            plt.xlabel("P(next_bit = 1)")
            plt.ylabel("Frequency")
            plt.savefig(f"hat_p_i_distribution.png") # Save the plot
            print(f"Saved hat_p_i distribution plot.")
            print(f"Mean hat_p_i: {np.mean(hat_p_i_values):.4f}")
            print(f"Std dev hat_p_i: {np.std(hat_p_i_values):.4f}")
            print(f"Fraction near 0.5 (|p - 0.5| < 0.1): {np.mean(np.abs(np.array(hat_p_i_values) - 0.5) < 0.1):.4f}")
            plt.close()  # Close the first plot

            # Create a new figure for the entropy plot
            plt.figure(figsize=(8, 6))
            plt.hist(entropies, bins=20)
            plt.title(f"Entropy dist, mean: {np.mean(entropies):.4f}, std dev: {np.std(entropies):.4f}")
            plt.xlabel("Entropy")
            plt.ylabel("Frequency")
            plt.savefig(f"entropy_distribution.png") # Save the plot
            plt.close()  # Close the second plot

            print(f"Rejection count: {rejection_count}")
            print(f"Rejection rate: {rejection_count / len(binary_tokens)}")

        return output_tokens, output_text
    
    def embed_char(self, x_j, pushforward_probs, debug=True):
        """
        Choose a bucket according to the pushforward distribution.
        
        Compute Bernoulli(min(1, |Sigma_PRC| * pushforward_probs[x_j])).
        If the result is 1, return x_j.
        Otherwise, return a random bucket according to the pushforward distribution.
        
        We only choose a bucket if it weight is at least 1/|Sigma_PRC|.

        If probability of prc bucket is greater than 1/|Sigma_PRC|, we accept it with probability 1.
        Otherwise, we sample it with probability 2 * (probability of prc bucket) - 1/|Sigma_PRC|.
        Else, we sample it among random buckets with probability at least 1/|Sigma_PRC| (which means we will not sample from the prc bucket).
        """
        ones_tensor = torch.ones(1, device=pushforward_probs.device)
        p = torch.min(ones_tensor, len(pushforward_probs) * pushforward_probs[x_j])
        if torch.bernoulli(p) == 1:
            return x_j
        else:
            q_i = pushforward_probs - 1/len(pushforward_probs)
            # only positive q_i
            q_i = torch.clamp(q_i, min=0)
            # normalize
            q_i = q_i / torch.sum(q_i)
            
            if debug:
                self.rejection_count += 1
            return torch.multinomial(q_i, num_samples=1).item()

    def watermarked_generate_by_token(self, prompt, num_tokens, debug=True):
        """
        Generates text by hashing the vocabulary into two buckets, then sampling from the bucket indicated by the prc-bit.
        """
        if self.hash_function is None:
            # generate a random hash function
            self.hash_function = lambda x: torch.randint(0, 2, (1,)).item()

        output_tokens = []
        output_text = ""

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.original_model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits / self.temperature, dim=0)
            # We'll keep this line for compatibility with other code
            original_token_probs = {i: probs[i].item() for i in range(len(probs))}

        with tqdm(total=num_tokens, desc="Generating tokens", disable=not debug) as pbar:
            while len(output_tokens) < num_tokens:
                # let x_i be the current PRC codeword bit
                x_i = self.prc_codeword[self.prc_index].item() 

                # compute pushforward distribution using the hash function
                hash_tensor = torch.tensor([self.hash_function(i) for i in range(len(probs))], device=self.device)
                
                pushforward_probs = torch.zeros(2, device=self.device)
                pushforward_probs.scatter_add_(0, hash_tensor, probs)

                breakpoint()
                
                y_i = self.embed_char(x_i, pushforward_probs) # bucket chosen

                # sample token among tokens in the bucket y_i
                bucket_mask = (hash_tensor == y_i)
                
                bucket_probs = probs.clone()
                bucket_probs = torch.where(bucket_mask, bucket_probs, torch.zeros_like(bucket_probs))
                
                # normalize
                bucket_probs = bucket_probs / bucket_probs.sum()
                
                # Sample a token from the bucket
                token_id = torch.multinomial(bucket_probs, num_samples=1).item()
                output_tokens.append(token_id)
                
                # Decode the token to text
                decoded_str = self.tokenizer.decode([token_id])
                output_text += decoded_str
                
                # generate next token
                next_token_tensor = torch.tensor([[token_id]], device=self.device)
                with torch.no_grad():
                    outputs = self.original_model(
                        input_ids=next_token_tensor,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits / self.temperature, dim=0)
                
                # update PRC index
                self.prc_index += 1
                if self.prc_index == len(self.prc_codeword):
                    if debug:
                        print("Generating new PRC codeword")
                    self.prc_index = 0
                    X_pm1 = Encode(self.encoding_key)
                    self.prc_codeword = ((1 - X_pm1) / 2).long()
                
                if debug:
                    pbar.update(1)
                
        if debug:
            print(f"Rejection count: {self.rejection_count}")
            print(f"Rejection rate: {self.rejection_count / num_tokens}")

        # Return the generated tokens and text
        return output_tokens, output_text
            