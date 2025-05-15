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
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, frequencies=None, encoding=None, decoding=None, temperature=1.0, vocab_size=None, top_p=0.9):
        """
        Args:
            original_model: The original (non-binary) language model.
            encoding_key: The key for the PRC encoding.
            tokenizer: The tokenizer for the model.
            frequencies: A dictionary mapping original tokens to frequencies.
            encoding:  A dictionary mapping original tokens to binary strings (prefix-free).
            decoding: A dictionary mapping binary strings to original tokens.
            temperature: The temperature for the model.
            vocab_size: The vocabulary size.
            top_p: The cumulative probability for top-p sampling.
        """
        self.original_model = original_model
        self.tokenizer = tokenizer
        self.device = next(original_model.parameters()).device
        self.encoding_key = encoding_key
        self.decoding_key = decoding_key
        X_pm1 = Encode(encoding_key)
        self.prc_codeword = ((1 - X_pm1) / 2).long()
        self.temperature = temperature

        self.vocab_size = vocab_size
        self.token_hashes = torch.randint(0, 2, (self.vocab_size,), device=self.device)
        self.hash_function = lambda x: self.token_hashes[x]
        self.rejection_count = 0

        assert len(self.prc_codeword) == n
        self.n = n

        self.prc_index = 0
        self.top_p = top_p

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
        Generates text using the watermarked, binarized model.
        
        This function does not successfully produce watermarked text. The rejection rate is too high. 
        
        Args:
            prompt: The initial prompt for generation
            num_bits: The number of binary bits to generate
            debug: Whether to generate debug plots and statistics (default True for backward compatibility)
        """
        binary_tokens = []
        output_tokens = []
        output_text = ""
        hash_tensor = self.token_hashes

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
            if debug:
                self.rejections.append(False)
            return x_j
        else:
            q_i = pushforward_probs - 1/len(pushforward_probs)
            # only positive q_i
            q_i = torch.clamp(q_i, min=0)
            # normalize
            q_i = q_i / torch.sum(q_i)
            
            if debug:
                self.rejection_count += 1
                self.rejections.append(True)
            return torch.multinomial(q_i, num_samples=1).item()

    def watermarked_generate_by_token(self, prompt, num_tokens, top_p=None, greedy=False, debug=True):
        """
        Generates text by hashing the vocabulary into two buckets, then sampling from the bucket indicated by the prc-bit.
        Only considers top-p tokens for hashing and sampling.

        Args:
            prompt: The initial prompt for generation.
            num_tokens: The number of tokens to generate.
            top_p: The cumulative probability for top-p sampling. Defaults to self.top_p.
            debug: Whether to generate debug plots and statistics.
        """
        output_tokens = []
        output_text = ""
        
        top_p = top_p if top_p is not None else self.top_p

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask for proper KV cache handling
        attention_mask = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits / self.temperature, dim=0)

        if debug:
            weight_zero_bucket = []
            weight_one_bucket = []
            num_top_p_tokens = []
            self.rejection_count = 0 # Initialize rejection_count for the generation run
            self.rejections = []

        with tqdm(total=num_tokens, desc="Generating tokens", disable=not debug) as pbar:
            while len(output_tokens) < num_tokens:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                top_p_mask = cumulative_probs < top_p
                # Include the first token that exceeds top_p, or all if sum < top_p
                if top_p_mask.sum() < len(sorted_probs): # check if there are tokens beyond top_p threshold
                    top_p_mask[top_p_mask.sum()] = True 
                else: # if all tokens are within top_p (e.g. top_p = 1.0 or very few tokens)
                    pass # top_p_mask already includes all tokens if sum < top_p

                top_p_indices = sorted_indices[top_p_mask]
                top_p_probs = sorted_probs[top_p_mask]
                
                # Normalize
                top_p_probs = top_p_probs / torch.sum(top_p_probs)

                # Get hashes for the top-p tokens
                current_hash_tensor = self.token_hashes[top_p_indices]

                # let x_i be the current PRC codeword bit
                x_i = self.prc_codeword[self.prc_index].item() 

                pushforward_probs = torch.zeros(2, device=self.device)
                pushforward_probs.scatter_add_(0, index=current_hash_tensor, src=top_p_probs)
                
                if debug:
                    num_top_p_tokens.append(len(top_p_indices))
                    weight_zero_bucket.append(pushforward_probs[0].item())
                    weight_one_bucket.append(pushforward_probs[1].item())
                
                y_i = self.embed_char(x_i, pushforward_probs, debug=debug) # bucket chosen

                # sample token among tokens in the bucket y_i from the top-p set
                bucket_mask_top_p = (current_hash_tensor == y_i)
                
                # Get probabilities of tokens in the chosen bucket from the top_p set
                bucket_probs_top_p = top_p_probs.clone() # Start with normalized top_p_probs
                # Zero out probabilities of tokens not in the chosen bucket
                bucket_probs_top_p = torch.where(bucket_mask_top_p, bucket_probs_top_p, torch.zeros_like(bucket_probs_top_p))
                
                # check if the chosen bucket is empty
                if torch.sum(bucket_probs_top_p) > 0:
                    # Normalize
                    bucket_probs_top_p = bucket_probs_top_p / torch.sum(bucket_probs_top_p)
                else:
                    if debug:
                        print(f"Warning: Bucket {y_i} was empty after top-p filtering. Sampling from all top-p tokens.")
                    if torch.sum(top_p_probs) > 0: # Ensure top_p_probs itself is not all zeros
                         bucket_probs_top_p = top_p_probs 
                    else:
                        assert False

                # greedy
                if greedy:
                    sampled_relative_index = torch.argmax(bucket_probs_top_p).item()
                else:
                # multinomial
                    sampled_relative_index = torch.multinomial(bucket_probs_top_p, num_samples=1).item()
                
                # Get the actual token_id
                token_id = top_p_indices[sampled_relative_index].item()

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
                    probs = torch.softmax(logits / self.temperature, dim=0) # Update probs for the next iteration
                
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
            # rejection rate
            print(f"Rejection count: {self.rejection_count}")
            print(f"Rejection rate: {self.rejection_count / num_tokens}")

            # plot rejections vs index
            plt.figure(figsize=(8, 6))
            plt.plot(self.rejections)
            plt.title("Rejections vs index")
            plt.xlabel("Index")
            plt.ylabel("Rejection")
            plt.savefig("rejections_vs_index.png")
            plt.close()
            
            # Plot the distribution of bucket weights
            plt.figure(figsize=(8, 6))
            plt.hist(weight_zero_bucket, bins=20, range=(0, 1))
            plt.title(f"Distribution of bucket 0 weights (mean: {np.mean(weight_zero_bucket):.4f})")
            plt.xlabel("Probability in bucket 0")
            plt.ylabel("Frequency")
            plt.savefig("bucket_0_distribution.png")
            plt.close()
            
            plt.figure(figsize=(8, 6))
            plt.hist(weight_one_bucket, bins=20, range=(0, 1))
            plt.title(f"Distribution of bucket 1 weights (mean: {np.mean(weight_one_bucket):.4f})")
            plt.xlabel("Probability in bucket 1")
            plt.ylabel("Frequency")
            plt.savefig("bucket_1_distribution.png")
            plt.close()
            
            # Print some statistics about bucket weights
            print(f"Mean weight in bucket 0: {np.mean(weight_zero_bucket):.4f}")
            print(f"Mean weight in bucket 1: {np.mean(weight_one_bucket):.4f}")
            print(f"Frequency of bucket 0 being empty (weight = 0): {np.mean(np.array(weight_zero_bucket) == 0.0):.4f}")
            print(f"Frequency of bucket 1 being empty (weight = 0): {np.mean(np.array(weight_one_bucket) == 0.0):.4f}")
            print(f"Frequency of buckets being heavily imbalanced (weight > 0.95): {np.mean(np.logical_or(np.array(weight_zero_bucket) > 0.95, np.array(weight_one_bucket) > 0.95)):.4f}")

        # Return the generated tokens and text
        return output_tokens, output_text