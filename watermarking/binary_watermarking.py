"""
Bit-level watermarking scheme from "Pseudorandom Error-Correcting Codes," Christ & Gunn 2024.

This module contains the implementation of a binary watermarking scheme that embeds 
watermark bits at the bit level using Huffman encoding.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import math
from tqdm import tqdm

from src.prc import Encode, KeyGen

class BinaryWatermarkModel:
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, 
                 frequencies=None, encoding=None, decoding=None, temperature=1.0):
        """
        Args:
            original_model: The original language model.
            encoding_key: The key for the PRC encoding.
            decoding_key: The key for the PRC decoding.
            tokenizer: The tokenizer for the model.
            frequencies: A dictionary mapping original tokens to frequencies.
            encoding: A dictionary mapping original tokens to binary strings (prefix-free).
            decoding: A dictionary mapping binary strings to original tokens.
            temperature: The temperature for the model.
            n: Length of the PRC codeword.
        """
        self.original_model = original_model
        self.tokenizer = tokenizer
        self.device = next(original_model.parameters()).device
        self.encoding_key = encoding_key
        self.decoding_key = decoding_key
        X_pm1 = Encode(encoding_key)
        self.prc_codeword = ((1 - X_pm1) / 2).long()
        self.temperature = temperature

        assert len(self.prc_codeword) == n
        self.n = n
        self.prc_index = 0

        assert frequencies is not None or (encoding is not None and decoding is not None)

        if frequencies is not None:
            from huffman import huffman_encode
            self.encoding = huffman_encode(frequencies)
            self.decoding = {code: token_id for token_id, code in self.encoding.items()}
        else:
            self.encoding = encoding
            self.decoding = decoding
            
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
        
        This function generates text by watermarking individual bits in a Huffman-encoded representation
        of the text. 
        
        Args:
            prompt: The initial prompt for generation
            num_bits: The number of binary bits to generate
            debug: Whether to generate debug plots and statistics (default True for backward compatibility)
        
        Returns:
            output_tokens: List of generated token IDs
            output_text: Generated text
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
            if hat_p_i_values and len(hat_p_i_values) > 0 and not np.isnan(np.sum(hat_p_i_values)):
                plt.hist(hat_p_i_values, bins=20, range=(0, 1))
                plt.title("Distribution of hat_p_i values encountered")
                print(f"Mean hat_p_i: {np.mean(hat_p_i_values):.4f}")
                print(f"Std dev hat_p_i: {np.std(hat_p_i_values):.4f}")
                print(f"Fraction near 0.5 (|p - 0.5| < 0.1): {np.mean(np.abs(np.array(hat_p_i_values) - 0.5) < 0.1):.4f}")
            else:
                plt.text(0.5, 0.5, "No valid hat_p_i data to plot", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title("Distribution of hat_p_i values (No Data)")
            plt.xlabel("P(next_bit = 1)")
            plt.ylabel("Frequency")
            plt.savefig(f"hat_p_i_distribution.png") # Save the plot
            print(f"Saved hat_p_i distribution plot.")
            plt.close()  # Close the first plot

            # Create a new figure for the entropy plot
            plt.figure(figsize=(8, 6))
            if entropies and len(entropies) > 0 and not np.isnan(np.sum(entropies)):
                plt.hist(entropies, bins=20)
                plt.title(f"Entropy dist, mean: {np.mean(entropies):.4f}, std dev: {np.std(entropies):.4f}")
            else:
                plt.text(0.5, 0.5, "No valid entropy data to plot", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title("Entropy Distribution (No Data)")
            plt.xlabel("Entropy")
            plt.ylabel("Frequency")
            plt.savefig(f"entropy_distribution.png") # Save the plot
            plt.close()  # Close the second plot

            print(f"Rejection count: {rejection_count}")
            print(f"Rejection rate: {rejection_count / len(binary_tokens)}")

        return output_tokens, output_text, binary_tokens 