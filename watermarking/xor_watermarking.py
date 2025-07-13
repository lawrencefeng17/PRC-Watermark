"""
XOR-based watermarking scheme.

This module contains the implementation of a watermarking scheme that generates
n tokens per codeword bit and uses XOR of bucket hashes to embed watermark bits.
The method generates groups of n tokens, computes XOR of their bucket hashes,
and keeps the group only if the XOR matches the target codeword bit.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from tqdm import tqdm

from src.prc import Encode, KeyGen

class XORWatermarkModel:
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, 
                 vocab_size=None, temperature=1.0, top_p=0.9, group_size=2, model_id=None):
        """
        Args:
            original_model: The original language model.
            encoding_key: The key for the PRC encoding.
            decoding_key: The key for the PRC decoding.
            n: The length of the PRC codeword.
            tokenizer: The tokenizer for the model.
            vocab_size: The vocabulary size.
            temperature: The temperature for the model.
            top_p: The cumulative probability for top-p sampling.
            group_size: Number of tokens to generate per codeword bit.
            model_id: The model ID for determining cache type.
        """
        self.original_model = original_model
        self.tokenizer = tokenizer
        self.device = next(original_model.parameters()).device
        self.dtype = next(original_model.parameters()).dtype
        self.encoding_key = encoding_key
        self.decoding_key = decoding_key
        X_pm1 = Encode(encoding_key)
        self.prc_codeword = ((1 - X_pm1) / 2).long()
        self.temperature = temperature
        self.model_id = model_id

        self.vocab_size = vocab_size
        self.token_hashes = torch.randint(0, 2, (self.vocab_size,), device=self.device)
        self.hash_function = lambda x: int(self.token_hashes[x])
        
        self.group_size = group_size
        self.retry_count = 0
        self.total_groups = 0

        assert len(self.prc_codeword) == n
        self.n = n

        self.prc_index = 0
        self.top_p = top_p
        
    def _initialize_cache(self, input_ids, max_tokens):
        """Initialize KV cache based on model type."""
        # Check if this is a Gemma-3 model that needs HybridCache
        if self.model_id and "gemma-3" in self.model_id.lower():
            try:
                from transformers import HybridCache
                # Calculate max cache length (input + max_tokens)
                max_cache_length = input_ids.shape[1] + max_tokens
                
                past_key_values = HybridCache(
                    config=self.original_model.config,
                    max_batch_size=1,
                    max_cache_len=max_cache_length,
                    device=self.device,
                    dtype=self.dtype
                )
                print(f"Using HybridCache for Gemma-3 model with max_cache_len={max_cache_length}")
                return past_key_values
            except ImportError:
                print("HybridCache not available, falling back to regular caching")
                return None
        else:
            return None

    def generate_token_group(self, input_ids, attention_mask, past_key_values, target_bit, top_p=None, greedy=False, debug=False, skip_xor_check=False):
        """
        Generate a group of tokens and check if their XOR matches the target bit.
        
        Args:
            skip_xor_check: If True, always return tokens regardless of XOR match
        
        Returns:
            tokens: List of token IDs if successful, None if XOR doesn't match (unless skip_xor_check=True)
            new_input_ids: Updated input_ids for next generation
            new_attention_mask: Updated attention mask
            new_past_key_values: Updated KV cache
            retry_needed: True if XOR didn't match target bit (False if skip_xor_check=True)
        """
        top_p = top_p if top_p is not None else self.top_p
        group_tokens = []
        group_hashes = []
        
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        current_past_key_values = past_key_values
        
        # Generate group_size tokens
        for i in range(self.group_size):
            with torch.no_grad():
                outputs = self.original_model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=current_past_key_values
                )
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits / self.temperature, dim=-1)
                # Update KV cache for all models
                current_past_key_values = outputs.past_key_values

            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            top_p_mask = cumulative_probs < top_p
            # Include the first token that exceeds top_p, or all if sum < top_p
            if top_p_mask.sum() < len(sorted_probs):
                top_p_mask[top_p_mask.sum()] = True 
            else:
                pass  # top_p_mask already includes all tokens if sum < top_p

            top_p_indices = sorted_indices[top_p_mask]
            top_p_probs = sorted_probs[top_p_mask]
            
            # Normalize
            top_p_probs = top_p_probs / torch.sum(top_p_probs)

            # Sample token from top-p distribution
            if greedy:
                sampled_relative_index = torch.argmax(top_p_probs).item()
            else:
                sampled_relative_index = torch.multinomial(top_p_probs, num_samples=1).item()
            
            token_id = top_p_indices[sampled_relative_index].item()
            token_hash = self.hash_function(token_id)
            
            group_tokens.append(token_id)
            group_hashes.append(token_hash)
            
            next_token_tensor = torch.tensor([[token_id]], device=self.device)
            current_input_ids = next_token_tensor
            current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token_tensor)], dim=1)
        
        # Compute XOR of group hashes
        group_xor = 0
        for hash_val in group_hashes:
            group_xor ^= hash_val
        
        # Check if XOR matches target bit
        if skip_xor_check or group_xor == target_bit:
            return group_tokens, current_input_ids, current_attention_mask, current_past_key_values, False
        else:
            return None, input_ids, attention_mask, past_key_values, True

    def watermarked_generate(self, input_ids, num_codeword_bits, top_p=None, greedy=False, debug=True, max_retries_per_group=100):
        """
        Generate watermarked text using XOR-based method.
        
        Args:
            input_ids: The initial input_ids for generation.
            num_codeword_bits: Number of codeword bits to embed (each requires group_size tokens).
            top_p: The cumulative probability for top-p sampling.
            greedy: Whether to use greedy sampling.
            debug: Whether to generate debug plots and statistics.
            max_retries_per_group: Maximum retries before giving up on a group.
            
        Returns:
            output_tokens: List of generated token ids
            output_text: Generated text
            xor_distribution_data: Data about XOR patterns for analysis
            retry_statistics: Statistics about retries per group
        """
        output_tokens = []
        xor_distribution_data = []
        retry_statistics = []
        
        top_p = top_p if top_p is not None else self.top_p

        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Get model configuration
        config = self.original_model.config
        max_position_embeddings = getattr(config, 'max_position_embeddings', None)
        if max_position_embeddings and input_ids.shape[1] > max_position_embeddings:
            input_ids = input_ids[:, -max_position_embeddings:]
            attention_mask = attention_mask[:, -max_position_embeddings:]

        # Initialize KV cache based on model type
        # Calculate total tokens we might generate (with retries)
        estimated_max_tokens = num_codeword_bits * self.group_size * 10
        past_key_values = self._initialize_cache(input_ids, estimated_max_tokens)
        
        if debug:
            self.retry_count = 0
            self.total_groups = 0

        with tqdm(total=num_codeword_bits, desc="Generating codeword bits", disable=not debug) as pbar:
            for bit_index in range(num_codeword_bits):
                # Get target bit from PRC codeword
                target_bit = self.prc_codeword[self.prc_index].item()
                
                # Try to generate a group that matches the target bit
                retries_for_this_group = 0
                group_generated = False
                
                kv_cache_before_group_attempt = copy.deepcopy(past_key_values)
                while not group_generated and retries_for_this_group < max_retries_per_group:
                    group_tokens, input_ids, attention_mask, updated_past_key_values, retry_needed = self.generate_token_group(
                        input_ids, attention_mask, kv_cache_before_group_attempt, target_bit, top_p, greedy, debug
                    )
                    
                    if not retry_needed:
                        # Success! Add tokens to output
                        past_key_values = updated_past_key_values
                        output_tokens.extend(group_tokens)
                        group_generated = True
                        
                        # Store XOR data for analysis
                        group_hashes = [self.hash_function(token) for token in group_tokens]
                        group_xor = 0
                        for hash_val in group_hashes:
                            group_xor ^= hash_val
                        
                        xor_distribution_data.append({
                            'group_tokens': group_tokens,
                            'group_hashes': group_hashes,
                            'group_xor': int(group_xor),
                            'target_bit': target_bit,
                            'retries': retries_for_this_group,
                            'failed': False
                        })
                    else:
                        retries_for_this_group += 1
                        if debug:
                            self.retry_count += 1
                
                if not group_generated:
                    print(f"Warning: Failed to generate group after {max_retries_per_group} retries for bit {bit_index}")
                    # Generate tokens anyway without XOR constraint
                    group_tokens, input_ids, attention_mask, past_key_values, _ = self.generate_token_group(
                        input_ids, attention_mask, past_key_values, target_bit, top_p, greedy, debug, skip_xor_check=True
                    )
                    output_tokens.extend(group_tokens)
                    
                    # Still store data for analysis
                    group_hashes = [self.hash_function(token) for token in group_tokens]
                    group_xor = 0
                    for hash_val in group_hashes:
                        group_xor ^= hash_val
                    
                    xor_distribution_data.append({
                        'group_tokens': group_tokens,
                        'group_hashes': group_hashes,
                        'group_xor': int(group_xor),
                        'target_bit': target_bit,
                        'retries': retries_for_this_group,
                        'failed': True
                    })
                
                retry_statistics.append(retries_for_this_group)
                
                if debug:
                    self.total_groups += 1
                
                # Update PRC index
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
            print(f"Total retries: {self.retry_count}")
            print(f"Total groups: {self.total_groups}")
            print(f"Average retries per group: {self.retry_count / self.total_groups if self.total_groups > 0 else 0:.2f}")
            print(f"Generated {len(output_tokens)} tokens for {num_codeword_bits} codeword bits")
            print(f"Tokens per codeword bit: {len(output_tokens) / num_codeword_bits:.2f}")
            
            # Plot retry statistics
            plt.figure(figsize=(10, 6))
            plt.hist(retry_statistics, bins=range(max(retry_statistics) + 2), alpha=0.7, edgecolor='black')
            plt.title(f"Distribution of Retries per Group (Group Size = {self.group_size})")
            plt.xlabel("Number of Retries")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"xor_retry_distribution_groupsize_{self.group_size}.png")
            plt.close()
            
            # Plot XOR success rate
            xor_values = [data['group_xor'] for data in xor_distribution_data]
            target_bits = [data['target_bit'] for data in xor_distribution_data]
            matches = [xor_val == target for xor_val, target in zip(xor_values, target_bits)]
            success_rate = sum(matches) / len(matches) if len(matches) > 0 else 0
            
            print(f"XOR match success rate: {success_rate:.4f}")
            
            # Plot XOR distribution
            plt.figure(figsize=(8, 6))
            plt.hist(xor_values, bins=[0, 1, 2], alpha=0.7, edgecolor='black')
            plt.title(f"Distribution of XOR Values (Group Size = {self.group_size})")
            plt.xlabel("XOR Value")
            plt.ylabel("Frequency")
            plt.xticks([0, 1])
            plt.grid(True, alpha=0.3)
            plt.savefig(f"xor_value_distribution_groupsize_{self.group_size}.png")
            plt.close()

        # Generate output text
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_tokens, output_text, xor_distribution_data, retry_statistics, success_rate
