"""
XOR-based watermarking scheme for language models.

This module implements a novel watermarking approach where we generate n tokens per codeword bit
and only keep them if the XOR of their bucket assignments matches the target PRC bit.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm

from src.prc import Encode, KeyGen

class XORWatermarkModel:
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, 
                 vocab_size=None, temperature=1.0, top_p=0.9, group_size=2):
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
            group_size: Number of tokens to generate per codeword bit (n in the method description).
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

        self.vocab_size = vocab_size
        self.token_hashes = torch.randint(0, 2, (self.vocab_size,), device=self.device)
        self.hash_function = lambda x: self.token_hashes[x]

        assert len(self.prc_codeword) == n
        self.n = n
        self.group_size = group_size

        self.prc_index = 0
        self.top_p = top_p
        
        # Statistics tracking
        self.retry_counts = []
        self.xor_attempts = []  # Track all XOR attempts for distribution analysis
        
    def generate_token_group(self, input_ids, attention_mask, past_key_values, top_p=None, greedy=False):
        """
        Generate a group of tokens using the current model state.
        
        Returns:
            tokens: List of token IDs
            bucket_values: List of bucket assignments (0 or 1) for each token
            new_past_key_values: Updated KV cache
        """
        top_p = top_p if top_p is not None else self.top_p
        
        tokens = []
        bucket_values = []
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        current_past_key_values = past_key_values
        
        for _ in range(self.group_size):
            # Forward pass
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
                current_past_key_values = outputs.past_key_values

            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            top_p_mask = cumulative_probs < top_p
            # Include the first token that exceeds top_p, or all if sum < top_p
            if top_p_mask.sum() < len(sorted_probs):
                top_p_mask[top_p_mask.sum()] = True

            top_p_indices = sorted_indices[top_p_mask]
            top_p_probs = sorted_probs[top_p_mask]
            
            # Normalize
            top_p_probs = top_p_probs / torch.sum(top_p_probs)

            # Sample token
            if greedy:
                sampled_relative_index = torch.argmax(top_p_probs).item()
            else:
                sampled_relative_index = torch.multinomial(top_p_probs, num_samples=1).item()
            
            token_id = top_p_indices[sampled_relative_index].item()
            bucket_value = self.hash_function(token_id)
            
            tokens.append(token_id)
            bucket_values.append(bucket_value)
            
            # Update input for next token in group
            next_token_tensor = torch.tensor([[token_id]], device=self.device)
            current_input_ids = next_token_tensor
            current_attention_mask = torch.ones_like(next_token_tensor)
        
        return tokens, bucket_values, current_past_key_values

    def watermarked_generate(self, input_ids, num_tokens, top_p=None, greedy=False, debug=True):
        """
        Generate watermarked text using XOR-based method.
        
        For each codeword bit, generate groups of tokens until their XOR matches the target bit.
        
        Args:
            input_ids: The initial input_ids for generation.
            num_tokens: The number of tokens to generate.
            top_p: The cumulative probability for top-p sampling.
            greedy: Whether to use greedy sampling.
            debug: Whether to generate debug plots and statistics.
            
        Returns:
            output_tokens: List of generated token ids
            output_text: Generated text
            retry_counts: Number of retries needed for each codeword bit
            xor_attempts: All XOR values attempted (for distribution analysis)
            accepted_groups: List of token groups that were accepted
        """
        output_tokens = []
        accepted_groups = []
        
        top_p = top_p if top_p is not None else self.top_p

        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Get model configuration for context length limits
        config = self.original_model.config
        max_position_embeddings = getattr(config, 'max_position_embeddings', None)
        if max_position_embeddings and input_ids.shape[1] > max_position_embeddings:
            input_ids = input_ids[:, -max_position_embeddings:]
            attention_mask = attention_mask[:, -max_position_embeddings:]

        # Reset statistics for this generation
        self.retry_counts = []
        self.xor_attempts = []

        # KV cache
        past_key_values = None
        
        # Calculate how many complete groups we can generate
        num_groups = num_tokens // self.group_size
        if num_groups == 0:
            raise ValueError(f"num_tokens ({num_tokens}) must be at least group_size ({self.group_size})")
        
        with tqdm(total=num_groups, desc=f"Generating {self.group_size}-token groups", disable=not debug) as pbar:
            for group_idx in range(num_groups):
                # Get target bit for this group
                target_bit = self.prc_codeword[self.prc_index].item()
                
                retry_count = 0
                while True:
                    # Generate a group of tokens
                    tokens, bucket_values, past_key_values = self.generate_token_group(
                        input_ids, attention_mask, past_key_values, top_p, greedy
                    )
                    
                    # Calculate XOR of bucket values
                    xor_result = 0
                    for bucket_val in bucket_values:
                        xor_result ^= bucket_val
                    
                    # Track this XOR attempt
                    self.xor_attempts.append(xor_result)
                    retry_count += 1
                    
                    # Check if XOR matches target bit
                    if xor_result == target_bit:
                        # Accept this group
                        output_tokens.extend(tokens)
                        accepted_groups.append({
                            'tokens': tokens,
                            'bucket_values': bucket_values,
                            'xor_result': xor_result,
                            'target_bit': target_bit,
                            'retry_count': retry_count
                        })
                        break
                    # If XOR doesn't match, retry (continue loop)
                
                self.retry_counts.append(retry_count)
                
                # Update input_ids and attention_mask for next group
                # Use the last group of tokens as context for the next group
                new_tokens_tensor = torch.tensor([tokens], device=self.device)
                input_ids = new_tokens_tensor
                attention_mask = torch.ones_like(new_tokens_tensor)
                
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
                    pbar.set_postfix({
                        'avg_retries': np.mean(self.retry_counts),
                        'last_retries': retry_count
                    })

        if debug:
            self._generate_debug_plots()
            self._print_statistics()

        # Generate output text
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_tokens, output_text, self.retry_counts, self.xor_attempts, accepted_groups

    def _generate_debug_plots(self):
        """Generate debug plots for XOR distribution and retry statistics."""
        
        # Plot XOR distribution
        plt.figure(figsize=(10, 6))
        xor_values = np.array(self.xor_attempts)
        unique_values, counts = np.unique(xor_values, return_counts=True)
        
        plt.subplot(2, 2, 1)
        plt.bar(unique_values, counts)
        plt.title(f"XOR Distribution (Group Size: {self.group_size})")
        plt.xlabel("XOR Result")
        plt.ylabel("Frequency")
        plt.xticks([0, 1])
        
        # Add percentage labels
        total_attempts = len(xor_values)
        for i, (val, count) in enumerate(zip(unique_values, counts)):
            plt.text(val, count + max(counts)*0.01, f'{100*count/total_attempts:.1f}%', 
                    ha='center', va='bottom')
        
        # Plot retry count distribution
        plt.subplot(2, 2, 2)
        plt.hist(self.retry_counts, bins=20, edgecolor='black', alpha=0.7)
        plt.title("Retry Count Distribution")
        plt.xlabel("Number of Retries")
        plt.ylabel("Frequency")
        
        # Plot retry counts over time
        plt.subplot(2, 2, 3)
        plt.plot(self.retry_counts, marker='o', markersize=2)
        plt.title("Retry Counts Over Time")
        plt.xlabel("Group Index")
        plt.ylabel("Number of Retries")
        
        # Plot cumulative XOR distribution
        plt.subplot(2, 2, 4)
        cumulative_0 = np.cumsum(xor_values == 0)
        cumulative_1 = np.cumsum(xor_values == 1)
        x_axis = np.arange(1, len(xor_values) + 1)
        
        plt.plot(x_axis, cumulative_0 / x_axis, label='XOR = 0', alpha=0.7)
        plt.plot(x_axis, cumulative_1 / x_axis, label='XOR = 1', alpha=0.7)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Expected (0.5)')
        plt.title("Cumulative XOR Proportions")
        plt.xlabel("Attempt Number")
        plt.ylabel("Proportion")
        plt.legend()
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"xor_watermark_analysis_groupsize_{self.group_size}.png", dpi=150)
        plt.close()

    def _print_statistics(self):
        """Print detailed statistics about the XOR watermarking process."""
        print(f"\n=== XOR Watermarking Statistics (Group Size: {self.group_size}) ===")
        
        # XOR distribution
        xor_values = np.array(self.xor_attempts)
        xor_0_count = np.sum(xor_values == 0)
        xor_1_count = np.sum(xor_values == 1)
        total_attempts = len(xor_values)
        
        print(f"Total XOR attempts: {total_attempts}")
        print(f"XOR = 0: {xor_0_count} ({100*xor_0_count/total_attempts:.1f}%)")
        print(f"XOR = 1: {xor_1_count} ({100*xor_1_count/total_attempts:.1f}%)")
        
        # Expected vs actual
        expected_prob = 0.5  # For random tokens, XOR should be ~50/50
        actual_prob_0 = xor_0_count / total_attempts
        print(f"Expected P(XOR=0): {expected_prob:.3f}, Actual: {actual_prob_0:.3f}")
        
        # Retry statistics
        retry_counts = np.array(self.retry_counts)
        print(f"\nRetry Statistics:")
        print(f"Average retries per group: {np.mean(retry_counts):.2f}")
        print(f"Median retries per group: {np.median(retry_counts):.2f}")
        print(f"Max retries for any group: {np.max(retry_counts)}")
        print(f"Groups requiring >5 retries: {np.sum(retry_counts > 5)} ({100*np.sum(retry_counts > 5)/len(retry_counts):.1f}%)")
        
        # Efficiency analysis
        theoretical_expected_retries = 1 / expected_prob  # Expected retries if truly random
        actual_expected_retries = np.mean(retry_counts)
        print(f"\nEfficiency Analysis:")
        print(f"Theoretical expected retries (if random): {theoretical_expected_retries:.2f}")
        print(f"Actual expected retries: {actual_expected_retries:.2f}")
        print(f"Efficiency ratio: {theoretical_expected_retries/actual_expected_retries:.3f}")

    def analyze_xor_distribution(self, input_ids, num_samples=1000, top_p=None):
        """
        Analyze the XOR distribution without watermarking constraints.
        
        This generates many token groups and computes their XOR distribution
        to understand the baseline randomness.
        
        Args:
            input_ids: Starting input for generation
            num_samples: Number of token groups to generate for analysis
            top_p: Top-p sampling parameter
            
        Returns:
            xor_distribution: Dictionary with XOR value frequencies
            samples: List of all XOR values computed
        """
        top_p = top_p if top_p is not None else self.top_p
        
        # Initialize
        attention_mask = torch.ones_like(input_ids).to(self.device)
        past_key_values = None
        xor_samples = []
        
        print(f"Analyzing XOR distribution for group size {self.group_size}...")
        
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            for _ in range(num_samples):
                # Generate a group of tokens
                tokens, bucket_values, past_key_values = self.generate_token_group(
                    input_ids, attention_mask, past_key_values, top_p, greedy=False
                )
                
                # Calculate XOR
                xor_result = 0
                for bucket_val in bucket_values:
                    xor_result ^= bucket_val
                
                xor_samples.append(xor_result)
                pbar.update(1)
        
        # Analyze distribution
        xor_samples = np.array(xor_samples)
        xor_0_count = np.sum(xor_samples == 0)
        xor_1_count = np.sum(xor_samples == 1)
        
        distribution = {
            0: xor_0_count,
            1: xor_1_count,
            'prob_0': xor_0_count / num_samples,
            'prob_1': xor_1_count / num_samples,
            'total_samples': num_samples
        }
        
        print(f"XOR Distribution Analysis (Group Size {self.group_size}):")
        print(f"P(XOR = 0) = {distribution['prob_0']:.4f}")
        print(f"P(XOR = 1) = {distribution['prob_1']:.4f}")
        print(f"Expected retries for target bit: {1/min(distribution['prob_0'], distribution['prob_1']):.2f}")
        
        return distribution, xor_samples.tolist() 