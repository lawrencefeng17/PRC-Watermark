import numpy as np
import torch
import matplotlib.pyplot as plt

import math
import pickle
from tqdm import tqdm

from src.prc import Encode, Encode_No_OTP, Decode, KeyGen

from transformers import AutoTokenizer, AutoModelForCausalLM

class IndependentHashModel:
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, vocab_size=None, temperature=1.0, top_p=0.9):
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
        # Instead of creating hash functions on-the-fly, we'll pre-generate a fixed number
        # of hash functions to use for positions and cycle through them if needed
        self.num_precomputed_hashes = min(n, 1000)  # Pre-compute up to 1000 position hashes
        self.position_token_hashes = {}
        self._precompute_position_hashes()
        
        self.prc_index = 0
        self.n = n
        self.top_p = top_p
    
    def _precompute_position_hashes(self):
        """
        Pre-compute position-specific hash tensors for efficiency.
        """
        for position in range(self.num_precomputed_hashes):
            # Generate a deterministic random hash based on position
            generator = torch.Generator(device=self.device)
            generator.manual_seed(position)
            self.position_token_hashes[position] = torch.randint(
                0, 2, (self.vocab_size,), 
                device=self.device, 
                generator=generator
            )
    
    def get_hash_function_for_position(self, position):
        """
        Gets or creates a hash function for a specific position.
        Each position has its own independent random hash mapping from tokens to {0,1}.
        """
        # Use modulo to cycle through pre-computed hash functions if we exceed the number
        position_key = position % self.num_precomputed_hashes
        
        # Return a hash function that uses the hash tensor for this position
        return lambda x: self.position_token_hashes[position_key][x].item()
    
    def get_hash_tensor_for_position(self, position):
        """
        Returns the hash tensor for a specific position.
        """
        position_key = position % self.num_precomputed_hashes
        return self.position_token_hashes[position_key]
    
    def hash_token(self, position, token_id):
        """
        Hash a token using the hash tensor for a specific position.
        This is a non-lambda version that can be used after loading saved hash tensors.
        """
        position_key = position % self.num_precomputed_hashes
        if position_key in self.position_token_hashes:
            return self.position_token_hashes[position_key][token_id].item()
        else:
            # Fallback should not happen with precomputed hashes
            return (token_id % 2)
    
    def embed_char(self, x_j, pushforward_probs, debug=True):
        """
        Choose a bucket according to the pushforward distribution.
        
        Compute Bernoulli(min(1, |Sigma_PRC| * pushforward_probs[x_j])).
        If the result is 1, return x_j.
        Otherwise, return a random bucket according to the pushforward distribution.
        
        We only choose a bucket if it weight is at least 1/|Sigma_PRC|.
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
        Generates text by hashing the vocabulary into two buckets using position-dependent hash functions,
        then sampling from the bucket indicated by the prc-bit.
        Only considers top-p tokens for hashing and sampling.

        Args:
            prompt: The initial prompt for generation.
            num_tokens: The number of tokens to generate.
            top_p: The cumulative probability for top-p sampling. Defaults to self.top_p.
            debug: Whether to generate debug plots and statistics.
            
        Returns:
            output_tokens: List of generated token ids
            output_text: Generated text
            pushforward_probs_seq: List of pushforward_probs for each token
            prc_bits_used_seq: List of PRC bits used for each token
            hashed_output_tokens_seq: List of hash values of output tokens
            position_hash_tensors: Dictionary mapping positions to token hash tensors
            rejection_count: Number of rejections during generation
            entropies_for_rejection_analysis: List of entropies for rejection analysis
        """
        output_tokens = []
        output_text = ""
        pushforward_probs_seq = []
        prc_bits_used_seq = []
        hashed_output_tokens_seq = []
        
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
        else:
            self.rejection_count = 0 # Always init rejection_count even without debug mode

        with tqdm(total=num_tokens, desc="Generating tokens", disable=not debug) as pbar:
            for position in range(num_tokens):
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

                # Get position-specific hash tensor directly - much faster than using lambdas
                position_hash_tensor = self.get_hash_tensor_for_position(position)
                
                # Vectorized hash lookup - much faster than list comprehension
                current_hash_tensor = position_hash_tensor[top_p_indices]

                # let x_i be the current PRC codeword bit
                x_i = self.prc_codeword[self.prc_index].item() 

                pushforward_probs = torch.zeros(2, device=self.device)
                # Ensure top_p_probs has the same dtype as pushforward_probs
                top_p_probs_compatible = top_p_probs.to(dtype=pushforward_probs.dtype)
                pushforward_probs.scatter_add_(0, index=current_hash_tensor, src=top_p_probs_compatible)
                
                # Store the PRC bit for this position
                prc_bits_used_seq.append(x_i)
                
                # Store pushforward probs for each token generation
                pushforward_probs_seq.append(pushforward_probs.cpu().numpy())
                
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
                
                # Store the hash of the generated token using the position-specific hash tensor
                token_hash = position_hash_tensor[token_id].item()
                hashed_output_tokens_seq.append(token_hash)
                
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
            
            # Calculate binary entropy for each pushforward_probs
            entropies_for_rejection_analysis = []
            for probs_pair in pushforward_probs_seq:
                p0 = probs_pair[0]
                p1 = probs_pair[1]
                entropy = 0
                # Avoid log(0) issues; 0 * log(0) is 0
                if p0 > 1e-9: # Using a small epsilon to avoid float precision issues with 0
                    entropy -= p0 * np.log2(p0)
                if p1 > 1e-9: # Using a small epsilon
                    entropy -= p1 * np.log2(p1)
                entropies_for_rejection_analysis.append(entropy)
            
            entropies_for_rejection_analysis = np.array(entropies_for_rejection_analysis)
            rejections_array = np.array(self.rejections, dtype=float) # Convert booleans to float for mean

            # Create entropy bins
            num_bins = 20
            # Max entropy for binary is 1.0. Min is 0.
            bins = np.linspace(0, 1.0, num_bins + 1)
            
            # Ensure bins cover the full range, even if all entropies are e.g. 0 or 1
            if entropies_for_rejection_analysis.size > 0:
                min_entropy = np.min(entropies_for_rejection_analysis)
                max_entropy = np.max(entropies_for_rejection_analysis)
                # Adjust bins slightly to include min and max values if they are exactly on bin edges
                # and to ensure all data is captured by digitize.
                # Bins for np.digitize are (inclusive, exclusive] for right=False (default)
                # or [inclusive, exclusive) for right=True.
                # We want to include 0 and 1.
                bins = np.linspace(min_entropy - 1e-9, max_entropy + 1e-9, num_bins + 1)
                if min_entropy < 0: bins[0] = min_entropy -1e-9 # handle potential float inaccuracies
                if max_entropy > 1: bins[-1] = max_entropy + 1e-9 # handle potential float inaccuracies
                if bins[0] > 0 and min_entropy < bins[0]: bins[0] = min_entropy -1e-9
                if bins[-1] < 1 and max_entropy > bins[-1]: bins[-1] = max_entropy + 1e-9

            # Ensure bins are monotonically increasing if adjustments made them not so (edge case)
            for i in range(len(bins) -1):
                if bins[i+1] <= bins[i]:
                    bins[i+1] = bins[i] + 1e-9 # ensure strictly increasing

            # Handle case with no data or all data points having the same entropy
            if entropies_for_rejection_analysis.size == 0:
                print("No data for entropy vs rejection plot.")
                rejection_rates_in_bins = np.zeros(num_bins)
                bin_centers = np.zeros(num_bins)
            elif np.all(entropies_for_rejection_analysis == entropies_for_rejection_analysis[0]):
                 print("All entropy values are the same. Plotting a single point for entropy vs rejection.")
                 rejection_rates_in_bins = np.array([np.mean(rejections_array) if rejections_array.size > 0 else 0])
                 bin_centers = np.array([entropies_for_rejection_analysis[0]])
            else:
                # np.digitize returns 1-based indices.
                binned_indices = np.digitize(entropies_for_rejection_analysis, bins, right=False)
                
                rejection_rates_in_bins = np.zeros(num_bins)
                bin_centers = np.zeros(num_bins)
                
                for i in range(1, num_bins + 1): # Bins are 1-indexed by digitize
                    mask = (binned_indices == i)
                    if np.any(mask):
                        rejection_rates_in_bins[i-1] = np.mean(rejections_array[mask])
                    else:
                        rejection_rates_in_bins[i-1] = np.nan # Use NaN if bin is empty
                    bin_centers[i-1] = (bins[i-1] + bins[i]) / 2
            
            plt.figure(figsize=(10, 6))
            # Filter out NaN values for plotting if some bins were empty
            valid_indices = ~np.isnan(rejection_rates_in_bins)
            if np.any(valid_indices):
                plt.plot(bin_centers[valid_indices], rejection_rates_in_bins[valid_indices], marker='o', linestyle='-')
            else: # Case where all bins might be NaN (e.g. very few data points)
                print("Not enough distinct data points to plot entropy vs rejection rate across bins.")

            plt.title("Rejection Rate vs. Bucket Entropy")
            plt.xlabel("Binary Entropy of Bucket Probabilities (pushforward_probs)")
            plt.ylabel("Average Rejection Rate")
            plt.ylim(0, 1) # Rejection rate is between 0 and 1
            plt.xlim(0, 1) # Entropy is between 0 and 1
            plt.grid(True)
            plt.savefig("rejection_rate_vs_entropy.png")
            plt.close()

            # Plot the distribution of bucket weights
            plt.figure(figsize=(8, 6))
            if weight_zero_bucket and len(weight_zero_bucket) > 0 and not np.isnan(np.sum(weight_zero_bucket)):
                plt.hist(weight_zero_bucket, bins=20, range=(0, 1))
                plt.title(f"Distribution of bucket 0 weights (mean: {np.mean(weight_zero_bucket):.4f})")
            else:
                plt.text(0.5, 0.5, "No valid bucket 0 weight data to plot", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title("Distribution of bucket 0 weights (No Data)")
            plt.xlabel("Probability in bucket 0")
            plt.ylabel("Frequency")
            plt.savefig("bucket_0_distribution.png")
            plt.close()
            
            plt.figure(figsize=(8, 6))
            if weight_one_bucket and len(weight_one_bucket) > 0 and not np.isnan(np.sum(weight_one_bucket)):
                plt.hist(weight_one_bucket, bins=20, range=(0, 1))
                plt.title(f"Distribution of bucket 1 weights (mean: {np.mean(weight_one_bucket):.4f})")
            else:
                plt.text(0.5, 0.5, "No valid bucket 1 weight data to plot", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title("Distribution of bucket 1 weights (No Data)")
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

        if not debug:
            entropies_for_rejection_analysis = None
            
        # Return the generated tokens, text, metrics, and position hash tensors
        return output_tokens, output_text, pushforward_probs_seq, prc_bits_used_seq, hashed_output_tokens_seq, self.position_token_hashes, self.rejection_count, entropies_for_rejection_analysis 

    def prepare_for_detection(self, position_hash_tensors, num_precomputed_hashes=None):
        """
        Prepare the model for detection by setting up the hash tensors.
        
        Args:
            position_hash_tensors: Dictionary mapping positions to token hash tensors
            num_precomputed_hashes: Number of precomputed hash tensors to use for modulo
            
        Returns:
            self: The model instance for method chaining
        """
        # Update the position hash tensors
        self.position_token_hashes = position_hash_tensors
        
        # Update the number of precomputed hashes if provided
        if num_precomputed_hashes is not None:
            self.num_precomputed_hashes = num_precomputed_hashes
        else:
            self.num_precomputed_hashes = len(position_hash_tensors)
            
        return self 