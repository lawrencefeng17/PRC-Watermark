"""
Tree XOR watermarking scheme.

Instead of performing rejection sampling on groups of tokens, 
we consider all possible next substrings of length group_size.

For every substring, we compute the XOR of the bucket hashes of the tokens.

These substrings of length group_size correspond to paths from the top of the tree to a leaf.
The probability of any substring is the product of the probabilities of the tokens in the substring.

The XOR of each substring determines which bucket the substring belongs to. 
Then, we use the embedding scheme from CG24 to choose which bucket to sample from in a 
distribution-preserving manner.

Then, we renormalize the probabilities of the substrings in the chosen bucket,
then sample from the renormalized distribution.

Instead of considering every possible substring of length group_size,
which will form a tree of size |V|^group_size, we can either restrict ourselves to the top-k tokens,
such that each node has k children. Alternatively, we can force the search to have width w,
so that the number of nodes in any layer is at most w, maintaining only the top-w substrings so far.
"""
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import inspect
from tqdm import tqdm
from transformers import HybridCache
from typing import Any
from dataclasses import dataclass
from src.prc import Encode, KeyGen
import heapq

@dataclass
class Node:
    prefix: list[int]          
    logp:   float              
    # Remove pkv and pos to save memory - we'll recompute from scratch
    
    def bucket(self, hash_fn) -> int:
        x = 0
        for t in self.prefix[1:]:
            x ^= hash_fn(t)
        return x


class TreeXORWatermarkModel:
    def __init__(self, original_model, encoding_key, decoding_key, n, tokenizer=None, 
                 vocab_size=None, group_size=4, debug=True):
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
        """
        self.original_model = original_model
        self.tokenizer = tokenizer
        self.device = next(original_model.parameters()).device
        self.dtype = next(original_model.parameters()).dtype
        self.encoding_key = encoding_key
        self.decoding_key = decoding_key
        X_pm1 = Encode(encoding_key)
        self.prc_codeword = ((1 - X_pm1) / 2).long()

        self.vocab_size = vocab_size
        self.token_hashes = torch.randint(0, 2, (self.vocab_size,), device=self.device)
        self.hash_function = lambda x: int(self.token_hashes[x])
        
        self.group_size = group_size

        assert len(self.prc_codeword) == n
        self.n = n

        sig = inspect.signature(self.original_model.forward)
        self.needs_cache_position = "cache_position" in sig.parameters
        self.use_hybrid_cache = any(
            k in original_model.name_or_path.lower()
            for k in ["gemma-2", "gemma-3"]
        )

        self.prc_index = 0
        self.debug = debug
        
        # Debug tracking variables
        self.rejection_count = 0
        self.rejections = []
        
    def _model_step(self, token_ids: torch.Tensor, pkv, pos: int | None):
        kwargs = dict(
            input_ids = token_ids,            # shape (B,1)
            past_key_values = pkv,
            use_cache = True,
            return_dict = True,
        )
        if self.needs_cache_position:
            kwargs["cache_position"] = torch.tensor([pos], device=self.device)
        outs = self.original_model(**kwargs)
        return outs.logits[:, -1], outs.past_key_values

    def _expand_frontier(self, frontier: list[Node], base_sequence: torch.Tensor) -> list[Node]:
        """
        Expand a node into a list of children, using recomputation instead of KV caching.
        This trades computation for memory efficiency.
        """
        all_children = []
        
        # Process each node in the frontier
        for parent in frontier:
            # Reconstruct full sequence for this node
            if len(parent.prefix) == 1:
                # Root node - just use base sequence
                full_sequence = base_sequence
            else:
                # Build sequence: base_sequence + parent.prefix[1:]
                parent_tokens = torch.tensor(parent.prefix[1:], device=self.device).unsqueeze(0)
                full_sequence = torch.cat([base_sequence, parent_tokens], dim=1)
            
            # Run model on full sequence (no KV caching)
            with torch.no_grad():
                outputs = self.original_model(
                    input_ids=full_sequence,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]  # Get last token logits
            
            # Get top-k candidates
            topk_probs, topk_idx = torch.topk(F.softmax(logits / self.temperature, -1), self.top_k)
            
            # Create children
            for p, idx in zip(topk_probs.squeeze(0).tolist(), topk_idx.squeeze(0).tolist()):
                child = Node(
                    prefix=parent.prefix + [idx],
                    logp=parent.logp + math.log(p),
                )
                all_children.append(child)
        
        if self.beam_size is not None:
            all_children = heapq.nlargest(self.beam_size, all_children, key=lambda n: n.logp)
        
        # Clear intermediate data
        torch.cuda.empty_cache()
        
        return all_children

    def _form_buckets(self, leaves: list[Node]):
        """
        Form buckets from leaves.
        """
        buckets = {0: [], 1: []}
        for leaf in leaves:
            b = leaf.bucket(self.hash_function)
            buckets[b].append((leaf, math.exp(leaf.logp)))

        pushforward_probs = torch.tensor([sum(p for _, p in buckets[0]), sum(p for _, p in buckets[1])], device=self.device)
        pushforward_probs = pushforward_probs / torch.sum(pushforward_probs)
        return buckets, pushforward_probs

    def _choose_bucket(self, pushforward_probs: torch.Tensor, target_bit: int, debug=True):
        """
        Choose a bucket from leaves with rejection tracking.
        """
        ones_tensor = torch.ones(1, device=self.device)
        p = torch.min(ones_tensor, len(pushforward_probs) * pushforward_probs[target_bit])
        
        if torch.bernoulli(p) == 1:
            if debug:
                self.rejections.append(False)
            return target_bit
        else:
            if debug:
                self.rejection_count += 1
                self.rejections.append(True)
            
            q_i = pushforward_probs - torch.tensor(1/len(pushforward_probs), device=self.device)
            # only positive q_i
            q_i = torch.clamp(q_i, min=0)
            # normalize
            q_i = q_i / torch.sum(q_i)
            
            return torch.multinomial(q_i, num_samples=1).item()

    def _sample_from_bucket(self, bucket: list[tuple[Node, float]]):
        """
        Sample from a bucket.
        """
        nodes, probs = zip(*bucket)
        probs = torch.tensor(probs, device=self.device)
        probs = probs / torch.sum(probs)
        idx = torch.multinomial(probs, 1).item()
        return nodes[idx]

    def generate_next_substring(self, input_ids, attention_mask, past_key_values, target_bit, token_offset=0, full_sequence=None):
        """
        Generate the next substring of length group_size.
        """
        # Construct base sequence from full_sequence if provided, otherwise reconstruct
        if full_sequence is not None:
            base_sequence = full_sequence
        else:
            # This shouldn't happen with our new approach, but fallback
            base_sequence = input_ids
            
        # 1. Build tree
        root = Node(prefix=[input_ids.item()], logp=0.0)  # Removed pkv and pos
        frontier = [root]

        for depth in range(self.group_size):
            # Clear previous frontier to free memory
            if depth > 0:
                del old_frontier
                torch.cuda.empty_cache()
            
            old_frontier = frontier  # Keep reference to delete later
            frontier = self._expand_frontier(frontier, base_sequence)
            
            # Aggressive memory cleanup between levels
            if depth < self.group_size - 1:  # Don't cleanup on last iteration
                torch.cuda.empty_cache()

        buckets, pushforward_probs = self._form_buckets(frontier)
        bucket_index = self._choose_bucket(pushforward_probs, target_bit, debug=self.debug)
        bucket = buckets[bucket_index]
        leaf = self._sample_from_bucket(bucket)

        # Final cleanup
        del frontier, buckets 
        torch.cuda.empty_cache()

        new_tokens = leaf.prefix[1:]
        next_input = torch.tensor([[new_tokens[-1]]], device=self.device)
        
        # Since we're not using KV caching in tree search, we need to recompute 
        # the KV cache for the main generation loop
        if past_key_values is not None:
            # Reconstruct full sequence with new tokens
            updated_full_sequence = torch.cat([
                base_sequence,
                torch.tensor([new_tokens], device=self.device)
            ], dim=1)
            
            # Recompute KV cache for the updated sequence
            with torch.no_grad():
                outputs = self.original_model(
                    input_ids=updated_full_sequence,
                    use_cache=True,
                    return_dict=True,
                )
                new_past_key_values = outputs.past_key_values
        else:
            new_past_key_values = None
        
        # Return pushforward_probs for debug tracking
        return new_tokens, next_input, new_past_key_values, pushforward_probs

    def watermarked_generate(self, input_ids, num_codeword_bits, temperature=1.0, top_k=20, beam_width=None):
        """
        Generate watermarked text using tree XOR watermarking.
        """
        self.temperature = temperature
        self.top_k = top_k
        self.beam_size = beam_width
        
        output_tokens = []
        xor_distribution_data = []
        
        # Debug tracking variables
        if self.debug:
            self.rejection_count = 0
            self.rejections = []
            pushforward_probs_seq = []
        
        # Keep track of the full sequence for memory-efficient tree search
        full_sequence = input_ids.clone()
        
        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        past_key_values = None
        if self.use_hybrid_cache:
            past_key_values = HybridCache(
                config=self.original_model.config,
                max_batch_size=1,
                max_cache_len=input_ids.shape[1] + num_codeword_bits * self.group_size,
                device=self.device,
                dtype=self.dtype,
            )

        prefill_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

        input_ids = input_ids[:, -1:]
        attention_mask = None

        with tqdm(total=num_codeword_bits, desc="Generating codeword bits", disable=not self.debug) as pbar:
            for bit_index in range(num_codeword_bits):
                # Get target bit from PRC codeword
                target_bit = self.prc_codeword[self.prc_index].item()
                
                # Clear cache before each group generation
                torch.cuda.empty_cache()
                
                group_tokens, input_ids, past_key_values, pushforward_probs = self.generate_next_substring(
                    input_ids, attention_mask, past_key_values, target_bit, 
                    token_offset=prefill_len + len(output_tokens),
                    full_sequence=full_sequence
                )
                
                # Track pushforward probabilities for debug
                if self.debug:
                    pushforward_probs_seq.append(pushforward_probs.cpu().numpy())
                
                # Update full sequence with new tokens
                full_sequence = torch.cat([
                    full_sequence,
                    torch.tensor([group_tokens], device=self.device)
                ], dim=1)
                
                output_tokens.extend(group_tokens)
                
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
                    'pushforward_probs': pushforward_probs.cpu().numpy() if self.debug else None,
                })
 
                
                # Update PRC index
                self.prc_index += 1
                if self.prc_index == len(self.prc_codeword):
                    if self.debug:
                        print("Generating new PRC codeword")
                    self.prc_index = 0
                    X_pm1 = Encode(self.encoding_key)
                    self.prc_codeword = ((1 - X_pm1) / 2).long()
                
                if self.debug:
                    pbar.update(1)

        if self.debug:
            # Debug plots and statistics
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Rejection rate statistics
            print(f"Rejection count: {self.rejection_count}")
            print(f"Rejection rate: {self.rejection_count / num_codeword_bits if num_codeword_bits > 0 else 0}")

            # Plot rejections vs index
            if self.rejections:
                plt.figure(figsize=(8, 6))
                plt.plot(self.rejections)
                plt.title("Rejections vs Group Index")
                plt.xlabel("Group Index")
                plt.ylabel("Rejection (0=No, 1=Yes)")
                plt.savefig("tree_xor_rejections_vs_index.png")
                plt.close()
            
            # Calculate bucket entropy for each group using actual pushforward probabilities
            bucket_entropies = []
            
            for data in xor_distribution_data:
                if data['pushforward_probs'] is not None:
                    # Use actual pushforward probabilities
                    p0, p1 = data['pushforward_probs'][0], data['pushforward_probs'][1]
                    
                    # Calculate entropy
                    entropy = 0
                    if p0 > 1e-9:
                        entropy -= p0 * np.log2(p0)
                    if p1 > 1e-9:
                        entropy -= p1 * np.log2(p1)
                    bucket_entropies.append(entropy)
            
            # Plot rejection rate vs entropy
            if bucket_entropies and self.rejections:
                plt.figure(figsize=(10, 6))
                
                # Create entropy bins
                num_bins = 10
                entropies_array = np.array(bucket_entropies)
                rejections_array = np.array(self.rejections, dtype=float)
                
                if len(entropies_array) > 0 and not np.all(entropies_array == entropies_array[0]):
                    bins = np.linspace(np.min(entropies_array), np.max(entropies_array), num_bins + 1)
                    binned_indices = np.digitize(entropies_array, bins, right=False)
                    
                    rejection_rates_in_bins = []
                    bin_centers = []
                    
                    for i in range(1, num_bins + 1):
                        mask = (binned_indices == i)
                        if np.any(mask):
                            rejection_rates_in_bins.append(np.mean(rejections_array[mask]))
                            bin_centers.append((bins[i-1] + bins[i]) / 2)
                    
                    if bin_centers:
                        plt.plot(bin_centers, rejection_rates_in_bins, marker='o', linestyle='-')
                        plt.title("Rejection Rate vs. Bucket Entropy")
                        plt.xlabel("Binary Entropy of Bucket Probabilities")
                        plt.ylabel("Average Rejection Rate")
                        plt.ylim(0, 1)
                        plt.grid(True)
                else:
                    plt.text(0.5, 0.5, "Insufficient entropy variation for binning", 
                             ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title("Rejection Rate vs. Bucket Entropy (Insufficient Data)")
                
                plt.savefig("tree_xor_rejection_rate_vs_entropy.png")
                plt.close()
            
            # Plot bucket entropy distribution
            if bucket_entropies:
                plt.figure(figsize=(8, 6))
                plt.hist(bucket_entropies, bins=20, range=(0, 1), alpha=0.7)
                plt.title(f"Distribution of Bucket Entropies (mean: {np.mean(bucket_entropies):.4f})")
                plt.xlabel("Binary Entropy")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.savefig("tree_xor_bucket_entropy_distribution.png")
                plt.close()
            
            # Plot bucket probability distributions (similar to token watermarking)
            if pushforward_probs_seq:
                bucket_0_probs = [probs[0] for probs in pushforward_probs_seq]
                bucket_1_probs = [probs[1] for probs in pushforward_probs_seq]
                
                plt.figure(figsize=(8, 6))
                plt.hist(bucket_0_probs, bins=20, range=(0, 1), alpha=0.7)
                plt.title(f"Distribution of Bucket 0 Probabilities (mean: {np.mean(bucket_0_probs):.4f})")
                plt.xlabel("Probability in Bucket 0")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.savefig("tree_xor_bucket_0_distribution.png")
                plt.close()
                
                plt.figure(figsize=(8, 6))
                plt.hist(bucket_1_probs, bins=20, range=(0, 1), alpha=0.7)
                plt.title(f"Distribution of Bucket 1 Probabilities (mean: {np.mean(bucket_1_probs):.4f})")
                plt.xlabel("Probability in Bucket 1")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.savefig("tree_xor_bucket_1_distribution.png")
                plt.close()
            
            # Print statistics
            if bucket_entropies:
                print(f"Mean bucket entropy: {np.mean(bucket_entropies):.4f}")
                print(f"Std bucket entropy: {np.std(bucket_entropies):.4f}")
            if pushforward_probs_seq:
                bucket_0_probs = [probs[0] for probs in pushforward_probs_seq]
                bucket_1_probs = [probs[1] for probs in pushforward_probs_seq]
                print(f"Mean probability in bucket 0: {np.mean(bucket_0_probs):.4f}")
                print(f"Mean probability in bucket 1: {np.mean(bucket_1_probs):.4f}")
                print(f"Frequency of bucket 0 being empty (prob = 0): {np.mean(np.array(bucket_0_probs) == 0.0):.4f}")
                print(f"Frequency of bucket 1 being empty (prob = 0): {np.mean(np.array(bucket_1_probs) == 0.0):.4f}")

        log_data = {
            "mean_bucket_0_prob": np.mean(bucket_0_probs),
            "mean_bucket_1_prob": np.mean(bucket_1_probs),
            "rejection_count": self.rejection_count,
            "rejection_rate": self.rejection_count / num_codeword_bits if num_codeword_bits > 0 else 0,
        }
        
        # Generate output text
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_tokens, output_text, xor_distribution_data, log_data