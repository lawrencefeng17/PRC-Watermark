#!/usr/bin/env python
"""
Script to analyze the distribution of 0s and 1s in Huffman-encoded tokens.
This script works with the Huffman encoding used in the watermarking scheme
described in "Pseudorandom Error-Correcting Codes," Christ & Gunn 2024.
"""

import argparse
import pickle
import json
import os
import numpy as np
from collections import defaultdict

# Try importing matplotlib, but handle the case where it fails
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print("Warning: matplotlib could not be imported. Plotting is disabled.")

from transformers import AutoTokenizer

def analyze_huffman_encoding_bit_distribution(encoding, token_frequencies=None, tokenizer=None, top_n=100, plot=True):
    """
    Analyzes the distribution of 0s and 1s in Huffman-encoded tokens.
    
    Args:
        encoding: A dictionary mapping token_ids to their binary Huffman codes.
        token_frequencies: A dictionary mapping token_ids to their frequencies. If None, all tokens are assumed to have equal frequency.
        tokenizer: Optional tokenizer to decode token IDs for better visualization.
        top_n: Number of most frequent tokens to analyze in detail.
        plot: Whether to create visualizations of the results.
        
    Returns:
        A dictionary containing the analysis results.
    """
    # Create default token frequencies if not provided
    if token_frequencies is None:
        print("No token frequencies provided. Assuming equal frequency for all tokens.")
        token_frequencies = {token_id: 1 for token_id in encoding.keys()}
    
    # Calculate overall bit distribution
    all_bits = []
    for token_id, code in encoding.items():
        all_bits.extend(code)
    
    total_zeros = all_bits.count('0')
    total_ones = all_bits.count('1')
    total_bits = len(all_bits)
    
    print(f"Overall distribution in all codes:")
    print(f"  Total bits: {total_bits}")
    print(f"  Zeros: {total_zeros} ({total_zeros/total_bits:.2%})")
    print(f"  Ones: {total_ones} ({total_ones/total_bits:.2%})")
    
    # Sort tokens by frequency
    sorted_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:top_n]
    
    # Analyze bit distribution in top tokens
    top_token_bits = []
    top_token_analysis = []
    
    print(f"\nBit distribution in top {top_n} most frequent tokens:")
    for token_id, freq in top_tokens:
        if token_id not in encoding:
            continue
            
        code = encoding[token_id]
        zeros = code.count('0')
        ones = code.count('1')
        total = len(code)
        
        top_token_bits.extend(code)
        
        token_text = f"Token {token_id}"
        if tokenizer:
            token_text = f"'{tokenizer.decode([token_id])}' (ID: {token_id})"
        
        print(f"  {token_text}, Freq: {freq}, Code: {code}")
        print(f"    Length: {total}, Zeros: {zeros} ({zeros/total:.2%}), Ones: {ones} ({ones/total:.2%})")
        
        top_token_analysis.append({
            'token_id': token_id,
            'token_text': token_text,
            'frequency': freq,
            'code': code,
            'length': total,
            'zeros': zeros,
            'ones': ones,
            'zero_ratio': zeros/total if total > 0 else 0,
            'one_ratio': ones/total if total > 0 else 0
        })
    
    # Calculate bit distribution in top tokens
    top_zeros = top_token_bits.count('0')
    top_ones = top_token_bits.count('1')
    top_total = len(top_token_bits)
    
    if top_total > 0:
        print(f"\nOverall distribution in top {top_n} tokens:")
        print(f"  Total bits: {top_total}")
        print(f"  Zeros: {top_zeros} ({top_zeros/top_total:.2%})")
        print(f"  Ones: {top_ones} ({top_ones/top_total:.2%})")
    
    # Distribution by code length
    code_lengths = [len(code) for code in encoding.values()]
    avg_code_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0
    
    print(f"\nCode length statistics:")
    print(f"  Average code length: {avg_code_length:.2f}")
    print(f"  Min code length: {min(code_lengths) if code_lengths else 0}")
    print(f"  Max code length: {max(code_lengths) if code_lengths else 0}")
    
    # Calculate weighted bit distribution (by token frequency)
    weighted_zeros = 0
    weighted_ones = 0
    total_weighted_bits = 0
    
    for token_id, freq in token_frequencies.items():
        if token_id in encoding:
            code = encoding[token_id]
            zeros = code.count('0')
            ones = code.count('1')
            
            weighted_zeros += zeros * freq
            weighted_ones += ones * freq
            total_weighted_bits += len(code) * freq
    
    if total_weighted_bits > 0:
        print(f"\nFrequency-weighted bit distribution (more representative of actual usage):")
        print(f"  Total weighted bits: {total_weighted_bits}")
        print(f"  Weighted zeros: {weighted_zeros} ({weighted_zeros/total_weighted_bits:.2%})")
        print(f"  Weighted ones: {weighted_ones} ({weighted_ones/total_weighted_bits:.2%})")
    
    # Analyze the first bit of each code (most significant bit)
    first_bit_zeros = 0
    first_bit_ones = 0
    weighted_first_bit_zeros = 0
    weighted_first_bit_ones = 0
    
    for token_id, code in encoding.items():
        if len(code) > 0:
            if code[0] == '0':
                first_bit_zeros += 1
                if token_id in token_frequencies:
                    weighted_first_bit_zeros += token_frequencies[token_id]
            else:
                first_bit_ones += 1
                if token_id in token_frequencies:
                    weighted_first_bit_ones += token_frequencies[token_id]
    
    total_first_bits = first_bit_zeros + first_bit_ones
    total_weighted_first_bits = weighted_first_bit_zeros + weighted_first_bit_ones
    
    if total_first_bits > 0:
        print(f"\nDistribution of first bit in Huffman codes:")
        print(f"  Zeros: {first_bit_zeros} ({first_bit_zeros/total_first_bits:.2%})")
        print(f"  Ones: {first_bit_ones} ({first_bit_ones/total_first_bits:.2%})")
    
    if total_weighted_first_bits > 0:
        print(f"\nFrequency-weighted distribution of first bit:")
        print(f"  Zeros: {weighted_first_bit_zeros} ({weighted_first_bit_zeros/total_weighted_first_bits:.2%})")
        print(f"  Ones: {weighted_first_bit_ones} ({weighted_first_bit_ones/total_weighted_first_bits:.2%})")
    
    # Analyze bit position distribution (is there a bias at specific positions?)
    max_code_length = max(code_lengths) if code_lengths else 0
    position_zeros = [0] * max_code_length
    position_ones = [0] * max_code_length
    position_counts = [0] * max_code_length
    
    for token_id, code in encoding.items():
        for i, bit in enumerate(code):
            position_counts[i] += 1
            if bit == '0':
                position_zeros[i] += 1
            else:
                position_ones[i] += 1
    
    if max_code_length > 0:
        print(f"\nBit distribution by position (up to position {min(10, max_code_length)}):")
        for i in range(min(10, max_code_length)):
            zero_ratio = position_zeros[i] / position_counts[i] if position_counts[i] > 0 else 0
            one_ratio = position_ones[i] / position_counts[i] if position_counts[i] > 0 else 0
            print(f"  Position {i+1}: Zeros: {position_zeros[i]} ({zero_ratio:.2%}), Ones: {position_ones[i]} ({one_ratio:.2%})")
    
    if plot and has_matplotlib and len(top_token_analysis) > 0:
        # Plot bit distribution by token frequency
        plt.figure(figsize=(10, 6))
        
        # Plot zero ratio vs frequency
        zero_ratios = [analysis['zero_ratio'] for analysis in top_token_analysis]
        frequencies = [analysis['frequency'] for analysis in top_token_analysis]
        
        plt.scatter(frequencies, zero_ratios, alpha=0.7)
        plt.title('Ratio of 0s in Huffman Code vs Token Frequency')
        plt.xlabel('Token Frequency')
        plt.ylabel('Ratio of 0s in Code')
        plt.grid(True, alpha=0.3)
        plt.savefig('huffman_zero_ratio_vs_frequency.png')
        print("Saved plot: huffman_zero_ratio_vs_frequency.png")
        
        # Plot code length vs frequency
        plt.figure(figsize=(10, 6))
        code_lengths = [analysis['length'] for analysis in top_token_analysis]
        
        plt.scatter(frequencies, code_lengths, alpha=0.7)
        plt.title('Huffman Code Length vs Token Frequency')
        plt.xlabel('Token Frequency')
        plt.ylabel('Code Length')
        plt.grid(True, alpha=0.3)
        plt.savefig('huffman_length_vs_frequency.png')
        print("Saved plot: huffman_length_vs_frequency.png")
        
        # Distribution of zeros and ones
        plt.figure(figsize=(10, 6))
        bins = 10
        plt.hist([analysis['zero_ratio'] for analysis in top_token_analysis], bins=bins, alpha=0.5, label='Zeros')
        plt.hist([analysis['one_ratio'] for analysis in top_token_analysis], bins=bins, alpha=0.5, label='Ones')
        plt.title('Distribution of Bit Ratios in Top Tokens')
        plt.xlabel('Ratio in Code')
        plt.ylabel('Number of Tokens')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('huffman_bit_ratio_distribution.png')
        print("Saved plot: huffman_bit_ratio_distribution.png")
        
        # Plot bit distribution by position
        if max_code_length > 0:
            plt.figure(figsize=(12, 6))
            positions = list(range(1, min(20, max_code_length) + 1))
            zero_ratios_by_pos = [position_zeros[i-1] / position_counts[i-1] if position_counts[i-1] > 0 else 0 
                                for i in positions]
            one_ratios_by_pos = [position_ones[i-1] / position_counts[i-1] if position_counts[i-1] > 0 else 0 
                                for i in positions]
            
            plt.bar([p - 0.2 for p in positions], zero_ratios_by_pos, width=0.4, label='Zeros', alpha=0.7)
            plt.bar([p + 0.2 for p in positions], one_ratios_by_pos, width=0.4, label='Ones', alpha=0.7)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Equal distribution')
            plt.title('Bit Distribution by Position in Huffman Code')
            plt.xlabel('Position in Code')
            plt.ylabel('Ratio of Bits')
            plt.xticks(positions)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('huffman_bit_distribution_by_position.png')
            print("Saved plot: huffman_bit_distribution_by_position.png")
    
    return {
        'total_stats': {
            'total_bits': total_bits,
            'zeros': total_zeros,
            'ones': total_ones,
            'zero_ratio': total_zeros/total_bits if total_bits > 0 else 0,
            'one_ratio': total_ones/total_bits if total_bits > 0 else 0
        },
        'top_token_stats': {
            'total_bits': top_total,
            'zeros': top_zeros,
            'ones': top_ones,
            'zero_ratio': top_zeros/top_total if top_total > 0 else 0,
            'one_ratio': top_ones/top_total if top_total > 0 else 0
        },
        'weighted_stats': {
            'total_bits': total_weighted_bits,
            'zeros': weighted_zeros,
            'ones': weighted_ones,
            'zero_ratio': weighted_zeros/total_weighted_bits if total_weighted_bits > 0 else 0,
            'one_ratio': weighted_ones/total_weighted_bits if total_weighted_bits > 0 else 0
        },
        'first_bit_stats': {
            'zeros': first_bit_zeros,
            'ones': first_bit_ones,
            'zero_ratio': first_bit_zeros/total_first_bits if total_first_bits > 0 else 0,
            'one_ratio': first_bit_ones/total_first_bits if total_first_bits > 0 else 0,
            'weighted_zeros': weighted_first_bit_zeros,
            'weighted_ones': weighted_first_bit_ones,
            'weighted_zero_ratio': weighted_first_bit_zeros/total_weighted_first_bits if total_weighted_first_bits > 0 else 0,
            'weighted_one_ratio': weighted_first_bit_ones/total_weighted_first_bits if total_weighted_first_bits > 0 else 0
        },
        'code_length_stats': {
            'avg_length': avg_code_length,
            'min_length': min(code_lengths) if code_lengths else 0,
            'max_length': max(code_lengths) if code_lengths else 0
        },
        'top_token_analysis': top_token_analysis,
        'position_stats': {
            'position_zeros': position_zeros,
            'position_ones': position_ones,
            'position_counts': position_counts
        }
    }

def analyze_encoded_text_bits(encoding, text, tokenizer):
    """
    Analyzes the distribution of 0s and 1s in encoded text.
    
    Args:
        encoding: A dictionary mapping token_ids to their binary Huffman codes.
        text: Text to analyze.
        tokenizer: Tokenizer to convert text to token IDs.
        
    Returns:
        A dictionary containing the bit distribution analysis.
    """
    token_ids = tokenizer.encode(text)
    binary_representation = ''
    
    for token_id in token_ids:
        if token_id in encoding:
            binary_representation += encoding[token_id]
    
    zeros = binary_representation.count('0')
    ones = binary_representation.count('1')
    total = len(binary_representation)
    
    if total == 0:
        print("Warning: No bits to analyze in the encoded text.")
        return {
            'total_bits': 0,
            'zeros': 0,
            'ones': 0,
            'zero_ratio': 0,
            'one_ratio': 0,
            'chunk_stats': []
        }
    
    print(f"\nBit distribution in encoded text:")
    print(f"  Total bits: {total}")
    print(f"  Zeros: {zeros} ({zeros/total:.2%})")
    print(f"  Ones: {ones} ({ones/total:.2%})")
    
    # Check for position bias in the stream
    chunk_size = 100
    chunks = [binary_representation[i:i+chunk_size] for i in range(0, len(binary_representation), chunk_size)]
    
    chunk_stats = []
    for i, chunk in enumerate(chunks):
        chunk_zeros = chunk.count('0')
        chunk_ones = chunk.count('1')
        chunk_total = len(chunk)
        
        if chunk_total > 0:
            chunk_stats.append({
                'chunk_id': i,
                'total': chunk_total,
                'zeros': chunk_zeros,
                'ones': chunk_ones,
                'zero_ratio': chunk_zeros/chunk_total,
                'one_ratio': chunk_ones/chunk_total
            })
    
    # Plot chunk distribution
    if has_matplotlib and len(chunk_stats) > 0:
        plt.figure(figsize=(12, 6))
        chunk_ids = [stat['chunk_id'] for stat in chunk_stats]
        zero_ratios = [stat['zero_ratio'] for stat in chunk_stats]
        
        plt.plot(chunk_ids, zero_ratios, marker='o', alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Equal distribution')
        plt.title('Ratio of 0s in Encoded Text by Chunks')
        plt.xlabel('Chunk ID')
        plt.ylabel('Ratio of 0s')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('encoded_text_bit_distribution.png')
        print("Saved plot: encoded_text_bit_distribution.png")
    
    return {
        'total_bits': total,
        'zeros': zeros,
        'ones': ones,
        'zero_ratio': zeros/total,
        'one_ratio': ones/total,
        'chunk_stats': chunk_stats
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze Huffman encoding bit distribution')
    parser.add_argument('--encoding_file', type=str, default='encoding.pkl', help='Path to the encoding pickle file')
    parser.add_argument('--counts_file', type=str, default=None, help='Path to the token counts pickle file (optional)')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct', 
                        help='Model ID for tokenizer (if different from the one used to create the encoding)')
    parser.add_argument('--top_n', type=int, default=100, help='Number of top tokens to analyze in detail')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    parser.add_argument('--analyze_text', type=str, default='', help='Text to analyze bit distribution in encoded form')
    parser.add_argument('--verbose', action='store_true', help='Print detailed token information')
    parser.add_argument('--generate_counts', action='store_true', help='Generate token counts from a sample corpus')
    parser.add_argument('--corpus_file', type=str, default='pride_and_prejudice.txt', help='Sample corpus to use for token counts if generating')
    args = parser.parse_args()
    
    # Load encoding
    print(f"Loading encoding from {args.encoding_file}...")
    with open(args.encoding_file, 'rb') as f:
        encoding = pickle.load(f)
    
    # Load or generate token counts
    token_counts = None
    if args.counts_file and os.path.exists(args.counts_file):
        print(f"Loading token counts from {args.counts_file}...")
        with open(args.counts_file, 'rb') as f:
            token_counts = pickle.load(f)
    elif args.generate_counts and os.path.exists(args.corpus_file):
        print(f"Generating token counts from {args.corpus_file}...")
        # Load tokenizer
        print(f"Loading tokenizer for {args.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
        # Generate token counts
        token_counts = defaultdict(int)
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    input_ids = tokenizer.encode(line)
                    for token_id in input_ids:
                        token_counts[token_id] += 1
                except Exception as e:
                    print(f"Warning: Error encoding line: {e}")
        
        # Save token counts
        with open('token_counts_generated.pkl', 'wb') as f:
            pickle.dump(dict(token_counts), f)
        print("Token counts generated and saved to token_counts_generated.pkl")
    else:
        print("No token counts file provided or found. Using equal frequencies for analysis.")
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None
    
    # Analyze Huffman encoding
    print("\nAnalyzing Huffman encoding bit distribution...")
    analysis_results = analyze_huffman_encoding_bit_distribution(
        encoding=encoding,
        token_frequencies=token_counts,
        tokenizer=tokenizer,
        top_n=args.top_n,
        plot=not args.no_plot
    )
    
    # Save results to file
    with open('huffman_analysis.json', 'w') as f:
        # Convert any non-serializable objects to strings
        analysis_dict = {
            'total_stats': analysis_results['total_stats'],
            'top_token_stats': analysis_results['top_token_stats'],
            'weighted_stats': analysis_results['weighted_stats'],
            'first_bit_stats': analysis_results['first_bit_stats'],
            'code_length_stats': analysis_results['code_length_stats'],
            'top_token_analysis': [
                {k: str(v) if k == 'token_text' else v for k, v in item.items()}
                for item in analysis_results['top_token_analysis'][:20]  # Save only first 20 for brevity
            ]
        }
        json.dump(analysis_dict, f, indent=2)
    print(f"Analysis results saved to huffman_analysis.json")
    
    # Analyze text if provided
    if args.analyze_text and tokenizer:
        print(f"\nAnalyzing bit distribution in encoded text: '{args.analyze_text}'")
        text_analysis = analyze_encoded_text_bits(encoding, args.analyze_text, tokenizer)
        
        with open('text_bit_analysis.json', 'w') as f:
            json.dump(text_analysis, f, indent=2)
        print(f"Text analysis results saved to text_bit_analysis.json")

if __name__ == "__main__":
    main() 