# PRC Watermarking

This package provides different watermarking schemes for text generation models based on Pseudorandom Codes (PRC). The package implements three watermarking approaches:

1. **Binary Watermarking**: Embeds watermarks at the bit level in the Huffman encoding of tokens
2. **Token Watermarking**: Embeds watermarks by influencing token selection using PRC bits
3. **Independent Hash Watermarking**: Uses position-specific hash functions for token-level watermarking

## Package Structure

- `__init__.py`: Main interface that exports all classes and functions
- `binary_watermarking.py`: Implementation of bit-level watermarking
- `token_watermarking.py`: Implementation of token-level watermarking
- `independent_token_watermarking.py`: Implementation of position-specific hash functions for token-level watermarking
- `detection.py`: Detection functions for all watermarking schemes
- `run_watermarking.py`: Main script to run the watermarking experiments

## How the Watermarking Works

### Binary Watermarking

1. Converts tokens to a binary string using Huffman encoding
2. Watermarks the binary string using PRC bits
3. Generates text by decoding the watermarked binary string back to tokens

### Token Watermarking

1. Hashes vocabulary into two buckets (0 and 1)
2. For each position, selects the bucket indicated by the PRC bit
3. Samples a token from the selected bucket

### Independent Hash Watermarking

1. Uses a unique hash function for each position in the text
2. Hashes vocabulary into two buckets (0 and 1) using the position-specific hash function
3. For each position, selects the bucket indicated by the PRC bit
4. Samples a token from the selected bucket
