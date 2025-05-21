"""
PRC Watermarking library

This package provides different watermarking schemes for text generation models:

1. Binary Watermarking: Embeds watermarks at the bit level in the Huffman encoding of tokens
2. Token Watermarking: Embeds watermarks by influencing token selection using PRC bits  
3. Independent Hash Watermarking: Uses position-specific hash functions for token-level watermarking

All schemes are based on the Pseudorandom Code (PRC) watermarking approach
described in Christ & Gunn (2024).
"""

# Import watermarking models
from .binary_watermarking import BinaryWatermarkModel
from .token_watermarking import TokenWatermarkModel
from .independent_token_watermarking import IndependentHashModel

# Import detection functions
from .detection import (
    detect_binary_watermark,
    detect_binary_text_watermark,
    detect_token_watermark,
    detect_independent_hash_watermark,
    regenerate_independent_hash_functions
)

__all__ = [
    # Watermarking models
    'BinaryWatermarkModel',
    'TokenWatermarkModel',
    'IndependentHashModel',
    
    # Detection functions
    'detect_binary_watermark',
    'detect_binary_text_watermark',
    'detect_token_watermark',
    'detect_independent_hash_watermark',
    'regenerate_independent_hash_functions'
] 