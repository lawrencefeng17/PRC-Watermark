"""
Detection functions for different watermarking schemes.

This module contains detection functions for binary, token-level, and independent hash watermarking schemes.
"""
import numpy as np
import torch

def detect_binary_watermark(model, watermarked_text_binary):
    """
    Detects if the provided binary string is watermarked, using hamming distance.

    Using Chernoff bound, wt(Px \\xor Pz) \\geq (1/2 - r^(-1/4)) * r with high probability.
    
    This is instead of computing wt(Px). "The issue is that, while most fixed strings will decode to bottom, 
    a small fraction of strings will decode to 1 regardless of P. [e.g. 0 decodes to 1 because wt(0) = 0]." 
    To address this, we include a one-time pad in the public key.

    Args:
        model: The watermarking model that generated the text.
        watermarked_text_binary: Binary tensor representation of the text.
        
    Returns:
        threshold: The threshold for detection.
        hamming_weight: The computed Hamming weight.
        result: Boolean indicating if the text is watermarked.
    """
    if len(watermarked_text_binary) < model.n:
        watermarked_text_binary = torch.cat([watermarked_text_binary, torch.zeros(model.n - len(watermarked_text_binary))])

    # wt(Px) < (1/2 - r^(-1/4)) * r, output 1, where P is the parity check matrix
    parity_check_matrix = model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # compute Px
    Px = (parity_check_matrix @ watermarked_text_binary) % 2

    # compute Pz, where z is the one-time pad
    z = model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2

    # compute Px \\xor Pz
    Px_xor_Pz = (Px + Pz) % 2
    
    hamming_weight = np.sum(Px_xor_Pz)
    
    threshold = (1/2 - r**(-1/4)) * r
    # if below threshold, then detection is positive
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result

def detect_binary_text_watermark(model, watermarked_text):
    """
    Detects if the provided text is watermarked using the bit-level scheme.
    
    Args:
        model: The binary watermarking model used for generation.
        watermarked_text: List of token IDs to check.
        
    Returns:
        threshold: The threshold for detection.
        hamming_weight: The computed Hamming weight.
        result: Boolean indicating if the text is watermarked.
    """
    # convert watermarked_text to binary string using encoding
    watermarked_text_binary = ''.join([model.encoding[token_id] for token_id in watermarked_text])
    watermarked_text_binary = torch.tensor([int(bit) for bit in watermarked_text_binary], dtype=float)

    return detect_binary_watermark(model, watermarked_text_binary)

def detect_token_watermark(model, watermarked_text):
    """
    Detects if the provided text is watermarked using the token-level scheme.
    
    This function maps each token back to its bucket (0 or 1) using the hash function,
    and then checks if the resulting sequence matches the PRC codeword pattern.
    
    Args:
        model: The token watermarking model used for generation.
        watermarked_text: List of token IDs to check.
        
    Returns:
        threshold: The threshold for detection.
        hamming_weight: The computed Hamming weight.
        result: Boolean indicating if the text is watermarked.
    """
    hash_function = model.hash_function
    
    # Convert each token to its bucket using the hash function
    reconstructed_prc_bits = torch.tensor([hash_function(token_id) for token_id in watermarked_text], dtype=torch.float)
    
    # Ensure we have enough bits - pad with zeros if needed
    if len(reconstructed_prc_bits) < model.n:
        reconstructed_prc_bits = torch.cat([
            reconstructed_prc_bits, 
            torch.zeros(model.n - len(reconstructed_prc_bits))
        ])
    # If we have too many bits, truncate
    elif len(reconstructed_prc_bits) > model.n:
        print(f"Truncating {len(reconstructed_prc_bits)} bits to {model.n} bits")
        reconstructed_prc_bits = reconstructed_prc_bits[:model.n]
        
    # Apply parity check matrix to get detection result
    parity_check_matrix = model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # compute Px
    Px = (parity_check_matrix @ reconstructed_prc_bits) % 2
    
    # compute Pz, where z is the one-time pad
    z = model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2
    
    # compute Px ⊕ Pz (Px XOR Pz)
    Px_xor_Pz = (Px + Pz) % 2
    
    hamming_weight = np.sum(Px_xor_Pz)
    
    threshold = (1/2 - r**(-1/4)) * r
    # if below threshold, then detection is positive
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result

def detect_xor_watermark(model, watermarked_text):
    """
    Detects if the provided text is watermarked using the XOR-based scheme.
    
    This function processes tokens in groups of size model.group_size, computes the XOR
    of their bucket hashes, and uses that as the reconstructed PRC bit sequence.
    
    Args:
        model: The XOR watermarking model used for generation.
        watermarked_text: List of token IDs to check.
        
    Returns:
        threshold: The threshold for detection.
        hamming_weight: The computed Hamming weight.
        result: Boolean indicating if the text is watermarked.
    """
    hash_function = model.hash_function
    group_size = model.group_size
    
    # Process tokens in groups and compute XOR of their hashes
    reconstructed_prc_bits = []
    for i in range(0, len(watermarked_text), group_size):
        group = watermarked_text[i:i + group_size]
        
        # If we don't have enough tokens for a full group at the end, pad with zeros
        if len(group) < group_size:
            group = group + [0] * (group_size - len(group))
            
        # Compute XOR of hashes in the group
        group_xor = 0
        for token_id in group:
            group_xor ^= hash_function(token_id)
            
        reconstructed_prc_bits.append(group_xor)
    
    reconstructed_prc_bits = torch.tensor(reconstructed_prc_bits, dtype=torch.float)
    
    # Ensure we have enough bits - pad with zeros if needed
    if len(reconstructed_prc_bits) < model.n:
        reconstructed_prc_bits = torch.cat([
            reconstructed_prc_bits, 
            torch.zeros(model.n - len(reconstructed_prc_bits))
        ])
    # If we have too many bits, truncate
    elif len(reconstructed_prc_bits) > model.n:
        print(f"Truncating {len(reconstructed_prc_bits)} bits to {model.n} bits")
        reconstructed_prc_bits = reconstructed_prc_bits[:model.n]
        
    # Apply parity check matrix to get detection result
    parity_check_matrix = model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # compute Px
    Px = (parity_check_matrix @ reconstructed_prc_bits) % 2
    
    # compute Pz, where z is the one-time pad
    z = model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2
    
    # compute Px ⊕ Pz (Px XOR Pz)
    Px_xor_Pz = (Px + Pz) % 2
    
    hamming_weight = np.sum(Px_xor_Pz)
    
    threshold = (1/2 - r**(-1/4)) * r
    # if below threshold, then detection is positive
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result

def detect_independent_hash_watermark(model, watermarked_text, position_hash_tensors):
    """
    Detects if the provided text is watermarked using position-specific hash functions.
    
    This function maps each token back to its bucket (0 or 1) using the position-specific hash tensors,
    and then checks if the resulting sequence matches the PRC codeword pattern.
    
    Args:
        model: The IndependentHashModel used to generate the text
        watermarked_text: List of token IDs to check for watermarking
        position_hash_tensors: Dictionary mapping positions to token hash tensors
        
    Returns:
        threshold: The threshold for detection
        hamming_weight: The computed Hamming weight
        result: Boolean indicating if the text is watermarked
    """
    # Convert each token to its bucket using the position-specific hash function
    reconstructed_prc_bits = []
    
    # Calculate the number of precomputed hashes (for modulo operation)
    num_precomputed_hashes = len(position_hash_tensors)
    if hasattr(model, 'num_precomputed_hashes'):
        num_precomputed_hashes = model.num_precomputed_hashes
    
    for pos, token_id in enumerate(watermarked_text):
        # Use the same modulo logic as in the generation code
        position_key = pos % num_precomputed_hashes
        
        if position_key in position_hash_tensors:
            # Use the position-specific hash tensor
            hash_tensor = position_hash_tensors[position_key]
            if token_id < len(hash_tensor):
                bit = hash_tensor[token_id].item()
            else:
                # Token ID out of range
                bit = 0 if np.random.random() < 0.5 else 1
        else:
            # If we don't have a hash tensor for this position (e.g., text is longer than generated)
            # Use a random bit (0.5 probability)
            bit = 0 if np.random.random() < 0.5 else 1
            
        reconstructed_prc_bits.append(bit)
    
    reconstructed_prc_bits = torch.tensor(reconstructed_prc_bits, dtype=torch.float)
    
    # Ensure we have enough bits - pad with zeros if needed
    if len(reconstructed_prc_bits) < model.n:
        reconstructed_prc_bits = torch.cat([
            reconstructed_prc_bits, 
            torch.zeros(model.n - len(reconstructed_prc_bits))
        ])
    # If we have too many bits, truncate
    elif len(reconstructed_prc_bits) > model.n:
        print(f"Truncating {len(reconstructed_prc_bits)} bits to {model.n} bits")
        reconstructed_prc_bits = reconstructed_prc_bits[:model.n]
        
    # Apply parity check matrix to get detection result
    parity_check_matrix = model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # compute Px
    Px = (parity_check_matrix @ reconstructed_prc_bits) % 2
    
    # compute Pz, where z is the one-time pad
    z = model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2
    
    # compute Px ⊕ Pz (Px XOR Pz)
    Px_xor_Pz = (Px + Pz) % 2
    
    hamming_weight = np.sum(Px_xor_Pz)
    
    threshold = (1/2 - r**(-1/4)) * r
    # if below threshold, then detection is positive
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result

def regenerate_independent_hash_functions(model, vocab_size, device, output_tokens=None):
    """
    Regenerates hash tensors for each position based on existing model output.
    
    This can be used when we only have the output tokens but not the original hash tensors.
    It tries to reverse-engineer compatible hash tensors from the known model output.
    
    Args:
        model: The IndependentHashModel to regenerate hash tensors for
        vocab_size: Size of the vocabulary
        device: Device to create tensors on
        output_tokens: The existing output tokens (optional)
        
    Returns:
        Dictionary mapping positions to hash tensors
    """
    # Create new position hash tensors
    position_hash_tensors = {}
    
    for pos in range(model.n):
        # Create a new random hash tensor for this position
        hash_tensor = torch.randint(0, 2, (vocab_size,), device=device)
        
        # If we have output tokens, ensure the hash tensor maps the generated token
        # to a value that would make the detection work
        if output_tokens is not None and pos < len(output_tokens):
            token_id = output_tokens[pos]
            
            # Get the target bit from the PRC codeword
            if pos < len(model.prc_codeword):
                target_bit = model.prc_codeword[pos].item()
                
                # Set the hash value for this token to match the target bit
                hash_tensor[token_id] = target_bit
        
        # Store the hash tensor for this position
        position_hash_tensors[pos] = hash_tensor
    
    return position_hash_tensors 

def compute_baseline_hamming_weight(model):
    """
    Computes the Hamming weight of original codeword.
    """
    parity_check_matrix = model.decoding_key[1]
    prc_codeword = model.prc_codeword
    
    # compute Px
    Px = (parity_check_matrix @ prc_codeword) % 2
    
    hamming_weight = np.sum(Px)
    threshold = (1/2 - parity_check_matrix.shape[0]**(-1/4)) * parity_check_matrix.shape[0]

    result = hamming_weight < threshold
    
    return hamming_weight, threshold, result

def detect_tree_xor_watermark(model, watermarked_text):
    """
    Detects if the provided text is watermarked using a group-based XOR scheme.

    This function is agnostic to the generation method (rejection sampling or tree-based)
    and works by verifying the underlying watermark structure. It processes tokens in
    groups, computes the XOR of their hashes, and checks if the resulting bit
    sequence correlates with the secret PRC codeword.

    Args:
        model: The watermarking model containing the hash function and decoding key.
        watermarked_text: A list of token IDs to check.

    Returns:
        threshold (float): The detection threshold.
        hamming_weight (int): The computed Hamming weight of the syndrome.
        result (bool): True if the text is detected as watermarked, False otherwise.
    """
    hash_function = model.hash_function
    group_size = model.group_size
    
    # Process tokens in groups and compute the XOR of their hashes
    reconstructed_prc_bits = []
    for i in range(0, len(watermarked_text), group_size):
        group = watermarked_text[i:i + group_size]
        
        # If the last group is incomplete, it's ignored as it can't form a full bit.
        if len(group) < group_size:
            continue
            
        # Compute the XOR of the hashes for the tokens in the group
        group_xor = 0
        for token_id in group:
            group_xor ^= hash_function(token_id)
            
        reconstructed_prc_bits.append(group_xor)
    
    reconstructed_prc_bits = torch.tensor(reconstructed_prc_bits, dtype=torch.float)
    
    # If not enough bits were generated to perform the check, we cannot detect.
    if len(reconstructed_prc_bits) < model.n:
        print(f"Warning: Not enough tokens to form a full PRC block. Need {model.n * group_size}, have {len(watermarked_text)}.")
        return None, None, False

    # Truncate if we have more bits than the PRC codeword length
    if len(reconstructed_prc_bits) > model.n:
        reconstructed_prc_bits = reconstructed_prc_bits[:model.n]
        
    # Apply the parity check matrix for detection
    parity_check_matrix = model.decoding_key[1]
    r = parity_check_matrix.shape[0]
    
    # Compute Px (syndrome of the reconstructed bits)
    Px = (parity_check_matrix @ reconstructed_prc_bits) % 2
    
    # Compute Pz (syndrome of the one-time pad)
    z = model.decoding_key[2]
    Pz = (parity_check_matrix @ z) % 2
    
    # The final syndrome is the XOR of the two
    final_syndrome = (Px + Pz) % 2
    
    hamming_weight = np.sum(final_syndrome)
    
    # Calculate the statistical threshold for detection
    threshold = (0.5 - r**(-0.25)) * r

    # The watermark is detected if the Hamming weight is below the threshold
    result = hamming_weight < threshold
    
    return threshold, hamming_weight, result   