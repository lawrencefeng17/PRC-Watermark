import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

def generate_top_p_from_binarized_logic(model, tokenizer, prompt, top_p=0.9, temperature=1.0, max_tokens=200):
    """
    Generates text using top-p sampling logic adapted from binarized.py,
    without any watermarking components.
    """
    device = model.device
    
    # Initial encoding
    # Assuming a chat model, let's apply a generic template if one isn't applied.
    # For simplicity here, we'll just encode the raw prompt.
    # For Llama-3 Instruct, a chat template should be applied before this function.
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    output_tokens = input_ids[0].tolist() # Start with prompt tokens
    
    past_key_values = None

    with tqdm(total=max_tokens, desc="Generating tokens (standalone top-p)") as pbar:
        for _ in range(max_tokens):
            # Prepare model inputs
            if past_key_values is None: # First iteration
                current_input_ids = input_ids
            else: # Subsequent iterations
                current_input_ids = torch.tensor([[output_tokens[-1]]], device=device)

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits[:, -1, :] # Get logits for the last token
                past_key_values = outputs.past_key_values

            # Apply temperature
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Top-p sampling logic from binarized.py
            # Ensure probs is 1D
            current_probs = probs[0] 
            
            sorted_probs, sorted_indices = torch.sort(current_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            top_p_mask = cumulative_probs < top_p
            # Include the first token that makes cumulative probability >= top_p
            if top_p_mask.sum() < len(sorted_probs): # if not all tokens are already included
                top_p_mask[top_p_mask.sum()] = True
            
            # Get the tokens and their probabilities that fall within the top_p nucleus
            nucleus_indices = sorted_indices[top_p_mask]
            nucleus_probs = sorted_probs[top_p_mask]
            
            # Normalize the probabilities within the nucleus
            if torch.sum(nucleus_probs) > 1e-9: # Avoid division by zero
                normalized_nucleus_probs = nucleus_probs / torch.sum(nucleus_probs)
            else: # Fallback if all nucleus_probs are zero (highly unlikely with softmax)
                # This might happen if top_p is extremely small and hits zero-prob tokens
                # Or if temperature is very high making probs uniform and tiny.
                # Default to sampling from the original top_p_indices if sum is zero.
                # Or, could take argmax of original probs, or error.
                # For now, if nucleus is empty/zero, break or handle as an error/warning.
                print("Warning: Nucleus probabilities sum to zero. Picking highest probability token from nucleus_indices or breaking.")
                if len(nucleus_indices) > 0:
                    token_id = nucleus_indices[0].item() # Fallback to greedy from selected set
                else: # No tokens even selected by top_p, extremely unlikely
                    print("Error: No tokens selected by top_p. Stopping generation.")
                    break 
                if token_id == tokenizer.eos_token_id:
                    break
                output_tokens.append(token_id)
                pbar.update(1)
                continue


            # Ensure no NaN/Inf in normalized_nucleus_probs
            if not torch.all(torch.isfinite(normalized_nucleus_probs)):
                print("Warning: Non-finite values in normalized_nucleus_probs. Defaulting to greedy.")
                sampled_relative_index = torch.argmax(normalized_nucleus_probs).item()
            else:
                 # Ensure probabilities are non-negative before multinomial
                normalized_nucleus_probs = torch.clamp(normalized_nucleus_probs, min=0)
                # Re-normalize just in case clamp changed the sum slightly, and add epsilon for safety
                normalized_nucleus_probs = normalized_nucleus_probs / (torch.sum(normalized_nucleus_probs) + 1e-9)

                # Sample a token from the nucleus
                sampled_relative_index = torch.multinomial(normalized_nucleus_probs, num_samples=1).item()
            
            token_id = nucleus_indices[sampled_relative_index].item()
            
            if token_id == tokenizer.eos_token_id:
                break
                
            output_tokens.append(token_id)
            pbar.update(1)
            
    # Decode the generated tokens (excluding the prompt)
    generated_text = tokenizer.decode(output_tokens[input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text, output_tokens[input_ids.shape[1]:]


def main():
    parser = argparse.ArgumentParser(description='Standalone Top-P Sampling Test')
    parser.add_argument('--prompt', type=str, default='Once upon a time, in a land far away,')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--temperature', type=float, default=0.7) # Default from binarized.py
    parser.add_argument('--top_p', type=float, default=0.9)    # Default from binarized.py
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--apply_chat_template', action='store_true', help="Apply Llama-3 Instruct chat template to the prompt.")

    args = parser.parse_args()
    
    print(f"Loading model: {args.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval() # Set to evaluation mode
    
    print(f"Model loaded on {device}")

    current_prompt = args.prompt
    if args.apply_chat_template and "Instruct" in args.model_id:
        messages = [{"role": "user", "content": args.prompt}]
        current_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Applied chat template. New prompt: {current_prompt}")
    
    print(f"\\nPrompt: {current_prompt}")
    print(f"Generating with top_p={args.top_p}, temperature={args.temperature}, max_tokens={args.max_tokens}...")
    
    generated_text, generated_token_ids = generate_top_p_from_binarized_logic(
        model=model,
        tokenizer=tokenizer,
        prompt=current_prompt,
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    print("\\n--- Generated Text ---")
    print(generated_text)
    print(f"\\nNumber of tokens generated: {len(generated_token_ids)}")

if __name__ == "__main__":
    main() 