import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

def generate_top_p_from_binarized_logic(model, tokenizer, prompt, top_p=0.9, temperature=1.0, max_tokens=200, continue_from=None):
    """
    Generates text using top-p sampling logic adapted from binarized.py,
    without any watermarking components.
    
    Args:
        continue_from (str, optional): If provided, this text will be treated as the initial 
                                     generation that should be continued.
    """
    device = model.device
    
    # Handle continuation by combining prompt and continue_from
    if continue_from is not None:
        full_prompt = prompt + " " + continue_from
    else:
        full_prompt = prompt
    
    # Initial encoding
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    # Get model configuration
    config = model.config
    max_position_embeddings = getattr(config, 'max_position_embeddings', None)
    if max_position_embeddings and input_ids.shape[1] > max_position_embeddings:
        print(f"Truncating input_ids to {max_position_embeddings} tokens")
        input_ids = input_ids[:, -max_position_embeddings:]

    
    # Initialize generation config
    attention_mask = torch.ones_like(input_ids)
    
    # For storing generated tokens
    all_tokens = input_ids[0].tolist()

    past_key_values = None

    with tqdm(total=max_tokens, desc="Generating tokens (standalone top-p)") as pbar:
        for _ in range(max_tokens):
            # Forward pass with proper attention mask
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Get vocabulary size from the model config
                vocab_size = model.config.vocab_size
                current_probs = probs[0, :vocab_size]
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(current_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Apply nucleus sampling
                nucleus_mask = cumulative_probs <= top_p
                if nucleus_mask.sum() < len(sorted_probs):
                    nucleus_mask[nucleus_mask.sum()] = True
                
                # Get the allowed tokens and their probabilities
                allowed_tokens = sorted_indices[nucleus_mask]
                allowed_probs = sorted_probs[nucleus_mask]
                
                # Normalize probabilities
                allowed_probs = allowed_probs / allowed_probs.sum()
                
                # Sample token
                try:
                    idx = torch.multinomial(allowed_probs, num_samples=1)
                    next_token = allowed_tokens[idx].item()
                except RuntimeError:
                    # Fallback to most likely token if sampling fails
                    next_token = allowed_tokens[0].item()
                
                # Break if we generate an EOS token
                if next_token == tokenizer.eos_token_id:
                    break
                
                # Update input_ids and attention_mask for next iteration
                next_token_tensor = torch.tensor([[next_token]], device=device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_tensor)], dim=1)
                
                # Store the generated token
                all_tokens.append(next_token)
                pbar.update(1)
    
    # Get only the newly generated tokens
    # generated_tokens = all_tokens[len(prompt):]
    generated_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    
    return generated_text, all_tokens


def main():
    parser = argparse.ArgumentParser(description='Standalone Top-P Sampling Test')
    parser.add_argument('--prompt', type=str, default="""Write an extensive, winding summary and analysis of the Brothers Karamazov. It should be at least 2000 words long.""")
    parser.add_argument('--continue_from', type=str, default=None, help='Text to continue from. Will be appended to the prompt before generation.')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--temperature', type=float, default=1.0) 
    parser.add_argument('--top_p', type=float, default=0.995)
    parser.add_argument('--apply_chat_template', action='store_true', help="Apply Llama-3 Instruct chat template to the prompt.")
    parser.add_argument('--continue_from_file', type=str, default=None, help='File to continue from. Will be appended to the prompt before generation.')

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
    if "Llama-3" in args.model_id and "Instruct" in args.model_id:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": args.prompt
            }
        ]
        current_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Applied chat template. New prompt: {current_prompt}")
    elif "gemma-3" in args.model_id:
        if "it" in args.model_id:
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."},]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": args.prompt},]
                    },
                ],
            ]
            current_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )[0]
    print(f"\\nPrompt: {current_prompt}")
    print(f"Generating with top_p={args.top_p}, temperature={args.temperature}, max_tokens={args.max_tokens}...")
    
    if args.continue_from_file:
        with open(args.continue_from_file, 'r') as f:
            continue_from = f.read()
    else:
        continue_from = None

    generated_text, generated_token_ids = generate_top_p_from_binarized_logic(
        model=model,
        tokenizer=tokenizer,
        prompt=current_prompt,
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        continue_from=continue_from
    )
    
    print("\\n--- Generated Text ---")
    print(generated_text)
    print(f"\\nNumber of tokens generated: {len(generated_token_ids)}")

if __name__ == "__main__":
    main() 