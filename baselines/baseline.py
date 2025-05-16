import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_with_top_p(model, tokenizer, prompt, top_p=0.9, temperature=1.0, max_tokens=200):
    """Improved token-by-token generation with top-p sampling"""
    device = model.device
    
    # Format prompt properly for instruction model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Initial encoding and forward pass
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    output_tokens = []
    
    # Generate tokens one by one
    with torch.no_grad():
        # Initial forward pass to get past_key_values
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        # Token generation loop
        for _ in range(max_tokens):
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Apply top-p (nucleus) sampling - more precisely
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False  
            
            # Create a mask for indices to keep
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=0, 
                index=sorted_indices, 
                src=sorted_indices_to_remove
            )
            
            # Set removed tokens to zero probability and renormalize
            next_token_logits = next_token_logits.clone()
            next_token_logits[0, indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs[0], num_samples=1)
            
            # Stop if we hit the EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Append to output tokens
            output_tokens.append(next_token.item())
            
            # Prepare inputs for next iteration
            next_token_tensor = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_tensor)], dim=-1)
            
            # Forward pass with past_key_values for efficiency
            outputs = model(
                input_ids=next_token_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Update key elements for next iteration
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
    
    # Decode the generated tokens
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return output_text, output_tokens

def main():
    parser = argparse.ArgumentParser(description='Generate text with custom top-p sampling')
    parser.add_argument('--prompt', type=str, default='Tell me a story about a wizard.')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.995)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--fp16', action='store_true', help='Use half precision')
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate precision
    dtype = torch.float16 if args.fp16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=device
    )
    
    print(f"Model loaded on {device} with {dtype} precision")
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerating with top_p={args.top_p}, temperature={args.temperature}, max_tokens={args.max_tokens}...")
    
    output_text, output_tokens = generate_with_top_p(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    print("\nGenerated text:")
    print(output_text)
    print(f"\nNumber of tokens generated: {len(output_tokens)}")

if __name__ == "__main__":
    main()