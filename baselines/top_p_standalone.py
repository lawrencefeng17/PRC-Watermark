import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

def generate(model, tokenizer, prompt, top_p=0.9, temperature=1.0, max_tokens=200, continue_from=None):
    """
    Top-p sampling with manual key/value cache reuse, explicit position_ids,
    and a guard against exceeding max_position_embeddings.
    """
    device = model.device

    # Build full prompt (incl. any continuation)
    full_prompt = prompt + (f" {continue_from}" if continue_from else "")
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    # Truncate to model max length if needed
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos and input_ids.size(1) > max_pos:
        input_ids = input_ids[:, -max_pos:]
        attention_mask = torch.ones_like(input_ids)

    # Track total tokens so far
    total_len = input_ids.size(1)
    all_tokens = input_ids[0].tolist()
    past_key_values = None

    with tqdm(total=max_tokens, desc="Generating tokens (manual cache)") as pbar:
        for _ in range(max_tokens):
            # Guard: if we would exceed max_pos, stop generation
            if max_pos and total_len >= max_pos:
                print(f"\nReached max_position_embeddings ({max_pos}); stopping early.")
                break

            # Prepare inputs for this step
            if past_key_values is None:
                step_input_ids      = input_ids
                step_attention_mask = attention_mask
                position_ids        = None
            else:
                # only the last token + correct position_ids
                step_input_ids      = torch.tensor([[all_tokens[-1]]], device=device)
                step_attention_mask = torch.ones_like(step_input_ids)
                position_ids        = torch.arange(
                                         total_len - 1, total_len,
                                         dtype=torch.long,
                                         device=device
                                      ).unsqueeze(0)

            # Forward
            with torch.no_grad():
                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=step_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :] / temperature
                past_key_values = outputs.past_key_values

            # Top-p sampling
            probs = F.softmax(logits, dim=-1)[0]
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs <= top_p
            if mask.sum() < sorted_probs.size(0):
                mask[mask.sum()] = True
            candidates = sorted_idx[mask]
            candidate_probs = sorted_probs[mask] / sorted_probs[mask].sum()
            try:
                pick = torch.multinomial(candidate_probs, 1).item()
            except RuntimeError:
                pick = candidates[0].item()

            # Stop on EOS
            if pick == tokenizer.eos_token_id:
                break

            # Append and advance
            all_tokens.append(pick)
            total_len += 1
            pbar.update(1)

    text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    return text, all_tokens

def main():
    parser = argparse.ArgumentParser(description='Standalone Top-P Sampling Test')
    parser.add_argument('--prompt', type=str, default="""Write an extensive, winding summary and analysis of the Brothers Karamazov. It should be at least 2000 words long.""")
    parser.add_argument('--continue_from', type=str, default=None, help='Text to continue from. Will be appended to the prompt before generation.')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--temperature', type=float, default=1.0) 
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--continue_from_file', type=str, default=None, help='File to continue from. Will be appended to the prompt before generation.')
    parser.add_argument('--output_file', type=str, default=None, help='File to save the generated text.')

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
    if "meta-llama" in args.model_id:
        if "Instruct" in args.model_id:
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
        else:
            current_prompt = args.prompt
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
        else:
            current_prompt = args.prompt
    else:
        current_prompt = args.prompt

    print(f"\\nPrompt: {current_prompt}")
    print(f"Generating with top_p={args.top_p}, temperature={args.temperature}, max_tokens={args.max_tokens}...")
    
    if args.continue_from_file:
        with open(args.continue_from_file, 'r') as f:
            continue_from = f.read()
    else:
        continue_from = None

    generated_text, generated_token_ids = generate(
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

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(generated_text)
        print(f"Generated text saved to {args.output_file}")

if __name__ == "__main__":
    main() 