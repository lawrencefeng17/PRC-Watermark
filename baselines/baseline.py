#!/usr/bin/env python
# baseline_top_p.py
import argparse, inspect, os, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------
# nucleus sampling helper
# -----------------------------------------------------------
def sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    """Return a single token id sampled with nucleus (p) sampling."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumprobs > top_p).nonzero(as_tuple=False)
    if cutoff.numel():
        last_idx = cutoff[0, 0]
        mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
        mask[: last_idx + 1] = True
        filtered_probs = sorted_probs[mask]
        filtered_idx = sorted_idx[mask]
    else:                         # extremely low top_p
        filtered_probs, filtered_idx = sorted_probs, sorted_idx
    filtered_probs = filtered_probs / filtered_probs.sum()  # renorm
    return filtered_idx[torch.multinomial(filtered_probs, 1)].item()


# -----------------------------------------------------------
# main generation routine
# -----------------------------------------------------------
@torch.no_grad()
def generate_top_p(
    model,
    tokenizer,
    prompt: str,
    top_p: float = 0.9,
    temperature: float = 1.0,
    max_tokens: int = 200,
    device: str = "cuda",
):
    """Minimal top-p loop that auto-detects cache style."""
    # ---------- encode prompt ----------
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # ---------- figure out which cache we need ----------
    sig = inspect.signature(model.forward)
    needs_cache_position = "cache_position" in sig.parameters
    hybrid_cache_cls = None
    if "gemma-2" in model.name_or_path.lower() or "gemma-3" in model.name_or_path.lower():
        try:
            from transformers import HybridCache

            hybrid_cache_cls = HybridCache
        except ImportError as e:
            raise RuntimeError("Install a recent transformers for HybridCache") from e

    # ---------- create (static) cache object ----------
    past_key_values = None
    if hybrid_cache_cls is not None:
        max_cache_len = input_ids.shape[1] + max_tokens
        past_key_values = hybrid_cache_cls(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=model.dtype,
        )

    model.eval()
    generated = input_ids.clone()

    # ---------- one full forward pass to fill the cache ----------
    outs = model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True,
    )
    past_key_values = outs.past_key_values  # SAME object after first call
    next_token_logits = outs.logits[:, -1, :] / temperature
    next_token = sample_top_p(F.softmax(next_token_logits, dim=-1)[0], top_p)
    generated = torch.cat(
        [generated, torch.tensor([[next_token]], device=device)], dim=-1
    )

    # ---------- stream generation ----------
    for step in range(1, max_tokens):
        kwargs = dict(
            input_ids=generated[:, -1:],  # only the last token
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if needs_cache_position:
            # Gemma-3 needs absolute offset of this token
            cache_pos = torch.tensor([input_ids.shape[1] + step - 1], device=device)
            kwargs["cache_position"] = cache_pos

        outs = model(**kwargs)
        next_token_logits = outs.logits[:, -1, :] / temperature
        next_token = sample_top_p(F.softmax(next_token_logits, dim=-1)[0], top_p)

        generated = torch.cat(
            [generated, torch.tensor([[next_token]], device=device)], dim=-1
        )
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# -----------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Baseline top-p sampler")
    parser.add_argument("--model_id")
    parser.add_argument("-g", action="store_true", help="Use gemma-3-1b-it")
    parser.add_argument("-l", action="store_true", help="Use llama-3.2-1b-instruct")
    parser.add_argument("--prompt", default="Write an extensive, winding summary and analysis of the Brothers Karamazov.")
    parser.add_argument("--top_p", type=float, default=1.00)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.g:
        args.model_id = "google/gemma-3-1b-it"
    elif args.l:
        args.model_id = "meta-llama/Llama-3.2-1B-Instruct"
    else:
        raise ValueError("Please specify a model with -g or -l")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

    text = generate_top_p(
        model,
        tokenizer,
        args.prompt,
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
