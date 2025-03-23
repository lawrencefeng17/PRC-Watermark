import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import sys
import time

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset

from src.prc import KeyGen, Encode, Decode, Detect
import src.pseudogaussians as prc_gaussians
from src.baseline.gs_watermark import Gaussian_Shading_chacha
from src.baseline.treering_watermark import tr_detect, tr_get_noise
from inversion import stable_diffusion_pipe, generate

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
parser.add_argument('--dataset_id', type=str, default='databricks/databricks-dolly-15k')
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('--prompt', type=str, default='Tell me a fantastical story about a wizard.')
parser.add_argument('--stream', action='store_true', help='Enable streaming generation output')
parser.add_argument('--stream_delay', type=float, default=0.05, help='Delay between tokens when streaming (in seconds)')
args = parser.parse_args()
print(args)

hf_cache_dir = '/home/lawrence/.cache/huggingface/hub' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n = 4 * 64 * 64  # the length of a PRC codeword (for stable diffusion)
n = 512
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'

# LOAD MODEL
dataset = load_dataset(dataset_id, split='train')

print("Loading model...")
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'

    # Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = config.vocab_size
eos_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded on {device}")

# SETUP
def setup(vocab_size):
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
        with open(f'keys/{exp_id}.pkl', 'wb') as f:  # Save the keys to a file
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:  # Load the keys from a file
            encoding_key, decoding_key = pickle.load(f)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{exp_id}.pkl')
    # random mapping from model vocab to {-1, 1}
    mapping = torch.randint(0, 2, (vocab_size,)).to(device)
    mapping = mapping * 2 - 1
    return encoding_key, decoding_key, mapping

# WATERMARK
def watermark(prompt, encoding_key, phi, stream=False, stream_delay=0.05):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()
    
    x = Encode(encoding_key)
    i = 1 # token index
    j = 1 # codeword index
    # token by token generation until EOS token
    tokens = []
    
    if stream:
        print(prompt, end='', flush=True)

    max_iterations = 1000
    for _ in tqdm(range(max_iterations), disable=stream):
        with torch.no_grad():
            outputs = model(generated_ids)
            
        p_i = outputs.logits[:, -1, :] # logits conditioned on all previous tokens
        
        next_token_id = torch.argmax(p_i, dim=-1).unsqueeze(-1)

        generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)
        
        if next_token_id.item() == eos_token_id:
            break

        x_j = x[j]
        t_i = EmbedChar(x_j, p_i, phi) # watermarked token

        i += 1
        j += 1
        
        if j > n: 
            j = 1
            # generate new PRC codeword
            x = Encode(encoding_key)

        tokens.append(t_i.unsqueeze(0))
        
        # Stream the token if streaming is enabled
        if stream and tokens:
            last_token = tokenizer.decode([t_i.item()])
            print(last_token, end='', flush=True)
            time.sleep(stream_delay)  # Add a small delay between tokens for readability

    if tokens:
        token_ids = torch.cat(tokens, dim=0).unsqueeze(0)
        watermarked_prompt = tokenizer.decode(token_ids.squeeze())
    else:
        watermarked_prompt = ""
    
    if stream:
        print()  # Add a newline at the end of streaming
        
    return watermarked_prompt

def EmbedChar(x_j, p_i, phi):
    # pushforward distribution : phi o p_i
    p_i = torch.softmax(p_i, dim=-1).to(device)
    # Ensure phi has the correct shape [1, vocab_size]
    phi = phi.view(1, -1).to(device)
    prob_bucket_0 = (p_i * (phi == 0)).sum()
    prob_bucket_1 = (p_i * (phi == 1)).sum()
    pushforward_logits = torch.log(torch.tensor([prob_bucket_0, prob_bucket_1], device=device))
    pushforward_distribution = torch.distributions.Categorical(logits=pushforward_logits)

    if torch.rand(1, device=device) < torch.min(torch.tensor(1.0, device=device), 2 * pushforward_distribution.probs[x_j.long()]):
        y_i = x_j # y_i is in the vocabulary of the PRC
    else:
        # sample y_i from q_i where q_i is [phi_p_i  - 1/2]_+
        q_i = torch.distributions.Categorical(logits=torch.clamp(pushforward_distribution.logits - 1/2, min=0))
        y_i = q_i.sample()

    # now sample token from the bucket
    # Create a mask for tokens that belong to the specified bucket (y_i)
    bucket_mask = (phi == y_i).squeeze()
    
    # Get the original probabilities for tokens in this bucket
    bucket_probs = p_i.clone()
    # Zero out probabilities for tokens not in this bucket
    bucket_probs[:, ~bucket_mask] = 0
    # Normalize the probabilities within the bucket
    bucket_probs = bucket_probs / bucket_probs.sum()
    # Create a categorical distribution over tokens in the bucket
    bucket_distribution = torch.distributions.Categorical(probs=bucket_probs.squeeze())
    
    t_i = bucket_distribution.sample()

    return t_i

def detect(decoding_key, watermarked_prompt):
    for i in range(len(watermarked_prompt)):
        for j in range(i, min(i + n - 1, len(watermarked_prompt))):
            if Decode(decoding_key, watermarked_prompt[i:j+1]) is not None:
                return True
    return False

def main():
    prompt = "Tell me a fantastical story about a wizard."
    encoding_key, decoding_key, mapping = setup(vocab_size)
    print("Watermarking prompt...")
    watermarked_prompt = watermark(prompt, encoding_key, phi=mapping, stream=args.stream, stream_delay=args.stream_delay)
    
    if not args.stream:
        print("Generated text:")
        print(watermarked_prompt)
    
    # Check if watermark is detectable
    is_watermarked = detect(decoding_key, watermarked_prompt)
    print(f"Watermark detected: {is_watermarked}")
    
    breakpoint()

if __name__ == "__main__":
    main()