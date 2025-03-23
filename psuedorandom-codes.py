"""
Implementation of Watermarking Schemes described in "Pseudorandom Error-Correcting Codes," Christ & Gunn 2024.

See page 50 for the scheme.
"""
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