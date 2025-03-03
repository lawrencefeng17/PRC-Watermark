"""
For PRC watermarking Only, will add Tree-Ring and Gaussian Shading watermarking later
"""

import argparse
import os
import pickle
import torch
from PIL import Image
from tqdm import tqdm
from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, exact_inversion
from pathlib import Path
parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('-f', type=str, default=None) # specifc image to decode

parser.add_argument('--test_path', type=str, default='original_images')
args = parser.parse_args()
print(args)

hf_cache_dir = '/home/lawrence/.cache/huggingface/hub' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'

with open(f'keys/{exp_id}.pkl', 'rb') as f:
    encoding_key, decoding_key = pickle.load(f)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

cur_inv_order = 0
var = 1.5
combined_results = []

if args.f is not None:
    test_files = [args.f]
else:
    test_files = os.listdir(f'results/{exp_id}/{args.test_path}')

for file in test_files:
    img = Image.open(file)
    file_name = Path(file).stem
    reversed_latents = exact_inversion(img,
                                       prompt='',
                                       test_num_inference_steps=args.inf_steps,
                                       inv_order=cur_inv_order,
                                       pipe=pipe
                                       )
    reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
    torch.save(reversed_prc, f'tensors/{file_name}.pt')
    detection_result = Detect(decoding_key, reversed_prc)
    decoding_result = (Decode(decoding_key, reversed_prc) is not None)
    combined_result = detection_result or decoding_result
    combined_results.append(combined_result)
    print(f'{file_name}: Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')

with open('decoded.txt', 'w') as f:
    for result in combined_results:
        f.write(f'{result}\n')

print(f'Decoded results saved to decoded.txt')