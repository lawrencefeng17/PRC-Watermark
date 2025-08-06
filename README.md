# PRC-Watermark

Fork of Image Watermarking with PRC

## What's in here?

* Implementation of Watermarking Scheme for Language Models from "Pseudorandom Error-Correcting Codes" by Christ & Gunn 2024.
  * This includes their proposed binarization method.
  * This includes a bucketing method proposed by Golowich & Moitra "Edit-Distance Robust Watermarks for Language Models"

* attacks on image watermarking with PRC, testing robustness, similarity of latents after edits
* we can scrub the PRC watermark on images by seeding a proxy diffusion model with the reversed latent of a watermarked image using the proxy diffusion model

## Repository Structure

The repository has been organized into the following structure:

* `src/` - Original PRC implementation for image watermarking
* `watermarking/` - New implementation of PRC watermarking for language models
  * See [watermark README](watermarking/README.md) for details on usage and implementation
  * `watermarking/experiments/` - Experiments on language model watermarking
* `experiments/` - Experiments on language model watermarking
* `plots/` - Visualizations and plots from experiments
* `attacks/` - Implementation of attacks on watermarking schemes
* `baselines/` - Implementation of baselines for language model generation

## Running Code

### Text Watermarking

To run the text watermarking procedure, use the following command:

```
python watermarking/run_watermarking.py --model_id "meta-llama/Llama-3.2-1B-Instruct" --n 2048 --prc_t 3 --temperature 1 --debug --new --top_p 0.995 --methods token
```

The flags for watermarking/run_watermarking.py are:
* `--model_id` - The model to use for generation.
* `--n` - The length of the PRC code.
* `--prc_t` - The sparsity of the PRC code.
* `--temperature` - The temperature to use for generation.
* `--debug` - Whether to enable debug mode.
* `--new` - Whether to force the generation of a new piece of text.
* `--top_p` - The top-p value to use for generation.
* `--methods` - The methods to run. The default is to run all methods. The three methods are binary, tokens, and independent_hash.
* `--greedy` - Greedy sampling as opposed to multinomial sampling.

This will watermark the text with a PRC code of length 2048. The parity check matrix is t-sparse, given by prc_t. The temperature is the temperature of the LLM. 

Debug flag will enable the generation of graphs and other statistics. New flag will force the generation of a new piece of text (watermarked text can be saved and reused).

### Baselines

To run the top-p baseline, use the following command:

```
python baselines/top_p_standalone.py --model_id "meta-llama/Llama-3.2-1B-Instruct" --prompt "Write a thrilling story about a murder investigation in an old mansion." --max_tokens 1024 --continue_from_file continue_from_file.txt
```

The flags for baselines/top_p_standalone.py are:
* `--prompt` - The prompt to use for generation.
* `--model_id` - The model to use for generation.
* `--max_tokens` - The maximum number of tokens to generate.
* `--temperature` - The temperature to use for generation.
* `--top_p` - The top-p value to use for generation.
* `--continue_from_file` - The file to continue from. Will be appended to the prompt before generation.

### Sweeps

To run a sweep across prompt (categories), top-p values, use the following command:

```
python run_all_comparisons.py
```

## Reduction to the Binary Alphabet (CG24)

Without loss of generality, let's consider language models over a binary alphabet. Here is the construction. Let $\mathcal{T}$ be the vocabulary of $\mathsf{Model}$. $\mathsf{Model}$ generates a probability vector $p \in [0,1]^{|\mathcal{T}|}$ . We want to construct a new $\mathsf{Model}'$ which outputs $p' \in [0,1]^2.$ 

Let $\mathsf{Enc}$ be any prefix-free encoding. $\mathsf{Model}'$ will compute its distribution $\mathsf{p}'$ over the next binary token by querying $\mathsf{Model}$ for its distribution $\mathsf{p}$ over $\mathcal{T}$, and computing the distribution over the next bit of the binary encoding of the next token. 

Say $\mathsf{Model}'$ is given the binary token sequence $(t_{1}', \dots, t_\ell')$. 

Let $((t_{1}, \dots, t_{i}), s) \leftarrow \mathsf{Dec}(t_{1}', \dots, t_\ell')$, and $\mathsf{p} = \mathsf{Model}(\text{prompt}, (t_{1}, \dots, t_{i}))$. That is, when we decode the given binary string, we get some valid tokens $t_{1}$ through $t_i$, and some extra bits (bits we are still generating) $s$.

Then, to compute the probability of the next binary token being 0, we compute:

$$\mathsf{p}'(0) = \sum\limits_{t \in \mathcal{T}, \mathsf{Enc}(t)[1:len(s) + 1] = s||0} \mathsf{p}(t)$$

That is, take the SUM of (probabilities of tokens) whose encoding begins with the bits $s$ generated so far. 

## Implementation Details

### Issues

#### Embedding the PRC symbol

* The rejection rate is too high, at around 40% when watermarking bit by bit.
* Instead, we take inspiration from GM24 and use a bucket-based approach.
* We hash the vocabulary into two buckets, then perform a biased sample in favor of the bucket indicated by the PRC-bit.
* The problem is that most text generated is not 2048 tokens.

#### Practicalities of quality LLM generation

* In practice, top-p or top-k sampling is used to avoid degenerate text. The current theoretical approaches rely on sampling from the full token distribution, but this is not practical. While distribution preserving, it doesn't actually produce quality text.
* I've implemented and tested sampling from the distribution in a top-p style, combining it with the bucket-based approach by GM24. First, tokens are removed from consideration as donein top-p sampling. Then the remaining tokens are hashed into buckets.

### Results of Implementatoin

Produced using sweep code `run_all_comparisons.py`

* Gemma-3-1b-it
* 3 categories of prompts (coding, story generation, and book report)
* t=3, n=1024, top-p=[0.995, 0.99, 0.98, 0.95, 0.9]
* 10 generations per setting (averaged)
* All text produced was coherent.

![Hamming Weight](/substitution_rate_experiments/hamming_weight_threshold_comparison.png) 
 
![Rejection Rate](/substitution_rate_experiments/rejection_rate_vs_top_p_boxplot.png)

![Entropy vs Rejection Rate](/substitution_rate_experiments/entropy_vs_rejection_scatter.png)

The subsituttion rate needed to make this scheme practical is high, around 40%.

## A new embedding algorithm

See `watermarking/tree_xor_watermarking.py` for implementation details of a tunable beam-decoding-like method which allows the substitution rate to drop to near 0% at the cost of 4x the response length.
