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

## Running the Language Model Watermarking

To run the text watermarking procedure, use the following command:

```
python watermarking/run_watermarking.py --model_id "meta-llama/Llama-3.2-1B-Instruct" --n 2048 --prc_t 3 --temperature 1 --debug --new --top_p 0.995
```

This will watermark the text with a PRC code of length 2048. The parity check matrix is t-sparse, given by prc_t. The temperature is the temperature of the LLM. 

Debug will enable the generation of graphs and other statistics. New will force the generation of a new piece of text (watermarked text can be saved and reused).

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
* I'm currently experimenting with thresholding the distribution in a top-p style, combining it with the bucket-based approach by GM24.
