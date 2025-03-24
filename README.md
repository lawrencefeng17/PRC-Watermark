Fork of Image Watermarking with PRC

# What's in here?

* attacks on image watermarking with PRC, testing robustness, similarity of latents after edits
* we can scrub the PRC watermark on images by seeding a proxy diffusion model with the reversed latent of a watermarked image using the proxy diffusion model

* *In Progress in cg24.py:* Implementation of Watermarking Scheme for Language Models from "Pseudorandom Error-Correcting Codes" by CG24.

## Reduction to the Binary Alphabet (CG24)
Without loss of generality, let's consider language models over a binary alphabet. Here is the construction. Let $\mathcal{T}$ be the vocabulary of $\mathsf{Model}$. $\mathsf{Model}$ generates a probability vector $p \in [0,1]^{|\mathcal{T}|}$ . We want to construct a new $\mathsf{Model}'$ which outputs $p' \in [0,1]^2.$ 

Let $\mathsf{Enc}$ be any prefix-free encoding. $\mathsf{Model}'$ will compute its distribution $\mathsf{p}'$ over the next binary token by querying $\mathsf{Model}$ for its distribution $\mathsf{p}$ over $\mathcal{T}$, and computing the distribution over the next bit of the binary encoding of the next token. 

Say $\mathsf{Model}'$ is given the binary token sequence $(t_{1}', \dots, t_\ell')$. 

Let $((t_{1}, \dots, t_{i}), s) \leftarrow \mathsf{Dec}(t_{1}', \dots, t_\ell')$, and $\mathsf{p} = \mathsf{Model}(\text{prompt}, (t_{1}, \dots, t_{i}))$. That is, when we decode the given binary string, we get some valid tokens $t_{1}$ through $t_i$, and some extra bits (bits we are still generating) $s$.

Then, to compute the probability of the next binary token being 0, we compute:
$$
\mathsf{p}'(0) = \sum\limits_{t \in \mathcal{T}, \mathsf{Enc}(t)[1:len(s) + 1] = s||0} \mathsf{p}(t)
$$
That is, take the SUM of (probabilities of tokens) whose encoding begins with the bits $s$ generated so far. 


