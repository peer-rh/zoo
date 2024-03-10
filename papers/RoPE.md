---
aliases:
  - roformer
tags:
  - ai/method
  - "#paper/read"
4 Sentence Summary: The RoPE method is a relative position embedding method, which does not directly embed in the context. It does so by using rotational matrices and shows that there is low-to-none performance loss to absolute positioning and other relative methods, while being efficient and even depending on the task performs better and with faster convergence
Citation: "J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv, Nov. 08, 2023. doi: [10.48550/arXiv.2104.09864](https://doi.org/10.48550/arXiv.2104.09864)."
---
- Most Pretrained [[transformer]] are trained on either *absolute positional embeddings* or *relative positional embeddings*
	- The downside with those relative methods is that they usually added embeddings in the context, making the $kv$-cache (and also Linear transformers) unusable
- The paper introduces a novel embedding method based on *Rotational Matrices* called **RoPE**
- One way to describe relative positional embedding is to show $\braket{ f_{q}(x_{m}, m) , f_{k}(x_{n}, n) }=g(x_{m}, x_{n}, m-n)$, so that it is only dependent on input and difference in position
- The proposed solution for 2d-Case is $f_{\{ q, k \}}(x_{m},m)=e^{i\theta m}(W_{\{ q,k \}}x_{m})$, which can also be described in $\mathbb{R}$, via the *Givens-Rotation* $R_{m}=\begin{pmatrix} \cos \theta m & -\sin \theta m\\sin \theta m & \cos \theta m \end{pmatrix}$. 
- This can be generalised to the $d$-dim. case, where $f_{\{ q, k \}}(x_{m},m)=R^{d}_{\Theta,m}(W_{\{ q,k \}}x_{m})$, $$R_{\Theta,m}^{d}=R_{m}=\begin{pmatrix} 
\cos \theta_{0} m & -\sin \theta_{0} m & 0 & \dots & \dots\\ \sin \theta_{0} m & \cos \theta_{0} m & 0 & \dots & \dots  \\
0 & 0 & \ddots & 0 & 0 \\
\dots & \dots & 0 & \cos \theta_{\frac{d}{2}} m & -\sin \theta_{\frac{d}{2}} m \\... & \dots & 0 & \sin \theta_{\frac{d}{2}} m & \cos \theta_{\frac{d}{2}} m \\\end{pmatrix}$$, where $\Theta \in\mathbb{R}^{d/2}$
- Since $R^{d}$ is *sparse*, this can be further sped up
- Additionally $q_{m}^{T}k_{n}=x_{m}^{T}W_{q}^{T}R^{d}_{\Theta, m-n}W_{k}x_{n}$, since $R$ is *orthogonal*
### Results
- This operation enables *long-term decay* and even outperforms other method on long sequences
	- The authors however don't fully know why it outperforms other models
- Additionally the model (RoFormer) shows faster convergence and performed either up-par with BERT and greatly outperformed it on 1 of 6 datasets