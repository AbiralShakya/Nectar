# Nectar

Nectar considers energy of the mixture of experts model (fundamental to scaling for modern LLMs, Diffusion models, etc) and reroutes experts during inference (a new paradigm in test time adaption techniques) based upon energy profiles.

to consider: kernel optimizations for dequantization & ttt specifically (fused kernels, large chunk updates, tile packing), 
    look into TTT architecture and MIT paper: https://arxiv.org/pdf/2505.23884 (Test time training done right) to consider (convexity proofs annd logic, large chun)


Nectar is essentially an analogy to switch transformer but hardware aware (for now), or another interpretation is considering TTT research on model weight updates but now as larger amount of experts are being intergrated in next gen LLM and GenAI, bringing that test time self supervised learning to re routing itself.

TODO: consider model sharding within GPU for memory management (good experimentation area)