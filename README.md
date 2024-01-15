# Online Distillation

Code for distilling language models, i.e. training a smaller model on the outputs of a larger model. Supports offline distillation (over a fixed dataset) and online distillation (sample from student on a set of prompts/inputs) and aims to reproduce the results from [GKD](https://arxiv.org/abs/2306.13649).

During online distillation, we generate from the student model and then run inference with the larger teacher model to get ground truth logits. Then we run the student model again then update the model.
By default the student model is DDP (since it is small) and the teacher model is FDSP, though it is always run in inference mode. The default loss is reverse KL divergence, though JSD can/should also be used. This implementation does *not* backprop through sampling of the student model.

Used to produce a superior 1.3b draft model for Llama2-13B: https://huggingface.co/philkrav/tinyllama-1.3b-draft-llama-13b-chat which can be used for better speculative decoding, a la [DistillSpec](https://arxiv.org/abs/2310.08461); more improved draft models to come.

