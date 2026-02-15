# LLMCode

Educational repo with experiments around NN basics, GPT decoders, and quantization.

## `src/basic`

Core educational modules for foundational deep learning concepts.

- `src/basic/activations.py`: activation function experiments.
- `src/basic/batchNorm.py`: batch normalization behavior and implementation notes.
- `src/basic/custom_nn_module.py`: custom `nn.Module` patterns.
- `src/basic/dropout.py`: dropout behavior experiments.
- `src/basic/lora_layer.py`: simple LoRA-style layer adaptation example.
- `src/basic/straight_through_estimator.py`: STE-based quantization-style gradient flow demo.
- `src/basic/train_0.py`: basic training loop scaffold.
- `src/basic/cnn/cnn.py`: CNN example implementation.
- `src/basic/loss/loss.py`: loss function experiments.
- `src/basic/rnn/rnn.py`: RNN/LSTM/GRU examples.

## `src/llms`

Decoder-only GPT experiments, training scripts, and run analysis utilities.

- `src/llms/gpt_basic.py`: baseline GPT decoder implementation.
- `src/llms/gpt_optimized.py`: optimized decoder variant (SDPA, KV cache, RoPE, RMSNorm, SwiGLU, tied embeddings).
- `src/llms/llm_0_KVcache.py`: KV-cache-focused decoder experimentation.
- `src/llms/llm_2.py`: intermediate decoder variant.
- `src/llms/llm_3.py`: intermediate decoder variant.
- `src/llms/train.py`: main character-level training script used for model comparisons.
- `src/llms/train_advanced.py`: alternate/extended training workflow.
- `src/llms/plot_training.py`: log parsing and metric visualization across runs.

## `src/quantization`

Quantization path from basics to QAT training and quantized inference/deployment.

- `src/quantization/deploy_quantized.py`: deployment helper for quantized checkpoints.
- `src/quantization/inference_quantized.py`: quantized text generation and memory reporting.
- `src/quantization/QAT/quant.py`: fake-quant / quantized layer replacement utilities.
- `src/quantization/QAT/train_qat.py`: quantization-aware training entrypoint.
- `src/quantization/QAT/test.py`: QAT validation and latency sanity tests.
- `src/quantization/quantization_basics/quantlinear.py`: baseline quantized linear layer implementation.
- `src/quantization/quantization_basics/simple_quantization.py`: simple quantization walkthrough.
- `src/quantization/quantization_basics/test.py`: tests for quantization basics modules.
