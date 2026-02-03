import torch
import bitsandbytes as bnb
from deploy_quantized import deploy_qat_model

# Assuming you have GPTDecoder available
from src.llms.llm_0 import GPTDecoder


def generate_text_quantized(model, context, decode_fn, max_new_tokens=400, temperature=0.8, device='cpu'):
    """
    Generate text using quantized model (works for both 4-bit and 8-bit)
    
    Args:
        model: Quantized model
        context: Starting token ids (torch.Tensor)
        decode_fn: Function to decode token ids to text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        device: 'cpu' or 'cuda'
    """
    model.eval()
    context = context.to(device)
    
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature
        )
    
    return decode_fn(generated[0].tolist())


# Example usage
if __name__ == "__main__":
    # Configuration
    vocab_size = 65  # Adjust to your vocab size
    d_model = 128
    n_heads = 4
    n_layers = 4
    max_seq_len = 512
    device = 'cuda'
    
    # Choose quantization level
    NUM_BITS = 4  # Change to 4 for 4-bit quantization
    
    # Character encoding (adjust to your tokenizer)
    chars = sorted(list(set(open('data/input.txt', 'r').read())))  # Adjust path
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    
    def encode(s):
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    
    def decode(t):
        return "".join([itos[i] for i in t])
    
    # Load and deploy quantized model
    model_kwargs = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'max_seq_len': max_seq_len
    }
    
    print(f"Deploying {NUM_BITS}-bit model...")
    model_quantized = deploy_qat_model(
        checkpoint_path=f'checkpoints/qat/{NUM_BITS}bit/last.pt',
        model_class=GPTDecoder,  # Import your model class
        model_kwargs=model_kwargs,
        num_bits=NUM_BITS,
        device=device
    )
    
    # Generate text
    print("\nGenerating text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = generate_text_quantized(
        model_quantized, 
        context, 
        decode, 
        max_new_tokens=400,
        temperature=0.8,
        device=device
    )
    
    # Check memory usage
    print("\n" + "="*50)
    print("Memory Usage:")
    print("="*50)

    param_count = sum(p.numel() for p in model_quantized.parameters())
    print(f"Total parameters: {param_count / 1e6:.2f} Million")

    # bitsandbytes does actual bit-packing
    if NUM_BITS == 4:
        # 4-bit: 0.5 bytes per param (packed)
        bytes_per_param = 0.5
        print(f"Storage: 4-bit packed (0.5 bytes/param)")
    elif NUM_BITS == 8:
        # 8-bit: 1 byte per param
        bytes_per_param = 1.0
        print(f"Storage: int8 (1 byte/param)")

    memory_mb = (param_count * bytes_per_param) / (1024 * 1024)
    print(f"Estimated memory: ~{memory_mb:.2f} MB")

    # Compare with FP32
    fp32_memory_mb = (param_count * 4) / (1024 * 1024)
    reduction = (1 - memory_mb / fp32_memory_mb) * 100
    print(f"FP32 would use: ~{fp32_memory_mb:.2f} MB")
    print(f"Memory reduction: ~{reduction:.1f}%")
