import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
except ImportError:
    print("Please install bitsandbytes: pip install bitsandbytes")
    exit(1)


def replace_linear_with_bnb_8bit(model):
    """
    Replace all nn.Linear layers with bitsandbytes 8-bit quantized layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create 8-bit quantized linear layer
            bnb_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,  # Use int8 weights
                threshold=6.0
            )
            
            # Copy weights and bias
            with torch.no_grad():
                bnb_layer.weight.data = module.weight.data
                if module.bias is not None:
                    bnb_layer.bias.data = module.bias.data
            
            setattr(model, name, bnb_layer)
        else:
            # Recursively apply to child modules
            replace_linear_with_bnb_8bit(module)


def replace_linear_with_bnb_4bit(model):
    """
    Replace all nn.Linear layers with bitsandbytes 4-bit quantized layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create 4-bit quantized linear layer
            bnb_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float32,
                compress_statistics=True,
                quant_type='nf4'  # 'nf4' or 'fp4'
            )
            
            # Copy weights and bias
            with torch.no_grad():
                bnb_layer.weight.data = module.weight.data
                if module.bias is not None:
                    bnb_layer.bias.data = module.bias.data
            
            setattr(model, name, bnb_layer)
        else:
            # Recursively apply to child modules
            replace_linear_with_bnb_4bit(module)


def deploy_qat_model(checkpoint_path, model_class, model_kwargs, num_bits=4, device='cpu'):
    """
    Load QAT trained model and convert to 4-bit or 8-bit for deployment
    
    Args:
        checkpoint_path: Path to QAT checkpoint
        model_class: Model class (e.g., GPTDecoder)
        model_kwargs: Dict of kwargs for model initialization
        num_bits: 4 or 8
        device: 'cpu' or 'cuda'
    """
    # Initialize model
    model = model_class(**model_kwargs)
    
    # Load QAT checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict (handle both QAT and non-QAT checkpoints)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Convert to quantized format
    print(f"Converting to {num_bits}-bit quantization...")
    if num_bits == 8:
        replace_linear_with_bnb_8bit(model)
    elif num_bits == 4:
        replace_linear_with_bnb_4bit(model)
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    
    model = model.to(device)
    
    print(f"✅ Model deployed with {num_bits}-bit quantization")
    return model


def save_quantized_model(model, save_path):
    """
    Save the quantized model
    """
    torch.save(model.state_dict(), save_path)
    print(f"💾 Quantized model saved to {save_path}")


def load_quantized_model(model_class, model_kwargs, checkpoint_path, num_bits=4, device='cpu'):
    """
    Load a saved quantized model
    """
    # Initialize model with quantized layers
    model = model_class(**model_kwargs)
    
    if num_bits == 8:
        replace_linear_with_bnb_8bit(model)
    elif num_bits == 4:
        replace_linear_with_bnb_4bit(model)
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


# Example usage
if __name__ == "__main__":
    # Example: Deploy your QAT trained model
    # from your_model_file import GPTDecoder
    
    # model_kwargs = {
    #     'vocab_size': 65,
    #     'd_model': 128,
    #     'n_heads': 4,
    #     'n_layers': 4,
    #     'max_seq_len': 512
    # }
    
    # # Deploy 8-bit model from QAT checkpoint
    # model_8bit = deploy_qat_model(
    #     checkpoint_path='checkpoints/qat/8bit/last.pt',
    #     model_class=GPTDecoder,
    #     model_kwargs=model_kwargs,
    #     num_bits=8,
    #     device='cpu'
    # )
    
    # # Deploy 4-bit model from QAT checkpoint
    # model_4bit = deploy_qat_model(
    #     checkpoint_path='checkpoints/qat/4bit/last.pt',
    #     model_class=GPTDecoder,
    #     model_kwargs=model_kwargs,
    #     num_bits=4,
    #     device='cpu'
    # )
    
    # # Save the quantized models
    # save_quantized_model(model_8bit, 'model_8bit_deployed.pt')
    # save_quantized_model(model_4bit, 'model_4bit_deployed.pt')
    
    # # Later, load for inference
    # model = load_quantized_model(
    #     model_class=GPTDecoder,
    #     model_kwargs=model_kwargs,
    #     checkpoint_path='model_8bit_deployed.pt',
    #     num_bits=8,
    #     device='cpu'
    # )
    
    # # Run inference
    # with torch.no_grad():
    #     output = model(input_tensor)
    
    print("Deployment functions ready for both 4-bit and 8-bit!")
    print("Uncomment the example usage section and adapt to your model.")
