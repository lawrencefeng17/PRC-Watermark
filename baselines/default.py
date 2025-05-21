import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(model_id, prompt, top_p=0.9, temperature=1.0, max_new_tokens=1024):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto",
        force_download=True
    )
    
    # Format the prompt properly for an instruction model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Use the tokenizer's chat template to format properly
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate with the model's built-in generation function
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Decode only the new tokens (exclude prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text

# Example usage
text = generate_text(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    prompt="Tell me a story about a wizard.",
    top_p=0.995,  # Your high top_p value
    temperature=1.0
)
print(text)