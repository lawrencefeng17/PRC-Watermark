import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from pathlib import Path
import re

def analyze_text_coherence(model, tokenizer, text):
    """Ask the LLM to identify where the text becomes incoherent."""
    
    # Construct the prompt
    analysis_prompt = f"""Analyze the coherence of the following text. Your task is to:
1. Read through the text carefully
2. Find the EXACT point where the text first shows signs of incoherence (nonsensical, ungrammatical, random symbols, misspelled words, non-English characters, etc.)
3. Copy the FIRST few words where the incoherence begins
4. If the text remains coherent throughout, explicitly state that

Text to analyze:
{text}

Respond ONLY with these three lines (no other text):
COHERENT: [true/false]
BREAKDOWN_POINT: [paste the exact text where incoherence begins]
EXPLANATION: [brief explanation of why this marks the breakdown point]"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes text coherence. You MUST respond ONLY with the three lines specified, exactly as shown in the format. Do not include the format examples in your response."
        },
        {
            "role": "user",
            "content": analysis_prompt
        }
    ]
    
    # Convert messages to model input
    model_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate analysis
    with torch.no_grad():
        outputs = model.generate(
            model_input,
            max_new_tokens=512,
            temperature=0.1,  # Low temperature for more focused response
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the response using regex
    try:
        # Find the last occurrence of each key in case the model repeats itself
        coherent_matches = list(re.finditer(r'COHERENT:\s*(true|false)', analysis, re.IGNORECASE))
        breakdown_matches = list(re.finditer(r'BREAKDOWN_POINT:\s*(.+?)(?=EXPLANATION:|$)', analysis, re.IGNORECASE | re.DOTALL))
        explanation_matches = list(re.finditer(r'EXPLANATION:\s*(.+?)(?=COHERENT:|$)', analysis, re.IGNORECASE | re.DOTALL))
        
        if not (coherent_matches and breakdown_matches and explanation_matches):
            print("Could not parse response format. Raw response:")
            print(analysis)
            return None
            
        # Use the last match for each field and clean up the response
        breakdown_text = breakdown_matches[-1].group(1).strip()
        # Remove any example format text if present
        if '[' in breakdown_text and ']' in breakdown_text:
            print("Warning: Found format example in model response, ignoring it.")
            return None
            
        result = {
            "is_fully_coherent": coherent_matches[-1].group(1).lower() == "true",
            "breakdown_point": breakdown_text,
            "explanation": explanation_matches[-1].group(1).strip()
        }
        
        # Convert "None" to None for breakdown point
        if result["breakdown_point"].lower() == "none":
            result["breakdown_point"] = None
            
        return result
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        print("Raw response:")
        print(analysis)
        return None

def count_coherent_tokens(tokenizer, full_text, breakdown_substring):
    """Count how many tokens were generated before incoherence."""
    if breakdown_substring is None or breakdown_substring.startswith('['):
        return len(tokenizer.encode(full_text)), full_text
        
    # Find the position of the breakdown substring
    breakdown_pos = full_text.find(breakdown_substring)
    if breakdown_pos == -1:
        print(f"Warning: Could not find breakdown point in text: {breakdown_substring}")
        return len(tokenizer.encode(full_text)), full_text
    
    # Get the coherent part of the text
    coherent_text = full_text[:breakdown_pos].strip()
    
    # Count tokens
    num_tokens = len(tokenizer.encode(coherent_text))
    
    return num_tokens, coherent_text

def main():
    parser = argparse.ArgumentParser(description='Analyze text coherence using LLM')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input text file')
    parser.add_argument('--model_id', type=str, default='google/gemma-3-1b-it', help='Model ID from HuggingFace')
    parser.add_argument('--output_dir', type=str, help='Directory to save analysis results (defaults to input file directory)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    model.eval()
    
    # Read input text
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Analyze coherence
    print("Analyzing text coherence...")
    analysis = analyze_text_coherence(model, tokenizer, text)
    
    if analysis:
        # Count coherent tokens and get coherent text
        num_coherent_tokens, coherent_text = count_coherent_tokens(tokenizer, text, analysis.get('breakdown_point'))
        analysis['num_coherent_tokens'] = num_coherent_tokens
        
        # Save results
        output_base = Path(args.input_file).stem
        
        # Save analysis results
        analysis_file = output_dir / f"{output_base}_coherence_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        # Save coherent portion of text
        if not analysis['is_fully_coherent']:
            coherent_file = output_dir / f"{output_base}_coherent_portion.txt"
            with open(coherent_file, 'w', encoding='utf-8') as f:
                f.write(coherent_text)
        
        print("\nAnalysis Results:")
        print(f"Fully coherent: {analysis['is_fully_coherent']}")
        if not analysis['is_fully_coherent']:
            print(f"Text becomes incoherent after {num_coherent_tokens} tokens")
            print(f"Breakdown point: {analysis['breakdown_point']}")
            print(f"Explanation: {analysis['explanation']}")
            print(f"Coherent portion saved to: {coherent_file}")
        print(f"Analysis saved to: {analysis_file}")
    else:
        print("Analysis failed. Please check the model's response.")

if __name__ == "__main__":
    main() 