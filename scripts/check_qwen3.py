"""
Quick verification that Qwen3-8B loads and runs on the local RTX 4050.

Tests:
  1. Model loads at 4-bit quantization (should fit in ~5 GB VRAM)
  2. Generates a short response without OOM
  3. Reports tokens/sec for sizing the experiment

If this fails, we have to either:
  - Use Qwen3-4B at 8-bit (smaller fit)
  - Use Qwen3-1.7B at fp16 (even smaller)
  - Use Qwen2.5-7B (older but well-tested)

We use the chat template, generate ~200 tokens, time it.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-8B"

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
      if torch.cuda.is_available() else "")

print(f"\nLoading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model in 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
load_time = time.time() - t0
print(f"  loaded in {load_time:.1f}s")
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Quick test prompt -- DDI-style to match what we'll actually do
messages = [
    {
        "role": "user",
        "content": (
            "Drug A: Warfarin (anticoagulant, narrow therapeutic index, "
            "metabolized by CYP2C9).\n"
            "Drug B: Fluconazole (antifungal, potent CYP2C9 inhibitor).\n\n"
            "In one paragraph, explain the pharmacological mechanism of "
            "interaction between these two drugs and which direction the "
            "effect goes. Be concise."
        ),
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # disable thinking mode for fast/concise responses
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
input_token_count = inputs.input_ids.shape[1]
print(f"\nInput tokens: {input_token_count}")

print("\nGenerating (max 200 new tokens)...")
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
gen_time = time.time() - t0

new_tokens = outputs[0][input_token_count:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
new_token_count = len(new_tokens)

print(f"\n--- Response ({new_token_count} tokens, {gen_time:.1f}s, "
      f"{new_token_count/gen_time:.1f} tok/sec) ---")
print(response)
print("-" * 60)

print(f"\nFinal VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\nIf this all worked, Qwen3-8B is ready for the pilot experiment.")
