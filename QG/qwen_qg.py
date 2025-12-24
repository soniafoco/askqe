import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast

# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LLM SETUP
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
model.config.use_cache = True


def call_llm(prompt, max_new_tokens=128):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # rimuove eventuali prefissi "assistant" / "user"
    for prefix in ["assistant\n", "user\n"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text