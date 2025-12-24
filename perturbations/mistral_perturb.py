import json
from prompt import prompts
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# MODEL
# =========================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
print("Model loaded OK")

LANGUAGE_MAP = {
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "tl": "Tagalog",
    "zh": "Chinese"
}

languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]


def call_mistral(prompt, max_new_tokens=128):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    return response.strip()


for language in languages:
    for perturbation in perturbations:
        print("Perturbation: ", perturbation)
        
        input_file = f"data/processed/en-{language}.jsonl"
        output_file = f"data/perturbation/en-{language}/{perturbation}.jsonl"

        with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
            for line in file:
                data = json.loads(line)
                if f"{language}" in data:
                    target_lang = LANGUAGE_MAP.get(language, language)
                    sentence = data[f"{language}"]
                    prompt = prompts[f"{perturbation}_{language}"].replace("{{original}}", sentence).replace("{{target_lang}}", target_lang)
                    print(prompt)
                    response = call_mistral(prompt)
                    print("> ", response)
                    print("=" * 80)
                    
                    data["perturbation"] = perturbation
                    data[f"pert_{language}"] = response
                    out_file.write(json.dumps(data, ensure_ascii=False) + "\n")