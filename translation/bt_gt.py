import json
from deep_translator import GoogleTranslator


languages = ["es", "fr", "hi", "tl", "zh-CN"]

for language in languages:
    translator = GoogleTranslator(source=language, target='en')
    """
    perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]
    for perturbation in perturbations:
        input_file = f"../contratico/en-{language}/{perturbation}.jsonl"
        output_file = f"en-{language}/bt-{perturbation}.jsonl"
    """
    input_file = f"data/processed/en-{language}.jsonl"
    output_file =  f"data/bt/en-{language}-gt.jsonl"

    updated_jsonl = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if "en" in data:  # Check if pert_es exists
                print("Source: ", data["en"])
                print("Translation: ", data["gt_en"])
                try:
                    backtranslated_text = translator.translate(data[f"{language}"])
                    print("Backtranslation: ", backtranslated_text)
                    data["bt"] = backtranslated_text  # Add translated English text
                except Exception as e:
                    print(f"Translation failed for: {data['pert_es']} with error: {e}")
                    data["bt"] = ""
        updated_jsonl.append(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in updated_jsonl:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')