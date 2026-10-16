import time

from src.common import atomic_json, choose_groups, paths, read_json, read_rows
from src.data import translated_key


def translate(config, system, device):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    spec = config["translators"][system]
    base, _ = paths(config)
    path = base / "translations.json"
    cache = read_json(path) if path.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    model = AutoModelForSeq2SeqLM.from_pretrained(spec["id"], revision=spec["revision"]).to(device).eval()
    for dataset in config["datasets"]:
        for language, info in config["languages"].items():
            source = base / dataset / f"{language}.jsonl"
            if not source.exists():
                continue
            if spec["kind"] == "m2m" and language not in spec["languages"]:
                raise ValueError(f"Unsupported translation language {language}")
            tokenizer.src_lang = info["nllb"] if spec["kind"] == "nllb" else language
            target_token = tokenizer.convert_tokens_to_ids("eng_Latn") if spec["kind"] == "nllb" else tokenizer.get_lang_id("en")
            for row in choose_groups(read_rows(source), config["translation_groups"], config["seed"] + 1):
                for text in [row["a"], row["b"], *row["options"]]:
                    key = translated_key(system, spec, language, text)
                    if key in cache and cache[key]["status"] == "ok":
                        continue
                    inputs = tokenizer(text, return_tensors="pt", truncation=False).to(device)
                    if inputs.input_ids.shape[1] > spec["max_input_tokens"]:
                        raise ValueError(f"Translation input too long: {key}")
                    start = time.perf_counter()
                    try:
                        with torch.inference_mode():
                            output = model.generate(**inputs, forced_bos_token_id=target_token, do_sample=False, num_beams=1, max_new_tokens=spec["max_new_tokens"])
                        ids = output[0].tolist()
                        if ids[-1] != tokenizer.eos_token_id:
                            raise ValueError("Translation reached token limit without EOS")
                        result = tokenizer.decode(ids, skip_special_tokens=True).strip()
                        if not result:
                            raise ValueError("Empty translation")
                        cache[key] = {"status": "ok", "translation": result, "source": text, "language": language, "system": system,
                                      "revision": spec["revision"], "input_tokens": inputs.input_ids.shape[1], "output_tokens": len(ids), "seconds": time.perf_counter() - start}
                    except Exception as error:
                        cache[key] = {"status": "error", "error": str(error)}
                        atomic_json(path, cache)
                        raise
                    atomic_json(path, cache)
