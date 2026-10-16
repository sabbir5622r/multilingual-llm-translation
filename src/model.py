import re
import time

INSTRUCTIONS = {
    "en": {"rc": "Choose the correct answer using the passage. Reply with only A, B, C, or D.",
           "nli": "Classify the relation. A: entailment; B: neutral; C: contradiction. Reply with only A, B, or C."},
    "bn": {"rc": "অনুচ্ছেদ অনুযায়ী সঠিক উত্তর বেছে নিন। শুধু A, B, C অথবা D লিখুন।",
           "nli": "সম্পর্ক নির্ধারণ করুন। A: অনুসিদ্ধান্ত; B: নিরপেক্ষ; C: বিরোধিতা। শুধু A, B অথবা C লিখুন।"}
}
PROMPT_VERSION = "components-v1"


def prompt(row):
    instruction = INSTRUCTIONS[row["instruction"]]["rc" if row["dataset"] == "belebele" else "nli"]
    if row["dataset"] == "belebele":
        choices = "\n".join(f"{chr(65+i)}: {value}" for i, value in enumerate(row["options"]))
        return f'{instruction}\n\nPassage:\n{row["a"]}\n\nQuestion:\n{row["b"]}\n{choices}\n\nAnswer:'
    return f'{instruction}\n\nPremise:\n{row["a"]}\n\nHypothesis:\n{row["b"]}\n\nAnswer:'


def parse_answer(text, count):
    match = re.fullmatch(r"\s*([A-D])\s*[.]?\s*", text)
    return ord(match[1]) - 65 if match and ord(match[1]) - 65 < count else None


class Engine:
    def __init__(self, spec, device_map="auto", tokenizer_only=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
        self.spec = spec
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        self.model = None
        if not tokenizer_only:
            loader = Gemma3ForConditionalGeneration if spec["loader"] == "gemma3" else AutoModelForCausalLM
            self.model = loader.from_pretrained(spec["id"], revision=spec["revision"], torch_dtype=getattr(torch, spec["dtype"]), device_map=device_map).eval()

    def encode(self, row):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt(row)}], tokenize=False, add_generation_prompt=True, **self.spec.get("chat_kwargs", {}))
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)

    def predict(self, row, config):
        torch = self.torch
        inputs = self.encode(row)
        count = inputs.input_ids.shape[1]
        maximum = min(config["max_input_tokens"], getattr(self.model.config, "max_position_embeddings", config["max_input_tokens"]))
        if count + config["max_new_tokens"] > maximum:
            raise ValueError(f'Input exceeds context budget: {row["uid"]}, {count} tokens')
        inputs = inputs.to(self.model.device)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        for index in devices:
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"], pad_token_id=self.tokenizer.eos_token_id)
        for index in devices:
            torch.cuda.synchronize(index)
        seconds = time.perf_counter() - start
        generated = output[0, count:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        prediction = parse_answer(raw, 4 if row["dataset"] == "belebele" else 3)
        if prediction is not None and row["options"]:
            prediction = row["order"][prediction]
        return {"prediction": prediction, "raw": raw, "input_tokens": count, "output_tokens": len(generated),
                "seconds": seconds, "peak_allocated_bytes": {str(i): torch.cuda.max_memory_allocated(i) for i in devices},
                "peak_reserved_bytes": {str(i): torch.cuda.max_memory_reserved(i) for i in devices}}
