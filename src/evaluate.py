import importlib.metadata
import os
import platform
import time
from pathlib import Path

from src.common import atomic_json, digest, paths, read_json, read_rows
from src.model import Engine, PROMPT_VERSION


def selected_rows(config, dataset, language, track):
    base, _ = paths(config)
    rows = read_rows(base / "conditions.jsonl")
    manifest = read_json(base / "conditions.manifest.json")
    if manifest["config_hash"] != digest(config) or manifest["hash"] != digest(rows):
        raise ValueError("Condition manifest mismatch")
    selected = [r for r in rows if r["dataset"] == dataset and r["language"] == language and r["track"] == track]
    if not selected:
        raise ValueError("No matching conditions")
    return selected


def check_tokens(config, model, dataset, language, track):
    rows = selected_rows(config, dataset, language, track)
    engine = Engine(config["models"][model], tokenizer_only=True)
    lengths = [int(engine.encode(row).input_ids.shape[1]) for row in rows]
    report = {"rows": len(rows), "min": min(lengths), "max": max(lengths), "over_budget": sum(n + config["max_new_tokens"] > config["max_input_tokens"] for n in lengths)}
    print(report)
    if report["over_budget"]:
        raise ValueError("Token budget exceeded; revise frozen protocol explicitly")


def evaluate(config, model, dataset, language, track, smoke, device_map):
    import torch
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = selected_rows(config, dataset, language, track)
    if smoke:
        examples = list(dict.fromkeys(r["example"] for r in rows))[:smoke]
        rows = [r for r in rows if r["example"] in examples]
    _, results = paths(config)
    directory = results / (f"smoke_{smoke}" if smoke else "full") / model / dataset / language / track
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "running.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        run(config, model, rows, directory, device_map, bool(smoke))
    finally:
        lock.unlink()


def run(config, model, rows, directory, device_map, smoke):
    import torch
    expected = [r["uid"] for r in rows]
    signature = digest([config, model, PROMPT_VERSION, rows, device_map])
    state_path = directory / "checkpoint.json"
    state = read_json(state_path) if state_path.exists() else {"signature": signature, "status": "running", "predictions": {}, "sessions": []}
    if state["signature"] != signature:
        raise ValueError("Resume signature mismatch")
    if state["status"] == "complete":
        if set(state["predictions"]) != set(expected):
            raise ValueError("Corrupt complete checkpoint")
        print(f"Already complete: {directory}")
        return
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "accelerate", "huggingface-hub")}
    session = {"started_unix": time.time(), "python": platform.python_version(), "packages": packages,
               "cuda": torch.version.cuda,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "hourly_rate": config["hourly_rate"]}
    state["sessions"].append(session)
    start = time.perf_counter()
    try:
        engine = Engine(config["models"][model], device_map)
        session["load_seconds"] = time.perf_counter() - start
        for row in rows:
            if row["uid"] in state["predictions"]:
                continue
            result = engine.predict(row, config)
            state["predictions"][row["uid"]] = {**row, **result, "model": model, "model_id": config["models"][model]["id"], "revision": config["models"][model]["revision"]}
            session["elapsed_seconds"] = time.perf_counter() - start
            session["estimated_cost"] = session["elapsed_seconds"] / 3600 * config["hourly_rate"]
            atomic_json(state_path, state)
        if set(state["predictions"]) != set(expected):
            raise ValueError("Incomplete prediction set")
        state["status"] = "complete"
        state["smoke"] = smoke
        state["expected"] = len(rows)
        state.pop("error", None)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        raise
    finally:
        session["elapsed_seconds"] = time.perf_counter() - start
        session["gpu_device_hours"] = session["elapsed_seconds"] / 3600 * len(session["gpus"])
        session["estimated_cost"] = session["elapsed_seconds"] / 3600 * config["hourly_rate"]
        atomic_json(state_path, state)
    print(f"Complete: {directory}")
