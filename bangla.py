import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import string
import time
import unicodedata
from collections import Counter

from src.common import (
    atomic_json,
    choose_groups,
    digest,
    paths,
    read_json,
    read_rows,
    write_rows,
)
from src.data import translated_key


SOURCES = {
    "indicxnli_bn": {
        "id": "Divyanshu/indicxnli",
        "revision": "635b8660242e82b2ac96787c17f71953833bf1dd",
        "subset": "bn",
    },
    "tydiqa_bn": {
        "id": "google-research-datasets/tydiqa",
        "revision": "da78f23f9119363459acbaf46bf89426ff26c259",
        "subset": "secondary_task",
    },
}


def load_source(spec, split):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        spec["id"],
        repo_type="dataset",
        revision=spec["revision"],
    )
    prefix = f'{spec["subset"]}/{split}'
    selected = [
        path
        for path in files
        if path.endswith(".parquet")
        and (
            path.startswith(prefix + "/")
            or path.startswith(prefix + "-")
        )
    ]

    if not selected:
        raise ValueError(f"No pinned files for {spec['id']}/{split}")

    urls = [
        f'hf://datasets/{spec["id"]}@{spec["revision"]}/{path}'
        for path in sorted(selected)
    ]

    return list(
        load_dataset(
            "parquet",
            data_files={split: urls},
            split=split,
        )
    )


def prepare(config):
    from src.data import prepare as prepare_existing

    prepare_existing(config)

    base, _ = paths(config)
    manifest = read_json(base / "manifest.json")
    split = config["datasets"]["xnli"]["split"]

    english_spec = {
        **config["datasets"]["xnli"],
        "subset": "en",
    }
    english = load_source(english_spec, split)
    bengali = load_source(SOURCES["indicxnli_bn"], split)

    if len(english) != len(bengali):
        raise ValueError("IndicXNLI alignment size mismatch")

    if any(
        int(first["label"]) != int(second["label"])
        for first, second in zip(english, bengali)
    ):
        raise ValueError("IndicXNLI label alignment failed")

    selected = read_rows(base / "xnli/en.jsonl")
    rows = []

    for row in selected:
        source = bengali[int(row["id"])]
        rows.append(
            {
                **row,
                "a": source["premise"],
                "b": source["hypothesis"],
            }
        )

    write_rows(base / "xnli/bn.jsonl", rows)

    manifest["datasets"]["xnli"]["languages"]["bn"] = {
        "status": "ready",
        "rows": len(rows),
        "hash": digest(rows),
        "source": SOURCES["indicxnli_bn"],
        "provenance": (
            "Existing English-to-Bengali machine translation; "
            "not official Bengali XNLI"
        ),
        "alignment": (
            "Released row order plus full label-vector check; "
            "semantic alignment requires audit"
        ),
    }

    raw = load_source(SOURCES["tydiqa_bn"], "validation")
    qa = []
    offset_mismatches = []

    for row in raw:
        if not row["id"].startswith("bengali-"):
            continue

        answers = row["answers"]
        texts = answers["text"]

        if not texts or any(
            not text or text not in row["context"]
            for text in texts
        ):
            raise ValueError(
                "Released TyDi QA answer text missing from context"
            )

        for start, text in zip(answers["answer_start"], texts):
            actual = row["context"][start:start + len(text)]

            if actual != text:
                offset_mismatches.append(
                    {
                        "id": row["id"],
                        "released_start": start,
                        "answer_text": text,
                    }
                )

        qa.append(
            {
                "id": row["id"],
                "group": digest(row["context"]),
                "a": row["context"],
                "b": row["question"],
                "options": [],
                "gold": -1,
                "answers": list(dict.fromkeys(texts)),
            }
        )

    if not qa or len({row["id"] for row in qa}) != len(qa):
        raise ValueError("Empty or duplicate Bengali TyDi QA IDs")

    exclude = config.get("exclude_first_groups", 0)
    blocked = (
        {
            row["group"]
            for row in choose_groups(qa, exclude, config["seed"])
        }
        if exclude
        else set()
    )

    qa = choose_groups(
        [row for row in qa if row["group"] not in blocked],
        config["sample_groups"],
        config["seed"],
    )

    write_rows(base / "tydiqa/bn.jsonl", qa)

    atomic_json(
        base / "tydiqa/span_checks.json",
        {
            "offset_mismatches": offset_mismatches,
            "policy": (
                "Score unchanged released answer text; require it "
                "to occur in context. Character offsets are not used."
            ),
        },
    )

    manifest["datasets"]["tydiqa"] = {
        "source": SOURCES["tydiqa_bn"],
        "split": "validation",
        "rows": len(qa),
        "hash": digest(qa),
        "provenance": (
            "Native Bengali Gold Passage QA; "
            "no released aligned English counterpart"
        ),
        "license": "Apache-2.0",
    }

    atomic_json(base / "manifest.json", manifest)

    print(
        {
            "indicxnli_bn": len(rows),
            "tydiqa_bn": len(qa),
        }
    )


def translate_qa(config, system, device):
    from src.translate import translate

    settings = json.loads(json.dumps(config))
    settings["datasets"] = {"tydiqa": {}}
    settings["languages"] = {
        "bn": config["languages"]["bn"],
    }

    translate(settings, system, device)


def qa_subset(config):
    base, _ = paths(config)
    rows = read_rows(base / "tydiqa/bn.jsonl")

    return choose_groups(
        rows,
        config["translation_groups"],
        config["seed"] + 1,
    )


def replacements(config, system, row):
    base, _ = paths(config)
    cache = read_json(base / "translations.json")
    result = {}

    for component in ("a", "b"):
        key = translated_key(
            system,
            config["translators"][system],
            "bn",
            row[component],
        )
        item = cache[key]

        if item["status"] != "ok":
            raise ValueError("QA translation failed")

        result[component] = item["translation"]

    return result


def references(config, system):
    base, _ = paths(config)
    output = []

    for row in qa_subset(config):
        english = replacements(config, system, row)

        output.append(
            {
                "example": row["id"],
                "system": system,
                "translation_hash": digest(english),
                "passage_bn": row["a"],
                "question_bn": row["b"],
                "answers_bn": json.dumps(
                    row["answers"],
                    ensure_ascii=False,
                ),
                "passage_en": english["a"],
                "question_en": english["b"],
                "answers_en_json": "",
                "preserved": "",
                "reviewer": "",
            }
        )

    path = (
        base
        / "tydiqa"
        / f"references_{system}.template.csv"
    )

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(output[0]),
        )
        writer.writeheader()
        writer.writerows(output)

    print(path)


def build_qa(config, system):
    base, _ = paths(config)
    path = base / "tydiqa" / f"references_{system}.csv"

    with path.open(encoding="utf-8", newline="") as stream:
        annotations = list(csv.DictReader(stream))

    reviewed = {
        row["example"]: row
        for row in annotations
    }

    if len(reviewed) != len(annotations):
        raise ValueError("Duplicate QA references")

    selected = qa_subset(config)
    control_ids = {
        row["id"]
        for row in choose_groups(
            selected,
            config["control_groups"],
            config["seed"] + 2,
        )
    }

    rows = []

    for source in selected:
        english = replacements(config, system, source)
        reference = reviewed.get(source["id"])

        if (
            not reference
            or reference["system"] != system
            or not reference["reviewer"].strip()
            or reference["translation_hash"] != digest(english)
        ):
            raise ValueError(
                "Missing, unreviewed, or stale QA reference"
            )

        if reference["preserved"] not in ("yes", "no"):
            raise ValueError("preserved must be yes or no")

        valid = reference["preserved"] == "yes"
        answers = (
            json.loads(reference["answers_en_json"])
            if valid
            else []
        )

        if valid and (
            not isinstance(answers, list)
            or not answers
            or any(
                not isinstance(answer, str)
                or not answer.strip()
                or answer not in english["a"]
                for answer in answers
            )
        ):
            raise ValueError(
                "English references must be exact spans "
                "of the translated passage"
            )

        for condition in ("LL", "EL", "LE", "EE"):
            first = (
                english
                if condition[0] == "E"
                else source
            )
            second = (
                english
                if condition[1] == "E"
                else source
            )

            variants = [("main", "en")]

            if source["id"] in control_ids:
                variants.extend(
                    [
                        ("removed", "en"),
                        ("instruction", "bn"),
                    ]
                )

            for control, instruction in variants:
                row = {
                    "dataset": "tydiqa",
                    "language": "bn",
                    "example": source["id"],
                    "group": source["group"],
                    "track": f"mt_{system}",
                    "condition": condition,
                    "control": control,
                    "instruction": instruction,
                    "a": (
                        ""
                        if control == "removed"
                        else first["a"]
                    ),
                    "b": second["b"],
                    "reference_valid": valid,
                    "answers": (
                        answers
                        if condition[0] == "E"
                        else source["answers"]
                    ),
                    "answer_language": (
                        "en"
                        if condition[0] == "E"
                        else "bn"
                    ),
                }

                row["uid"] = digest(
                    [
                        source["id"],
                        system,
                        condition,
                        control,
                        instruction,
                    ]
                )
                rows.append(row)

    write_rows(
        base / "tydiqa" / f"conditions_{system}.jsonl",
        rows,
    )

    print(
        {
            "qa_conditions": len(rows),
            "unscorable_examples": sum(
                row["condition"] == "LL"
                and row["control"] == "main"
                and not row["reference_valid"]
                for row in rows
            ),
        }
    )


def normalize(text, language):
    text = "".join(
        character
        for character in text.lower()
        if character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
    )

    if language == "en":
        text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())


def score(prediction, answers, language):
    prediction = normalize(prediction or "", language)
    exact, f1 = 0.0, 0.0

    for answer in answers:
        answer = normalize(answer, language)
        exact = max(exact, float(prediction == answer))

        predicted_tokens = prediction.split()
        answer_tokens = answer.split()
        common = sum(
            (
                Counter(predicted_tokens)
                & Counter(answer_tokens)
            ).values()
        )

        if common:
            f1 = max(
                f1,
                2 * common
                / (len(predicted_tokens) + len(answer_tokens)),
            )

    return exact, f1


def qa_prompt(row):
    instructions = {
        "en": (
            "Copy the shortest answer span exactly from the passage. "
            "Use the passage language, even if the question uses "
            "another language. Reply only with the answer text."
        ),
        "bn": (
            "অনুচ্ছেদ থেকে সংক্ষিপ্ততম উত্তরটি হুবহু লিখুন। "
            "প্রশ্ন অন্য ভাষায় থাকলেও অনুচ্ছেদের ভাষায় উত্তর দিন। "
            "শুধু উত্তরের লেখাটি দিন।"
        ),
    }

    return (
        f'{instructions[row["instruction"]]}\n\n'
        f'Passage:\n{row["a"]}\n\n'
        f'Question:\n{row["b"]}\n\n'
        "Answer:"
    )


def qa_tokens(config, model, system):
    from src.model import Engine

    base, _ = paths(config)
    engine = Engine(
        config["models"][model],
        tokenizer_only=True,
    )
    lengths = []

    for row in read_rows(
        base / "tydiqa" / f"conditions_{system}.jsonl"
    ):
        text = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": qa_prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
            **engine.spec.get("chat_kwargs", {}),
        )

        tokens = engine.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        lengths.append(len(tokens.input_ids))

    print(
        {
            "rows": len(lengths),
            "max_input_tokens": max(lengths),
            "output_budget": 64,
        }
    )

    if any(
        length + 64 > config["max_input_tokens"]
        for length in lengths
    ):
        raise ValueError(
            "QA token budget exceeded; revise protocol explicitly"
        )


def evaluate_qa(
    config,
    model,
    system,
    smoke,
    hourly_rate,
):
    import torch
    from src.model import Engine

    base, results = paths(config)
    rows = read_rows(
        base / "tydiqa" / f"conditions_{system}.jsonl"
    )

    if smoke:
        selected = set(
            list(
                dict.fromkeys(
                    row["example"]
                    for row in rows
                )
            )[:smoke]
        )
        rows = [
            row
            for row in rows
            if row["example"] in selected
        ]

    directory = (
        results
        / "qa"
        / (f"smoke_{smoke}" if smoke else "full")
        / model
        / system
    )
    directory.mkdir(parents=True, exist_ok=True)

    lock = directory / "running.lock"
    descriptor = os.open(
        lock,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    )
    os.close(descriptor)

    path = directory / "checkpoint.json"
    state = None
    session = None
    start = time.perf_counter()

    try:
        signature = digest(
            [
                config,
                model,
                rows,
                "bangla-qa-v1",
                hourly_rate,
            ]
        )

        state = (
            read_json(path)
            if path.exists()
            else {
                "signature": signature,
                "config": config,
                "sources": SOURCES,
                "expected": [
                    row["uid"]
                    for row in rows
                ],
                "status": "running",
                "predictions": {},
                "sessions": [],
            }
        )

        if state["signature"] != signature:
            raise ValueError("QA resume signature mismatch")

        expected = {row["uid"] for row in rows}

        if state["status"] == "complete":
            if set(state["predictions"]) != expected:
                raise ValueError("Corrupt QA checkpoint")

            print("Already complete")
            return

        session = {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "datasets",
                )
            },
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "hourly_rate": hourly_rate,
        }
        state["sessions"].append(session)

        torch.manual_seed(config["seed"])
        engine = Engine(config["models"][model])
        session["load_seconds"] = time.perf_counter() - start

        for row in rows:
            if row["uid"] in state["predictions"]:
                continue

            text = engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": qa_prompt(row)}],
                tokenize=False,
                add_generation_prompt=True,
                **engine.spec.get("chat_kwargs", {}),
            )

            inputs = engine.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
            )
            count = inputs.input_ids.shape[1]
            limit = 64

            context_limit = min(
                config["max_input_tokens"],
                getattr(
                    engine.model.config,
                    "max_position_embeddings",
                    config["max_input_tokens"],
                ),
            )

            if count + limit > context_limit:
                raise ValueError(
                    f'QA context budget exceeded: {row["uid"]}'
                )

            inputs = inputs.to(engine.model.device)

            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)

            begin = time.perf_counter()

            with torch.inference_mode():
                output = engine.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=limit,
                    pad_token_id=engine.tokenizer.eos_token_id,
                )

            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)

            elapsed = time.perf_counter() - begin
            generated = output[0, count:]

            raw = engine.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()

            eos = engine.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]

            invalid = (
                not raw
                or (
                    len(generated) >= limit
                    and int(generated[-1]) not in eos
                )
            )

            exact, f1 = score(
                None if invalid else raw,
                row["answers"],
                row["answer_language"],
            )

            scorable = (
                row["reference_valid"]
                or row["condition"] == "LL"
            )

            state["predictions"][row["uid"]] = {
                **row,
                "model": model,
                "raw": raw,
                "invalid_output": invalid,
                "exact_match": exact if scorable else None,
                "token_f1": f1 if scorable else None,
                "input_tokens": count,
                "output_tokens": len(generated),
                "seconds": elapsed,
                "peak_allocated_bytes": {
                    str(index): torch.cuda.max_memory_allocated(
                        index
                    )
                    for index in range(torch.cuda.device_count())
                },
                "peak_reserved_bytes": {
                    str(index): torch.cuda.max_memory_reserved(
                        index
                    )
                    for index in range(torch.cuda.device_count())
                },
            }

            atomic_json(path, state)

        if set(state["predictions"]) != expected:
            raise ValueError("Incomplete QA run")

        state["status"] = "complete"
        state.pop("error", None)

    except Exception as error:
        if state is not None:
            state["status"] = "failed"
            state["error"] = repr(error)

        raise

    finally:
        if state is not None:
            if session is not None:
                session["wall_seconds"] = (
                    time.perf_counter() - start
                )
                session["estimated_cost"] = (
                    session["wall_seconds"] / 3600 * hourly_rate
                )
                session["gpu_device_hours"] = (
                    session["wall_seconds"]
                    / 3600
                    * len(session["gpus"])
                )

            atomic_json(path, state)

        lock.unlink()


def analyze_qa(config):
    import pandas as pd
    from src.analysis import interval

    _, results = paths(config)
    all_rows = []

    for path in sorted(
        (results / "qa/full").glob("*/*/checkpoint.json")
    ):
        state = read_json(path)

        if (
            state["status"] != "complete"
            or set(state["predictions"])
            != set(state["expected"])
        ):
            raise ValueError(f"Incomplete QA run: {path}")

        all_rows.extend(state["predictions"].values())

    if not all_rows:
        raise ValueError("No full QA runs")

    frame = pd.DataFrame(all_rows)

    if frame.duplicated(["model", "uid"]).any():
        raise ValueError("Duplicate QA predictions")

    destination = results / "qa/analysis"
    destination.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        destination / "all_predictions.csv",
        index=False,
    )

    frame[frame.condition == "LL"].groupby(
        ["model", "track", "control", "instruction"]
    )[["exact_match", "token_f1"]].agg(
        ["mean", "count"]
    ).to_csv(
        destination / "native_baseline_all.csv"
    )

    frame[~frame.reference_valid].to_csv(
        destination / "unscorable_predictions.csv",
        index=False,
    )

    valid = frame[frame.reference_valid].copy()

    if valid.empty:
        raise ValueError(
            "No translation-valid paired QA examples"
        )

    output = []

    for names, part in valid.groupby(
        ["model", "track", "control", "instruction"]
    ):
        label = dict(
            zip(
                ["model", "track", "control", "instruction"],
                names,
            )
        )

        for metric in ("exact_match", "token_f1"):
            table = part.pivot(
                index=["example", "group"],
                columns="condition",
                values=metric,
            ).reindex(
                columns=["LL", "EL", "LE", "EE"]
            )

            if table.isna().any().any():
                raise ValueError("Missing paired QA condition")

            values = table.to_numpy(float)
            groups = table.index.get_level_values(
                "group"
            ).to_numpy()

            measures = {
                condition: values[:, index]
                for index, condition in enumerate(
                    ["LL", "EL", "LE", "EE"]
                )
            }

            measures.update(
                effect_EL=values[:, 1] - values[:, 0],
                effect_LE=values[:, 2] - values[:, 0],
                effect_EE=values[:, 3] - values[:, 0],
                interaction=(
                    values[:, 3]
                    - values[:, 1]
                    - values[:, 2]
                    + values[:, 0]
                ),
            )

            if metric == "exact_match":
                for index, condition in enumerate(
                    ["EL", "LE", "EE"],
                    1,
                ):
                    measures[f"rescued_{condition}"] = (
                        (1 - values[:, 0])
                        * values[:, index]
                    )
                    measures[f"harmed_{condition}"] = (
                        values[:, 0]
                        * (1 - values[:, index])
                    )

                    conditional = [
                        (
                            "rescue_rate",
                            values[:, 0] == 0,
                            values[:, index],
                        ),
                        (
                            "harm_rate",
                            values[:, 0] == 1,
                            1 - values[:, index],
                        ),
                    ]

                    for name, mask, rates in conditional:
                        if not mask.any():
                            continue

                        low, high = interval(
                            rates[mask],
                            groups[mask],
                            config["bootstrap"],
                            config["seed"],
                        )

                        output.append(
                            {
                                **label,
                                "metric": metric,
                                "contrast": f"{name}_{condition}",
                                "estimate": rates[mask].mean(),
                                "low": low,
                                "high": high,
                                "n": int(mask.sum()),
                            }
                        )

            for name, measure in measures.items():
                low, high = interval(
                    measure,
                    groups,
                    config["bootstrap"],
                    config["seed"],
                )

                output.append(
                    {
                        **label,
                        "metric": metric,
                        "contrast": name,
                        "estimate": measure.mean(),
                        "low": low,
                        "high": high,
                        "n": len(table),
                    }
                )

    pd.DataFrame(output).to_csv(
        destination / "qa_metrics.csv",
        index=False,
    )

    atomic_json(
        destination / "coverage.json",
        {
            "configured_models": list(config["models"]),
            "observed_models": sorted(frame.model.unique()),
            "complete_study_claim": False,
            "predictions": len(frame),
            "scorable_predictions": len(valid),
            "invalid_output_fraction": float(
                frame.invalid_output.mean()
            ),
            "notes": (
                "Each supplied checkpoint is complete; missing "
                "models/systems are not implied complete. "
                "QA text disagreement is not semantic inconsistency."
            ),
        },
    )

    print(destination)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "qa-translate",
            "qa-references",
            "qa-build",
            "qa-tokens",
            "qa-evaluate",
            "qa-analyze",
        ],
    )
    parser.add_argument(
        "--config",
        default="configs/pilot.lock.json",
    )
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--system", default="nllb")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument(
        "--hourly-rate",
        type=float,
        default=0,
    )

    args = parser.parse_args()
    config = read_json(args.config)

    if args.command == "prepare":
        prepare(config)
    elif args.command == "qa-translate":
        translate_qa(config, args.system, args.device)
    elif args.command == "qa-references":
        references(config, args.system)
    elif args.command == "qa-build":
        build_qa(config, args.system)
    elif args.command == "qa-tokens":
        qa_tokens(config, args.model, args.system)
    elif args.command == "qa-evaluate":
        evaluate_qa(
            config,
            args.model,
            args.system,
            args.smoke,
            args.hourly_rate,
        )
    elif args.command == "qa-analyze":
        analyze_qa(config)


if __name__ == "__main__":
    main()