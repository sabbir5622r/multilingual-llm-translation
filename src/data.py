
from src.common import atomic_json, choose_groups, digest, paths, read_json, read_rows, write_rows


def freeze(config, destination):
    from huggingface_hub import HfApi
    api = HfApi()
    for section in ("datasets", "models", "translators"):
        for item in config[section].values():
            info = api.repo_info(item["id"], repo_type="dataset" if section == "datasets" else "model", revision=item["revision"])
            item["revision"] = info.sha
            item["card"] = info.card_data.to_dict() if info.card_data is not None else {}
    atomic_json(destination, config)


def normalize(dataset, row, index):
    if dataset == "belebele":
        key = f'{row["link"]}:{row["question_number"]}'
        return {"id": key, "group": str(row["link"]), "a": row["flores_passage"],
                "b": row["question"], "options": [row[f"mc_answer{i}"] for i in range(1, 5)],
                "gold": int(row["correct_answer_num"]) - 1}
    return {"id": str(index), "group": digest(row["premise"]), "a": row["premise"],
            "b": row["hypothesis"], "options": [], "gold": int(row["label"])}


def load_source(name, spec, language):
    from datasets import load_dataset
    if name != "xnli":
        return load_dataset(spec["id"], language, split=spec["split"], revision=spec["revision"])
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(spec["id"], repo_type="dataset", revision=spec["revision"])
    prefix = f'{language}/{spec["split"]}-'
    selected = [f'hf://datasets/{spec["id"]}@{spec["revision"]}/{path}' for path in files if path.startswith(prefix) and path.endswith(".parquet")]
    if not selected:
        raise ValueError(f"No pinned split files for {name}/{language}")
    return load_dataset("parquet", data_files={spec["split"]: selected}, split=spec["split"])


def prepare(config):
    from datasets import get_dataset_config_names, load_dataset
    base, _ = paths(config)
    manifest = {"config_hash": digest(config), "datasets": {}}
    for name, spec in config["datasets"].items():
        available = get_dataset_config_names(spec["id"], revision=spec["revision"])
        english = [normalize(name, r, i) for i, r in enumerate(load_source(name, spec, spec["english"]))]
        english_map = {r["id"]: r for r in english}
        if len(english_map) != len(english):
            raise ValueError("Duplicate English IDs")
        excluded = {r["group"] for r in choose_groups(english, config.get("exclude_first_groups", 0), config["seed"])} if config.get("exclude_first_groups", 0) else set()
        eligible = [r for r in english if r["group"] not in excluded]
        selected = choose_groups(eligible, config["sample_groups"], config["seed"])
        selected_ids = {r["id"] for r in selected}
        write_rows(base / name / "en.jsonl", selected)
        report = {"revision": spec["revision"], "split": spec["split"], "license": spec["license"], "provenance": spec["provenance"], "languages": {}}
        for language, language_spec in config["languages"].items():
            code = language_spec.get(name)
            if code is None:
                report["languages"][language] = {"status": "not_in_official_coverage"}
                continue
            if code not in available:
                raise ValueError(f"Unavailable configuration {name}/{code}")
            target = [normalize(name, r, i) for i, r in enumerate(load_source(name, spec, code))]
            if len(target) != len(english):
                raise ValueError(f"Alignment size mismatch {name}/{language}")
            if name == "xnli":
                for t, e in zip(target, english):
                    t["group"] = e["group"]
            target_map = {r["id"]: r for r in target}
            if len(target_map) != len(target) or set(target_map) != set(english_map):
                raise ValueError(f"Alignment ID mismatch {name}/{language}")
            for key, t in target_map.items():
                if t["gold"] != english_map[key]["gold"]:
                    raise ValueError(f"Alignment label mismatch {name}/{language}/{key}")
                if not all(isinstance(x, str) and x.strip() for x in [t["a"], t["b"], *t["options"]]):
                    raise ValueError("Empty component")
            rows = [target_map[r["id"]] for r in selected]
            write_rows(base / name / f"{language}.jsonl", rows)
            report["languages"][language] = {"status": "ready", "rows": len(rows), "hash": digest(rows), "alignment": "released IDs" if name == "belebele" else "official parallel row position plus label verification; no independent source IDs"}
        report["english_hash"] = digest(selected)
        manifest["datasets"][name] = report
    atomic_json(base / "manifest.json", manifest)


def translated_key(system, spec, language, text):
    return digest([system, spec, language, text])


def build(config, systems):
    base, _ = paths(config)
    manifest = read_json(base / "manifest.json")
    if manifest["config_hash"] != digest(config):
        raise ValueError("Prepared data belong to another configuration")
    rows = []
    cache = read_json(base / "translations.json") if (base / "translations.json").exists() else {}
    for dataset in config["datasets"]:
        english = {r["id"]: r for r in read_rows(base / dataset / "en.jsonl")}
        for language in config["languages"]:
            path = base / dataset / f"{language}.jsonl"
            if not path.exists():
                continue
            targets = read_rows(path)
            translated_ids = {r["id"] for r in choose_groups(targets, config["translation_groups"], config["seed"] + 1)}
            control_ids = {r["id"] for r in choose_groups(targets, config["control_groups"], config["seed"] + 2)}
            for target in targets:
                tracks = [("aligned", english[target["id"]])]
                if target["id"] in translated_ids:
                    for system in systems:
                        spec = config["translators"][system]
                        def fetch(text):
                            key = translated_key(system, spec, language, text)
                            item = cache[key]
                            if item["status"] != "ok":
                                raise ValueError(f"Failed translation {key}")
                            return item["translation"]
                        tracks.append((f"mt_{system}", {"a": fetch(target["a"]), "b": fetch(target["b"]), "options": [fetch(x) for x in target["options"]]}))
                for track, replacement in tracks:
                    for condition in ("LL", "EL", "LE", "EE"):
                        first = replacement if condition[0] == "E" else target
                        second = replacement if condition[1] == "E" else target
                        variants = [("main", "en", 0)]
                        if target["id"] in control_ids:
                            variants.append(("removed", "en", 0))
                            for instruction in config["instruction_languages"]:
                                if instruction != "en" and instruction == language:
                                    variants.append(("instruction", instruction, 0))
                            if dataset == "belebele":
                                variants.extend(("order", "en", shift) for shift in (1, 2, 3))
                        for control, instruction, shift in variants:
                            options = second["options"]
                            order = list(range(len(options)))
                            order = order[shift:] + order[:shift]
                            row = {"dataset": dataset, "language": language, "example": target["id"], "group": target["group"],
                                   "track": track, "condition": condition, "control": control, "instruction": instruction,
                                   "shift": shift, "a": "" if control == "removed" else first["a"], "b": second["b"],
                                   "options": [options[i] for i in order], "order": order, "gold": target["gold"]}
                            row["uid"] = digest([dataset, language, target["id"], track, condition, control, instruction, shift])
                            rows.append(row)
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate condition IDs")
    write_rows(base / "conditions.jsonl", rows)
    atomic_json(base / "conditions.manifest.json", {"config_hash": digest(config), "hash": digest(rows), "rows": len(rows), "systems": systems})
