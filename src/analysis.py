import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common import atomic_json, choose_groups, digest, paths, read_json, read_rows

CONDITIONS = ["LL", "EL", "LE", "EE"]


def interval(values, groups, repeats, seed):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    parts = [values[groups == group] for group in unique]
    sums = np.array([p.sum(axis=0) for p in parts])
    sizes = np.array([len(p) for p in parts])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sample = rng.integers(0, len(parts), len(parts))
        draws.append(sums[sample].sum(axis=0) / sizes[sample].sum())
    return np.quantile(draws, [0.025, 0.975], axis=0)


def effects(correct, predictions):
    baseline = correct[:, 0]
    measures = {"effect_EL": correct[:, 1] - baseline, "effect_LE": correct[:, 2] - baseline,
                "effect_EE": correct[:, 3] - baseline,
                "interaction": correct[:, 3] - correct[:, 1] - correct[:, 2] + baseline}
    for index, name in enumerate(CONDITIONS):
        measures[f"accuracy_{name}"] = correct[:, index]
        if index:
            measures[f"rescued_{name}"] = (1 - baseline) * correct[:, index]
            measures[f"harmed_{name}"] = baseline * (1 - correct[:, index])
            measures[f"disagreement_{name}"] = (predictions[:, index] != predictions[:, 0]).astype(float)
    return measures


def load_results(config):
    _, results = paths(config)
    rows = []
    for path in sorted((results / "full").glob("*/*/*/*/checkpoint.json")):
        state = read_json(path)
        if state["status"] != "complete" or state.get("smoke") or len(state["predictions"]) != state["expected"]:
            raise ValueError(f"Incomplete run: {path}")
        rows.extend(state["predictions"].values())
    if not rows:
        raise ValueError("No complete full runs")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["model", "uid"]).any():
        raise ValueError("Duplicate predictions")
    frame["prediction"] = frame["prediction"].fillna(-1).astype(int)
    frame["correct"] = (frame.prediction == frame.gold).astype(float)
    return frame


def paired_table(frame):
    index = ["model", "dataset", "language", "track", "control", "instruction", "shift", "example", "group"]
    table = frame.pivot(index=index, columns="condition", values=["correct", "prediction"])
    if table.isna().any().any() or any(c not in table["correct"] for c in CONDITIONS):
        raise ValueError("Missing component condition")
    return table


def summarize(config, frame, destination):
    table = paired_table(frame)
    output = []
    keys = ["model", "dataset", "language", "track", "control", "instruction", "shift"]
    for names, part in table.groupby(level=keys):
        correct = part["correct"][CONDITIONS].to_numpy(float)
        predictions = part["prediction"][CONDITIONS].to_numpy(int)
        measures = effects(correct, predictions)
        label = dict(zip(keys, names))
        groups = part.index.get_level_values("group").to_numpy()
        for metric, values in measures.items():
            lo, hi = interval(values, groups, config["bootstrap"], config["seed"])
            output.append({**label, "metric": metric, "estimate": float(values.mean()), "low": float(lo), "high": float(hi), "n": len(values), "clusters": len(set(groups))})
        for i, condition in enumerate(CONDITIONS[1:], 1):
            for metric, mask, value in [("rescue_rate", correct[:, 0] == 0, correct[:, i]), ("harm_rate", correct[:, 0] == 1, 1-correct[:, i])]:
                if mask.any():
                    lo, hi = interval(value[mask], groups[mask], config["bootstrap"], config["seed"])
                else:
                    lo, hi = None, None
                output.append({**label, "metric": f"{metric}_{condition}", "estimate": float(value[mask].mean()) if mask.any() else None, "low": lo, "high": hi, "n": int(mask.sum()), "clusters": len(set(groups[mask]))})
    pd.DataFrame(output).to_csv(destination / "effects.csv", index=False)
    scores = []
    for names, part in frame.groupby(keys + ["condition"]):
        record = dict(zip(keys + ["condition"], names))
        record.update(chance_accuracy=0.25 if record["dataset"] == "belebele" else 1/3, accuracy=part.correct.mean(), invalid_rate=(part.prediction == -1).mean(), n=len(part))
        record["macro_f1"] = f1_score(part.gold, part.prediction, labels=[0, 1, 2], average="macro", zero_division=0) if record["dataset"] == "xnli" else None
        if record["dataset"] == "xnli":
            groups = part.group.unique()
            pieces = [part[part.group == group] for group in groups]
            rng = np.random.default_rng(config["seed"])
            draws = []
            for _ in range(config["bootstrap"]):
                sample = pd.concat([pieces[i] for i in rng.integers(0, len(pieces), len(pieces))])
                draws.append(f1_score(sample.gold, sample.prediction, labels=[0, 1, 2], average="macro", zero_division=0))
            record["macro_f1_low"], record["macro_f1_high"] = np.quantile(draws, [.025, .975])
        scores.append(record)
    pd.DataFrame(scores).to_csv(destination / "scores.csv", index=False)
    return pd.DataFrame(output)


def compare(config, frame, destination):
    records = []
    main = frame[(frame.control == "main") & (frame.instruction == "en") & (frame["shift"] == 0)]
    for (dataset, language), subset in main.groupby(["dataset", "language"]):
        models = sorted(subset.model.unique())
        for first, second in itertools.combinations(models, 2):
            a = subset[subset.model == first]
            b = subset[subset.model == second]
            joined = a.merge(b, on=["example", "track", "condition"], suffixes=("_a", "_b"), validate="one_to_one")
            if len(joined) != len(a) or len(joined) != len(b):
                raise ValueError("Model comparison requires identical example sets")
            for (track, condition), part in joined.groupby(["track", "condition"]):
                values = (part.correct_a - part.correct_b).to_numpy()
                lo, hi = interval(values, part.group_a.to_numpy(), config["bootstrap"], config["seed"])
                records.append({"dataset": dataset, "language": language, "first": first, "second": second, "track": track, "condition": condition,
                                "difference": values.mean(), "low": lo, "high": hi, "n": len(part)})
    comparisons = pd.DataFrame(records)
    comparisons.to_csv(destination / "model_comparisons.csv", index=False)
    stability = []
    if records:
        for names, part in comparisons.groupby(["dataset", "language", "first", "second", "track"]):
            baseline = part[part.condition == "LL"].iloc[0]
            for row in part.itertuples():
                stability.append({**dict(zip(["dataset", "language", "first", "second", "track"], names)), "condition": row.condition,
                                  "baseline_difference": baseline.difference, "condition_difference": row.difference,
                                  "sign_reversal": bool(baseline.difference * row.difference < 0),
                                  "resolved_opposite_intervals": bool((baseline.low > 0 and row.high < 0) or (baseline.high < 0 and row.low > 0))})
    pd.DataFrame(stability).to_csv(destination / "ranking_stability.csv", index=False)
    contrasts = []
    table = paired_table(main)
    for names, subset in table.groupby(level=["model", "dataset", "language"]):
        tracks = subset.index.get_level_values("track").unique()
        aligned = subset.xs("aligned", level="track")
        for track in tracks:
            if track == "aligned":
                continue
            translated = subset.xs(track, level="track")
            matched = aligned.reindex(translated.index)
            if matched.isna().any().any():
                raise ValueError("Track alignment failure")
            ea = effects(matched["correct"][CONDITIONS].to_numpy(float), matched["prediction"][CONDITIONS].to_numpy(int))
            eb = effects(translated["correct"][CONDITIONS].to_numpy(float), translated["prediction"][CONDITIONS].to_numpy(int))
            for metric in ea:
                values = eb[metric] - ea[metric]
                lo, hi = interval(values, translated.index.get_level_values("group").to_numpy(), config["bootstrap"], config["seed"])
                contrasts.append({"model": names[0], "dataset": names[1], "language": names[2], "track": track, "metric": metric, "mt_minus_aligned": values.mean(), "low": lo, "high": hi, "n": len(values)})
    pd.DataFrame(contrasts).to_csv(destination / "track_comparisons.csv", index=False)


def analyze(config, audit):
    _, results = paths(config)
    frame = load_results(config)
    destination = results / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    base, _ = paths(config)
    expected_rows = read_rows(base / "conditions.jsonl")
    coverage = []
    for model in config["models"]:
        observed = set(frame.loc[frame.model == model, "uid"])
        expected = {r["uid"] for r in expected_rows}
        coverage.append({"model": model, "expected": len(expected), "observed": len(observed), "missing": len(expected-observed), "unexpected": len(observed-expected), "complete": expected == observed})
    pd.DataFrame(coverage).to_csv(destination / "coverage.csv", index=False)
    atomic_json(destination / "study_status.json", {"all_configured_runs_complete": all(r["complete"] for r in coverage), "intervals": "unadjusted exploratory 95% paired cluster bootstrap"})
    effects_frame = summarize(config, frame, destination)
    compare(config, frame, destination)
    compare_controls(config, frame, destination)
    frame.to_csv(destination / "predictions.csv", index=False)
    if audit:
        annotations = pd.read_csv(audit, dtype=str).fillna("")
        required = ["label_preserved", "negation", "quantities", "entities", "temporal", "clues", "ambiguity"]
        valid = annotations[(annotations[required] == "yes").all(axis=1) & (annotations.reviewer.str.len() > 0)]
        keys = ["dataset", "language", "example", "track"]
        if annotations.duplicated(keys).any():
            raise ValueError("Duplicate audit annotations")
        audited = frame.merge(valid[keys], on=keys, how="inner", validate="many_to_one")
        if audited.empty:
            raise ValueError("No fully valid reviewed examples")
        target = destination / "audited_valid"
        target.mkdir(exist_ok=True)
        summarize(config, audited, target)
        atomic_json(target / "coverage.json", {"reviewed_rows": len(annotations), "valid_rows": len(valid), "prediction_rows": len(audited), "audit_hash": digest(annotations.to_dict("records"))})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = effects_frame[(effects_frame.control == "main") & (effects_frame.metric == "interaction")]
    fig, axis = plt.subplots(figsize=(9, max(3, len(main) * .25)))
    for i, row in enumerate(main.itertuples()):
        axis.errorbar(row.estimate, i, xerr=[[row.estimate-row.low], [row.high-row.estimate]], fmt="o", color="#246899")
    axis.set_yticks(range(len(main)), [f"{r.model}/{r.dataset}/{r.language}/{r.track}" for r in main.itertuples()])
    axis.axvline(0, color="gray", linewidth=.7)
    axis.set_xlabel("Accuracy interaction (paired cluster bootstrap, 95% CI)")
    fig.tight_layout()
    fig.savefig(destination / "interaction.pdf")
    fig.savefig(destination / "interaction.png", dpi=250)
    plt.close(fig)


def audit_sample(config):
    base, _ = paths(config)
    rows = [r for r in read_rows(base / "conditions.jsonl") if r["condition"] == "EE" and r["control"] == "main"]
    targets = {}
    for dataset in config["datasets"]:
        for language in config["languages"]:
            path = base / dataset / f"{language}.jsonl"
            if path.exists():
                for r in read_rows(path):
                    targets[(dataset, language, r["id"])] = r
    sample = choose_groups(rows, config["control_groups"], config["seed"] + 3)
    output = []
    for row in sample:
        source = targets[(row["dataset"], row["language"], row["example"])]
        output.append({**{k: row[k] for k in ("dataset", "language", "example", "track", "group")},
                       "source_a": source["a"], "source_b": source["b"], "source_options": str(source["options"]),
                       "english_a": row["a"], "english_b": row["b"], "english_options": str(row["options"]), "gold": row["gold"],
                       **{k: "" for k in ("reviewer", "label_preserved", "negation", "quantities", "entities", "temporal", "clues", "ambiguity", "notes")}})
    pd.DataFrame(output).to_csv(base / "audit.csv", index=False)


def compare_controls(config, frame, destination):
    keys = ["model", "dataset", "language", "track", "condition", "example"]
    main = frame[frame.control == "main"]
    output = []
    for (control, instruction, shift), part in frame[frame.control != "main"].groupby(["control", "instruction", "shift"]):
        joined = part.merge(main, on=keys, suffixes=("_control", "_main"), validate="one_to_one")
        if len(joined) != len(part):
            raise ValueError("Control has no matched main condition")
        for names, group in joined.groupby(["model", "dataset", "language", "track", "condition"]):
            values = (group.correct_control - group.correct_main).to_numpy()
            lo, hi = interval(values, group.group_main.to_numpy(), config["bootstrap"], config["seed"])
            output.append({**dict(zip(["model", "dataset", "language", "track", "condition"], names)),
                           "control": control, "instruction": instruction, "shift": int(shift),
                           "accuracy_delta": values.mean(), "low": lo, "high": hi,
                           "prediction_disagreement": (group.prediction_control != group.prediction_main).mean(), "n": len(group)})
    pd.DataFrame(output).to_csv(destination / "control_comparisons.csv", index=False)
