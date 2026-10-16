import json

from src.common import atomic_json, paths, read_json, read_rows


def budget(config):
    base, results = paths(config)
    conditions = read_rows(base / "conditions.jsonl")
    report = []
    for path in sorted(results.glob("smoke_*/*/*/*/*/checkpoint.json")):
        state = read_json(path)
        rows = list(state["predictions"].values())
        if state["status"] != "complete" or not rows:
            continue
        sample = rows[0]
        target = [r for r in conditions if all(r[k] == sample[k] for k in ("dataset", "language", "track"))]
        measured = sum(r["seconds"] for r in rows)
        projected = measured / len(rows) * len(target)
        sessions = state["sessions"]
        report.append({"model": sample["model"], "dataset": sample["dataset"], "language": sample["language"], "track": sample["track"],
                       "sample_predictions": len(rows), "planned_predictions": len(target), "measured_inference_seconds": measured,
                       "projected_inference_hours": projected / 3600, "projected_cost_at_configured_rate": projected / 3600 * config["hourly_rate"],
                       "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
                       "measured_session_seconds": sum(s["elapsed_seconds"] for s in sessions),
                       "warning": "Small smoke extrapolation excludes translation, download, queueing, and future failures. Measure a larger pilot before budgeting."})
    if not report:
        raise ValueError("Run a complete smoke test first")
    atomic_json(results / "budget.json", report)
    print(json.dumps(report, indent=2))
