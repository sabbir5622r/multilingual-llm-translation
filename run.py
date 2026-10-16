import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "prepare", "translate", "build", "tokens", "evaluate", "audit", "analyze", "budget"])
    parser.add_argument("--config", default="configs/pilot.lock.json")
    parser.add_argument("--output", default="configs/pilot.lock.json")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--dataset", default="belebele")
    parser.add_argument("--language", default="bn")
    parser.add_argument("--track", default="aligned")
    parser.add_argument("--system", default="nllb")
    parser.add_argument("--systems", nargs="*", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--audit-file")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.command != "freeze":
        for section in ("datasets", "models", "translators"):
            for item in config[section].values():
                if len(item["revision"]) != 40:
                    raise ValueError("Resolve immutable revisions with freeze first")
    if args.command == "freeze":
        from src.data import freeze
        freeze(config, args.output)
    elif args.command == "prepare":
        from src.data import prepare
        prepare(config)
    elif args.command == "translate":
        from src.translate import translate
        translate(config, args.system, args.device)
    elif args.command == "build":
        from src.data import build
        build(config, args.systems)
    elif args.command == "tokens":
        from src.evaluate import check_tokens
        check_tokens(config, args.model, args.dataset, args.language, args.track)
    elif args.command == "evaluate":
        from src.evaluate import evaluate
        evaluate(config, args.model, args.dataset, args.language, args.track, args.smoke, args.device_map)
    elif args.command == "audit":
        from src.analysis import audit_sample
        audit_sample(config)
    elif args.command == "budget":
        from src.budget import budget
        budget(config)
    elif args.command == "analyze":
        from src.analysis import analyze
        analyze(config, args.audit_file)


if __name__ == "__main__":
    main()
