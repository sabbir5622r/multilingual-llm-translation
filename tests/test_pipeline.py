import tempfile
import unittest
from pathlib import Path

from src.common import atomic_json, digest, paths, read_rows, write_rows
from src.data import build, normalize, translated_key
from src.model import parse_answer


class PipelineTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse_answer(" B. ", 4), 1)
        for text in ["Answer: B", "B because", "AB", "D"]:
            self.assertIsNone(parse_answer(text, 3))

    def test_alignment(self):
        row = {"link": "passage1", "question_number": 2, "flores_passage": "text", "question": "q", "correct_answer_num": "2", **{f"mc_answer{i}": str(i) for i in range(1, 5)}}
        normalized = normalize("belebele", row, 0)
        self.assertEqual(normalized["id"], "passage1:2")
        self.assertEqual(normalized["gold"], 1)

    def test_component_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"root": directory, "phase": "pilot", "seed": 42, "translation_groups": 1, "control_groups": 1,
                      "instruction_languages": ["en", "bn"], "datasets": {"belebele": {}}, "languages": {"bn": {}}, "translators": {"test": {"revision": "fixed"}}}
            base, _ = paths(config)
            target = {"id": "1", "group": "g", "a": "passage_L", "b": "question_L", "options": ["l1", "l2", "l3", "l4"], "gold": 1}
            english = {**target, "a": "passage_E", "b": "question_E", "options": ["e1", "e2", "e3", "e4"]}
            write_rows(base / "belebele/en.jsonl", [english])
            write_rows(base / "belebele/bn.jsonl", [target])
            atomic_json(base / "manifest.json", {"config_hash": digest(config)})
            cache = {translated_key("test", config["translators"]["test"], "bn", text): {"status": "ok", "translation": "mt_" + text} for text in [target["a"], target["b"], *target["options"]]}
            atomic_json(base / "translations.json", cache)
            build(config, ["test"])
            rows = read_rows(base / "conditions.jsonl")
            self.assertEqual(len(rows), 48)
            main = {r["condition"]: r for r in rows if r["control"] == "main" and r["track"] == "aligned"}
            self.assertEqual(main["EL"]["a"], "passage_E")
            self.assertEqual(main["EL"]["b"], "question_L")
            self.assertEqual(main["LE"]["options"], english["options"])
            rotated = next(r for r in rows if r["control"] == "order" and r["shift"] == 1)
            self.assertEqual(rotated["order"][0], 1)
            self.assertEqual(rotated["gold"], 1)
            self.assertTrue(all(r["a"] == "" for r in rows if r["control"] == "removed"))
            self.assertEqual(len({r["uid"] for r in rows}), len(rows))

    def test_statistics(self):
        import numpy as np
        from src.analysis import effects, interval
        correct = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=float)
        result = effects(correct, correct)
        self.assertEqual(result["interaction"].mean(), 0)
        self.assertEqual(result["rescued_EE"].sum(), 1)
        self.assertEqual(result["harmed_EE"].sum(), 1)
        lo, hi = interval(np.ones(4), np.array(["a", "a", "b", "b"]), 100, 42)
        self.assertEqual((lo, hi), (1, 1))

class AnalysisTests(unittest.TestCase):
    def test_complete_analysis(self):
        import pandas as pd
        from src.analysis import summarize, compare
        rows = []
        for model in ["a", "b"]:
            for track in ["aligned", "mt_test"]:
                for example in range(4):
                    for i, condition in enumerate(["LL", "EL", "LE", "EE"]):
                        rows.append({"model": model, "dataset": "xnli", "language": "hi", "track": track,
                                     "control": "main", "instruction": "en", "shift": 0, "example": str(example),
                                     "group": str(example // 2), "condition": condition, "gold": example % 3,
                                     "prediction": (example + (i % 2)) % 3, "correct": float(i % 2 == 0)})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            frame = pd.DataFrame(rows)
            config = {"bootstrap": 50, "seed": 42}
            summarize(config, frame, destination)
            compare(config, frame, destination)
            result = pd.read_csv(destination / "track_comparisons.csv")
            self.assertTrue((result.mt_minus_aligned == 0).all())
            self.assertTrue((pd.read_csv(destination / "model_comparisons.csv").difference == 0).all())
            self.assertTrue((pd.read_csv(destination / "ranking_stability.csv").sign_reversal == False).all())


if __name__ == "__main__":
    unittest.main()
