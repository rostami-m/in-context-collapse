import copy
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.export_region_aggregate_tables import (KS, MODEL, REGIONS, ROOT, SOURCES,
                                                    build_fragment, build_fragments, number, source_rows)


def fixture(task):
    payload = {"model": MODEL, "task": task, "arm": "remove", "ks": list(KS)}
    for key, _ in REGIONS:
        payload[key] = {condition: {str(k): Decimal("0.625") for k in KS}
                        for condition in ("standard", "remap")}
    return payload


class RegionAggregateTableTests(unittest.TestCase):
    def test_rounding_and_invalid_values(self):
        self.assertEqual(number(Decimal("0.6325")), "0.633")
        self.assertEqual(number(0), "0.000")
        self.assertEqual(number(Decimal("0.6325"), leading_zero=False), ".633")
        self.assertEqual(number(0, leading_zero=False), ".000")
        self.assertEqual(number(1, leading_zero=False), "1.000")
        for value in (True, "0.5", Decimal("NaN"), Decimal("Infinity"),
                      Decimal("-0.01"), Decimal("1.01")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value)

    def test_wrong_identity_or_grid_is_rejected(self):
        original = fixture("shapes4")
        for field, value in (("model", "other"), ("task", "cifar4"), ("arm", "induce"),
                             ("ks", [0, 1, 2, 4, 8])):
            payload = copy.deepcopy(original)
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                source_rows(payload, "shapes4", "Shapes")

    def test_missing_or_extra_curve_cell_is_rejected(self):
        for change in ("missing", "extra"):
            payload = fixture("shapes4")
            if change == "missing":
                del payload["late:both"]["standard"]["16"]
            else:
                payload["late:both"]["standard"]["32"] = Decimal("0.5")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "curve cells"):
                source_rows(payload, "shapes4", "Shapes")

    def test_ci_fields_are_not_used(self):
        payload = fixture("shapes4")
        expected = source_rows(payload, "shapes4", "Shapes")
        for key, _ in REGIONS:
            payload[key]["standard_ci"] = {"0": "not recoverable"}
            payload[key]["remap_ci"] = {"0": 999}
        self.assertEqual(source_rows(payload, "shapes4", "Shapes"), expected)

    def test_all_real_cells_match_raw_means(self):
        output = build_fragment(ROOT / "data/figdata")
        rows = [line for line in output.splitlines() if " & " in line and not line.startswith("%")]
        self.assertEqual(len(rows), 12)
        compared = 0
        for index, (name, task, label) in enumerate(SOURCES):
            raw = json.loads((ROOT / "data/figdata" / name).read_text(), parse_float=Decimal)
            for offset, (key, region) in enumerate(REGIONS):
                cells = rows[index * len(REGIONS) + offset].removesuffix(r" \\").split(" & ")
                self.assertEqual(cells[:2], [label, region])
                for k, cell in zip(KS, cells[2:], strict=True):
                    for condition, value in zip(("standard", "remap"), cell.split("/"), strict=True):
                        self.assertEqual(Decimal(value), raw[key][condition][str(k)])
                        compared += 1
        self.assertEqual(compared, 144)
        self.assertIn("Shapes & Late & 1.000/.000 & .942/.250", output)
        self.assertNotIn(r"\pm", output)

    def test_compact_cells_match_raw_k16_means(self):
        output = build_fragments(ROOT / "data/figdata")["region_standard_remap_k16.tex"]
        rows = [line for line in output.splitlines() if " & " in line and not line.startswith("%")]
        self.assertEqual(len(rows), 6)
        payloads = [json.loads((ROOT / "data/figdata" / name).read_text(), parse_float=Decimal)
                    for name, _, _ in SOURCES]
        for row, (key, region) in zip(rows, REGIONS, strict=True):
            cells = row.removesuffix(r" \\").split(" & ")
            self.assertEqual(cells[0], region)
            expected = [payload[key][condition]["16"] for payload in payloads
                        for condition in ("standard", "remap")]
            self.assertEqual([Decimal(value) for value in cells[1:]], expected)

    def test_cli_check_detects_stale_rows_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "region_standard_remap.tex"
            command = [sys.executable, str(ROOT / "scripts/export_region_aggregate_tables.py"),
                       "--out-dir", temporary]
            generated = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            checked = subprocess.run(command + ["--check"], capture_output=True, text=True, check=False)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            path.write_text("stale rows\n")
            stale = subprocess.run(command + ["--check"], capture_output=True, text=True, check=False)
            self.assertEqual(stale.returncode, 1)
            self.assertIn("Generated region rows differ", stale.stderr)
            self.assertEqual(path.read_text(), "stale rows\n")


if __name__ == "__main__":
    unittest.main()
