import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from incontext_collapse import figures


class QualitativeFigureTests(unittest.TestCase):
    @staticmethod
    def image():
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_classification_title_includes_unchanged_controls(self):
        image = self.image()
        entries = [{"img": image, "true": "circle", "preds": {
            "0": {"pred": "circle", "correct": True},
            "8": {"pred": prediction, "correct": correct},
        }} for prediction, correct in (("circle", True), ("square", False))]
        payload = {"ks": [0, 8], "tasks": {"shapes4": {"entries": entries}}}
        with patch.object(figures, "load", return_value=payload), \
                patch.object(figures, "save_figure") as save:
            figures.fig_gallery()
        figure, name = save.call_args.args
        self.assertEqual(name, "fig_gallery.pdf")
        self.assertIn("unchanged controls", figure.texts[0].get_text())
        self.assertNotIn("wrong once", figure.texts[0].get_text())
        self.assertIn(r"\checkmark", figure.axes[0].get_xlabel())
        self.assertIn(r"\times", figure.axes[1].get_xlabel())

    def test_vqa_preserves_answers_without_inferred_correctness(self):
        answer = "Harry Potter and the Goblet of Fire."
        entry = {"img": self.image(), "q": "What is the title of the book?",
                 "gold": ["first reference", "second reference"], "preds": {
                     "2": {"pred": "short answer", "score": 1},
                     "8": {"pred": answer, "score": 0},
                 }}
        payload = {"k_lo": 2, "k_hi": 8, "tasks": {
            dataset: {"flips": [entry, entry]}
            for dataset in ("vqav2", "textvqa", "scienceqa")
        }}
        with patch.object(figures, "load", return_value=payload), \
                patch.object(figures, "save_figure") as save:
            figures.fig_vqa_failcase()
        figure, name = save.call_args.args
        self.assertEqual(name, "fig_vqa_failcase.pdf")
        for axis in figure.axes:
            label = " ".join(axis.get_xlabel().split())
            self.assertIn("one ref.: first reference", label)
            self.assertIn("K=2: short answer", label)
            self.assertIn(f"K=8: {answer}", label)
            self.assertNotIn(r"\checkmark", label)
            self.assertNotIn(r"\times", label)
        self.assertNotIn("failure", figure.texts[0].get_text())
        self.assertNotIn("wrong", figure.texts[0].get_text())


if __name__ == "__main__":
    unittest.main()
