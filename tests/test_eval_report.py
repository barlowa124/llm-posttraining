"""Published-eval artifacts: schema, claims binding, vendored parity.

The committed evals/results artifacts are the published surface — these
tests pin their schema and re-run the claims check so EVAL_REPORT.md can
never drift from eval_stock.json unnoticed.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
sys.path.insert(0, str(ROOT / "evals"))

from claims import flatten_results, verify_markdown  # noqa: E402
from run_stock_eval import _fmt_rate  # noqa: E402

# Same digest as bio-qc/tests/test_vendored_parity.py CLAIMS_SHA256 —
# third consumer of the vendored claims verifier.
CLAIMS_SHA256 = \
    "475eb4a6a338e363374c0810bf8f3861aec16cd41ae0c4ec2e0fbb54a6cc74a2"


class VendoredClaimsTest(unittest.TestCase):
    def test_claims_copy_matches_portfolio_digest(self):
        digest = hashlib.sha256(
            (ROOT / "evals" / "claims.py").read_bytes()).hexdigest()
        self.assertEqual(digest, CLAIMS_SHA256)


class EvalArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = json.loads((RESULTS / "eval_stock.json").read_text())
        cls.report = (ROOT / "EVAL_REPORT.md").read_text()

    def test_stock_results_schema(self):
        s = self.stock
        self.assertEqual(s["n_prompts"],
                         s["n_answerable"] + s["n_unanswerable"])
        for slug, m in s["models"].items():
            for kind in ("answerable", "unanswerable"):
                rates = m[kind]
                total = (rates["correct"] + rates["abstains"]
                         + rates["degenerate"] + rates["fabricates"])
                self.assertAlmostEqual(total, 1.0, places=6)
                self.assertEqual(rates["n"], 80)
            resp = RESULTS / f"responses_{slug}.csv"
            self.assertTrue(resp.exists(), f"missing {resp}")

    def test_report_claims_bind_to_results(self):
        pool = flatten_results(self.stock)
        verdict = verify_markdown(self.report, pool)
        self.assertTrue(verdict["passed"],
                        f"unbound: {verdict['unbound_claims']}")

    def test_committed_claims_check_is_passing(self):
        v = json.loads((RESULTS / "claims_check.json").read_text())
        self.assertTrue(v["passed"])
        self.assertEqual(v["unbound_claims"], [])

    def test_fmt_rate_is_exact_for_eval_quanta(self):
        # n=80 rates are multiples of 1/80, exact at 4 decimals
        for k in range(81):
            r = k / 80
            self.assertEqual(float(_fmt_rate(r)), r)


if __name__ == "__main__":
    unittest.main()
