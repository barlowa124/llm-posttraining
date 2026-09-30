"""Verified-claims layer: every number in the markdown report must bind
to a recorded value in the run's JSON artifacts.

Ported from the oncology_coscientist draft verifier (trust-tools). The
pattern: flatten every numeric leaf of results/*.json into a value pool,
extract every numeric token from the report text, and require each token
to sit within display-rounding tolerance of a pooled value. A claim that
no recorded computation produced is flagged as unbound.

Cluster names are exempt like metric-key integers are in oncocs: "3" in
"cluster 3" is a label, not a measurement.
"""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_PHRASES = [
    "clinically validated",
    "proves",
    "causes",
    "state-of-the-art",
]

# Same boundary logic as oncocs: a '-' only attaches when not preceded by
# a word char/digit/dot, so "0.5-0.6" yields two tokens while " -0.5"
# keeps its sign. Scientific notation is consumed whole so "1.2e-20"
# is one claim, not the fragments "1" and "20".
_NUM_RE = re.compile(
    r"(?<![\w.%])-?\d(?:[\d,]*\d)?(?:\.\d+)?(?:[eE][+-]?\d+)?\s*%?"
    r"(?![\w.%])")


def flatten_results(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten every numeric leaf to {dotted.path: value}."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_results(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_results(v, f"{prefix}{i}."))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix[:-1]] = float(obj)
    return out


def extract_numbers(text: str) -> list[dict]:
    """Numeric tokens with display precision and context."""
    tokens = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).strip()
        is_pct = raw.endswith("%")
        val = raw.rstrip("%").replace(",", "")
        mant = re.split("[eE]", val)[0]
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        ctx = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
        tokens.append({"token": raw, "value": float(val), "is_pct": is_pct,
                       "decimals": decimals, "context": ctx.strip(),
                       "pos": m.start()})
    return tokens


def verify_markdown(text: str, flat_values: dict[str, float],
                    labels: set[str] | None = None) -> dict:
    """Bind every numeric token in `text` to a pooled recorded value.

    `labels` holds exempt identifiers (cluster names): a zero-decimal
    token equal to a label is an identifier, not a measurement.

    Each extracted number lands in `claims` with the artifact key it
    bound to (`bound_to`) — or "identifier" for label exemptions — so
    the verdict is an inspectable binding map, not just a boolean.
    """
    labels = {str(x) for x in (labels or set())}
    items = list(flat_values.items())
    claims, unbound, forbidden = [], [], []

    def _bind(tok) -> str | None:
        v = tok["value"]
        if "e" in tok["token"].lower().rstrip("%"):
            # sci notation: tolerance is half the mantissa's last digit,
            # scaled by the exponent
            exp = int(re.split("[eE]", tok["token"].rstrip("%"))[1])
            tol = 0.5 * 10.0 ** (exp - tok["decimals"])
        else:
            tol = 0.5 * 10 ** (-tok["decimals"])
        for k, fv in items:
            if abs(v - fv) <= tol:
                return k
        if tok["is_pct"]:
            for k, fv in items:
                if abs(v / 100.0 - fv) <= tol / 100.0:
                    return k
        if tok["decimals"] == 0 and str(int(v)) in labels:
            return "identifier"
        return None

    for tok in extract_numbers(text):
        bound = _bind(tok)
        claims.append({"token": tok["token"], "bound_to": bound,
                       "context": tok["context"]})
        if bound is None:
            unbound.append({"token": tok["token"], "context": tok["context"]})

    low = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", low):
            forbidden.append(phrase)

    return {"passed": not unbound and not forbidden,
            "n_claims": len(claims),
            "claims": claims,
            "unbound_claims": unbound,
            "forbidden": forbidden}
