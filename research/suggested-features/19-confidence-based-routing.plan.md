# Feature 19: Confidence-Based Routing

**Status:** Optional | **CLI Flag:** `--confidence-routing` | **Complexity:** Low-Medium

---

## Overview

The router model (Feature 16) outputs a softmax probability vector over domains. This feature adds the decision logic: if the maximum softmax confidence exceeds a configurable threshold (default 0.7), route to the predicted specialist. If confidence is below threshold, fall back to the general model. Temperature scaling is applied to calibrate the router's raw logits before thresholding. All routing decisions are logged for analysis.

---

## Motivation

Raw softmax outputs from a classifier are often overconfident (high logit magnitude → high softmax → false certainty). Without calibration and thresholding:

- A poorly matched prompt might be routed to the wrong specialist with high "confidence"
- The system has no way to express "I'm not sure which specialist to use"
- There is no fallback mechanism for out-of-domain prompts

Confidence-based routing adds a principled uncertainty quantification layer:
- Calibrated confidences better reflect true accuracy
- Per-domain thresholds allow tuning for precision vs recall tradeoffs
- Fallback to general model is safe for ambiguous inputs
- Decision logging enables offline analysis and threshold tuning

---

## Architecture / Design

### Flow Diagram

```
Prompt tokens
     ↓
RouterModel.forward()  →  raw logits
     ↓
Temperature scaling:  logits / temperature
     ↓
Softmax  →  probabilities [p_react, p_nextjs, ..., p_general_ts]
     ↓
max_confidence = max(probabilities)
domain = argmax(probabilities)
     ↓
   [max_confidence >= threshold[domain]?]
      YES → route to specialist[domain]
      NO  → route to general_ts fallback
     ↓
Log routing decision
```

### Temperature Scaling

Temperature scaling is a post-hoc calibration technique that divides logits by a scalar T before softmax. T > 1 softens (spreads out) probabilities; T < 1 sharpens them. The optimal T is found by minimizing NLL on a validation set.

```python
calibrated_probs = softmax(logits / temperature)
```

T is fit once after training and stored alongside the router checkpoint.

---

## Implementation Steps

### Step 1: RoutingDecision Dataclass

```python
# cola_coder/router/routing.py
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class RoutingDecision:
    domain: str
    confidence: float
    all_probs: dict[str, float]
    used_fallback: bool
    fallback_reason: Optional[str]
    temperature: float
    threshold_used: float
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "confidence": round(self.confidence, 4),
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "threshold_used": self.threshold_used,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
        }
```

### Step 2: ConfidenceRouter Class

```python
# cola_coder/router/confidence_router.py
import time
import json
import pathlib
import torch
import torch.nn.functional as F
from .routing import RoutingDecision
from .model import RouterModel
from ..registry import SpecialistRegistry
from ..registry.schema import SpecialistEntry

DOMAIN_NAMES = ["react", "nextjs", "graphql", "prisma", "zod", "testing", "general_ts"]

class ConfidenceRouter:
    def __init__(
        self,
        router_model: RouterModel,
        registry: SpecialistRegistry,
        temperature: float = 1.0,
        global_threshold: float = 0.7,
        log_path: Optional[str] = "logs/routing_decisions.jsonl",
    ):
        self.model = router_model
        self.registry = registry
        self.temperature = temperature
        self.global_threshold = global_threshold
        self._log_path = log_path
        self._decision_log: list[RoutingDecision] = []

        if log_path:
            pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def route(
        self,
        input_ids: torch.Tensor,
        device: str = "cpu",
    ) -> RoutingDecision:
        """
        Given tokenized prompt, return a RoutingDecision with chosen domain.
        """
        t0 = time.perf_counter()

        with torch.no_grad():
            input_ids = input_ids.to(device)
            logits = self.model(input_ids)
            # Temperature scaling
            calibrated = logits / self.temperature
            probs = F.softmax(calibrated, dim=-1).squeeze(0)

        probs_dict = {name: probs[i].item() for i, name in enumerate(DOMAIN_NAMES)}
        max_conf, max_idx = probs.max(dim=0)
        predicted_domain = DOMAIN_NAMES[max_idx.item()]
        max_confidence = max_conf.item()

        # Get per-domain threshold from registry
        entry = self.registry.get_specialist(predicted_domain)
        threshold = entry.confidence_threshold if entry else self.global_threshold

        latency_ms = (time.perf_counter() - t0) * 1000

        if max_confidence >= threshold and entry is not None and entry.is_available():
            decision = RoutingDecision(
                domain=predicted_domain,
                confidence=max_confidence,
                all_probs=probs_dict,
                used_fallback=False,
                fallback_reason=None,
                temperature=self.temperature,
                threshold_used=threshold,
                latency_ms=latency_ms,
            )
        else:
            fallback_entry = self.registry.get_default()
            reason = (
                f"confidence {max_confidence:.3f} < threshold {threshold:.3f}"
                if max_confidence < threshold
                else "specialist checkpoint unavailable"
            )
            decision = RoutingDecision(
                domain=fallback_entry.name if fallback_entry else "general_ts",
                confidence=max_confidence,
                all_probs=probs_dict,
                used_fallback=True,
                fallback_reason=reason,
                temperature=self.temperature,
                threshold_used=threshold,
                latency_ms=latency_ms,
            )

        self._log_decision(decision)
        return decision

    def _log_decision(self, decision: RoutingDecision):
        self._decision_log.append(decision)
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
```

### Step 3: Temperature Calibration

```python
# cola_coder/router/calibration.py
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from .model import RouterModel

def calibrate_temperature(
    model: RouterModel,
    val_loader,
    device: str = "cuda",
    search_range: tuple = (0.1, 10.0),
) -> float:
    """
    Find optimal temperature T that minimizes NLL on validation set.
    Returns the optimal temperature scalar.
    """
    model.eval().to(device)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    def nll(temperature):
        scaled = logits_tensor / temperature
        loss = F.cross_entropy(scaled, labels_tensor).item()
        return loss

    result = minimize_scalar(nll, bounds=search_range, method="bounded")
    optimal_t = result.x
    print(f"Optimal temperature: {optimal_t:.4f} (NLL: {result.fun:.4f})")
    return float(optimal_t)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute ECE to measure calibration quality."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop_in_bin = in_bin.float().mean().item()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean().item()
            conf_in_bin = confidences[in_bin].float().mean().item()
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return ece
```

### Step 4: CLI Display with Confidence Bar

```python
# cli additions
def display_routing_decision(decision: RoutingDecision, console):
    """Rich CLI display of routing decision with confidence bar."""
    from rich.panel import Panel
    from rich.table import Table

    status = "[green]SPECIALIST[/green]" if not decision.used_fallback else "[yellow]FALLBACK[/yellow]"
    console.print(Panel(
        f"[bold]Routed to:[/bold] [cyan]{decision.domain}[/cyan]  "
        f"[bold]Status:[/bold] {status}  "
        f"[bold]Latency:[/bold] {decision.latency_ms:.1f}ms",
        title="Routing Decision"
    ))

    table = Table(show_header=False, box=None)
    table.add_column("Domain", style="dim", width=15)
    table.add_column("Bar", width=22)
    table.add_column("Score", justify="right")

    sorted_probs = sorted(decision.all_probs.items(), key=lambda x: -x[1])
    for domain, prob in sorted_probs:
        bar_len = int(prob * 20)
        bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (20 - bar_len)
        highlight = "[bold]→ [/bold]" if domain == decision.domain else "  "
        table.add_row(f"{highlight}{domain}", bar, f"{prob:.3f}")

    console.print(table)
    if decision.used_fallback and decision.fallback_reason:
        console.print(f"[dim]Fallback reason: {decision.fallback_reason}[/dim]")


@app.command()
def analyze_routing_log(
    log_path: str = typer.Option("logs/routing_decisions.jsonl"),
):
    """Analyze saved routing decisions log."""
    import json
    from collections import Counter
    records = [json.loads(l) for l in open(log_path)]
    domains = Counter(r["domain"] for r in records)
    fallback_rate = sum(1 for r in records if r["used_fallback"]) / len(records)
    avg_conf = sum(r["confidence"] for r in records) / len(records)
    console.print(f"Total decisions: {len(records)}")
    console.print(f"Fallback rate: {fallback_rate:.1%}")
    console.print(f"Avg confidence: {avg_conf:.3f}")
    for domain, count in domains.most_common():
        console.print(f"  {domain:15s} {count:5d} ({count/len(records):.1%})")
```

---

## Key Files to Modify

- `cola_coder/router/routing.py` — RoutingDecision dataclass
- `cola_coder/router/confidence_router.py` — ConfidenceRouter class
- `cola_coder/router/calibration.py` — temperature calibration
- `cola_coder/cli.py` — `analyze-routing-log` command, display helpers
- `cola_coder/generate.py` — integrate ConfidenceRouter into generation pipeline
- `configs/router.yaml` — add `temperature`, `global_threshold` fields
- `logs/` — log directory for routing decisions

---

## Testing Strategy

```python
# tests/test_confidence_routing.py
import torch

def make_mock_router(probs: list[float]):
    """Create a router model that always returns fixed logits."""
    import torch.nn as nn
    class MockRouter(nn.Module):
        def __init__(self, probs):
            super().__init__()
            # Inverse softmax approximation
            import math
            self._logits = torch.tensor([math.log(p + 1e-8) for p in probs]).unsqueeze(0)
        def forward(self, x):
            return self._logits.expand(x.shape[0], -1)
    return MockRouter(probs)

def test_routes_to_specialist_when_confident():
    # React gets 0.9 confidence — should route to react
    probs = [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
    router = make_mock_router(probs)
    # ... assert decision.domain == "react" and not decision.used_fallback

def test_falls_back_when_not_confident():
    # All domains near equal — should fallback
    probs = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
    router = make_mock_router(probs)
    # ... assert decision.used_fallback == True

def test_temperature_scaling_softens_probs():
    logits = torch.tensor([[5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    hot = F.softmax(logits / 2.0, dim=-1)
    cold = F.softmax(logits / 0.5, dim=-1)
    assert hot[0, 0] < cold[0, 0]  # Higher T → less confident

def test_routing_decision_logged(tmp_path):
    log_path = str(tmp_path / "routing.jsonl")
    # ... make decisions, verify log_path has correct JSONL entries
```

---

## Performance Considerations

- **Temperature calibration is offline:** Run once on validation set, store T in checkpoint. No runtime overhead.
- **Threshold tuning:** Provide a sweep utility that shows precision/recall curves at different thresholds to help choose optimal value.
- **Log rotation:** Routing log can grow large in production. Implement daily rotation or a max-size cap.
- **Per-domain thresholds:** Some domains (e.g., Prisma) are very distinctive and can use a lower threshold (0.5). Others (e.g., General TS vs React) may need higher (0.8).
- **Async logging:** Write routing decisions to a background queue to avoid blocking generation. Use `queue.Queue` + background writer thread.

---

## Dependencies

- Feature 16 (RouterModel)
- Feature 18 (SpecialistRegistry)
- `scipy` (for temperature calibration `minimize_scalar`)
- `rich` (for CLI confidence bar display)

---

## Estimated Complexity

| Task                          | Effort  |
|-------------------------------|---------|
| RoutingDecision dataclass     | 0.5h    |
| ConfidenceRouter class        | 3h      |
| Temperature calibration       | 2h      |
| CLI display + log analysis    | 1.5h    |
| Tests                         | 1.5h    |
| Integration with generator    | 1h      |
| **Total**                     | **~9.5h** |

Overall complexity: **Low-Medium** (math is simple, integration work is the bulk)

---

## 2026 Best Practices

- **ECE as calibration metric:** Expected Calibration Error is now standard for measuring classifier calibration quality. Target ECE < 0.05 after temperature scaling.
- **Platt scaling alternative:** For very small validation sets, Platt scaling (logistic regression on logits) may outperform temperature scaling.
- **Selective prediction:** If max confidence < 0.5 for ALL domains after calibration, consider rejecting the routing entirely and asking the user which domain they intend (interactive CLI mode).
- **Reliability diagrams:** Generate calibration reliability diagrams (confidence vs accuracy binned plot) for visual inspection — saves time vs debugging raw numbers.
- **Threshold A/B testing:** Log routing decisions with threshold version tags; enables offline comparison of threshold strategies without redeploying.
- **Per-session calibration:** In long-running sessions, track rolling accuracy of routing decisions and nudge temperature dynamically if accuracy drops.
