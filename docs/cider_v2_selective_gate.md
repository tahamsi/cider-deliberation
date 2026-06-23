# CIDeR-v2.1 Selective Correction Gate

CIDeR-v2.1 retains adaptive exposure and contamination vetoes, but replaces the original all-or-nothing correction gate with an evidence-weighted selective gate.

## Acceptance logic

A switch is easier to accept when:

- the original answer is low-confidence, low-quality, invalid, or unsupported;
- another independent agent already proposed the replacement;
- other post-exposure agents converged on the replacement;
- the verifier agrees;
- the revised rationale improves materially and is not copied verbatim.

A switch is harder to accept when:

- the original answer is high-confidence, high-quality, and independently supported;
- the verifier disagrees;
- the proposal merely copies the visible majority;
- answer validity regresses.

Accepted borderline corrections receive an `acceptance_weight` in `[0.60, 1.00]`. The trajectory aggregator multiplies correction support by this weight instead of granting every accepted switch full credit.

## Compatibility

Set `v2_gate_mode: strict` to restore the original CIDeR-v2 gate. The campaign runner exposes this as `--gate_mode strict`, and its mechanism stage includes `legacy_strict_gate` as a direct comparison.

## Main parameters

| Parameter | Default | Purpose |
|---|---:|---|
| `v2_switch_accept_threshold` | 0.52 | Base selective acceptance threshold |
| `v2_soft_evidence_gain` | 0.10 | Minimum evidence for recovery paths |
| `v2_weak_initial_confidence` | 0.62 | Defines a weak original answer |
| `v2_strong_initial_confidence` | 0.82 | Defines a protected original answer |
| `v2_weak_initial_threshold_relief` | 0.08 | Relaxes the threshold for weak originals |
| `v2_peer_threshold_relief` | 0.08 | Relaxes the threshold for independent corroboration |
| `v2_post_exposure_threshold_relief` | 0.04 | Smaller relaxation for post-exposure corroboration |
| `v2_protected_initial_penalty` | 0.14 | Protects strong, independently supported originals |
| `v2_verifier_disagreement_penalty` | 0.10 | Raises the bar when the verifier disagrees |
| `v2_copy_similarity_threshold` | 0.72 | Activates the unsupported-copy veto |

## Diagnostics

Every result now records:

- `v2_gate_version`
- `v2_gate_summary`
- per-switch acceptance score and threshold
- independent and post-exposure corroboration counts
- weak/protected initial flags
- copy similarity
- fractional acceptance weight

Use:

```bash
python scripts/summarize_cider_v2_gate.py outputs/<campaign>
```

The summary reports beneficial-switch recall, harmful-switch acceptance, net accepted corrections, reason counts, and gate thresholds.
