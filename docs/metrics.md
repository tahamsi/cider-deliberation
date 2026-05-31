# Metrics

Primary metrics:

- Accuracy with exact match, multiple-choice index matching, and numeric extraction for free-form math answers.
- Calibration via Brier score and expected calibration error.
- Cost via token and USD estimates recorded by agents.
- Exposure density and mean exposure load from method exposure matrices.
- Agreement, dissent, and answer-switch rates from transcripts.
- CCS, causal correction score: exposed agents that move from an initially wrong answer to a final correct answer divided by exposed agents that started wrong.
- CHS, causal harm score: exposed agents that move from an initially correct answer to a final wrong answer divided by exposed agents that started correct.
- PCI, persuasion contamination index: exposed answer switches that copy the visible majority without correctness improvement divided by all exposed answer switches.
- WCR, wrong convergence rate: task rate where the final modal answer is wrong and final agreement is at least 0.75.
- PDS, persuasion drift score: exposed answer switches divided by exposed final agent states.

Statistical helpers include paired t-tests and bootstrap confidence intervals.

The causal thesis metrics are operational transcript proxies. They are reported
from saved exposure and response logs, and should not be described as randomized
causal identification unless the corresponding run randomized exposure.
