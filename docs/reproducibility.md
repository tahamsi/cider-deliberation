# Reproducibility

Every benchmark run records:

- random seed
- git commit when available
- model and agent configuration
- method names and parameters
- raw transcripts
- exposure matrices
- final predictions
- token and cost estimates
- aggregate metric CSV

Use the mock agent for deterministic local verification before API-backed experiments.
