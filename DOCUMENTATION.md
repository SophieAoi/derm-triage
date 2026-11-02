# Project context
Uncertainty-aware AI triage for teledermatology (final-year project). Classifies skin photos into urgency tiers (Urgent, Soon, Routine); low-confidence cases are escalated. Research prototype, not clinical.

# Rules
- Python, PyTorch. Pin versions in requirements.txt. Fix random seeds.
- Never commit raw images or datasets. data/ is gitignored.
- Splits must be by patient/lesion so no case appears in both train and test.
- DDI is an external test set only: never used for training or tuning.
- The taxonomy is a CSV: source_label, condition_group, urgency_tier, rationale, source. It is frozen before training. Do not invent clinical rationales or sources; leave them blank for me to fill in.
- Explain what you did after each step. Ask before large changes.
