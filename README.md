# Classification with Rejection for Dermatological Triage: Uncertainty-Aware Prioritization and Skin-Tone Equity (derm-triage)

Uncertainty-aware AI triage for teledermatology. This research prototype sorts skin-complaint photos into urgency tiers (Urgent, Soon, Routine). Low-confidence cases are escalated for human review instead of being filed as low priority. A queue simulation and a skin-tone (Fitzpatrick) audit measure the effect on waiting time and queue position.

> Research prototype only. Not a diagnostic tool and not clinically validated.

## Status

In progress. Currently building the data pipeline and the diagnosis-to-urgency taxonomy.

## Datasets

HAM10000 (ISIC), PAD-UFES-20, Fitzpatrick17k, and DDI (external test only). Raw images are not included in this repo.

## Planned work

1. Urgency taxonomy and dataset relabelling
2. Baseline triage models
3. Uncertainty and escalation (softmax entropy, Monte Carlo dropout, deep ensembles)
4. Queue simulation
5. Skin-tone equity audit
6. Web dashboard