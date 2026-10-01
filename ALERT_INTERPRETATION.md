# Alert interpretation

BESSPulse alerts are transparent `DERIVED DECISION-SUPPORT OUTPUT`, not another ML model. Alert
families cover delivery risk, anomaly, power/energy derating, rack/PCS unavailability, thermal and
tracking evidence, efficiency degradation, and commercial impact. Generation accepts only
observable/model/derived inputs; injected fault labels and severity are rejected by the strict
input schema and used only after replay for evaluation.

The default `priority_v2_weighted` score is
`confidence × (0.35 risk + 0.30 capacity + 0.25 commercial + 0.10 anomaly)`. The alternative
`priority_v1_product = confidence × risk × capacity × commercial` is retained. Risk is the maximum
reliability-adjusted 6h/12h/24h probability. Reliability weights are 1.0, 0.85, and 0.25; the 24h
model is contextual because Prompt 5 found poor out-of-time calibration. Anomaly score is clipped
separately and is never called a probability.

Capacity severity is the larger of affected-power/reference-power and affected-usable-energy /
reference-usable-energy. Commercial severity uses the Prompt 9 dimensionless RevenueAtRisk
fraction and safely becomes zero when opportunity is zero. Confidence is an operational score
from feature completeness, selected horizon reliability, and detector support; it is not a
statistical confidence interval. A technical severity at or above 0.75 imposes a 0.65 priority
floor so severe engineering conditions cannot disappear during low-value periods.

Project-assumption bands are INFO <0.20, LOW <0.40, MEDIUM <0.65, HIGH <0.85, and CRITICAL
otherwise. Alert evidence retains raw MW, MWh, probabilities, anomaly score, RevenueAtRisk,
limiting factor, source versions, price provenance, and the counterfactual disclaimer.

Two intervals are required to open and three clear intervals to resolve. Open 0.30 and close 0.20
thresholds provide hysteresis. Active issues update instead of duplicating; keys include component,
type, and a 30-minute merge bucket. Cooldown is configured for downstream enforcement. Alerts may
be OPEN or RESOLVED; acknowledgment remains out of scope.

The API returns these fields through `/api/v1/alerts` with bounded pagination and filters. Its
`anomaly_score_semantics` states that the score is not a probability, and structured provenance is
`DECISION_SUPPORT`. No fault-truth or replay-evaluation fields are exposed.
