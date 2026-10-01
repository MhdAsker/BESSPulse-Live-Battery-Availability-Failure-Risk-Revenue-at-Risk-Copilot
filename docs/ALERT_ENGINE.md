# Alert engine

`alerts.priority` calculates raw and normalized components, `rules` selects an observable alert
family, `evidence` preserves lineage, `service` owns persistence/hysteresis/deduplication, and
`storage` upserts lifecycle state. Commercial context can reorder technically similar issues, but
it never creates a trading-only alert and never modifies deterministic physical capability.

Historical replay generates alerts causally in timestamp order. Simulated fault truth is loaded
only after generation to calculate coverage and lead time. The replay does not tune thresholds on
those outcomes.
