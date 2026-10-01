"""Transparent vote ensemble retaining individual detector evidence."""

import pandas as pd

from models.anomaly.config import AnomalyConfig


def score_vote_ensemble(
    detector_outputs: list[pd.DataFrame], config: AnomalyConfig | None = None
) -> pd.DataFrame:
    cfg = config or AnomalyConfig()
    if not detector_outputs:
        raise ValueError("Ensemble requires detector outputs")
    keys = ["timestamp_utc", "component_id", "component_type"]
    merged = detector_outputs[0][[*keys, "anomaly_flag"]].rename(
        columns={"anomaly_flag": detector_outputs[0]["detector_name"].iloc[0]}
    )
    detector_names = [str(detector_outputs[0]["detector_name"].iloc[0])]
    for output in detector_outputs[1:]:
        name = str(output["detector_name"].iloc[0])
        detector_names.append(name)
        merged = merged.merge(
            output[[*keys, "anomaly_flag"]].rename(columns={"anomaly_flag": name}),
            on=keys,
            validate="one_to_one",
        )
    votes = merged[detector_names].sum(axis=1).astype(int)
    merged["detector_name"] = "vote_ensemble"
    merged["detector_version"] = "vote_ensemble_v1"
    merged["anomaly_score"] = votes / len(detector_names)
    merged["threshold"] = cfg.ensemble_vote_threshold / len(detector_names)
    merged["anomaly_flag"] = votes >= cfg.ensemble_vote_threshold
    merged["persistence_count"] = votes
    merged["supporting_signals"] = [
        [name for name in detector_names if bool(merged.at[index, name])] for index in merged.index
    ]
    merged["feature_set_version"] = "v1"
    merged["data_provenance"] = "DERIVED"
    return merged.drop(columns=detector_names)


def aggregate_site_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent site roll-up while retaining rack evidence separately."""
    frame = intervals.copy()
    frame["pcs_id"] = frame["component_id"].astype(str).str.split("-RACK").str[0]
    frame["flagged_score"] = frame["anomaly_score"].where(frame["anomaly_flag"])
    summary = frame.groupby("timestamp_utc", as_index=False).agg(
        active_anomaly_count=("anomaly_flag", "sum"),
        max_component_anomaly_score=("anomaly_score", "max"),
        number_racks_anomalous=("anomaly_flag", "sum"),
        critical_component_count=("flagged_score", "count"),
    )
    pcs = (
        frame.loc[frame["anomaly_flag"]]
        .groupby("timestamp_utc")["pcs_id"]
        .nunique()
        .rename("number_pcs_anomalous")
    )
    summary = summary.merge(pcs, on="timestamp_utc", how="left")
    summary["number_pcs_anomalous"] = summary["number_pcs_anomalous"].fillna(0).astype(int)
    summary["data_provenance"] = "DERIVED"
    return summary
