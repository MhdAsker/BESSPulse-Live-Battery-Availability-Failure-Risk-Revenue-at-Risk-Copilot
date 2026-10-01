# BESSPulse availability definitions

All quantities below are deterministic `DERIVED ENGINEERING ANALYTIC` values calculated from observable `SIMULATED` telemetry. They are not ML predictions, contractual availability guarantees, or real commercial-BESS measurements. Positive power is discharge; negative power is charge.

## Technical availability

**Question:** How much installed physical equipment is observably operational?

- Rack technical availability is `available rack count / configured rack count`.
- PCS technical availability is `available PCS count / configured PCS count`.
- Site technical availability uses the rack fraction because default racks are equal-sized; this is equivalent to installed-rack-power weighting.
- Range: `[0,1]`, direction-neutral, site/rack/PCS level.
- A rack is available only when required telemetry is present, not stale, its observable availability is true, its state is not offline/faulted, and its parent PCS is available.
- Unknown components remain in the installed denominator but are reported separately. They are never silently healthy.

Example: one of 32 equal racks unavailable gives site technical availability `31/32 = 0.96875`; it does not automatically make its PCS unavailable.

## Directional power availability

**Question:** How much instantaneous AC MW can participate in each direction?

Rack power is the minimum of nominal rack MW multiplied by observable thermal derating and directional AC energy divided by the configured telemetry interval. It is zero when unavailable. PCS power is:

```text
min(sum connected rack directional MW, inferred observable PCS converter MW limit)
```

Site power is the sum of PCS capability capped by configured site power. `discharge_power_availability` and `charge_power_availability` divide directional MW by the corresponding rated site MW. Units are MW/fraction; range is `[0,rating]`/`[0,1]`.

`available_power_mw` is contextual: current requested direction for an active request; minimum symmetric directional capability while idle. It never replaces the explicit directional fields.

Thermal derating may lower MW while technical availability remains one. The observable PCS factor is normalized against rack-derived capability to avoid counting the same rack/SOC constraint twice.

## Directional energy availability

**Question:** How much AC energy is deliverable or acceptable before an SOC bound?

For a rack with nameplate capacity `C`, simulated SOH proxy `h`, SOC `s`, limits `s_min/s_max`, discharge efficiency `eta_d`, and charge efficiency `eta_c`:

```text
usable_capacity_mwh = C * h
available_discharge_energy_mwh = usable_capacity_mwh * max(s - s_min, 0) * eta_d
available_charge_energy_mwh = usable_capacity_mwh * max(s_max - s, 0) / eta_c
```

The discharge value is deliverable AC energy. The charge value is acceptable AC input needed to fill the headroom. Efficiency is applied exactly once. Unavailable/unknown components contribute zero.

Directional energy availability divides by the corresponding full-window AC denominator: `rated MWh * (s_max-s_min) * eta_d` for discharge and `rated MWh * (s_max-s_min) / eta_c` for charge. Range is `[0,1]`.

`available_energy_mwh` is an explicit alias for discharge energy; `charge_headroom_mwh` aliases acceptable charge energy. This differs from legacy simulator `SiteTelemetry.available_energy_mwh`, which is stored DC energy above minimum SOC and remains unchanged for backward compatibility.

## Requested-power and energy availability

**Question:** Can current deterministic capability support the EMS request?

For an active request:

```text
requested_power_availability = min(directional_available_power / abs(request), 1)
requested_energy_availability = min(directional_available_energy / (abs(request) * configured_hours), 1)
sustainable_request_duration_hours = directional_available_energy / abs(request)
```

The default energy-sufficiency horizon is one hour. A request at or below 0.1 MW magnitude is idle: `requested_power_is_active=false` and request availability/duration are null, not 100% or failed.

Examples: a 20 MW request with 15 MW capability gives 0.75 instantaneous request availability. A supported 10 MW request with 5 MWh available has 0.5 one-hour energy availability and 0.5 hours sustainable duration.

## Sustainable power

For duration `d`, `sustainable_power = min(directional_power_limit, directional_energy / d)`. Snapshots expose 15-minute, one-hour, and two-hour charge/discharge power. This prevents instantaneous MW from implying indefinite duration.

## Performance is separate

`delivery_ratio = abs(actual power) / abs(requested power)` for active requests. `observed_delivery_success` uses the existing 0.95 threshold. `capability_delivery_gap = delivery_ratio - requested_power_availability` is diagnostic only. It is possible for request availability to be one while delivery fails.

## Limiting factors and missing data

Factors are derived from observable constraints only: `SOC_LOW`, `SOC_HIGH`, `RACK_UNAVAILABLE`, `PCS_UNAVAILABLE`, `PCS_UNKNOWN`, `THERMAL_DERATING`, `POWER_LIMIT`, `ENERGY_LIMIT`, `CAPACITY_FADE`, or `MULTIPLE`. Component IDs identify contributors but do not diagnose injected fault cause.

Missing required telemetry, invalid values, missing parent state, or data older than the configured staleness threshold produce `UNKNOWN` with zero dispatch contribution. Historical calculation uses exact timestamp rows only and never future-fills. Time summaries weight by elapsed intervals; the final row uses the median preceding cadence.
## Commercial use

Prompt 9 conservatively aggregates directional power, deliverable discharge energy, acceptable
charge energy, and technical availability by minimum within each market interval. It reconstructs
the underlying usable DC window by reversing each directional efficiency exactly once. Anomaly
and delivery-risk scores do not alter these deterministic limits.
