# Reduced-order battery model

## Scope and convention

The model represents operational capability, not electrochemistry. Site nameplate values are divided evenly across configured PCS units and racks. Positive AC power `P` is discharge and negative power is charge. One state update occurs per configured interval `Δt` hours.

## SOC and energy

For usable rack capacity `C = C_nominal × SOH`, discharge efficiency `ηd`, and charge efficiency `ηc`:

```text
discharge (P >= 0): ΔE_internal = -(P × Δt / ηd)
charge    (P <  0): ΔE_internal = -P × Δt × ηc
SOC_next = clamp(SOC + ΔE_internal / C - self_discharge, SOC_min, SOC_max)
```

AC charge and discharge counters each accumulate `abs(P) × Δt` in their respective direction. Cumulative throughput is their sum. Equivalent full cycles are `throughput / (2 × C_nominal)`. Power is pre-limited by remaining energy so clamping should only absorb floating-point edge effects.

## Power capability

Rack nameplate power is multiplied by availability, scheduled power derating, and thermal derating. Energy-limited charge/discharge power is also computed from the distance to configured SOC bounds. Actual power is the request limited to this capability, followed by any tracking-error factor. Aggregate requests are allocated proportional to available capability.

Thermal derating is 1 below `derating_start_c`, decreases linearly to 0 at `shutdown_temperature_c`, and is 0 above shutdown. PCS/site nameplate caps and rack energy bounds are always enforced.

## Efficiency

Directional nominal efficiency receives modest transparent penalties for distance from 25 °C, distance from 50% SOC, operation away from mid-load, and SOH loss. An efficiency-degradation fault applies a multiplier. The result is bounded to 0.70–0.99. Reported RTE is the square of interval directional efficiency; it is a proxy, not a measured cycle test.

## Temperature

Rack temperature relaxes toward ambient according to a configurable time constant and cooling effectiveness. Heating rises with squared present/recent normalized power; a small SOC-extreme term is included. Cooling-degradation changes relaxation. Thermal-drift is modeled as an observed temperature offset, reflecting a local sensor/hotspot symptom without adding hidden energy.

## Capacity degradation

SOH begins at 1 and decreases from four configurable proxies: calendar time, EFC throughput, exposure above 30 °C, and depth away from 50% SOC. Accelerated-capacity-fade multiplies that interval loss. SOH is floored at 0.5 for numerical protection. Coefficients are assumptions, not empirically calibrated ageing parameters.

## Voltage and current

Mean rack voltage is a linear SOC proxy, `V_nominal × (0.90 + 0.20 × SOC)`. Current is `P × 10^6 / V`. Voltage and temperature spreads are diagnostic proxies. They are not cell-resolved electrical models.

