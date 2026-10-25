"""Small synthetic integration example; these are not paper measurements."""

import json

from ipj.cost import dollars_energy
from ipj.energy.integrate import PowerSample, energy_joules

samples = [PowerSample(0.0, 100.0), PowerSample(1.0, 120.0), PowerSample(2.0, 100.0)]
joules = energy_joules(samples, start=0.0, end=2.0)
print(json.dumps({
    "provenance": "synthetic_example",
    "duration_s": 2.0,
    "energy_j": joules,
    "electricity_usd_per_kwh": 0.0871,
    "marginal_electricity_usd": dollars_energy(joules, usd_per_kwh=0.0871),
}, indent=2))
