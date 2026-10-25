"""Orchestration (Track C).

Two clusters live here. The trained-orchestrator pipeline — ``harness`` (action loop + Trajectory
schema), ``generate`` (Harbor-verified trace generation), ``filter`` (traces -> SFT dataset) — and
``routing``, the older live-routing cascade experiments (v1_query/v2_turn/v2_proxy/...). See README.md.
"""
