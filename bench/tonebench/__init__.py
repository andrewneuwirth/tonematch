"""tonebench: measure how well a tone matcher recovers a guitar tone.

Two tracks:
- synthetic: DI rendered through our own amp model with known settings, plus random studio production.
  The answer is known, so both settings error and sound error can be scored.
- real: Guitar-TECHS mic'd-amp recordings as targets. Settings are unknown, so only sound error is scored,
  against the real recording of a held-out passage of the same performance.
"""

SR = 48_000
