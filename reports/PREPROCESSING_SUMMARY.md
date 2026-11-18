# Preprocessing validation

| Partition | Trips | GPS points | Candidate rows | Candidate misses vs proxy |
|---|---:|---:|---:|---:|
| Train | 80 | 208,417 | 1,667,336 | 18,967 (9.10%) |
| Student test | 10 | 19,160 | 153,280 | 67 (0.35%) |

The default fitting subset contains 136,582 GPS points / 1,092,656 candidate rows.
It excludes ambiguous/far derived labels and candidate misses, not whole trips.
Evaluation retains all valid points. Every reported label is a derived proxy.

Two training trips (00000054 and 00000033) account for most candidate misses.
Investigate their alignment and reference/GPS consistency before final model comparisons.
There are directly nonjoining reference-arc pairs in 11 trips; originals remain unchanged.
Projection-versus-haversine segment-length distortion is at most 1.14% for segments >1 m.

Five focused tests passed. All 90 output artifacts passed shape, finite-value, candidate-label,
time-order, fold, and mask checks. Exact segment search was checked against a brute-force
search on 10 real observations. All 360 original file hashes were verified.

The loader was checked for disjoint fitting/validation trip IDs and retention of all 19,160
student-test observations. No ML models were trained and no Git changes were committed.

See `validation_summary.json`, `processed/inventory.csv`, and `PRE_PROCESSING_DECISIONS.md`.
Open `development_trip_00000001.html`
for the interactive map. Orange shows an untrained nearest-road baseline.
