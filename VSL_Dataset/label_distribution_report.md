# VSL Label Distribution Audit

- Vocabulary records: 7810
- Case-insensitive exact labels after prefix removal: 5194
- Unique Vimeo IDs in vocabulary records: 7810
- Manifest Vimeo IDs: 7810; ID sets match: True
- Labels with at least 6 unique Vimeo IDs for manual screening: 34

## Labels by Unique Video Count

| Unique Vimeo IDs per label | Label count |
| --- | ---: |
| 0 | 0 |
| 1 | 3688 |
| 2-5 | 1472 |
| 6-10 | 34 |
| >10 | 0 |

## Media Cross-Check

- JSONL vocabulary media occurrences: `{"image": 7810, "embedded_reference": 7810}`
- Distinct JSONL vocabulary media URLs by type: `{"embedded_reference": 7810, "image": 1}`
- Manifest rows by type/status: `{"embedded_reference:reference_only": 7810, "image:queued": 45}`

Regional counts overlap when one page lists multiple regions. Candidate status only prioritizes review; it does not establish licensing, accessibility, or training quality.

Detailed files: `label_distribution.csv`, `label_region_distribution.csv`, and `label_candidates_ge_6.csv`.
