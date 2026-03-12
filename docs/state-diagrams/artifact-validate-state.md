# Artifact Validate State

This diagram documents `gloggur artifact validate`, including the successful
`valid` path and the warning-only provenance branch.

| State | Transitions |
| --- | --- |
| `path_validation` | Verifies the requested artifact path exists and is a file. |
| `manifest_schema_hash_verification` | Reads the archive, validates `manifest.json`, schema version, totals, and optional file hashes. |
| `warning_codes` | Success-with-warnings path when provenance is missing but strict provenance was not required. |
| `valid` | Final success payload with archive metadata and hash-verification details. |

```mermaid
stateDiagram-v2
    [*] --> path_validation

    state "path_validation" as path_validation
    state "manifest_schema_hash_verification" as manifest_schema_hash_verification
    state "warning_codes" as warning_codes
    state "valid" as valid

    path_validation --> manifest_schema_hash_verification: artifact path is readable
    manifest_schema_hash_verification --> warning_codes: warning_codes+=artifact_manifest_provenance_missing
    warning_codes --> valid: valid=true
    manifest_schema_hash_verification --> valid: warning_codes=[]
```

## Notes

- `require_provenance=true` changes missing provenance from a warning into a
  hard failure before `valid` is emitted.
- `expected_manifest_sha256` is enforced inside
  `manifest_schema_hash_verification`.
- The final payload may have `valid=true` and non-empty `warning_codes` at the
  same time.
