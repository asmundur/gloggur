# Artifact Restore State

This diagram documents `gloggur artifact restore`, which always validates the
archive first and then restores through a staging directory before activation.

| State | Transitions |
| --- | --- |
| `valid` | Restore proceeds only after the embedded validation step reports `valid=true`. |
| `warning_codes` | Validation warnings are carried forward into the restore payload. |
| `destination_checks` | Verifies the destination path policy, overwrite behavior, and parent-directory readiness. |
| `staging_extraction` | Extracts cache members into a temporary restore directory. |
| `destination_activation` | Removes or replaces the final destination and promotes the staging directory into place. |
| `restored` | Final success payload with restored file counts and inherited validation warnings. |

```mermaid
stateDiagram-v2
    [*] --> valid

    state "valid" as valid
    state "warning_codes" as warning_codes
    state "destination_checks" as destination_checks
    state "staging_extraction" as staging_extraction
    state "destination_activation" as destination_activation
    state "restored" as restored

    valid --> warning_codes: validation_payload.warning_codes present
    valid --> destination_checks: validation_payload.warning_codes=[]
    warning_codes --> destination_checks: carry warning_codes forward

    destination_checks --> staging_extraction: destination accepted
    staging_extraction --> destination_activation: staged files restored
    destination_activation --> restored
```

## Notes

- Restore reuses the same validation logic as `artifact validate`, including
  provenance checks and expected manifest digest enforcement.
- `overwrite=false` blocks the transition out of `destination_checks` when the
  destination directory already exists.
- The final payload reports `restored=true` and includes any inherited
  `warning_codes`.
