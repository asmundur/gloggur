# Artifact Publish State

This diagram documents `gloggur artifact publish`, from source validation to
manifest/archive creation and the transport-specific publish path.

| State | Transitions |
| --- | --- |
| `source_validation` | Verifies the source cache path exists, is a directory, and contains `index.db`. |
| `manifest_archive_creation` | Builds manifest metadata and creates the deterministic tar.gz archive. |
| `local_path` | Copies the temporary archive to a filesystem destination. |
| `http_put` | Uploads the archive directly to an HTTP(S) destination. |
| `uploader_command` | Publishes through an external argv-style uploader template. |
| `published` | Final success payload with checksums, manifest metadata, and transport details. |

```mermaid
stateDiagram-v2
    [*] --> source_validation

    state "source_validation" as source_validation
    state "manifest_archive_creation" as manifest_archive_creation
    state "local_path" as local_path
    state "http_put" as http_put
    state "uploader_command" as uploader_command
    state "published" as published

    source_validation --> manifest_archive_creation: source cache is initialized
    manifest_archive_creation --> local_path: transport=local_path
    manifest_archive_creation --> http_put: transport=http_put
    manifest_archive_creation --> uploader_command: transport=uploader_command

    local_path --> published
    http_put --> published
    uploader_command --> published
```

## Notes

- Destination validation happens before transport branching and can fail on
  unsupported schemes, destination collisions, or source/destination overlap.
- The final payload always reports `published=true`, plus the selected
  transport, archive checksums, and manifest data.
- Transport failures use artifact-specific error codes from
  [../ERROR_CODES.md](../ERROR_CODES.md).
