# API Versioning Contract

The ISL Sign-to-Text backend guarantees stability for frontend integration through a strict versioning policy. 

## Current Status: `v1`

All endpoints and WebSocket contracts defined in the current `api/app.py` and `api/schemas.py` are considered **v1**. This contract is now frozen.

### Breaking Changes (Requires `v2` migration)
Any of the following changes will necessitate a major version bump (`v2`) and will not be applied to the `v1` endpoints:
- Changing the expected `feature_dimension` (e.g. from 506 to 620).
- Changing the `sequence_length` (e.g. from 20 to 30) if it breaks the frontend window logic.
- Renaming or removing JSON fields in REST payloads or WebSocket messages.
- Altering the core URL paths.

### Non-Breaking Additions (Minor Version Bump)
The following changes are permitted under the current `v1` contract as a minor bump (e.g., `v1.1`):
- Adding new REST endpoints (e.g. `/metrics`).
- Adding new optional fields to JSON request payloads (defaulting to backward-compatible values).
- Adding new fields to JSON response payloads.
- Adding new message types to the WebSocket protocol (e.g., `{"type": "debug"}`) that frontends can safely ignore if unrecognized.
