# ISL Sign-to-Text Error Catalog

This catalog documents the error codes returned by the API during REST and WebSocket interactions. Frontend integrations should handle these codes gracefully.

| Code | Name | Description | Suggested Frontend Action |
|------|------|-------------|---------------------------|
| **E001** | `Feature dimension mismatch` | The incoming feature vector does not match the expected `506` shape. | Ensure MediaPipe output is correctly mapped to 253 points + velocity (506). |
| **E002** | `Invalid schema version` | The payload's `schema_version` is unsupported by this backend version. | Check `/health` to verify supported schema version. Update frontend payload. |
| **E003** | `Missing sequence frames` | A prediction was requested, but the sequence lacked the minimum required frames. | Ensure the sequence buffer is full before sending to `/predict`. |
| **E004** | `Flood protection` | Too many WebSocket requests are in flight (Max: 2). The frame was dropped. | None. This is normal during fast streaming. The backend drops frames to maintain low latency. |
| **E005** | `Internal inference failure` | An unhandled exception occurred during PyTorch evaluation. | Check backend logs. |
| **E006** | `Invalid normalization values` | Features contain values significantly outside the expected `[-3.0, 3.0]` range. | Check camera calibration and MediaPipe landmark scaling logic. |
| **E007** | `NaN or Inf detected` | The incoming features contain `NaN` (Not a Number) or `Infinity`. | Ensure frontend division by zero (e.g. face anchor distance) is handled. |
| **E008** | `Model not loaded` | Inference was requested before the model finished loading. | Wait for the WebSocket connection to establish fully. |
| **E009** | `HDF5 loading failure` | The dataset or precomputed asset failed to load. | Contact backend engineering. |
| **E010** | `Unsupported API version` | The requested endpoint or WS protocol version is deprecated. | Check `docs/API_VERSIONING.md`. |
