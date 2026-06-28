# Security Policy — ISL Sign-to-Text

## Supported Versions

This is a Final Year Project (FYP) research codebase. The actively maintained version is:

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes    |
| 1.x     | ❌ No     |

---

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public GitHub issue**.

Instead, report it directly by opening a private issue or contacting the repository maintainer via GitHub.

**Response timeline:** Typically within 7 days for academic-severity issues.

---

## Known Security Considerations

### 1. CORS Policy

The API uses CORS middleware. In **development**, `DEBUG=true` allows all origins (`*`).
In **production** (default), `ALLOWED_ORIGINS` must be explicitly set via environment variable:

```bash
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

If `ALLOWED_ORIGINS` is unset in production, the API will reject cross-origin requests by default.

**Never** use `*` for production deployments.

### 2. No Authentication

The `/ws/translate` WebSocket endpoint and `/predict` endpoint have **no authentication**.

For production deployment:
- Place behind a reverse proxy (nginx, Caddy)
- Add API key validation middleware
- Rate-limit connections per IP

### 3. WebSocket Session Isolation

Each WebSocket connection is assigned a UUID-keyed session stored in `app.state.sessions`.
Sessions are cleaned up on disconnect. No cross-session data leakage occurs by design.

### 4. Input Validation

- All sequence shapes are validated against config-derived dimensions (not hardcoded)
- NaN/Inf detection is implemented in `/validate_features`
- JSON parse errors return a safe error message, not a stack trace

### 5. Model Files

Model `.pth` and `.onnx` files are **gitignored** and never committed to the repository.
They must be transferred out-of-band (e.g., shared drive, private artifact storage).

### 6. No Secrets in Codebase

Confirmed clean by audit:
- No hardcoded API keys, tokens, passwords, or credentials
- All sensitive configuration via environment variables (see `.env.example`)
- `.env` is gitignored

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API listen port |
| `DEBUG` | `false` | Enable top-5 debug responses |
| `ALLOWED_ORIGINS` | `""` | Comma-separated allowed CORS origins |

---

## Dependency Security

To scan for known CVEs in dependencies:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Or run Bandit for static analysis:

```bash
pip install bandit
bandit -r src api -ll
```
