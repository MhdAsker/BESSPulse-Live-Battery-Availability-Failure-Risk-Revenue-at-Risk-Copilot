# Security policy

## Supported scope

BESSPulse 1.0 is a read-only portfolio/research demo without authentication or tenant isolation. Do not expose mutation controls or private operational data to the public internet.

## Secret handling

Secrets belong in local environment files excluded by Git or in deployment-platform secret stores. Never put database credentials, Gemini keys, ENTSO-E tokens, Supabase server keys, or deployment hooks in Streamlit/browser configuration. Supabase management tokens are not required.

Run `python scripts/check_secrets.py` before committing. If a real credential is committed or pasted into an external system, revoke and rotate it; removal from the latest commit does not remove it from Git history.

## Reporting

Report vulnerabilities privately to the repository owner. Include affected version, reproduction steps, and impact without including live credentials.
