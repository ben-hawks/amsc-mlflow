# Setup, authentication and permissions

Source: the AmSC MLflow Integration Guide (Google Doc linked from the AmSC
`intro-to-mlflow-pytorch` README), the `intro-to-mlflow-pytorch` examples, and checks
against the staging server on 2026-10-02.

## Servers

| Server | URI | Notes |
|---|---|---|
| staging (default here) | `https://mlflow-staging.american-science-cloud.org` | named by the integration guide. Reachable from the public internet; returns `401 Unauthorized: Bearer token required` without a token |
| development | `https://mlflow.ms.dev.american-science-cloud.org` | used by the guide's setup snippet and every `intro-to-mlflow-pytorch` script. On 2026-10-02 it gave an empty reply from outside the AmSC network |

Set `MLFLOW_TRACKING_URI` explicitly in anything long-lived (job scripts, `.env`), so a
change of default doesn't silently move runs to another server. Runs, experiments and
registered models don't move between servers or workspaces.

## The Access Token

- Comes from **MyAmSC Profile → Raw Tokens → Access Token**. The ID Token next to it is
  rejected.
- It's a JWT. MLflow sends it as `Authorization: Bearer <token>` when
  `MLFLOW_TRACKING_TOKEN` is set. No other client configuration is needed.
- It expires. `amsc_mlflow.token_expiry()` decodes the `exp` claim (without verifying the
  signature, only to warn), and `check_connection.py` prints the expiry time.
- **Handling rules**:
  - the user sets it themselves, with `read -s` (bash) or `Read-Host -MaskInput`
    (PowerShell 7.1+);
  - never in source code, never on a command line (visible in `ps` and shell history),
    never printed, never committed;
  - `unset MLFLOW_TRACKING_TOKEN` when done;
  - for batch jobs, use a file only the user can read, named by `MLFLOW_TRACKING_TOKEN_FILE`
    (`references/hpc.md`).

  An agent using this skill must not ask for the token in conversation.

## Workspaces and permissions

- A workspace (MLflow >= 3.13, `mlflow.set_workspace()`) isolates experiments, registered
  models and prompts. AmSC's team workspace is `modelservices`. `amsc_mlflow.configure()`
  selects `MLFLOW_WORKSPACE`, defaulting to `modelservices` on AmSC servers.
  `MLFLOW_WORKSPACE=none` uses the server's default workspace.
- **Selecting a workspace doesn't grant access.** The account needs the workspace user role
  (USE permission), assigned by a platform admin. Without it, calls return `403 Permission
  denied` even though the token is fine.
- **Experiment ownership.** An experiment belongs to whoever created it. Other users
  creating runs in it can get 403. Use `<project>-<username>` for personal experiments, and
  ask the workspace admin about shared experiments.
- For a first connectivity test without `modelservices` access, `MLFLOW_WORKSPACE=default`
  works for most accounts. A run in `default` isn't visible from `modelservices`, and
  registry operations must use the workspace the run is in.
- An older client (no `set_workspace`) logs to the server's default workspace;
  `configure()` warns. Use mlflow >= 3.13. The AmSC examples pin 3.16.0, the AXESS demo
  pins 3.10.0, and this skill was tested with 3.16.1.

## Configuration files

`assets/env.example` lists every variable. Copy it to `.env` (git-ignored), and keep
the token out of it unless the file is chmod 600 and never committed.

## Error reference

| Error | Cause | Fix |
|---|---|---|
| `401 ... Bearer token required` | no token sent | set `MLFLOW_TRACKING_TOKEN` (or `MLFLOW_TRACKING_TOKEN_FILE`) |
| `401 ... Invalid Bearer Token` | expired, an ID Token, or corrupted | new Access Token |
| `403 Permission denied` | no workspace permission, or someone else's experiment | workspace role; own experiment name |
| `Active workspace '...' cannot be used because the remote server does not support workspaces` | workspace set against a server without them | `MLFLOW_WORKSPACE=none` |
| SSL certificate errors | site proxy or CA bundle | fix the CA bundle (`REQUESTS_CA_BUNDLE`); don't disable verification unless an admin says so |
