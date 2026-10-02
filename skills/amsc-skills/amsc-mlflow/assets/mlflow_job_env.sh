#!/bin/bash
# Source in a Slurm or PBS job script (before Python starts) to reach the AmSC MLflow server
# from compute nodes:
#
#     AMSC_SITE=frontier source <skill>/assets/mlflow_job_env.sh
#
# - AMSC_SITE: perlmutter | frontier | alcf | none. If unset, guessed from NERSC_HOST /
#   LMOD_SYSTEM_NAME / hostname. See references/hpc.md for the status of each site's
#   settings (Frontier and ALCF values are unverified by a run with this skill).
# - Reads the token from MLFLOW_TRACKING_TOKEN_FILE (default ~/.amsc/mlflow_token) if
#   MLFLOW_TRACKING_TOKEN isn't already set. Never echoes it.
# - Exports the multipart-upload settings, the server and the workspace (overridable).

: "${MLFLOW_TRACKING_URI:=https://mlflow-staging.american-science-cloud.org}"
: "${MLFLOW_WORKSPACE:=modelservices}"
: "${MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD:=true}"
: "${MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE:=5242880}"
: "${MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE:=104857600}"
export MLFLOW_TRACKING_URI MLFLOW_WORKSPACE MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD \
       MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE

if [ -z "${AMSC_SITE:-}" ]; then
    case "${NERSC_HOST:-}${LMOD_SYSTEM_NAME:-}$(hostname -f 2>/dev/null)" in
        *perlmutter*|*nersc*) AMSC_SITE=perlmutter ;;
        *frontier*|*olcf*|*ccs.ornl.gov*) AMSC_SITE=frontier ;;
        *aurora*|*polaris*|*alcf*) AMSC_SITE=alcf ;;
        *) AMSC_SITE=none ;;
    esac
fi
case "$AMSC_SITE" in
    perlmutter|none) ;;  # Perlmutter compute nodes have outbound access
    frontier)
        export http_proxy=http://proxy.ccs.ornl.gov:3128 https_proxy=http://proxy.ccs.ornl.gov:3128
        export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,*.ccs.ornl.gov" ;;
    alcf)
        export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 ;;
    *) echo "mlflow_job_env.sh: unknown AMSC_SITE=$AMSC_SITE" >&2 ;;
esac
export HTTP_PROXY="${http_proxy:-}" HTTPS_PROXY="${https_proxy:-}"
[ -n "$HTTP_PROXY" ] || unset HTTP_PROXY HTTPS_PROXY

: "${MLFLOW_TRACKING_TOKEN_FILE:=$HOME/.amsc/mlflow_token}"
export MLFLOW_TRACKING_TOKEN_FILE
if [ -z "${MLFLOW_TRACKING_TOKEN:-}" ] && [ -r "$MLFLOW_TRACKING_TOKEN_FILE" ]; then
    MLFLOW_TRACKING_TOKEN="$(tr -d '\r\n' < "$MLFLOW_TRACKING_TOKEN_FILE")"
    export MLFLOW_TRACKING_TOKEN
fi
[ -n "${MLFLOW_TRACKING_TOKEN:-}" ] || echo "mlflow_job_env.sh: no MLflow token (set MLFLOW_TRACKING_TOKEN_FILE); logging will fail with 401" >&2
