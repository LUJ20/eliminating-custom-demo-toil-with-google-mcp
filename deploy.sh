#!/usr/bin/env bash
# Gemini + MCP Use-Case Studio: one-step deploy to Cloud Run (default), or a local run with --local.
#
#   ./deploy.sh --project YOUR_PROJECT_ID [--allow user:a@example.com,group:team@example.com]
#               [--region us-central1] [--service gemini-mcp-studio] [--min-instances 1]
#               [--bucket NAME] [--location global] [--drive-folder DRIVE_FOLDER_URL_OR_ID]
#   ./deploy.sh --local --project YOUR_PROJECT_ID [--drive-folder DRIVE_FOLDER_URL_OR_ID] [--port 8502]
#
# Cloud Run: builds this folder into a container, runs it as its own service account behind IAP (only you and
# the --allow list can open it) and prints the URL. Re-run any time to update the app or add people.
# Values can also come from .env (see .env.example) or environment variables.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
DRIVE="${DRIVE_FOLDER:-}"
BUCKET="${GCS_BUCKET:-}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
BUCKET_LOCATION="${BUCKET_LOCATION:-US}"
PORT="${PORT:-8502}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE="${CLOUD_RUN_SERVICE:-gemini-mcp-studio}"
MIN_INSTANCES="${CLOUD_RUN_MIN_INSTANCES:-1}"
ALLOW="${IAP_ALLOW:-}"
MODE="cloud-run"

while [ $# -gt 0 ]; do
  case "$1" in
    --project)        PROJECT="$2"; shift 2 ;;
    --local)          MODE="local"; shift ;;
    --allow)          ALLOW="$2"; shift 2 ;;
    --region)         REGION="$2"; shift 2 ;;
    --service)        SERVICE="$2"; shift 2 ;;
    --min-instances)  MIN_INSTANCES="$2"; shift 2 ;;
    --drive-folder)   DRIVE="$2"; shift 2 ;;
    --bucket)         BUCKET="$2"; shift 2 ;;
    --location)       LOCATION="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    -h|--help)        sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
retry() {  # IAM needs up to a minute to see a new service account
  local i
  for i in 1 2 3 4 5 6; do "$@" >/dev/null 2>&1 && return 0; sleep 10; done
  "$@" >/dev/null  # the last try shows the error
}

# Keep only the folder ID from a Drive link (a URL with '&' would break sourcing .env later).
case "$DRIVE" in
  */folders/*) DRIVE="${DRIVE#*/folders/}"; DRIVE="${DRIVE%%[/?#]*}" ;;
  *id=*)       DRIVE="${DRIVE#*id=}"; DRIVE="${DRIVE%%[&#]*}" ;;
esac

command -v gcloud >/dev/null || die "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
gcloud auth print-access-token >/dev/null 2>&1 || die "Not signed in. Run: gcloud auth login"
[ -n "$PROJECT" ] || PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
[ -n "$PROJECT" ] || die "No project. Pass --project YOUR_PROJECT_ID"
[[ "$MIN_INSTANCES" =~ ^[0-9]+$ ]] || die "--min-instances must be a whole number"
BUCKET="${BUCKET:-${PROJECT}-gemini-mcp-studio}"

step "Project: $PROJECT | bucket: gs://$BUCKET | location: $LOCATION | target: $MODE"
gcloud config set project "$PROJECT" --quiet >/dev/null

step "Enabling APIs"
# Drive, Slides and IAM Credentials always: Google Slides publishing works locally and on Cloud Run, where the
# service account gets its Drive-scoped token from the metadata server or, failing that, the IAM Credentials API.
APIS="aiplatform.googleapis.com developerknowledge.googleapis.com storage.googleapis.com drive.googleapis.com slides.googleapis.com iamcredentials.googleapis.com"
if [ "$MODE" = "cloud-run" ]; then
  APIS="$APIS run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com iap.googleapis.com iam.googleapis.com"
fi
gcloud services enable $APIS --project "$PROJECT"

step "Ensuring bucket gs://$BUCKET"
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$BUCKET_LOCATION" \
    --uniform-bucket-level-access --public-access-prevention
fi

step "Writing .env (other settings in it are kept)"
set_env() {  # replace the KEY= line (or append it); every other line stays as it is
  if grep -q "^$1=" .env 2>/dev/null; then
    awk -v k="$1" -v v="$2" 'index($0, k "=") == 1 { print k "=" v; next } { print }' .env > .env.tmp
    mv .env.tmp .env
  else
    printf '%s=%s\n' "$1" "$2" >> .env
  fi
}
touch .env
set_env GOOGLE_CLOUD_PROJECT "$PROJECT"
set_env GOOGLE_CLOUD_LOCATION "$LOCATION"
set_env GCS_BUCKET "$BUCKET"
set_env DRIVE_FOLDER "$DRIVE"

# ------------------------------------------------------------------------------------------ Cloud Run
if [ "$MODE" = "cloud-run" ]; then
  NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
  SA="gemini-mcp-studio@${PROJECT}.iam.gserviceaccount.com"
  echo "Google Slides on Cloud Run: put the Drive folder in a shared drive and add $SA as Content manager, then paste the folder link in the sidebar (or pass --drive-folder)."

  step "Service account $SA (Agent Platform user, API consumer, objects in gs://$BUCKET only, token creator on itself)"
  gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1 \
    || gcloud iam service-accounts create gemini-mcp-studio --project "$PROJECT" \
         --display-name "Gemini + MCP Studio on Cloud Run"
  for role in roles/aiplatform.user roles/serviceusage.serviceUsageConsumer; do
    retry gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" --role "$role" \
      --condition None --quiet
  done
  retry gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$SA" \
    --role roles/storage.objectAdmin --quiet
  # Lets the service account mint its own Drive-scoped token through the IAM Credentials API (Google Slides),
  # the fallback when the metadata server does not hand out the Drive scope itself.
  retry gcloud iam service-accounts add-iam-policy-binding "$SA" --member "serviceAccount:$SA" \
    --role roles/iam.serviceAccountTokenCreator --project "$PROJECT" --condition None --quiet \
    || echo "WARN: could not grant roles/iam.serviceAccountTokenCreator to $SA on itself; Google Slides may fall back to gs://$BUCKET/."

  step "Letting Cloud Build deploy from source (Cloud Run Builder on the default compute service account)"
  retry gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" --role roles/run.builder \
    --condition None --quiet || echo "WARN: could not grant roles/run.builder; the build may fail on permissions."
  # IAP service agent: what `gcloud beta services identity create` does, without installing the beta component.
  curl -fsS -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    "https://serviceusage.googleapis.com/v1beta1/projects/${NUMBER}/services/iap.googleapis.com:generateServiceIdentity" \
    >/dev/null 2>&1 || true

  step "Building and deploying $SERVICE to Cloud Run in $REGION (about 5 minutes the first time)"
  ENV_VARS="GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$LOCATION,GCS_BUCKET=$BUCKET${DRIVE:+,DRIVE_FOLDER=$DRIVE}"
  gcloud run deploy "$SERVICE" --source . --project "$PROJECT" --region "$REGION" \
    --service-account "$SA" --no-allow-unauthenticated --iap \
    --min-instances "$MIN_INSTANCES" --max-instances 1 --no-cpu-throttling --cpu 2 --memory 4Gi \
    --timeout 3600 --session-affinity --execution-environment gen2 \
    --set-env-vars "$ENV_VARS" \
    --quiet

  step "Access through IAP"
  retry gcloud run services add-iam-policy-binding "$SERVICE" --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:service-${NUMBER}@gcp-sa-iap.iam.gserviceaccount.com" --role roles/run.invoker --quiet
  ME="$(gcloud config get-value account 2>/dev/null || true)"
  case "$ME" in
    *.gserviceaccount.com) ME="serviceAccount:$ME" ;;
    ?*) ME="user:$ME" ;;
  esac
  IFS=',' read -r -a MEMBERS <<< "${ME}${ALLOW:+,$ALLOW}"
  for m in "${MEMBERS[@]}"; do
    [ -n "$m" ] || continue
    retry gcloud iap web add-iam-policy-binding --project "$PROJECT" --region "$REGION" \
      --resource-type cloud-run --service "$SERVICE" --member "$m" --role roles/iap.httpsResourceAccessor --quiet
    echo "  can open: $m"
  done

  URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
  step "Done: $URL"
  echo "First visit: sign in with an allowed Google account. A new instance spends a few minutes picking the newest"
  echo "models when it starts; open it sooner and the first page waits for that, once."
  echo "Add people:  ./deploy.sh --project $PROJECT --allow user:NAME@example.com,group:TEAM@example.com"
  echo "Remove:      gcloud run services delete $SERVICE --project $PROJECT --region $REGION"
  exit 0
fi

# ------------------------------------------------------------------------------------------ --local
if [ -n "$DRIVE" ]; then
  step "Checking Drive credentials"
  # The app uses the first credential that carries the Drive scope: the gcloud user, then ADC.
  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    gcloud auth application-default set-quota-project "$PROJECT" >/dev/null 2>&1 \
      || echo "WARN: could not set the ADC quota project to $PROJECT."
  fi
  echo "If Drive publishing reports a missing Drive scope, run one of:"
  echo "  gcloud auth login --enable-gdrive-access"
  echo "  gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive"
  echo "Until then the app publishes to gs://$BUCKET/<project-name>/ instead."
fi

step "Python environment"
PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys, ssl; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || die "Python 3.10+ not found"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip >/dev/null
.venv/bin/python -m pip install -q -r requirements.txt

step "Starting app on http://localhost:$PORT (the first run picks the newest models, a few minutes)"
exec .venv/bin/streamlit run app.py --server.port "$PORT" --server.headless true
