#!/usr/bin/env bash
# GCP 인프라 1회성 부트스트랩. 이미 있는 리소스는 건너뛰므로 재실행해도 안전하다.
#
#   gcloud auth login
#   PROJECT_ID=<프로젝트ID> bash deploy/gcp-bootstrap.sh
#
# 끝나면 GitHub Secrets에 넣을 값을 출력한다.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?PROJECT_ID를 지정하세요}"
GITHUB_REPO="${GITHUB_REPO:-SungjinWi99/Youth-Policy-RAG}"
REGION="${REGION:-asia-northeast3}"
ZONE="${ZONE:-asia-northeast3-a}"
VM_NAME="${VM_NAME:-youth-policy-rag}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
DISK_GB="${DISK_GB:-30}"
REPO_NAME="${REPO_NAME:-youth-policy-rag}"
SA_NAME="${SA_NAME:-github-deployer}"
POOL_NAME="${POOL_NAME:-github-pool}"
PROVIDER_NAME="${PROVIDER_NAME:-github-provider}"
APP_DIR="${APP_DIR:-/opt/youth-policy-rag}"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
GITHUB_OWNER="${GITHUB_REPO%%/*}"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "==> API 활성화"
gcloud services enable \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  iap.googleapis.com \
  sts.googleapis.com

echo "==> Artifact Registry"
gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="청년정책 RAG 컨테이너 이미지"

echo "==> OS Login 활성화 (프로젝트 전역)"
# CI가 SSH 키를 프로젝트 메타데이터에 심는 대신 IAM으로 접근을 통제한다.
gcloud compute project-info add-metadata --metadata enable-oslogin=TRUE

echo "==> 고정 외부 IP"
gcloud compute addresses describe "${VM_NAME}-ip" --region="$REGION" >/dev/null 2>&1 || \
  gcloud compute addresses create "${VM_NAME}-ip" --region="$REGION"
STATIC_IP="$(gcloud compute addresses describe "${VM_NAME}-ip" --region="$REGION" --format='value(address)')"

echo "==> 방화벽"
# HTTP만 외부에 연다. 8000(FastAPI)·3000(Next.js)·4319(LangFeather)는 열지 않는다.
gcloud compute firewall-rules describe allow-http >/dev/null 2>&1 || \
  gcloud compute firewall-rules create allow-http \
    --allow=tcp:80 --target-tags=http-server --source-ranges=0.0.0.0/0
# SSH는 인터넷에 열지 않고 IAP 터널 대역만 허용한다.
gcloud compute firewall-rules describe allow-ssh-from-iap >/dev/null 2>&1 || \
  gcloud compute firewall-rules create allow-ssh-from-iap \
    --allow=tcp:22 --source-ranges=35.235.240.0/20

echo "==> VM"
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="${DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --address="$STATIC_IP" \
    --tags=http-server \
    --metadata="startup-script=#!/usr/bin/env bash
set -eux
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
# 루트의 docker가 Artifact Registry를 VM 서비스 계정으로 인증하게 한다.
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
mkdir -p ${APP_DIR}/data/chroma ${APP_DIR}/data/sqlite ${APP_DIR}/data/raw ${APP_DIR}/langfeather-data
"
fi

echo "==> VM 서비스 계정에 이미지 pull 권한"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/artifactregistry.reader --condition=None >/dev/null

echo "==> 배포용 서비스 계정"
gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="GitHub Actions deployer"

for ROLE in \
  roles/artifactregistry.writer \
  roles/compute.osAdminLogin \
  roles/compute.viewer \
  roles/iap.tunnelResourceAccessor
do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" --role="$ROLE" --condition=None >/dev/null
done

echo "==> Workload Identity Federation"
gcloud iam workload-identity-pools describe "$POOL_NAME" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$POOL_NAME" \
    --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers describe "$PROVIDER_NAME" \
  --location=global --workload-identity-pool="$POOL_NAME" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_NAME" \
    --location=global \
    --workload-identity-pool="$POOL_NAME" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
    --attribute-condition="assertion.repository_owner == '${GITHUB_OWNER}'"

POOL_ID="$(gcloud iam workload-identity-pools describe "$POOL_NAME" --location=global --format='value(name)')"
# 이 저장소의 워크플로만 이 서비스 계정을 가장할 수 있다.
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" >/dev/null

PROVIDER_ID="$(gcloud iam workload-identity-pools providers describe "$PROVIDER_NAME" \
  --location=global --workload-identity-pool="$POOL_NAME" --format='value(name)')"

cat <<EOF

===========================================================
완료. GitHub 저장소 설정에 아래를 넣으세요.

Settings > Secrets and variables > Actions > Variables
  GCP_PROJECT_ID       ${PROJECT_ID}

Settings > Secrets and variables > Actions > Secrets
  GCP_WIF_PROVIDER     ${PROVIDER_ID}
  GCP_SERVICE_ACCOUNT  ${SA_EMAIL}

서비스 주소: http://${STATIC_IP}
VM 접속:     gcloud compute ssh ${VM_NAME} --zone ${ZONE} --tunnel-through-iap
===========================================================
EOF
