#!/bin/bash
set -euo pipefail

: "${INPUT_GPU_ARCH:?}"
: "${FILTER_REPO:=ROCm/jax}"
: "${FILTER_RUN_ID:=}"

ROOT="jax-ci-test-logs/${FILTER_REPO}/"
CUTOFF="$(date -u -d '2 days ago' +%F)"

echo "Scanning S3 prefix: s3://${BUCKET}/${ROOT}"
echo "Cutoff date (UTC): ${CUTOFF}"

aws s3 ls "s3://${BUCKET}/${ROOT}" --recursive \
  | awk '/\/_SUCCESS$/ {print $4}' \
  | while read -r SUCCESS_KEY; do

    PREFIX="${SUCCESS_KEY%/_SUCCESS}"

    # Filter by run id (optional manual override)
    if [[ -n "${FILTER_RUN_ID:-}" ]] && [[ "${PREFIX}" != *"_${FILTER_RUN_ID}_"* ]]; then
      continue
    fi

    # Skip already ingested
    if aws s3 ls "s3://${BUCKET}/${PREFIX}/_INGESTED" >/dev/null 2>&1; then
      continue
    fi

    # Extract run_dir
    RUN_DIR="$(basename "$(dirname "${PREFIX}")")"
    RUN_DATE="${RUN_DIR:0:10}"

    # Skip older than cutoff
    if [[ "${RUN_DATE}" < "${CUTOFF}" ]]; then
      continue
    fi

    # Skip if logs.tar.gz missing
    if ! aws s3 ls "s3://${BUCKET}/${PREFIX}/logs.tar.gz" >/dev/null 2>&1; then
      echo "Skipping ${PREFIX}: logs.tar.gz not found"
      continue
    fi

    echo "Ingesting: ${PREFIX}"

    WD="$(mktemp -d)"

    aws s3 cp "s3://${BUCKET}/${PREFIX}/run-manifest.json" "${WD}/run-manifest.json"
    aws s3 cp "s3://${BUCKET}/${PREFIX}/logs.tar.gz" "${WD}/logs.tar.gz"

    mkdir -p "${WD}/logs_dir/extracted"
    cp "${WD}/run-manifest.json" "${WD}/logs_dir/"
    tar -xzf "${WD}/logs.tar.gz" -C "${WD}/logs_dir/extracted"

    if python3 ci/upload_pytest_to_db.py \
      --local_logs_dir "${WD}/logs_dir" \
      --run-tag "ci-run" \
      --gpu-tag "${INPUT_GPU_ARCH}" \
      --artifact_uri "s3://${BUCKET}/${PREFIX}"
    then
      printf '' | aws s3 cp - "s3://${BUCKET}/${PREFIX}/_INGESTED"
      echo "Marked _INGESTED: ${PREFIX}"
    else
      echo "Skipping ${PREFIX}: ingest failed"
    fi

    rm -rf "${WD}"
  done

echo "Done"

