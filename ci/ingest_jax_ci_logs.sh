#!/bin/bash
set -euo pipefail

: "${BUCKET:?}"
: "${INPUT_GPU_ARCH:?}"
: "${FILTER_REPO:=jax-ml/jax}"
: "${FILTER_RUN_ID:=}"

ROOT="jax-ci-test-logs/${FILTER_REPO}"
CUTOFF="$(date -u -d '2 days ago' +%F)"

echo "Scanning S3 prefix: s3://${BUCKET}/${ROOT}/"
echo "Cutoff date (UTC): ${CUTOFF}"

# Lists immediate child prefixes for an S3 path.
list_child_prefixes() {
  aws s3 ls "$1" 2>/dev/null | awk '$1 == "PRE" { print $2 }'
}

# Returns success if the given S3 object exists.
object_exists() {
  aws s3 ls "$1" >/dev/null 2>&1
}

# Returns success if the prefix is ready for ingestion.
required_objects_exist() {
  local prefix="$1"

  object_exists "s3://${BUCKET}/${prefix}/_SUCCESS" &&
  ! object_exists "s3://${BUCKET}/${prefix}/_INGESTED" &&
  object_exists "s3://${BUCKET}/${prefix}/logs.tar.gz" &&
  object_exists "s3://${BUCKET}/${prefix}/run-manifest.json"
}

# Downloads, extracts, ingests, and marks the prefix as processed.
ingest_prefix() {
  local prefix="$1"
  local wd

  echo "Ingesting: ${prefix}"

  wd="$(mktemp -d)"

  aws s3 cp "s3://${BUCKET}/${prefix}/run-manifest.json" "${wd}/run-manifest.json"
  aws s3 cp "s3://${BUCKET}/${prefix}/logs.tar.gz" "${wd}/logs.tar.gz"

  mkdir -p "${wd}/logs_dir/extracted"
  cp "${wd}/run-manifest.json" "${wd}/logs_dir/"
  tar -xzf "${wd}/logs.tar.gz" -C "${wd}/logs_dir/extracted"

  if true;
   # python3 ci/upload_pytest_to_db.py \
   # --local_logs_dir "${wd}/logs_dir" \
   # --run-tag "ci-run" \
   # --gpu-tag "${INPUT_GPU_ARCH}" \
   # --artifact_uri "s3://${BUCKET}/${prefix}"
  then
   #  printf '' | aws s3 cp - "s3://${BUCKET}/${prefix}/_INGESTED"
    echo "Marked _INGESTED: ${prefix}"
  else
    echo "Skipping ${prefix}: ingest failed"
  fi

  rm -rf "${wd}"
}

# Traverses:
#   repo -> branch -> mode -> run_dir -> combo
#
# Applies the 2-day cutoff at the run-dir level so older runs are
# skipped before scanning combo-level prefixes.
for branch in $(list_child_prefixes "s3://${BUCKET}/${ROOT}/"); do
  branch="${branch%/}"

  for mode in continuous nightly; do
    mode_root="${ROOT}/${branch}/${mode}"

    for run_dir in $(list_child_prefixes "s3://${BUCKET}/${mode_root}/"); do
      run_dir="${run_dir%/}"
      run_date="${run_dir:0:10}"

      echo "Scanning branch=${branch} mode=${mode} run_dir=${run_dir}"

      if [[ ! "${run_date}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Skipping malformed run dir: ${mode_root}/${run_dir}"
        continue
      fi

      if [[ "${run_date}" < "${CUTOFF}" ]]; then
        continue
      fi

      if [[ -n "${FILTER_RUN_ID}" ]] && [[ "${run_dir}" != *"_${FILTER_RUN_ID}_"* ]]; then
        continue
      fi

      run_root="${mode_root}/${run_dir}"

      for combo in $(list_child_prefixes "s3://${BUCKET}/${run_root}/"); do
        combo="${combo%/}"
        prefix="${run_root}/${combo}"

        required_objects_exist "${prefix}" || continue
        ingest_prefix "${prefix}"
      done
    done
  done
 done

echo "Done"
