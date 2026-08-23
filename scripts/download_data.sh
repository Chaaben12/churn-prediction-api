#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/training/data"
FILE="${DATA_DIR}/Telco-Customer-Churn.csv"
URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SHA256="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

mkdir -p "${DATA_DIR}"

if [ -f "${FILE}" ] && printf '%s  %s\n' "${SHA256}" "${FILE}" | sha256sum --check --status >/dev/null 2>&1; then
    echo "Dataset already present and valid: ${FILE}"
    exit 0
fi

echo "Downloading Telco Customer Churn dataset..."
curl --fail --silent --show-error --location "${URL}" --output "${FILE}"

if ! printf '%s  %s\n' "${SHA256}" "${FILE}" | sha256sum --check --status; then
    echo "ERROR: SHA256 mismatch - remote file changed, refusing to use it." >&2
    rm -f "${FILE}"
    exit 1
fi

echo "OK: ${FILE} ($(( $(wc -l < "${FILE}") - 1 )) data rows)"
