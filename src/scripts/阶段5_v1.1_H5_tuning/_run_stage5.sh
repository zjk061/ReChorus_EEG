#!/usr/bin/env bash
# Internal helper: run one stage-5 grid point.
# Usage: _run_stage5.sh <S5_ID> <lr> <dropout> <l2> [extra main.py args...]
set -euo pipefail

STAGE5_ID="${1:?S5_ID required}"
LR="${2:?lr required}"
DROPOUT="${3:?dropout required}"
L2="${4:?l2 required}"
shift 4

cd /root/autodl-tmp/src
source "$(dirname "$0")/_h5_stage5_common_args.sh"

echo "=============================================="
echo " Stage 5 ${STAGE5_ID}"
echo " lr=${LR}  dropout=${DROPOUT}  l2=${L2}"
if (("$#" > 0)); then
  echo " extra args: $*"
fi
echo "=============================================="

python main.py \
  "${H5_STAGE5_COMMON_ARGS[@]}" \
  --lr "${LR}" \
  --dropout "${DROPOUT}" \
  --l2 "${L2}" \
  "$@"
