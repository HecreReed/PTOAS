#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi

PTOAS_BIN=${PTOAS_BIN:-}
if [[ -z "${PTOAS_BIN}" ]]; then
  echo "error: PTOAS_BIN not set" >&2
  exit 2
fi

TESTDATA_DIR=${TESTDATA_DIR:-}
if [[ -z "${TESTDATA_DIR}" ]]; then
  echo "error: TESTDATA_DIR not set" >&2
  exit 2
fi

IN="${TESTDATA_DIR}/comm_p2p_dynamic_v0_roundtrip.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_comm_p2p_dynamic_out"}
mkdir -p "${OUT_DIR}"

BC="${OUT_DIR}/comm_p2p_dynamic_v0_roundtrip.ptobc"
ROUNDTRIP="${OUT_DIR}/comm_p2p_dynamic_v0_roundtrip.roundtrip.pto"

"${PTOBC_BIN}" encode "${IN}" -o "${BC}"
"${PTOBC_BIN}" decode "${BC}" -o "${ROUNDTRIP}"

[[ $(grep -Fc "pto.comm.tput(" "${ROUNDTRIP}") -eq 2 ]]
[[ $(grep -Fc "pto.comm.tget(" "${ROUNDTRIP}") -eq 2 ]]
grep -F "!pto.partition_tensor_view<?x4096xi8>" "${ROUNDTRIP}" >/dev/null
grep -F "atomic_add" "${ROUNDTRIP}" >/dev/null
grep -F "pto.comm.tput(%0, %1, buf(%2, %3)" "${ROUNDTRIP}" >/dev/null
grep -F "pto.comm.tget(%0, %1, buf(%2, %3)" "${ROUNDTRIP}" >/dev/null

"${PTOAS_BIN}" --pto-arch=a3 --emit-pto-ir "${ROUNDTRIP}" -o /dev/null
