#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

if [ $# -ne 2 ]; then
  echo "error: incorrect parameters"
  echo "usage: $0 <bisheng compiler path> <toolkit path>"
  exit 1
fi

COMPILER_PATH="$1"
TOOLKIT_PATH="$2"

if [ -f "$COMPILER_PATH" ]; then
  echo "bisheng compiler not exist."
  exit 1
fi

$COMPILER_PATH/*.run --noexec --extract=$COMPILER_PATH/t1

cp -r $COMPILER_PATH/t1/bisheng_compiler $TOOLKIT_PATH/cann/include/pto
cp -r $COMPILER_PATH/t1/hcc $TOOLKIT_PATH/cann/include/pto

rm -rf $COMPILER_PATH/t1

echo "install new bisheng compiler to $TOOLKIT_PATH/cann/include/pto success!"
