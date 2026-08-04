# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compatibility loader for PTODSL TileLib templates.

The template sources live in the top-level ``TileOps`` package.
"""

from __future__ import annotations

from pathlib import Path

from TileOps import load_template
import TileOps as _tileops

__path__ = [str(Path(_tileops.__file__).resolve().parent)]

__all__ = ["load_template"]
