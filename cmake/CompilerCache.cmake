# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Optional compiler cache for faster local rebuilds after the LLVM21 upgrade.
# Enable with -DPTOAS_USE_COMPILER_CACHE=ON (default). Explicit
# CMAKE_CXX_COMPILER_LAUNCHER / CMAKE_C_COMPILER_LAUNCHER always win.

option(PTOAS_USE_COMPILER_CACHE
  "Auto-detect ccache/sccache when CMAKE_*_COMPILER_LAUNCHER is unset"
  ON)

function(ptoas_try_enable_compiler_cache)
  if(NOT PTOAS_USE_COMPILER_CACHE)
    message(STATUS "PTOAS compiler cache: disabled (PTOAS_USE_COMPILER_CACHE=OFF)")
    return()
  endif()

  if(DEFINED CMAKE_CXX_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "")
    message(STATUS "PTOAS compiler cache: using CMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
    return()
  endif()
  if(DEFINED ENV{CMAKE_CXX_COMPILER_LAUNCHER} AND NOT "$ENV{CMAKE_CXX_COMPILER_LAUNCHER}" STREQUAL "")
    set(CMAKE_CXX_COMPILER_LAUNCHER "$ENV{CMAKE_CXX_COMPILER_LAUNCHER}" CACHE STRING
        "C++ compiler launcher from environment" FORCE)
    if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER OR CMAKE_C_COMPILER_LAUNCHER STREQUAL "")
      if(DEFINED ENV{CMAKE_C_COMPILER_LAUNCHER} AND NOT "$ENV{CMAKE_C_COMPILER_LAUNCHER}" STREQUAL "")
        set(CMAKE_C_COMPILER_LAUNCHER "$ENV{CMAKE_C_COMPILER_LAUNCHER}" CACHE STRING
            "C compiler launcher from environment" FORCE)
      else()
        set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_CXX_COMPILER_LAUNCHER}" CACHE STRING
            "C compiler launcher (matched CXX launcher)" FORCE)
      endif()
    endif()
    message(STATUS "PTOAS compiler cache: using env launcher ${CMAKE_CXX_COMPILER_LAUNCHER}")
    return()
  endif()

  find_program(PTOAS_COMPILER_CACHE_PROGRAM NAMES ccache sccache)
  if(NOT PTOAS_COMPILER_CACHE_PROGRAM)
    message(STATUS "PTOAS compiler cache: no ccache/sccache found; install one for faster rebuilds")
    return()
  endif()

  set(CMAKE_CXX_COMPILER_LAUNCHER "${PTOAS_COMPILER_CACHE_PROGRAM}" CACHE STRING
      "C++ compiler launcher auto-detected by PTOAS" FORCE)
  set(CMAKE_C_COMPILER_LAUNCHER "${PTOAS_COMPILER_CACHE_PROGRAM}" CACHE STRING
      "C compiler launcher auto-detected by PTOAS" FORCE)
  message(STATUS "PTOAS compiler cache: enabled (${PTOAS_COMPILER_CACHE_PROGRAM})")
endfunction()

ptoas_try_enable_compiler_cache()
