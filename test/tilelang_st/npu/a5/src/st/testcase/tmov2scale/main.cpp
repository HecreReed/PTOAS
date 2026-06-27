// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchTMOV2SCALE_f16_16x16x16(uint16_t *a, uint16_t *b, float *scale, float *c, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(uint16_t *, uint16_t *, float *, float *, void *);
    size_t lhsRows;
    size_t lhsCols;
    size_t rhsRows;
    size_t rhsCols;
    size_t scaleRows;
    size_t scaleCols;
    size_t outRows;
    size_t outCols;
};

static const TestCase kCases[] = {
    {"f16_16x16x16", LaunchTMOV2SCALE_f16_16x16x16, 16, 16, 16, 16, 1, 16, 16, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    size_t lhsBytes = tc.lhsRows * tc.lhsCols * sizeof(uint16_t);
    size_t rhsBytes = tc.rhsRows * tc.rhsCols * sizeof(uint16_t);
    size_t scaleBytes = tc.scaleCols * sizeof(float);  // 16 * 4 = 64 bytes
    size_t outBytes = tc.outRows * tc.outCols * sizeof(float);  // f32 output

    std::printf("[INFO] === case: %s (lhs=%zux%zu, rhs=%zux%zu, scale=%zux%zu, out=%zux%zu) ===\n",
        tc.name, tc.lhsRows, tc.lhsCols, tc.rhsRows, tc.rhsCols, tc.scaleRows, tc.scaleCols, tc.outRows, tc.outCols);

    std::string caseDir = std::string("./") + tc.name;

    void *lhsHost = nullptr, *rhsHost = nullptr, *scaleHost = nullptr, *outHost = nullptr;
    void *lhsDevice = nullptr, *rhsDevice = nullptr, *scaleDevice = nullptr, *outDevice = nullptr;

    aclrtMallocHost(&lhsHost, lhsBytes);
    aclrtMallocHost(&rhsHost, rhsBytes);
    aclrtMallocHost(&scaleHost, scaleBytes);
    aclrtMallocHost(&outHost, outBytes);

    aclrtMalloc(&lhsDevice, lhsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&rhsDevice, rhsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&scaleDevice, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), lhsBytes, lhsHost, lhsBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), rhsBytes, rhsHost, rhsBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/scale.bin").c_str(), scaleBytes, scaleHost, scaleBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/scale.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(lhsDevice, lhsBytes, lhsHost, lhsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(rhsDevice, rhsBytes, rhsHost, rhsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(scaleDevice, scaleBytes, scaleHost, scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(static_cast<uint16_t *>(lhsDevice), static_cast<uint16_t *>(rhsDevice),
                  static_cast<float *>(scaleDevice), static_cast<float *>(outDevice), stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (lhsDevice) aclrtFree(lhsDevice);
    if (rhsDevice) aclrtFree(rhsDevice);
    if (scaleDevice) aclrtFree(scaleDevice);
    if (outDevice) aclrtFree(outDevice);
    if (lhsHost) aclrtFreeHost(lhsHost);
    if (rhsHost) aclrtFreeHost(rhsHost);
    if (scaleHost) aclrtFreeHost(scaleHost);
    if (outHost) aclrtFreeHost(outHost);

    if (rc == 0) std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;
    int rc = 0, deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(envDevice);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter && std::strcmp(kCases[i].name, caseFilter) != 0) continue;
        if (RunCase(kCases[i], deviceId, stream) != 0) { rc = 1; break; }
    }

    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}