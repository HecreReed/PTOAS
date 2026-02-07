/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            const char *_recent = aclGetRecentErrMsg();                                          \
            if (_recent != nullptr && _recent[0] != '\0') {                                      \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            }                                                                                    \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

@LAUNCH_DECL@

int main() {
    @PARAM_DECLS@

    ACL_CHECK(aclInit(nullptr));
    int deviceId = 0;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    @ALLOC_HOST@
    @ALLOC_DEVICE@

    @READ_INPUTS@
    @COPY_TO_DEVICE@
    @LAUNCH_CALL@

    ACL_CHECK(aclrtSynchronizeStream(stream));
    @COPY_BACK@

    @WRITE_OUTPUT@

    @FREE_DEVICE@
    @FREE_HOST@
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
