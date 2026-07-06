# CMakeLists.txt — SwiGLU FFN + MoE 项目构建配置
#
# 构建目标：
#   swiglu_ffn    — 静态库（silu + matvec + swiglu_ffn）
#   generate_data — 静态库（generate_data）
#   test_utils    — 静态库（test_utils）
#   softmax       — 静态库（softmax）
#   topk          — 静态库（topk）
#   swiglu_moe    — 静态库（swiglu_moe）
#   test_ffn_small — 可执行文件（FFN 小维度测试）
#   test_ffn_large — 可执行文件（FFN 大维度测试）
#   test_ffn_bench — 可执行文件（FFN 性能评测）
#   test_moe_small — 可执行文件（MoE 小维度测试）
#   test_moe_large — 可执行文件（MoE 大维度测试）

cmake_minimum_required(VERSION 3.10)
project(swiglu_ffn C)

# 使用 C11 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 查找 OpenSSL（需要 libcrypto 用于 MD5 哈希）
find_package(OpenSSL REQUIRED)

# ============================================================================
# 静态库目标
# ============================================================================

# swiglu_ffn 核心库（silu + matvec + swiglu_ffn）
add_library(swiglu_ffn STATIC
    src/silu.c
    src/matvec.c
    src/swiglu_ffn.c
)
target_include_directories(swiglu_ffn PUBLIC include)
target_link_libraries(swiglu_ffn PUBLIC m)

# generate_data 数据生成库
add_library(generate_data STATIC
    src/generate_data.c
)
target_include_directories(generate_data PUBLIC include)
target_link_libraries(generate_data PUBLIC OpenSSL::Crypto)

# test_utils 测试工具库
add_library(test_utils STATIC
    src/test_utils.c
)
target_include_directories(test_utils PUBLIC include)

# softmax — Softmax 归一化
add_library(softmax STATIC
    src/softmax.c
)
target_include_directories(softmax PUBLIC include)

# topk — Top-k 选择
add_library(topk STATIC
    src/topk.c
)
target_include_directories(topk PUBLIC include)

# swiglu_moe — MoE 层编排（依赖 swiglu_ffn + softmax + topk）
add_library(swiglu_moe STATIC
    src/swiglu_moe.c
)
target_include_directories(swiglu_moe PUBLIC include)
target_link_libraries(swiglu_moe PUBLIC swiglu_ffn softmax topk m)

# ============================================================================
# 测试可执行文件
# ============================================================================

# test_ffn_small — FFN 小维度测试（d_model=4, d_ff=8）
add_executable(test_ffn_small
    test/test_ffn_small.c
)
target_link_libraries(test_ffn_small PRIVATE
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)

# test_ffn_large — FFN 大维度测试（d_model=2048, d_ff=7168）
add_executable(test_ffn_large
    test/test_ffn_large.c
)
target_link_libraries(test_ffn_large PRIVATE
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)

# test_ffn_bench — FFN 性能评测程序
add_executable(test_ffn_bench
    test/test_ffn_bench.c
)
target_link_libraries(test_ffn_bench PRIVATE
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)

# test_moe_small — MoE 小维度测试（4→8，4专家×top-2）
add_executable(test_moe_small
    test/test_moe_small.c
)
target_link_libraries(test_moe_small PRIVATE
    swiglu_moe
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)

# test_moe_large — MoE 大维度测试（2048→7168，8专家×top-2）
add_executable(test_moe_large
    test/test_moe_large.c
)
target_link_libraries(test_moe_large PRIVATE
    swiglu_moe
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)
