# CMakeLists.txt — SwiGLU FFN 项目构建配置
#
# 构建目标：
#   swiglu_ffn    — 静态库（silu + matvec + swiglu_ffn）
#   generate_data — 静态库（generate_data）
#   test_utils    — 静态库（test_utils）
#   test_small    — 可执行文件（d_model=4, d_ff=8 测试）
#   test_large    — 可执行文件（d_model=2048, d_ff=7168 测试）
#   test_bench    — 可执行文件（性能评测）

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

# ============================================================================
# 测试可执行文件
# ============================================================================

# test_small — 小维度测试（d_model=4, d_ff=8）
add_executable(test_small
    test/test_small.c
)
target_link_libraries(test_small PRIVATE
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)

# test_large — 大维度测试（d_model=2048, d_ff=7168）
add_executable(test_large
    test/test_large.c
)
target_link_libraries(test_large PRIVATE
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)

# test_bench — 性能评测程序
add_executable(test_bench
    test/test_bench.c
)
target_link_libraries(test_bench PRIVATE
    swiglu_ffn
    generate_data
    test_utils
    OpenSSL::Crypto
    m
)
