/*
 * generate_data.c — 测试数据生成模块实现
 *
 * 为 SwiGLU FFN 生成可复现的随机测试数据。
 * 使用 OpenSSL MD5 从 base_seed + 标签字符串派生独立种子，
 * 确保同一 base_seed 下四个生成函数产生不同但可复现的数据。
 */

#include "generate_data.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <openssl/md5.h>

/*
 * 从 base_seed 和标签字符串派生独立的随机种子
 *
 * 派生方式：
 *   1. 拼接 sprintf(buf, "%d_%s", base_seed, tag)
 *   2. 对拼接字符串计算 MD5 哈希
 *   3. 取 MD5 digest 前 4 字节，按小端序解释为 int
 *   4. 返回该 int 作为 srand 的种子
 *
 * 参数：
 *   base_seed — 基础种子（用户指定）
 *   tag       — 标签字符串（如 "input", "W_gate" 等）
 *
 * 返回：
 *   派生出的独立种子值
 */
static int derive_seed(int base_seed, const char* tag) {
    // 拼接种子字符串
    char buf[256];
    int written = snprintf(buf, sizeof(buf), "%d_%s", base_seed, tag);
    assert(written > 0 && (size_t)written < sizeof(buf));

    // 计算 MD5 哈希
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5((const unsigned char*)buf, strlen(buf), digest);

    // 取前 4 字节转为 int（小端序解释）
    int derived = ((int)digest[0]) |
                  ((int)digest[1] << 8) |
                  ((int)digest[2] << 16) |
                  ((int)digest[3] << 24);

    return derived;
}

/*
 * 填充向量或矩阵数据，值服从 [-1, 1) 均匀分布
 *
 * 使用公式：(float)rand() / RAND_MAX * 2.0f - 1.0f
 * rand() 返回 [0, RAND_MAX] 范围内的整数
 * (float)rand() / RAND_MAX 映射到 [0.0f, 1.0f]
 * 乘 2 减 1 后映射到 [-1.0f, 1.0f)
 */
static void fill_random(float* data, int count, int seed, const char* tag) {
    int derived_seed = derive_seed(seed, tag);
    srand((unsigned int)derived_seed);

    for (int i = 0; i < count; i++) {
        data[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

float* generate_input(int d_model, int seed) {
    float* input = (float*)malloc((size_t)d_model * sizeof(float));
    assert(input != NULL);
    fill_random(input, d_model, seed, "input");
    return input;
}

float* generate_W_gate(int d_model, int d_ff, int seed) {
    int count = d_model * d_ff;
    float* W_gate = (float*)malloc((size_t)count * sizeof(float));
    assert(W_gate != NULL);
    fill_random(W_gate, count, seed, "W_gate");
    return W_gate;
}

float* generate_W_up(int d_model, int d_ff, int seed) {
    int count = d_model * d_ff;
    float* W_up = (float*)malloc((size_t)count * sizeof(float));
    assert(W_up != NULL);
    fill_random(W_up, count, seed, "W_up");
    return W_up;
}

float* generate_W_down(int d_model, int d_ff, int seed) {
    int count = d_ff * d_model;
    float* W_down = (float*)malloc((size_t)count * sizeof(float));
    assert(W_down != NULL);
    fill_random(W_down, count, seed, "W_down");
    return W_down;
}
