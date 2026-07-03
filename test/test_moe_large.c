/*
 * test_moe_large.c — MoE 大维度测试
 *
 * 使用 d_model=2048, d_ff=7168, num_experts=8, top_k=2 的大维度测试 MoE 层。
 * 打印输出向量的前 5 个和后 5 个值，以及统计摘要（min/max/mean）。
 * 不打印输入向量、路由信息、中间专家输出。
 * 不进行数值正确性断言。
 */

#include "swiglu_moe.h"
#include "generate_data.h"
#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int d_model = 2048;
    int d_ff = 7168;
    int num_experts = 8;
    int top_k = 2;
    int seed = 42;

    printf("============================================================\n");
    printf("  SwiGLU MoE 大维度测试\n");
    printf("  d_model = %d, d_ff = %d, num_experts = %d, top_k = %d\n",
           d_model, d_ff, num_experts, top_k);
    printf("  seed = %d\n", seed);
    printf("============================================================\n\n");

    // 生成输入向量
    printf("正在生成输入向量...\n");
    fflush(stdout);
    float* input = generate_input(d_model, seed);

    // 生成路由器权重矩阵
    printf("正在生成路由器权重...\n");
    fflush(stdout);
    float* W_router = generate_W_router(d_model, num_experts, seed);

    // 为每个专家生成权重矩阵
    printf("正在生成 %d 个专家的权重矩阵...\n", num_experts);
    fflush(stdout);

    float* W_gate[num_experts];
    float* W_up[num_experts];
    float* W_down[num_experts];

    for (int e = 0; e < num_experts; e++) {
        W_gate[e] = generate_W_gate_expert(d_model, d_ff, seed, e);
        W_up[e]   = generate_W_up_expert(d_model, d_ff, seed, e);
        W_down[e] = generate_W_down_expert(d_model, d_ff, seed, e);
    }

    // 分配输出向量
    float* output = (float*)malloc((size_t)d_model * sizeof(float));
    if (output == NULL) {
        fprintf(stderr, "输出向量内存分配失败\n");
        return 1;
    }

    printf("数据生成完成，开始计算...\n\n");
    fflush(stdout);

    // 执行 SwiGLU MoE 前向计算
    swiglu_moe(input, W_router,
               (const float**)W_gate, (const float**)W_up, (const float**)W_down,
               num_experts, top_k, d_model, d_ff, output);

    // 打印输出向量（前 5 个和后 5 个元素）
    printf("输出向量 ");
    print_vector(output, d_model, 5, 5, "output");
    printf("\n");

    // 打印统计摘要
    print_stats(output, d_model, "输出向量");
    printf("\n");

    // 释放所有资源
    free(input);
    free(W_router);
    for (int e = 0; e < num_experts; e++) {
        free(W_gate[e]);
        free(W_up[e]);
        free(W_down[e]);
    }
    free(output);

    printf("测试完成。\n");
    return 0;
}
