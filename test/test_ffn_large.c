/*
 * test_ffn_large.c — FFN 大维度测试
 *
 * 使用 d_model=2048, d_ff=7168 的大维度测试，验证正确性。
 * 打印输出向量的前 5 个和后 5 个值，以及统计摘要（min/max/mean）。
 * 不打印输入向量，不打印中间向量，不进行数值正确性断言。
 */

#include "swiglu_ffn.h"
#include "generate_data.h"
#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int d_model = 2048;
    int d_ff = 7168;
    int seed = 42;

    printf("============================================================\n");
    printf("  SwiGLU FFN 大维度测试\n");
    printf("  d_model = %d, d_ff = %d, seed = %d\n", d_model, d_ff, seed);
    printf("============================================================\n\n");

    // 生成测试数据
    printf("正在生成测试数据...\n");
    fflush(stdout);

    float* input   = generate_input(d_model, seed);
    float* W_gate  = generate_W_gate(d_model, d_ff, seed);
    float* W_up    = generate_W_up(d_model, d_ff, seed);
    float* W_down  = generate_W_down(d_model, d_ff, seed);

    // 分配输出向量
    float* output = (float*)malloc((size_t)d_model * sizeof(float));
    if (output == NULL) {
        fprintf(stderr, "输出向量内存分配失败\n");
        return 1;
    }

    printf("数据生成完成，开始计算...\n\n");
    fflush(stdout);

    // 执行 SwiGLU FFN 前向计算
    swiglu_ffn(input, W_gate, W_up, W_down, d_model, d_ff, output);

    // 打印输出向量（前 5 个和后 5 个元素）
    printf("输出向量 ");
    print_vector(output, d_model, 5, 5, "output");
    printf("\n");

    // 打印统计摘要
    print_stats(output, d_model, "输出向量");
    printf("\n");

    // 释放所有资源
    free(input);
    free(W_gate);
    free(W_up);
    free(W_down);
    free(output);

    printf("测试完成。\n");
    return 0;
}
