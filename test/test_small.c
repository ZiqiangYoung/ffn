/*
 * test_small.c — 小维度 SwiGLU FFN 测试
 *
 * 使用 d_model=4, d_ff=8 的小维度测试，便于理解计算过程。
 * 打印输入向量和输出向量的全部 4 个值。
 * 不打印中间向量，不进行数值正确性断言。
 */

#include "swiglu_ffn.h"
#include "generate_data.h"
#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int d_model = 4;
    int d_ff = 8;
    int seed = 42;

    printf("============================================================\n");
    printf("  SwiGLU FFN 小维度测试\n");
    printf("  d_model = %d, d_ff = %d, seed = %d\n", d_model, d_ff, seed);
    printf("============================================================\n\n");

    // 生成测试数据
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

    // 执行 SwiGLU FFN 前向计算
    swiglu_ffn(input, W_gate, W_up, W_down, d_model, d_ff, output);

    // 打印结果
    printf("输入向量 ");
    print_vector(input, d_model, 4, 0, "input");

    printf("输出向量 ");
    print_vector(output, d_model, 4, 0, "output");

    // 释放所有资源
    free(input);
    free(W_gate);
    free(W_up);
    free(W_down);
    free(output);

    printf("\n测试完成。\n");
    return 0;
}
