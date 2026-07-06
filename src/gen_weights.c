/*
 * gen_weights.c — MoE FFN 权重生成工具
 *
 * 使用现有的 generate_data 模块生成 MoE 所有权重矩阵，
 * 以 raw float 二进制格式写入 MOE_WEIGHT_DIR 目录。
 *
 * 命令行接口：
 *   gen_weights [--seed=<int>] [--help]
 *
 * 维度参数全部来自 moe_config.h 宏定义，代码中不出现魔鬼数字。
 * 所有值服从 [-1, 1) 均匀分布，使用 MD5 派生种子（与 generate_data 一致）。
 */

#include "moe_config.h"
#include "generate_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/*
 * 打印帮助信息后以零退出
 */
static void print_help(const char* prog_name) {
    printf("用法: %s [选项]\n\n", prog_name);
    printf("生成 SwiGLU MoE FFN 的权重文件，写入 %s 目录\n\n", MOE_WEIGHT_DIR);
    printf("选项:\n");
    printf("  --seed=<int>    基础随机种子（默认: %d）\n", MOE_SEED_DEFAULT);
    printf("  --help          打印此帮助信息后退出\n\n");
    printf("生成文件:\n");
    printf("  共 %d 个 .bin 文件：\n", MOE_TOTAL_FILES);
    printf("  - W_router_%dx%d.bin\n", MOE_D_MODEL, MOE_ROUTER_COLS);
    for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
        printf("  - W_gate_%d_%dx%d.bin\n", e, MOE_D_MODEL, MOE_D_FF);
        printf("  - W_up_%d_%dx%d.bin\n", e, MOE_D_MODEL, MOE_D_FF);
        printf("  - W_down_%d_%dx%d.bin\n", e, MOE_D_MODEL, MOE_D_FF);
    }
    printf("\n所有值服从 [-1, 1) 均匀分布，小端序 raw float 二进制。\n");
}

/*
 * 解析 --seed=<int> 参数
 * 成功返回 1 并将值写入 *seed
 * 格式错误返回 -1
 * 参数不是 --seed 返回 0
 */
static int parse_seed_arg(const char* arg, int* seed) {
    const char* prefix = "--seed=";
    size_t prefix_len = strlen(prefix);

    if (strncmp(arg, prefix, prefix_len) != 0) {
        return 0;
    }

    // 确保等号后有值
    const char* val_str = arg + prefix_len;
    if (val_str[0] == '\0') {
        return -1;
    }

    // 验证所有字符都是数字或负号
    for (const char* p = val_str; *p != '\0'; p++) {
        if (*p != '-' && (*p < '0' || *p > '9')) {
            return -1;
        }
    }

    *seed = atoi(val_str);
    return 1;
}

/*
 * 确保输出目录存在，若不存在则尝试创建（权限 0755）
 */
static void ensure_dir(const char* dir_path) {
    struct stat st;
    if (stat(dir_path, &st) == 0) {
        // 目录已存在，检查是否为目录
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "错误：路径 \"%s\" 已存在但不是目录\n", dir_path);
            exit(1);
        }
        return;
    }

    // 目录不存在，尝试创建
    if (mkdir(dir_path, 0755) != 0) {
        fprintf(stderr, "错误：无法创建目录 \"%s\"：%s\n",
                dir_path, strerror(errno));
        exit(1);
    }
    printf("已创建目录: %s\n", dir_path);
}

/*
 * 将 float 数组以二进制形式写入文件
 */
static void write_float_bin(const char* path, const float* data, int count) {
    FILE* fp = fopen(path, "wb");
    assert(fp != NULL);
    size_t written = fwrite(data, sizeof(float), (size_t)count, fp);
    assert((int)written == count);
    fclose(fp);
}

int main(int argc, char* argv[]) {
    // ====================================================================
    // 解析命令行参数
    // ====================================================================
    int seed = MOE_SEED_DEFAULT;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }

        int result = parse_seed_arg(argv[i], &seed);
        if (result == -1) {
            fprintf(stderr, "错误：无效的 --seed 参数: %s\n", argv[i]);
            fprintf(stderr, "用法：%s [--seed=<int>] [--help]\n", argv[0]);
            return 1;
        } else if (result == 1) {
            continue;
        }

        // 无法识别的参数
        fprintf(stderr, "错误：无法识别的参数: %s\n", argv[i]);
        fprintf(stderr, "用法：%s [--seed=<int>] [--help]\n", argv[0]);
        return 1;
    }

    printf("============================================================\n");
    printf("  MoE FFN 权重生成工具\n");
    printf("  seed = %d\n", seed);
    printf("  d_model = %d, d_ff = %d, num_experts = %d\n",
           MOE_D_MODEL, MOE_D_FF, MOE_NUM_EXPERTS);
    printf("  输出目录: %s\n", MOE_WEIGHT_DIR);
    printf("============================================================\n\n");

    // ====================================================================
    // 确保输出目录存在
    // ====================================================================
    ensure_dir(MOE_WEIGHT_DIR);

    // ====================================================================
    // 生成并写入 W_router
    // W_router 形状 (MOE_D_MODEL, MOE_ROUTER_COLS)，行主序
    // ====================================================================
    {
        float* W_router = generate_W_router(MOE_D_MODEL, MOE_NUM_EXPERTS, seed);
        char path[512];
        snprintf(path, sizeof(path), "%s/W_router_%dx%d.bin",
                 MOE_WEIGHT_DIR, MOE_D_MODEL, MOE_ROUTER_COLS);
        write_float_bin(path, W_router, MOE_ROUTER_ELEMS);
        printf("已生成: %s\n", path);
        free(W_router);
    }

    // ====================================================================
    // 生成并写入每个专家的权重矩阵
    // 每个专家有 3 个矩阵：W_gate, W_up, W_down
    // ====================================================================
    for (int e = 0; e < MOE_NUM_EXPERTS; e++) {
        // W_gate 形状 (MOE_D_MODEL, MOE_D_FF)，行主序
        {
            float* W_gate = generate_W_gate_expert(MOE_D_MODEL, MOE_D_FF, seed, e);
            char path[512];
            snprintf(path, sizeof(path), "%s/W_gate_%d_%dx%d.bin",
                     MOE_WEIGHT_DIR, e, MOE_D_MODEL, MOE_D_FF);
            write_float_bin(path, W_gate, MOE_EXPERT_ELEMS);
            printf("已生成: %s\n", path);
            free(W_gate);
        }

        // W_up 形状 (MOE_D_MODEL, MOE_D_FF)，行主序
        {
            float* W_up = generate_W_up_expert(MOE_D_MODEL, MOE_D_FF, seed, e);
            char path[512];
            snprintf(path, sizeof(path), "%s/W_up_%d_%dx%d.bin",
                     MOE_WEIGHT_DIR, e, MOE_D_MODEL, MOE_D_FF);
            write_float_bin(path, W_up, MOE_EXPERT_ELEMS);
            printf("已生成: %s\n", path);
            free(W_up);
        }

        // W_down 形状 (MOE_D_FF, MOE_D_MODEL)，行主序
        {
            float* W_down = generate_W_down_expert(MOE_D_MODEL, MOE_D_FF, seed, e);
            char path[512];
            snprintf(path, sizeof(path), "%s/W_down_%d_%dx%d.bin",
                     MOE_WEIGHT_DIR, e, MOE_D_MODEL, MOE_D_FF);
            write_float_bin(path, W_down, MOE_DOWN_ELEMS);
            printf("已生成: %s\n", path);
            free(W_down);
        }
    }

    printf("\n共生成 %d 个权重文件到 %s\n", MOE_TOTAL_FILES, MOE_WEIGHT_DIR);
    printf("完成。\n");
    return 0;
}
