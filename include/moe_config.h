/*
 * moe_config.h — MoE FFN SPDK 移植的宏定义（ffn 侧）
 *
 * 所有固定参数均定义为宏，代码中禁止使用魔鬼数字。
 * 此文件中的同名宏须与 SPDK 侧 lib/moe_ffn/moe_config.h 值保持一致。
 * 两边各自持有独立副本，通过约定维持一致性（构建系统不同：CMake vs SPDK Makefile）。
 */

#ifndef MOE_CONFIG_H
#define MOE_CONFIG_H

/* ===== 维度参数（须与 SPDK 侧 moe_config.h 同名宏值一致）===== */
#define MOE_D_MODEL         2048
#define MOE_D_FF            7168
#define MOE_NUM_EXPERTS     8
#define MOE_TOP_K           2

/* ===== 种子 ===== */
#define MOE_SEED_DEFAULT    42            /* --seed 未指定时的默认值 */

/* ===== 权重文件输出目录 ===== */
#define MOE_WEIGHT_DIR      "/tmp/moe_weights"

/* ===== 派生值（由上述宏计算，不单独定义） ===== */
#define MOE_ROUTER_COLS     MOE_NUM_EXPERTS       /* W_router 输出列数 */
#define MOE_INPUT_BYTES     (MOE_D_MODEL * (int)sizeof(float))  /* 单个 input/output 字节数 */
#define MOE_ROUTER_ELEMS    (MOE_D_MODEL * MOE_ROUTER_COLS)     /* W_router 元素数 */
#define MOE_EXPERT_ELEMS    (MOE_D_MODEL * MOE_D_FF)            /* W_gate / W_up 元素数 */
#define MOE_DOWN_ELEMS      (MOE_D_FF * MOE_D_MODEL)            /* W_down 元素数 */
#define MOE_TOTAL_FILES     (1 + 3 * MOE_NUM_EXPERTS)           /* 权重文件总数 */

#endif /* MOE_CONFIG_H */
