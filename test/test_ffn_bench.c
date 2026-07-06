/*
 * test_ffn_bench.c — FFN 性能评测程序
 *
 * 以 latency（时延）为核心指标，评估 swiglu_ffn 的性能。
 * 评测策略与 fio（Flexible I/O Tester）保持一致：
 *   - 计时方式：clock_gettime(CLOCK_MONOTONIC) 纳秒级精度
 *   - 预热控制：按时间控制（--warmup），预热结束后 reset 统计
 *   - 运行控制：按时间控制（--runtime），到达后停止采集并输出报告
 *   - 百分位计算：对数分桶直方图，与 fio 分桶算法完全一致
 *   - 保底策略：至少 --min-calls 次调用
 */

#define _GNU_SOURCE   /* sched_setaffinity */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <getopt.h>
#include <sched.h>

#include "swiglu_ffn.h"
#include "generate_data.h"

/*
 * ============================================================================
 * 分桶直方图常量
 * ============================================================================
 *
 * 基于 fio 的对数分桶算法，但针对 benchmark 场景做了参数优化：
 *   - 将 M 从 6 提升到 8，桶宽度缩小 4 倍，分辨率大幅提升
 *   - 覆盖范围收窄到约 512ns ~ 17s（MSB 9~34），匹配 1ms~10s 的预期延迟范围
 *   - 总桶数 6912，内存 ~55 KB（仍然很小）
 *
 * 误差分析（M=8）：
 *   理论误差上界 = 1 / 2^(8+1) = 1/512 ≈ 0.2%（vs 原来的 0.78%）
 *
 * 分辨率对比（以 590ms 典型延迟为例）：
 *   原方案（M=6）：桶宽度 ≈ 8.4 ms，30ms 范围约 4 个桶
 *   新方案（M=8）：桶宽度 ≈ 2.1 ms，30ms 范围约 15 个桶
 *   分位数分辨率提升约 4 倍
 *
 * 分桶原理解析：
 *   设样本值 x 的最高有效位（MSB）为第 n 位（bit 0 起算），
 *   取第 n 位起往下的 M=8 位作为组内索引，丢弃剩余低位。
 *   每个组包含 2^8 = 256 个桶。
 *
 *   组 0~1：MSB 0~8，精确存储（val < 512 ns），512 个桶
 *   组 k（k≥2）：MSB = k+7，error_bits = k-1
 *   组 2：MSB 9，覆盖 512ns~1μs
 *   组 12：MSB 19，覆盖 262μs~524μs
 *   组 26：MSB 33，覆盖 8.6s~17.2s（最后一个正常组）
 */

#define BENCH_PLAT_BITS          8
#define BENCH_PLAT_VAL           (1 << BENCH_PLAT_BITS)     /* 256 */
#define BENCH_PLAT_GROUP_NR      27                         /* 组 0~26 = 27 组 */
#define BENCH_PLAT_NR            (BENCH_PLAT_GROUP_NR * BENCH_PLAT_VAL)  /* 6912 */

/* 保留旧常量名以保持代码可读性（后续代码统一用新名） */
#define FIO_IO_U_PLAT_BITS       BENCH_PLAT_BITS
#define FIO_IO_U_PLAT_VAL        BENCH_PLAT_VAL
#define FIO_IO_U_PLAT_GROUP_NR   BENCH_PLAT_GROUP_NR
#define FIO_IO_U_PLAT_NR         BENCH_PLAT_NR
#define FIO_IO_U_LIST_MAX_LEN    20

/*
 * ============================================================================
 * 百分位列表（对齐 fio 默认列表）
 * ============================================================================
 */

#define N_PERCENTILES 17

static const double PERCENTILE_LIST[N_PERCENTILES] = {
    1.0,  5.0,  10.0, 20.0, 30.0, 40.0, 50.0,
    60.0, 70.0, 80.0, 90.0, 95.0, 99.0, 99.5,
    99.9, 99.95, 99.99
};

/*
 * ============================================================================
 * 命令行参数配置
 * ============================================================================
 */

typedef struct {
    int d_model;
    int d_ff;
    int seed;
    int cpu;
    int warmup_sec;
    int runtime_sec;
    int min_calls;
    const char* json_output;
} bench_config_t;

/*
 * ============================================================================
 * 分桶直方图函数
 * ============================================================================
 */

/*
 * 将纳秒延迟值映射到分桶索引
 *
 * 算法（基于 fio plat_val_to_idx，M=8）：
 *   1. 计算 val 的 MSB（最高有效位，bit 0 起算）
 *   2. 若 MSB <= FIO_IO_U_PLAT_BITS（≤8），直接返回 val（小值精确存储）
 *   3. 否则：
 *      - error_bits = msb - FIO_IO_U_PLAT_BITS（要丢弃的低位数量）
 *      - base = (error_bits + 1) << FIO_IO_U_PLAT_BITS（本组之前的桶数）
 *      - offset = (FIO_IO_U_PLAT_VAL - 1) & (val >> error_bits)（组内偏移）
 *      - idx = min(base + offset, FIO_IO_U_PLAT_NR - 1)
 *
 * 示例（M=8）：
 *   组 0~1：MSB 0~8 → 索引 0~511，精确存储（val < 512ns），无舍入误差
 *   组 2：MSB 9   → 索引 512~767，丢弃 1 位（error_bits=1），桶宽 2ns
 *   组 3：MSB 10  → 索引 768~1023，丢弃 2 位（error_bits=2），桶宽 4ns
 *   ...
 *   组 12：MSB 19 → 索引 3072~3327，丢弃 11 位，桶宽 2048ns≈2μs
 *   组 26：MSB 33 → 索引 6656~6911，丢弃 25 位，桶宽 33.6ms
 */
static unsigned int plat_val_to_idx(unsigned long long val) {
    unsigned int msb, error_bits, base, offset, idx;

    // 计算 MSB（最高有效位位置，bit 0 起算）
    if (val == 0) {
        msb = 0;
    } else {
        msb = (unsigned int)(sizeof(val) * 8) - (unsigned int)__builtin_clzll(val) - 1;
    }

    // MSB <= FIO_IO_U_PLAT_BITS：小值，精确存储，不丢弃任何位
    if (msb <= FIO_IO_U_PLAT_BITS) {
        return (unsigned int)val;
    }

    // 计算要丢弃的低位数量
    error_bits = msb - FIO_IO_U_PLAT_BITS;

    // 计算本组之前各组的桶数之和
    // 组 0 和组 1 贡献 128 个桶，之后每组贡献 64 个桶
    base = (error_bits + 1) << FIO_IO_U_PLAT_BITS;

    // 丢弃 error_bits 个低位后，取剩余的 M 位作为组内偏移
    offset = (FIO_IO_U_PLAT_VAL - 1) & (val >> error_bits);

    // 确保索引不超过数组大小
    idx = (base + offset) < (FIO_IO_U_PLAT_NR - 1)
          ? (base + offset) : (FIO_IO_U_PLAT_NR - 1);

    return idx;
}

/*
 * 将分桶索引转换回代表值（桶范围的均值）
 *
 * 算法（基于 fio plat_idx_to_val，M=8）：
 *   1. 若 idx < FIO_IO_U_PLAT_VAL * 2（<512），直接返回 idx（小值精确）
 *   2. 否则：
 *      - error_bits = (idx >> FIO_IO_U_PLAT_BITS) - 1
 *      - base = 1ULL << (error_bits + FIO_IO_U_PLAT_BITS)（本组最小值）
 *      - k = idx % FIO_IO_U_PLAT_VAL（组内桶编号）
 *      - 返回 base + (k + 0.5) * (1 << error_bits)（桶范围中值）
 */
static unsigned long long plat_idx_to_val(unsigned int idx) {
    unsigned int error_bits;
    unsigned long long k, base;

    assert(idx < FIO_IO_U_PLAT_NR);

    // 组 0 和组 1：精确存储，直接返回
    if (idx < (unsigned int)(FIO_IO_U_PLAT_VAL << 1)) {
        return (unsigned long long)idx;
    }

    // 找到所属组，计算该组的最小值
    error_bits = (idx >> FIO_IO_U_PLAT_BITS) - 1;
    base = ((unsigned long long)1) << (error_bits + FIO_IO_U_PLAT_BITS);

    // 组内桶编号
    k = idx % FIO_IO_U_PLAT_VAL;

    // 返回桶范围的均值（中点加 0.5 修正）
    return base + (unsigned long long)(((double)k + 0.5) * (double)((unsigned long long)1 << error_bits));
}

/*
 * 从分桶直方图计算百分位值
 *
 * 参数：
 *   io_u_plat — 分桶直方图计数数组，长度 FIO_IO_U_PLAT_NR
 *   nr        — 样本总数
 *   plist     — 百分位列表（已排序）
 *   plen      — 百分位列表长度
 *   ovals     — 输出百分位值数组（由调用者分配，长度 plen）
 *   minv      — 输出：最小值
 *   maxv      — 输出：最大值
 */
static void calc_percentiles(
    const uint64_t* io_u_plat,
    unsigned long long nr,
    const double* plist,
    int plen,
    unsigned long long* ovals,
    unsigned long long* minv,
    unsigned long long* maxv)
{
    unsigned long long sum = 0;
    int j = 0;
    bool is_last = false;

    *minv = UINT64_MAX;
    *maxv = 0;

    // 遍历所有桶，累积样本数
    for (int i = 0; i < FIO_IO_U_PLAT_NR && !is_last; i++) {
        sum += io_u_plat[i];

        // 当累计样本数 ≥ 百分位阈值时，记录该桶代表值为百分位值
        while (((long double)sum) >= ((long double)plist[j] / 100.0 * (long double)nr)) {
            ovals[j] = plat_idx_to_val((unsigned int)i);

            // 更新最小/最大值
            if (ovals[j] < *minv) *minv = ovals[j];
            if (ovals[j] > *maxv) *maxv = ovals[j];

            if (j == plen - 1) {
                is_last = true;
                break;
            }
            j++;
        }
    }
}

/*
 * ============================================================================
 * Welford 在线算法 — 单次遍历计算均值和方差
 * ============================================================================
 *
 * 公式（n 从 1 开始计数，mean₀ = 0，M₂₀ = 0）：
 *   δ₁ = x - mean_{n-1}
 *   mean_n = mean_{n-1} + δ₁ / n
 *   δ₂ = x - mean_n
 *   M2_n = M2_{n-1} + δ₁ × δ₂
 *
 * 最终：
 *   avg = mean_N
 *   var = M2_N / N       （总体方差）
 *   stddev = sqrt(var)
 *   CV = stddev / avg
 *
 * 使用 long double 中间精度，减少大样本量下的浮点累积误差。
 */

typedef struct {
    long double mean;
    long double m2;
    unsigned long long n;
} welford_state_t;

/*
 * 喂入一个新样本
 */
static void welford_update(welford_state_t* ws, unsigned long long x) {
    ws->n++;
    long double delta1 = (long double)x - ws->mean;
    ws->mean += delta1 / (long double)ws->n;
    long double delta2 = (long double)x - ws->mean;
    ws->m2 += delta1 * delta2;
}

/*
 * 重置 Welford 状态（预热结束后清零统计）
 */
static void welford_reset(welford_state_t* ws) {
    ws->mean = 0.0L;
    ws->m2 = 0.0L;
    ws->n = 0;
}

/*
 * ============================================================================
 * 纳秒到可读单位转换（对齐 fio 策略）
 * ============================================================================
 */

typedef enum {
    UNIT_NSEC = 0,
    UNIT_USEC = 1,
    UNIT_MSEC = 2
} time_unit_t;

/*
 * 决定使用哪个时间单位进行文本输出
 *
 * 阈值（对齐 fio）：
 *   ns → μs：min > 2,000 && max > 99,999 && stddev > 1,000.0
 *   μs → ms：min > 2,000,000 && max > 99,999,999 && stddev > 1,000,000.0
 *
 * 三个条件全部满足才切换。先检查 ms 再检查 μs（从大到小）。
 */
static time_unit_t determine_unit(unsigned long long min_ns,
                                   unsigned long long max_ns,
                                   double stddev_ns) {
    if (min_ns > 2000000ULL && max_ns > 99999999ULL && stddev_ns > 1000000.0) {
        return UNIT_MSEC;
    }
    if (min_ns > 2000ULL && max_ns > 99999ULL && stddev_ns > 1000.0) {
        return UNIT_USEC;
    }
    return UNIT_NSEC;
}

/*
 * 获取单位对应的除数和单位名称
 */
static const char* unit_name(time_unit_t unit) {
    switch (unit) {
        case UNIT_NSEC: return "ns";
        case UNIT_USEC: return "us";
        case UNIT_MSEC: return "ms";
        default:        return "ns";
    }
}

static double unit_divisor(time_unit_t unit) {
    switch (unit) {
        case UNIT_NSEC: return 1.0;
        case UNIT_USEC: return 1000.0;
        case UNIT_MSEC: return 1000000.0;
        default:        return 1.0;
    }
}

/*
 * ============================================================================
 * 辅助函数
 * ============================================================================
 */

/*
 * 用随机值填充输入向量
 * 每次调用 swiglu_ffn 前调用此函数，确保每次输入不同
 * rand() 不在计时区间内
 */
static void fill_input_random(float* input, int d_model) {
    for (int i = 0; i < d_model; i++) {
        input[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

/*
 * 计算两个 timespec 之间的纳秒差值
 */
static inline unsigned long long timespec_diff_ns(
    const struct timespec* t0, const struct timespec* t1) {
    long long sec_diff = (long long)t1->tv_sec - (long long)t0->tv_sec;
    long long nsec_diff = (long long)t1->tv_nsec - (long long)t0->tv_nsec;
    return (unsigned long long)(sec_diff * 1000000000LL + nsec_diff);
}

/*
 * ============================================================================
 * JSON 输出
 * ============================================================================
 */

/*
 * 将百分位数值转为 JSON key（如 99.99 → "p99_99"）
 */
static const char* percentile_json_key(double pct) {
    static char key[32]; // p + up to 5 digits + up to 2 _ + null
    if (pct == floor(pct)) {
        // 整数百分位：直接 "p%d"
        snprintf(key, sizeof(key), "p%d", (int)pct);
    } else {
        // 带小数百分位：用 _ 替代 .
        // 如 99.5 → "p99_5", 99.99 → "p99_99"
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f", pct);
        // buf = "99.50" 时我们想要 "p99_5"
        // 更稳健的做法：手动构建
        int ipart = (int)pct;
        int fpart = (int)((pct - (double)ipart) * 100.0 + 0.5);
        // 去掉尾部多余的 0
        while (fpart > 0 && fpart % 10 == 0) {
            fpart /= 10;
        }
        snprintf(key, sizeof(key), "p%d_%d", ipart, fpart);
    }
    return key;
}

/*
 * 写入 JSON 性能报告到文件
 *
 * 返回：true 表示成功，false 表示失败
 */
static bool write_json_report(
    const char* path,
    const bench_config_t* cfg,
    unsigned long long min_ns,
    unsigned long long max_ns,
    double avg_ns,
    double stddev_ns,
    double cv,
    const unsigned long long* percentiles_ns,
    unsigned long long total_calls,
    double actual_runtime_sec)
{
    FILE* fp = fopen(path, "w");
    if (!fp) {
        return false;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"completed\": true,\n");
    fprintf(fp, "  \"parameters\": {\n");
    fprintf(fp, "    \"d_model\": %d,\n", cfg->d_model);
    fprintf(fp, "    \"d_ff\": %d,\n", cfg->d_ff);
    fprintf(fp, "    \"seed\": %d,\n", cfg->seed);
    fprintf(fp, "    \"cpu\": %d,\n", cfg->cpu);
    fprintf(fp, "    \"warmup_sec\": %d,\n", cfg->warmup_sec);
    fprintf(fp, "    \"runtime_sec\": %d,\n", cfg->runtime_sec);
    fprintf(fp, "    \"min_calls\": %d\n", cfg->min_calls);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"latency_ns\": {\n");
    fprintf(fp, "    \"min\": %llu,\n", (unsigned long long)min_ns);
    fprintf(fp, "    \"max\": %llu,\n", (unsigned long long)max_ns);
    fprintf(fp, "    \"avg\": %.1f,\n", avg_ns);
    fprintf(fp, "    \"stddev\": %.1f,\n", stddev_ns);
    fprintf(fp, "    \"cv\": %.3f,\n", cv);
    fprintf(fp, "    \"percentiles\": {\n");

    for (int i = 0; i < N_PERCENTILES; i++) {
        const char* key = percentile_json_key(PERCENTILE_LIST[i]);
        const char* comma = (i < N_PERCENTILES - 1) ? "," : "";
        fprintf(fp, "      \"%s\": %llu%s\n",
                key, (unsigned long long)percentiles_ns[i], comma);
    }

    fprintf(fp, "    }\n");
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"throughput\": {\n");
    fprintf(fp, "    \"total_calls\": %llu,\n", (unsigned long long)total_calls);
    fprintf(fp, "    \"runtime_sec\": %.3f,\n", actual_runtime_sec);
    fprintf(fp, "    \"rate_calls_per_sec\": %.1f\n",
            actual_runtime_sec > 0.0 ? (double)total_calls / actual_runtime_sec : 0.0);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return true;
}

/*
 * ============================================================================
 * 文本报告输出
 * ============================================================================
 */

/*
 * 格式化百分位显示字符串
 * 如 99.99 → "99.99%"
 */
static void format_percentile_display(double pct, char* buf, size_t bufsz) {
    if (pct == floor(pct)) {
        snprintf(buf, bufsz, "%.0f%%", pct);
    } else {
        snprintf(buf, bufsz, "%.2f%%", pct);
    }
}

/*
 * 打印文本格式的评测报告到 stdout
 */
static void print_text_report(
    const bench_config_t* cfg,
    unsigned long long min_ns,
    unsigned long long max_ns,
    double avg_ns,
    double stddev_ns,
    double cv,
    const unsigned long long* percentiles_ns,
    unsigned long long total_calls,
    double actual_runtime_sec,
    bool json_ok,
    const char* json_path)
{
    // 决定时间单位
    time_unit_t unit = determine_unit(min_ns, max_ns, stddev_ns);
    const char* unit_str = unit_name(unit);
    double div = unit_divisor(unit);

    // 计算每个百分位列宽度，用于对齐
    int pct_width = 6; // "99.99%"
    // 找出最长百分位字符串
    for (int i = 0; i < N_PERCENTILES; i++) {
        char buf[16];
        format_percentile_display(PERCENTILE_LIST[i], buf, sizeof(buf));
        int w = (int)strlen(buf);
        if (w > pct_width) pct_width = w;
    }

    printf("================================================================\n");
    printf("  SwiGLU FFN 性能评测报告\n");
    printf("  d_model = %d, d_ff = %d, seed = %d\n",
           cfg->d_model, cfg->d_ff, cfg->seed);
    printf("  CPU: %d, warmup = %ds, runtime = %ds, min_calls = %d\n",
           cfg->cpu, cfg->warmup_sec, cfg->runtime_sec, cfg->min_calls);
    printf("================================================================\n");
    printf("\n");

    // 延迟统计
    printf("延迟统计 (Latency):\n");
    printf("  min       = %8.3f %s\n", (double)min_ns / div, unit_str);
    printf("  max       = %8.3f %s\n", (double)max_ns / div, unit_str);
    printf("  avg       = %8.3f %s\n", avg_ns / div, unit_str);
    printf("  stddev    = %8.3f %s\n", stddev_ns / div, unit_str);
    printf("  CV        = %8.3f\n", cv);
    printf("\n");

    // 百分位分布
    printf("百分位分布 (Percentiles):\n");
    for (int i = 0; i < N_PERCENTILES; i++) {
        char pct_str[16];
        format_percentile_display(PERCENTILE_LIST[i], pct_str, sizeof(pct_str));
        printf("  %*s = %8.3f %s\n",
               pct_width, pct_str,
               (double)percentiles_ns[i] / div, unit_str);
    }
    printf("\n");

    // 吞吐量
    printf("吞吐量 (Throughput):\n");
    printf("  total_calls = %llu\n", (unsigned long long)total_calls);
    printf("  runtime     = %.3f s\n", actual_runtime_sec);
    printf("  rate        = %.2f calls/sec\n",
           actual_runtime_sec > 0.0
               ? (double)total_calls / actual_runtime_sec : 0.0);
    printf("\n");

    // JSON 写入状态
    if (json_ok) {
        printf("JSON 结果已写入: %s\n", json_path);
    }
}

/*
 * ============================================================================
 * 命令行参数解析
 * ============================================================================
 */

/*
 * 打印帮助信息
 */
static void print_help(void) {
    printf("用法: test_ffn_bench [选项]\n");
    printf("\n");
    printf("SwiGLU FFN 性能评测程序。以 latency（时延）为核心指标，\n");
    printf("评估 swiglu_ffn 从被调用到返回的单次墙钟时间。\n");
    printf("\n");
    printf("选项:\n");
    printf("  --d_model=<int>      模型维度（输入/输出维度），默认 2048\n");
    printf("  --d_ff=<int>         前馈网络中间维度，默认 7168\n");
    printf("  --seed=<int>         随机种子，默认 42\n");
    printf("  --cpu=<int>          绑定的 CPU 核心编号，默认 0\n");
    printf("  --warmup=<int>       预热时长（秒），默认 3\n");
    printf("  --runtime=<int>      正式运行时长（秒），默认 10\n");
    printf("  --min-calls=<int>    保底最小调用次数，默认 50\n");
    printf("  --json-output=<path> JSON 输出路径，默认 ./ffn_bench_result.json\n");
    printf("  --help               打印此帮助信息\n");
}

/*
 * 解析所有命令行参数，填充配置结构体
 *
 * 使用 POSIX getopt_long 进行参数解析，支持 --key=value 和 --key value 两种形式。
 *
 * 返回：0 成功，-1 参数错误，-2 需要退出（如 --help）
 */
static int parse_args(int argc, char** argv, bench_config_t* cfg) {
    // 设置默认值
    cfg->d_model = 2048;
    cfg->d_ff = 7168;
    cfg->seed = 42;
    cfg->cpu = 0;
    cfg->warmup_sec = 3;
    cfg->runtime_sec = 10;
    cfg->min_calls = 50;
    cfg->json_output = "./ffn_bench_result.json";

    /*
     * struct option 数组，定义所有长选项
     *
     * { 名称, 是否需要参数, 标志位（NULL=直接返回val）, val }
     * has_arg: no_argument=0, required_argument=1, optional_argument=2
     */
    static struct option long_options[] = {
        {"d_model",     required_argument, NULL, 'm'},
        {"d_ff",        required_argument, NULL, 'f'},
        {"seed",        required_argument, NULL, 's'},
        {"cpu",         required_argument, NULL, 'c'},
        {"warmup",      required_argument, NULL, 'w'},
        {"runtime",     required_argument, NULL, 'r'},
        {"min-calls",   required_argument, NULL, 'n'},
        {"json-output", required_argument, NULL, 'o'},
        {"help",        no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm': // --d_model
                cfg->d_model = atoi(optarg);
                break;
            case 'f': // --d_ff
                cfg->d_ff = atoi(optarg);
                break;
            case 's': // --seed
                cfg->seed = atoi(optarg);
                break;
            case 'c': // --cpu
                cfg->cpu = atoi(optarg);
                break;
            case 'w': // --warmup
                cfg->warmup_sec = atoi(optarg);
                break;
            case 'r': // --runtime
                cfg->runtime_sec = atoi(optarg);
                break;
            case 'n': // --min-calls
                cfg->min_calls = atoi(optarg);
                break;
            case 'o': // --json-output
                cfg->json_output = optarg;
                break;
            case 'h': // --help
                print_help();
                return -2;
            case '?': // 未知参数或缺少参数值（getopt_long 已打印错误）
                fprintf(stderr, "使用 --help 查看帮助信息。\n");
                return -1;
            default:
                fprintf(stderr, "错误: 未知选项\n");
                return -1;
        }
    }

    // 检查是否有多余的非选项参数
    if (optind < argc) {
        fprintf(stderr, "错误: 未知参数: %s\n", argv[optind]);
        fprintf(stderr, "使用 --help 查看帮助信息。\n");
        return -1;
    }

    return 0;
}

/*
 * ============================================================================
 * 主函数
 * ============================================================================
 */

int main(int argc, char** argv) {
    bench_config_t cfg;
    int ret = parse_args(argc, argv, &cfg);
    if (ret == -1) return 1;
    if (ret == -2) return 0; // --help

    /*
     * --------------------------------------------------------------------
     * CPU 亲和性设置
     * --------------------------------------------------------------------
     * 将当前进程绑定到指定 CPU 核心，消除跨核心迁移带来的调度噪声
     */
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cfg.cpu, &cpuset);
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            fprintf(stderr, "警告: 无法设置 CPU 亲和性到核心 %d\n", cfg.cpu);
            // 不终止程序，允许继续运行
        }
    }

    /*
     * --------------------------------------------------------------------
     * 数据准备：一次性生成权重矩阵（全程复用）
     * --------------------------------------------------------------------
     */
    float* W_gate = generate_W_gate(cfg.d_model, cfg.d_ff, cfg.seed);
    float* W_up   = generate_W_up(cfg.d_model, cfg.d_ff, cfg.seed);
    float* W_down = generate_W_down(cfg.d_model, cfg.d_ff, cfg.seed);

    // 分配 input 和 output 向量（栈上分配受限制，使用堆）
    float* input  = (float*)malloc((size_t)cfg.d_model * sizeof(float));
    float* output = (float*)malloc((size_t)cfg.d_model * sizeof(float));
    assert(input != NULL);
    assert(output != NULL);

    /*
     * --------------------------------------------------------------------
     * 分桶直方图初始化
     * --------------------------------------------------------------------
     */
    uint64_t* histogram = (uint64_t*)calloc(FIO_IO_U_PLAT_NR, sizeof(uint64_t));
    assert(histogram != NULL);

    /*
     * --------------------------------------------------------------------
     * Welford 统计状态初始化
     * --------------------------------------------------------------------
     */
    welford_state_t welford;
    welford_reset(&welford);

    unsigned long long running_min_ns = UINT64_MAX;
    unsigned long long running_max_ns = 0;

    // 用于时间控制的 timespec
    struct timespec warmup_start, now, phase_start;

    /*
     * --------------------------------------------------------------------
     * 预热阶段
     * --------------------------------------------------------------------
     * 循环调用 swiglu_ffn，采集时延但不计入最终统计。
     * 预热结束条件：调用次数 ≥ min_calls AND 墙钟时间 ≥ warmup_sec
     */
    clock_gettime(CLOCK_MONOTONIC, &warmup_start);

    unsigned long long warmup_calls = 0;
    while (1) {
        // 每次生成随机输入（不在计时区间内）
        fill_input_random(input, cfg.d_model);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        swiglu_ffn(input, W_gate, W_up, W_down, cfg.d_model, cfg.d_ff, output);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        unsigned long long latency_ns = timespec_diff_ns(&t0, &t1);

        // 预热期间也采集样本到直方图（但不计入统计），便于观察预热效果
        unsigned int idx = plat_val_to_idx(latency_ns);
        histogram[idx]++;

        warmup_calls++;

        // 检查是否满足预热退出条件
        if (warmup_calls >= (unsigned long long)cfg.min_calls) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (double)(now.tv_sec - warmup_start.tv_sec)
                           + (double)(now.tv_nsec - warmup_start.tv_nsec) / 1e9;
            if (elapsed >= (double)cfg.warmup_sec) {
                break;
            }
        }
    }

    /*
     * --------------------------------------------------------------------
     * 预热结束，重置统计
     * --------------------------------------------------------------------
     * 清空直方图和 Welford 状态，准备正式测量
     */
    memset(histogram, 0, FIO_IO_U_PLAT_NR * sizeof(uint64_t));
    welford_reset(&welford);
    running_min_ns = UINT64_MAX;
    running_max_ns = 0;

    /*
     * --------------------------------------------------------------------
     * 正式测量阶段
     * --------------------------------------------------------------------
     * 循环调用 swiglu_ffn，采集时延计入最终统计。
     * 测量结束条件：调用次数 ≥ min_calls AND 墙钟时间 ≥ runtime_sec
     */
    clock_gettime(CLOCK_MONOTONIC, &phase_start);

    unsigned long long total_calls = 0;
    while (1) {
        // 每次生成随机输入（不在计时区间内）
        fill_input_random(input, cfg.d_model);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        swiglu_ffn(input, W_gate, W_up, W_down, cfg.d_model, cfg.d_ff, output);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        unsigned long long latency_ns = timespec_diff_ns(&t0, &t1);

        // 更新分桶直方图
        unsigned int idx = plat_val_to_idx(latency_ns);
        histogram[idx]++;

        // 更新 Welford 在线统计
        welford_update(&welford, latency_ns);

        // 更新运行期 min/max
        if (latency_ns < running_min_ns) running_min_ns = latency_ns;
        if (latency_ns > running_max_ns) running_max_ns = latency_ns;

        total_calls++;

        // 检查是否满足测量退出条件
        if (total_calls >= (unsigned long long)cfg.min_calls) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (double)(now.tv_sec - phase_start.tv_sec)
                           + (double)(now.tv_nsec - phase_start.tv_nsec) / 1e9;
            if (elapsed >= (double)cfg.runtime_sec) {
                break;
            }
        }
    }

    // 计算实际运行时间
    clock_gettime(CLOCK_MONOTONIC, &now);
    double actual_runtime = (double)(now.tv_sec - phase_start.tv_sec)
                          + (double)(now.tv_nsec - phase_start.tv_nsec) / 1e9;

    /*
     * --------------------------------------------------------------------
     * 计算结果指标
     * --------------------------------------------------------------------
     */

    // Welford 最终结果
    double avg_ns = (double)welford.mean;
    double var = (welford.n > 0) ? (double)(welford.m2 / (long double)welford.n) : 0.0;
    double stddev_ns = sqrt(var);
    double cv = (avg_ns > 0.0) ? stddev_ns / avg_ns : 0.0;

    // 从直方图计算百分位
    unsigned long long percentiles_ns[N_PERCENTILES];
    unsigned long long pct_min, pct_max;
    calc_percentiles(histogram, total_calls, PERCENTILE_LIST, N_PERCENTILES,
                     percentiles_ns, &pct_min, &pct_max);

    // 百分位的 min/max 优先使用，但运行期统计的 min/max 作为补充
    // 对于最小值：如果分桶的最小值 > 运行期最小值，使用运行期值（后者更精确）
    if (running_min_ns < pct_min) pct_min = running_min_ns;
    if (running_max_ns > pct_max) pct_max = running_max_ns;

    /*
     * --------------------------------------------------------------------
     * 输出结果
     * --------------------------------------------------------------------
     */

    // JSON 输出
    bool json_ok = write_json_report(
        cfg.json_output, &cfg,
        pct_min, pct_max, avg_ns, stddev_ns, cv,
        percentiles_ns, total_calls, actual_runtime);

    // 文本报告输出到 stdout
    print_text_report(
        &cfg,
        pct_min, pct_max, avg_ns, stddev_ns, cv,
        percentiles_ns, total_calls, actual_runtime,
        json_ok, cfg.json_output);

    /*
     * --------------------------------------------------------------------
     * 清理资源
     * --------------------------------------------------------------------
     */
    free(W_gate);
    free(W_up);
    free(W_down);
    free(input);
    free(output);
    free(histogram);

    // 如果 JSON 写入失败，以非零退出码退出
    return json_ok ? 0 : 1;
}
