# SwiGLU FFN 性能评测需求规格文档

---

## 1. 项目概述

### 1.1 背景

项目 `swiglu_ffn` 作为优化比赛的基线代码，分发给学生后，学生将优化 `swiglu_ffn` 函数的内部实现。需要新增一个性能评测程序 `test_bench`，用于量化评估学生优化实现的性能。

### 1.2 评测目标

以 **latency（时延）** 为核心指标，评估 `swiglu_ffn` 从被调用到返回的单次墙钟时间。重点关注长尾时延（P99、P99.99），同时提供 avg、max、min 以及吞吐量和稳定性指标。

### 1.3 评测策略对齐

性能评测策略与 fio（Flexible I/O Tester）保持一致：

- **计时方式：** `clock_gettime(CLOCK_MONOTONIC)` 纳秒级精度
- **预热控制：** 按时间控制（`--warmup`），预热期间采集但不计入统计，预热结束 reset 统计
- **运行控制：** 按时间控制（`--runtime`），运行时长到达后停止采集并输出报告
- **百分位计算：** 对数分桶直方图（logarithmic bucketing），与 fio 分桶算法完全一致
- **保底策略：** 不论预热或运行阶段，均保底至少 `--min-calls` 次调用

### 1.4 文件命名

| 文件 | 路径 |
|------|------|
| 源代码 | `test/test_bench.c` |
| 可执行文件 | `test_bench` |
| 默认 JSON 输出 | `./ffn_bench_result.json` |

---

## 2. 核心功能需求

### 2.1 学生实现接入方式

**链接时替换。**

- 学生将自己的优化实现写入 `src/swiglu_ffn.c`
- 保持函数签名不变：
  ```c
  void swiglu_ffn(const float* input, const float* W_gate, const float* W_up,
                  const float* W_down, int d_model, int d_ff, float* output);
  ```
- 测试程序通过链接学生版本的 `libswiglu_ffn.a` 静态库来调用
- 不需要动态加载或宏替换机制

### 2.2 评测维度参数化

维度完全参数化，通过命令行参数传入，不使用宏或全局常量。

- `--d_model=<int>`：模型维度（输入/输出维度），默认 `2048`
- `--d_ff=<int>`：前馈网络中间维度，默认 `7168`
- `--seed=<int>`：随机种子，默认 `42`

一次运行只评测一组维度配置。多组维度由外部脚本多次调用。

### 2.3 数据生成策略

**权重矩阵固定，每次调用 input 随机变化（模拟推理场景）。**

- 权重矩阵 `W_gate`、`W_up`、`W_down` 在预热开始前一次性生成（使用 `generate_W_gate`、`generate_W_up`、`generate_W_down`，seed 参数化），全程复用
- input 向量分配在栈上或堆上一次，每次计时前原地覆写随机值

**计时区间严格不包含 rand()：**

```c
// 伪代码（不在计时区间内生成输入）
fill_input_random(input, d_model);          // 不在计时区间
clock_gettime(CLOCK_MONOTONIC, &t0);        // 计时起点
swiglu_ffn(input, W_gate, W_up, W_down,    // ← 仅此处在计时区间内
            d_model, d_ff, output);
clock_gettime(CLOCK_MONOTONIC, &t1);        // 计时终点
record_latency_sample(t1 - t0);             // 纳秒
```

- 每次 input 真正随机，无重复模式，不循环复用固定 buffer 中的 input（避免分支预测器适应特定输入模式）
- `rand()` 写过的 input 在 L1 cache 中热乎的，与真实推理场景一致

### 2.4 预热与正式测量

**按时间控制（对齐 fio） + 保底最小调用次数。**

| 阶段 | 行为 | 控制参数 |
|------|------|----------|
| 预热 | 循环调用 `swiglu_ffn`，采集时延但不计入最终统计 | `--warmup=<seconds>`（默认 3） |
| 正式测量 | 预热结束 reset 统计，继续循环调用，采集时延计入统计 | `--runtime=<seconds>`（默认 10） |

**保底机制：**

- `--min-calls=<int>`（默认 50）：预热阶段至少调用 `min_calls` 次后才开始检查预热时间是否到达；正式测量阶段至少调用 `min_calls` 次后才开始检查运行时间是否到达
- 防止极端大维度下单次调用极慢导致预热/测量样本过少

### 2.5 计时粒度

**逐次计时。** 每次 `swiglu_ffn` 调用记录一个独立 latency 样本（纳秒精度），所有样本喂入分桶直方图。

### 2.6 CPU 亲和性控制

**通过 `sched_setaffinity` 将进程绑定到指定 CPU 核心。**

- `--cpu=<N>`：绑定的 CPU 核心编号，默认 `0`
- 消除跨核心迁移带来的调度噪声，确保评测公平
- 纯用户态实现，不需要特权

---

## 3. 时延分桶直方图（百分位计算）

### 3.1 分桶方案

**完全对齐 fio 的对数分桶算法。**

| 参数 | 值 | 说明 |
|------|-----|------|
| `FIO_IO_U_PLAT_BITS` | `6` | 每组内索引位数 |
| `FIO_IO_U_PLAT_VAL` | `64`（1 << 6） | 每组桶数 |
| `FIO_IO_U_PLAT_GROUP_NR` | `29` | 组数 |
| `FIO_IO_U_PLAT_NR` | `1856`（29 × 64） | 总桶数 |
| 统计误差 | < 1%（1 / 2^7 = 0.78%） | 理论误差上界 |

### 3.2 分桶算法

**`plat_val_to_idx(val)`：** 将纳秒 latency 值映射到 bucket 索引。

1. 找到 `val` 的 MSB（最高有效位，从 bit 0 起算）
2. 若 `MSB <= FIO_IO_U_PLAT_BITS`（即 ≤6）：直接返回 `val` 作为索引（精确存储，无舍入误差）
3. 否则：丢弃 `MSB - FIO_IO_U_PLAT_BITS` 个低位（error bits），将余下的 `FIO_IO_U_PLAT_BITS` 位作为组内偏移
4. 确保索引不超过 `FIO_IO_U_PLAT_NR - 1`

**`plat_idx_to_val(idx)`：** 将 bucket 索引转换回对应的代表值（bucket 范围的均值）。

### 3.3 百分位计算

**`calc_percentiles(io_u_plat, nr_samples, plist)`：**

1. 对 `plist`（百分位列表）按升序排序
2. 遍历 bucket 数组累积样本计数
3. 当累计计数 ≥ `plist[j] / 100.0 × nr_samples` 时，记录该 bucket 的代表值为百分位值
4. 同时记录最小值和最大值

### 3.4 百分位列表

**对齐 fio 默认列表：**

```
1%, 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%,
99%, 99.5%, 99.9%, 99.95%, 99.99%
```

共 17 个百分位值。

### 3.5 纳秒到可读单位的自动转换

借鉴 fio 的 `nsec_to_usec` / `nsec_to_msec` 策略，文本输出中自动切换单位以确保可读性。切换阈值（对齐 fio）：

| 条件 | 切换方向 | 阈值（纳秒） |
|------|----------|-------------|
| `min > 2000` 且 `max > 99999` 且 `stddev > 1000.0` | ns → μs | `min > 2,000 ns` |
| `min > 2000000` 且 `max > 99999999` 且 `stddev > 1000000.0` | μs → ms | `min > 2,000,000 ns` |

- 三个条件全部满足才切换（保守策略：只在确实需要更大单位时才切换）
- 单位切换后所有展示值除以对应因子（1000 或 1000000），保留 3 位小数
- JSON 输出始终使用纳秒（不做单位转换）

---

## 4. 指标计算需求

### 4.1 Latency 指标

从分桶直方图和运行期累计统计中计算：

| 指标 | 来源 | 说明 |
|------|------|------|
| min | 运行期逐次更新 | 所有样本的最小值（ns） |
| max | 运行期逐次更新 | 所有样本的最大值（ns） |
| avg（mean）| 运行期 Welford 在线算法 | 所有样本的算术平均值（ns） |
| 百分位列表 | 分桶直方图后处理 | 17 个百分位值（ns/μs/ms 自适应） |

**Welford 在线算法**计算均值和方差（单次遍历，避免先存所有样本再求均值）。

使用 `long double` 中间精度以减少累积误差。公式如下（n 从 1 开始计数，mean₀ = 0，M₂₀ = 0）：

```
δ₁ = x - mean_{n-1}          // 当前值与旧均值的偏差
mean_n = mean_{n-1} + δ₁ / n // 更新均值
δ₂ = x - mean_n              // 当前值与新均值的偏差
M2_n = M2_{n-1} + δ₁ × δ₂    // 更新平方差累积
```

最终：
```
avg    = mean_N
var    = M2_N / N             // 总体方差
stddev = sqrt(var)            // 标准差
CV     = stddev / mean_N      // 变异系数
```

- 使用 `long double` 存储 `mean` 和 `M2` 中间值（降低大样本量下的浮点累积误差），最终结果转为 `float` 输出
- `min` 和 `max` 用 `uint64_t` 纳秒值直接比较更新

### 4.2 Throughput 指标

| 指标 | 计算 | 单位 |
|------|------|------|
| total_calls | 正式测量阶段总调用次数 | 次 |
| runtime | 正式测量阶段实际墙钟时间 | 秒 |
| throughput | `total_calls / runtime` | calls/sec |

### 4.3 稳定性指标

| 指标 | 计算 | 说明 |
|------|------|------|
| stddev | Welford 算法在线计算 | 标准差（ns） |
| CV | `stddev / mean` | 变异系数（无量纲），衡量相对波动程度 |

---

## 5. 输出规格

### 5.1 两路同时输出

| 通道 | 目标 | 格式 |
|------|------|------|
| stdout | 终端人类可读 | 文本表格 |
| 文件 | JSON 结构化 | JSON |

### 5.2 文本输出格式

**stdout 人类可读报告，参考 fio 风格：**

```
================================================================
  SwiGLU FFN 性能评测报告
  d_model = 2048, d_ff = 7168, seed = 42
  CPU: 0, warmup = 3s, runtime = 10s, min_calls = 50
================================================================

延迟统计 (Latency):
  min       = xxx.xxx ms
  max       = xxx.xxx ms
  avg       = xxx.xxx ms
  stddev    = xxx.xxx ms
  CV        = xxx

百分位分布 (Percentiles):
  1.00%  = xxx.xxx ms
  5.00%  = xxx.xxx ms
  ...
  99.99% = xxx.xxx ms

吞吐量 (Throughput):
  total_calls = xxxxxxx
  runtime     = xx.xxx s
  rate        = xxxxxx.xx calls/sec

JSON 结果已写入: ./ffn_bench_result.json
```

- 纳秒值自动转换为合适单位（ns/μs/ms），保留 3 位小数
- 百分位列对齐，按升序排列
- 不输出环境信息（CPU 型号 / 编译器版本等）

### 5.3 JSON 输出格式

```json
{
  "completed": true,
  "parameters": {
    "d_model": 2048,
    "d_ff": 7168,
    "seed": 42,
    "cpu": 0,
    "warmup_sec": 3,
    "runtime_sec": 10,
    "min_calls": 50
  },
  "latency_ns": {
    "min": 12345678,
    "max": 98765432,
    "avg": 45678901.5,
    "stddev": 1234567.8,
    "cv": 0.027,
    "percentiles": {
      "p1": 12345678,
      "p5": 23456789,
      "p10": 30000000,
      "p20": 35000000,
      "p30": 38000000,
      "p40": 40000000,
      "p50": 42000000,
      "p60": 44000000,
      "p70": 46000000,
      "p80": 49000000,
      "p90": 54000000,
      "p95": 60000000,
      "p99": 78000000,
      "p99_5": 82000000,
      "p99_9": 90000000,
      "p99_95": 94000000,
      "p99_99": 98000000
    }
  },
  "throughput": {
    "total_calls": 1234567,
    "runtime_sec": 10.002,
    "rate_calls_per_sec": 123425.5
  }
}
```

注意：
- JSON 中所有 latency 值统一使用**纳秒（ns）**单位（机器解析不要自动转换单位）
- 百分位 key 命名规则：`p` + 百分位数字，用 `_` 替代小数点
  - 整数百分位：`p1`（1%）、`p50`（50%）、`p99`（99%）
  - 带小数百分位：`p99_5`（99.5%）、`p99_9`（99.9%）、`p99_95`（99.95%）、`p99_99`（99.99%）
  - 判定方式：`p` 后所有 `_` 均视为小数点位置，如 `p99_99` = 99.99%

### 5.4 JSON 文件路径

- 默认路径：`./ffn_bench_result.json`
- 通过 `--json-output=<path>` 覆盖

---

## 6. 命令行接口

### 6.1 参数列表

```
test_bench [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--d_model=<int>` | int | `2048` | 模型维度 |
| `--d_ff=<int>` | int | `7168` | 前馈网络中间维度 |
| `--seed=<int>` | int | `42` | 随机种子 |
| `--cpu=<int>` | int | `0` | 绑定的 CPU 核心 |
| `--warmup=<int>` | int | `3` | 预热时长（秒） |
| `--runtime=<int>` | int | `10` | 正式运行时长（秒） |
| `--min-calls=<int>` | int | `50` | 保底最小调用次数 |
| `--json-output=<path>` | string | `./ffn_bench_result.json` | JSON 输出路径 |
| `--help` | flag | - | 打印帮助信息 |

### 6.2 参数解析

- POSIX 风格长选项（`--key=value`）
- 所有参数可选，均有合理默认值
- 不依赖外部参数解析库（如 getopt_long 不可用则自行实现简易解析）
- 无效参数打印错误信息并以非零退出
- `--help` 打印所有参数及默认值后以零退出

---

## 7. 非功能性需求

### 7.1 不做正确性校验

benchmark 程序不对 `swiglu_ffn` 的输出做任何数值正确性断言。正确性由 `test_small` 和 `test_large` 独立验证。

### 7.2 不做 SIGINT 优雅退出

不捕获 `SIGINT`（Ctrl+C），信号到达后进程直接终止，不输出 partial 结果。

### 7.3 不输出系统环境信息

不输出 CPU 型号、操作系统版本、编译器版本等环境信息。同一台评测机器上环境相同，不需要在报告中体现。

### 7.4 数值精度

- 计时使用 `clock_gettime(CLOCK_MONOTONIC, &ts)`，纳秒精度
- 百分位统计使用 `uint64_t` 存储纳秒值
- **Welford 在线算法的 `mean` 和 `M2` 中间变量使用 `long double`**（这是对 §4.1 全程 float 规则的特例，大样本量下 `long double` 减少累积误差），最终结果转换为 `float` 输出
- 分桶直方图计数器使用 `uint64_t`
- 其余所有计算使用 `float`（输入生成、swiglu_ffn 内部计算等）

### 7.5 C 语言标准

C11（`-std=c11`），通过 CMake 的 `CMAKE_C_STANDARD` 设置。

### 7.6 注释规范

- **注释语言：中文**
- 文件头部包含文件职责说明
- 关键计算步骤处注释对应数学公式
- 分桶算法处注释其数学原理
- 变量名和函数名使用英文（snake_case）

### 7.7 头文件保护

使用传统宏保护（`#ifndef` / `#define` / `#endif`），命名规范：`MODULE_NAME_H`。

### 7.8 代码风格

- 朴素实现，不进行性能优化
- 每个逻辑步骤独立成段，用空行和注释分隔
- 分桶直方图和百分位计算可在 `test_bench.c` 内实现，或抽象为独立模块

### 7.9 内存管理

- 权重矩阵和 input/output 向量均在堆上分配
- 所有 `malloc` 使用 `assert(ptr != NULL)` 检查
- 程序退出前释放所有分配的资源
- 分桶直方图数组（`uint64_t[1856]`）可放在静态区或堆上

---

## 8. 构建系统需求

### 8.1 CMake 新增目标

在现有 `CMakeLists.txt` 中新增：

| 目标 | 类型 | 源文件 | 链接 |
|------|------|--------|------|
| `test_bench` | 可执行文件 | `test/test_bench.c` | `swiglu_ffn` `generate_data` `test_utils` `m` `OpenSSL::Crypto` |

需要修改 `CMakeLists.txt`：
- 在 `add_executable` 块区域新增 `test_bench` 目标
- 使用 `target_link_libraries(test_bench PRIVATE ...)` 链接所需库

### 8.2 与现有测试程序的关系

| 程序 | 定位 | 状态 |
|------|------|------|
| `test_small` | 教学演示——小维度可视化 | 保持不变 |
| `test_large` | 教学演示——大维度正确性 | 保持不变 |
| `test_bench` | **新增**——性能评测 | 本需求文档定义 |

三者共存于 `test/` 目录下。

---

## 9. 测试需求与错误处理

### 9.1 基准行为验证

- 使用默认参数运行 `./test_bench`，程序正常退出，stdout 输出文本报告，生成 `ffn_bench_result.json`
- JSON 文件内容完整且格式正确（可通过 `jq` 或 Python `json.load()` 解析）
- JSON 中 `completed` 字段为 `true`
- 所有百分位值均为有效 `uint64_t` 纳秒值且非负且单调非降
- `total_calls > 0` 且 `throughput > 0`

### 9.2 错误处理

- 内存分配失败使用 `assert(ptr != NULL)` 终止（与项目现有惯例一致）
- JSON 文件无法写入时，程序仍完成 stdout 文本报告输出，但 stdout 中不显示"JSON 结果已写入"行，并以非零退出码退出

### 9.3 正确性验证

不做数值正确性断言。测试程序以不崩溃、JSON 格式正确为通过标准。

---

## 附录 A：决策记录

### A.1 链接时替换

**决策：** 学生通过替换 `src/swiglu_ffn.c` 接入评测。
**理由：** 与现有 CMake 结构兼容，函数签名统一，方便评测自动化。

### A.2 按时间控制预热与测量（对齐 fio）

**决策：** `--warmup --runtime` 按秒数控制 + `--min-calls` 保底。
**理由：** 与 fio 策略一致，自适应不同维度规模，保底机制防止极端情况样本过少。

### A.3 完整百分位列表

**决策：** 使用 fio 默认的 17 个百分位值。
**理由：** 让学生看到延迟全貌分布，信息量充足。

### A.4 时延指标 + 吞吐量 + 稳定性

**决策：** 报告 latency（百分比+avg/min/max）+ throughput（calls/sec）+ 稳定性（stddev/CV）。
**理由：** 以时延为核心，吞吐量和稳定性作为辅助评估维度。

### A.5 权重固定 + 每次随机 input（rand 不计入时延）

**决策：** 权重复用，每次计时前 rand() 覆写 input，计时仅包裹 swiglu_ffn。
**理由：** 对齐推理场景，rand() 噪声不计入时延，真正随机避免分支预测器作弊。

### A.6 分桶方案完全对齐 fio

**决策：** 复用 fio 的 6-bit 29 组对数分桶算法。
**理由：** 工业验证方案，<1% 误差，参数经过充分测试。

### A.7 两路输出（stdout 文本 + 文件 JSON）

**决策：** stdout 人类可读文本，文件输出结构化 JSON，默认路径可覆盖。
**理由：** 人眼直观 + 脚本可解析，分工明确。

### A.8 CPU 亲和性单核绑定

**决策：** `sched_setaffinity` 绑定到指定核心，默认 CPU 0，可配置。
**理由：** 消除调度噪声，评测公平，用户态实现。

### A.9 不做正确性校验

**决策：** benchmark 不对输出做正确性断言。
**理由：** 职责分离——正确性由 test_small/test_large 验证，benchmark 专注性能。

### A.10 不做 SIGINT 优雅退出

**决策：** 不捕获 SIGINT 信号，进程直接终止。
**理由：** 保持代码简洁。benchmark 运行时间由 `--runtime` 控制（默认 10s），用户预期明确，不需要 partial 结果场景。与 fio 的两次 SIGINT 策略不同——fio 面向交互式使用和长时间运行，ffn benchmark 面向批处理评测。

### A.11 不输出系统环境信息

**决策：** 报告中不包含环境信息。
**理由：** 评测在同一台机器上进行，环境信息对结果无区分度。

### A.12 单组维度

**决策：** 一次运行只评测一组维度。
**理由：** 简单，脚本循环调用更灵活。

### A.13 逐次计时

**决策：** 每次 swiglu_ffn 调用产出一个独立 latency 样本。
**理由：** 百毫秒级延迟下 clock_gettime 开销可忽略，保持分布完整性。

### A.14 Welford 在线算法

**决策：** 使用 Welford 单次遍历算法计算均值和方差。
**理由：** 避免存储全部样本，数值稳定。

### A.15 test_bench 命名

**决策：** 可执行文件和源码均命名为 `test_bench`。
**理由：** 语义明确，与 `test_small`/`test_large` 命名风格一致。

### A.16 Welford 算法使用 long double 中间精度

**决策：** Welford 在线算法的 `mean` 和 `M2` 累加器使用 `long double`，最终结果转为 `float`。
**理由：** 大样本量（百万次调用）下 `float` 累积误差可能显著（float 只有 ~7 位有效数字），`long double`（x86 上 80-bit，~19 位有效数字）足以保证统计精度。这是对 §4.1 float 规则的明确特例，其余计算保持 float。
