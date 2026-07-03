# SwiGLU MoE — 需求规格文档

---

## 1. 项目概述

### 1.1 背景

当前项目 `swiglu_ffn` 已实现 LLaMA 风格的单 SwiGLU FFN：

```
output = silu(input @ W_gate) ⊙ (input @ W_up) @ W_down
```

维度变化：`d_model → d_ff → d_ff → d_model`，无 bias、无 dropout、无归一化。

本需求文档描述在现有单 FFN 基础上扩展 **Sparse Mixture of Experts（稀疏 MoE）** 层。

### 1.2 MoE 定位

MoE 层是"上层编排者"——它通过一个路由器（Router / Gate）决定将输入 token 分配给哪些专家处理，并对专家输出做加权合成。每个专家就是一个完整的 `swiglu_ffn` 实例。

**MoE 层不修改现有 `swiglu_ffn` 的任何代码。** 所有现有模块、测试程序保持完整不变。

### 1.3 新增模块名称

| 模块 | 文件 |
|------|------|
| `softmax` | `include/softmax.h` + `src/softmax.c` |
| `topk` | `include/topk.h` + `src/topk.c` |
| `swiglu_moe` | `include/swiglu_moe.h` + `src/swiglu_moe.c` |

修改模块：`generate_data`（新增路由器权重和专家权重生成函数）。

---

## 2. Sparse MoE 数学定义

### 2.1 整体公式

采用 **Top-k Sparse MoE with Linear Router**（对标 Mixtral 8×7B）：

```
logits      = input @ W_router                                    — (d_model,) → (num_experts,)
indices, top_logits = topk(logits, top_k)                        — 选出值最大的 top_k 个
weights     = softmax(top_logits)                                 — 对选中的 top_k 个 logits 归一化

for i in 0 .. top_k - 1:
    expert_out[i] = swiglu_ffn(input, W_gate[indices[i]],
                               W_up[indices[i]], W_down[indices[i]],
                               d_model, d_ff)

output = Σ_{i=0}^{top_k-1} weights[i] × expert_out[i]            — (d_model,)
```

**维度变化：** `d_model → num_experts（路由）→ d_model × top_k（专家计算）→ d_model（加权求和）`

**关键性质：**
- 无共享专家（Shared Expert）
- 无辅助损失（Load Balancing Loss）—— 教学基线不涉及训练
- 无 capacity factor、无 expert drop —— 所有 token 都激活恰好 top_k 个专家
- 路由器无 bias

### 2.2 与单 FFN 的关系

单 FFN 是 MoE 在 `num_experts = 1, top_k = 1` 时的退化情况。但 MoE 层不通过参数组合模拟单 FFN——它作为独立的上层模块存在。

---

## 3. 新增子模块需求

### 3.1 Softmax 模块

**文件：** `include/softmax.h` + `src/softmax.c`

**接口：**

```c
void softmax(float* x, int n);
```

**计算：** 对数组 `x` 的 n 个元素执行 softmax，结果原地写回。

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=0}^{n-1} e^{x_j}}$$

**实现约束：**
- 使用 `<math.h>` 中的 `expf()` 计算指数
- 朴素实现：先计算所有 `expf(x[i])` 存入临时数组，累加求和，再逐元素除以和
- 临时数组在堆上 `malloc`，用完立即 `free`
- 不做数值稳定性优化（如减去 max 偏移）——有意保留可优化空间
- 该模块不依赖项目中其他任何模块

### 3.2 Top-k 选择模块

**文件：** `include/topk.h` + `src/topk.c`

**接口：**

```c
void topk_select(const float* values, int n, int k, int* indices, float* out_values);
```

**计算：** 从长度为 n 的数组 `values` 中选出值最大的 k 个元素。

- `indices`：输出参数，存放选中的 k 个索引，由调用者分配
- `out_values`：输出参数，存放选中的 k 个值，由调用者分配
- 当 `k >= n` 时，返回全部 n 个元素（按值降序排列）

**实现约束：**
- 使用朴素的全排序方法：构造 n 个 `(value, index)` 对，按 value 降序排列，取前 k 个
- 排序使用插入排序或冒泡排序（O(n²)），不使用 `qsort()` —— 有意保留性能优化空间
- 函数内部在堆上分配排序所需的临时数组，函数返回前释放
- 该模块不依赖项目中其他任何模块

### 3.3 SwiGLU MoE 模块

**文件：** `include/swiglu_moe.h` + `src/swiglu_moe.c`

**接口：**

```c
void swiglu_moe(const float* input, const float* W_router,
                const float* W_gate[], const float* W_up[], const float* W_down[],
                int num_experts, int top_k, int d_model, int d_ff, float* output);
```

**参数说明：**

| 参数 | 形状 / 类型 | 说明 |
|------|------------|------|
| `input` | `(d_model,)` | 输入向量，只读 |
| `W_router` | `(d_model, num_experts)` | 路由器权重矩阵，行主序，只读 |
| `W_gate[]` | 长度为 `num_experts` 的指针数组，每个指向 `(d_model, d_ff)` | 所有专家的门控投影权重，只读 |
| `W_up[]` | 长度为 `num_experts` 的指针数组，每个指向 `(d_model, d_ff)` | 所有专家的上投影权重，只读 |
| `W_down[]` | 长度为 `num_experts` 的指针数组，每个指向 `(d_ff, d_model)` | 所有专家的下投影权重，只读 |
| `num_experts` | `int` | 专家总数，运行时指定 |
| `top_k` | `int` | 每个 token 激活的专家数，运行时指定（必须 `1 ≤ top_k ≤ num_experts`） |
| `d_model` | `int` | 模型维度 |
| `d_ff` | `int` | 专家前馈网络中间维度 |
| `output` | `(d_model,)` | 输出向量，只写，由调用者分配 |

**详细计算步骤：**

```
第 1 步：路由 logits 计算
    logits = input @ W_router
    维度：(d_model,) → (num_experts,)

第 2 步：Top-k 选择
    选出 logits 中值最大的 top_k 个索引，存入 indices[]
    取出对应的 logits 值，存入 top_logits[]

第 3 步：Softmax 归一化
    weights = softmax(top_logits)
    对选中的 top_k 个 logits 做归一化，得到融合权重

第 4 步：专家计算（对每个选中的专家）
    对于 i = 0 .. top_k - 1:
        expert_out[i] = swiglu_ffn(input,
                                   W_gate[indices[i]],
                                   W_up[indices[i]],
                                   W_down[indices[i]],
                                   d_model, d_ff)
    每个专家输出形状 (d_model,)

第 5 步：加权求和
    output = Σ_{i=0}^{top_k-1} weights[i] × expert_out[i]
    逐一元素加权累加
```

**内部分配与释放顺序：**

```
malloc: logits        (num_experts 个 float)
        indices       (top_k 个 int)
        top_logits    (top_k 个 float)
        expert_out[i] (top_k 个 float* 各分配 d_model 个 float)

计算流程：

1. malloc logits → 计算 logits → free logits（logits 不再需要时立即释放）
2. indices 和 top_logits 已在调用者分配，填充即可
3. 原地 softmax(top_logits)
4. for i in 0..top_k-1:
     expert_out[i] = malloc(d_model * sizeof(float))
     swiglu_ffn(..., expert_out[i])   — 内部自行管理其临时向量
5. output 清零，for i in 0..top_k-1:
     for j in 0..d_model-1:
         output[j] += weights[i] * expert_out[i][j]
6. for i in 0..top_k-1:
     free expert_out[i]
   （所有专家输出在加权求和完成后统一释放）
```

**函数不分配任何由调用者负责释放的内存。** Input 和 output 指针允许指向同一地址（别名安全）。

输出加权累加前需先初始化为零：`memset(output, 0, d_model * sizeof(float))` 或逐元素赋零，因为加权求和是累加模式。

---

## 4. 现有模块修改

### 4.1 generate_data 模块扩展

**文件修改：** `include/generate_data.h` + `src/generate_data.c`

新增以下函数：

```c
// 生成路由器权重矩阵，形状 (d_model, num_experts)，行主序
float* generate_W_router(int d_model, int num_experts, int seed);

// 按 expert_id 生成指定专家的门控投影权重
float* generate_W_gate_expert(int d_model, int d_ff, int seed, int expert_id);

// 按 expert_id 生成指定专家的上投影权重
float* generate_W_up_expert(int d_model, int d_ff, int seed, int expert_id);

// 按 expert_id 生成指定专家的下投影权重
float* generate_W_down_expert(int d_model, int d_ff, int seed, int expert_id);
```

**种子派生规则：**
- `generate_W_router` 使用 tag `"W_router"`
- `generate_W_gate_expert` 使用 tag `"W_gate_0"`, `"W_gate_1"`, ...（由 `sprintf(buf, "W_gate_%d", expert_id)` 构造）
- `generate_W_up_expert` 使用 tag 同理：`"W_up_%d"`
- `generate_W_down_expert` 使用 tag 同理：`"W_down_%d"`

**生成规则与现有函数一致：**
- 所有值服从 `[-1, 1)` 均匀分布
- 使用 `(float)rand() / RAND_MAX * 2.0f - 1.0f` 生成
- 通过 `derive_seed(base_seed, tag)` 使用 MD5 派生种子
- 每个函数内部 `malloc` 返回数据，调用者负责 `free`

**现有生成函数不修改、不删除。** `generate_input`、`generate_W_gate`、`generate_W_up`、`generate_W_down` 保持完整可用，供现有单 FFN 测试继续使用。

---

## 5. 内存管理需求

### 5.1 分配策略

| 数据类别 | 分配方 | 释放方 | 位置 |
|----------|--------|--------|------|
| `input` / `output` | 调用者 | 调用者 | 调用者决定 |
| `W_router` | 调用者 | 调用者 | 调用者决定 |
| 专家权重 `W_gate[]` `W_up[]` `W_down[]` | 调用者 | 调用者 | 调用者决定 |
| `swiglu_moe` 内部临时向量（logits 等） | `swiglu_moe` 内 `malloc` | `swiglu_moe` 内 `free` | 堆 |
| `swiglu_moe` 内各专家输出 `expert_out[i]` | `swiglu_moe` 内 `malloc` | `swiglu_moe` 内 `free`（加权求和后统一释放） | 堆 |
| `swiglu_ffn` 内部临时向量 | `swiglu_ffn` 内 `malloc` | `swiglu_ffn` 内 `free` | 堆 |
| `topk_select` 内部临时数组 | `topk_select` 内 `malloc` | `topk_select` 内 `free` | 堆 |
| `softmax` 内部临时数组 | `softmax` 内 `malloc` | `softmax` 内 `free` | 堆 |
| 生成函数返回的数据 | 生成函数内 `malloc` | 调用者 `free` | 堆 |

### 5.2 分配失败处理

所有 `malloc` 调用后使用 `assert(ptr != NULL)` 检查。不实现 `goto cleanup` 等错误恢复路径。与现有代码风格一致。

### 5.3 禁止栈上分配

禁止栈上分配矩阵级别的空间。临时向量一律堆上分配。

---

## 6. 构建系统更新

### 6.1 新增 CMake 目标

| 目标 | 类型 | 源文件 | 链接依赖 |
|------|------|--------|----------|
| `softmax` | 静态库 | `src/softmax.c` | 无（仅 `<math.h>`） |
| `topk` | 静态库 | `src/topk.c` | 无 |
| `swiglu_moe` | 静态库 | `src/swiglu_moe.c` | `swiglu_ffn` `softmax` `topk` `m` |
| `test_moe_small` | 可执行文件 | `test/test_moe_small.c` | `swiglu_moe` `generate_data` `test_utils` `OpenSSL::Crypto` `m` |
| `test_moe_large` | 可执行文件 | `test/test_moe_large.c` | 同上 |

**现有目标不变：** `swiglu_ffn`、`generate_data`、`test_utils`、`test_small`、`test_large` 保持完整。

### 6.2 `swiglu_ffn` 库不修改

`swiglu_ffn` 静态库的源文件列表和依赖关系完全不变。MoE 层作为独立的上层库存在。

### 6.3 外部依赖

外部依赖不变：OpenSSL（libcrypto）、C 标准库、`<math.h>`。

### 6.4 更新后目录结构

```
swiglu_ffn/
├── CMakeLists.txt                    ← 修改：新增 5 个构建目标
├── include/
│   ├── silu.h                        ← 不变
│   ├── matvec.h                      ← 不变
│   ├── swiglu_ffn.h                  ← 不变
│   ├── softmax.h                     ← 新增
│   ├── topk.h                        ← 新增
│   ├── swiglu_moe.h                  ← 新增
│   ├── generate_data.h               ← 修改：新增 4 个函数声明
│   └── test_utils.h                  ← 不变
├── src/
│   ├── silu.c                        ← 不变
│   ├── matvec.c                      ← 不变
│   ├── swiglu_ffn.c                  ← 不变
│   ├── softmax.c                     ← 新增
│   ├── topk.c                        ← 新增
│   ├── swiglu_moe.c                  ← 新增
│   ├── generate_data.c               ← 修改：新增 4 个函数实现
│   └── test_utils.c                  ← 不变
└── test/
    ├── test_small.c                  ← 不变
    ├── test_large.c                  ← 不变
    ├── test_moe_small.c              ← 新增
    └── test_moe_large.c              ← 新增
```

---

## 7. 测试需求

### 7.1 test_moe_small（维度 4→8，4 专家 × top-2）

- `d_model = 4`，`d_ff = 8`，`num_experts = 4`，`top_k = 2`，`seed = 42`
- 使用生成函数创建 `input`、`W_router` 和 4 个专家的 `W_gate`、`W_up`、`W_down`
- 调用 `swiglu_moe` 计算输出
- **打印内容：** 输入向量全部 4 个值 + 输出向量全部 4 个值（不打印路由 logits、不打印选中专家索引/权重、不打印中间专家输出）
- 程序退出前释放所有分配的资源

### 7.2 test_moe_large（维度 2048→7168，8 专家 × top-2）

- `d_model = 2048`，`d_ff = 7168`，`num_experts = 8`，`top_k = 2`，`seed = 42`
- 使用生成函数创建 `input`、`W_router` 和 8 个专家的 `W_gate`、`W_up`、`W_down`
- 调用 `swiglu_moe` 计算输出
- **打印内容：**
  - 输出向量前 5 个值和后 5 个值（用 `...` 省略中间）
  - 输出向量的统计摘要：min、max、mean
  - 不打印输入向量、不打印路由信息、不打印中间专家输出
- 程序退出前释放所有分配的资源

### 7.3 正确性验证

与现有测试一致：不做数值正确性断言。测试程序以不崩溃、无 NaN、正常退出为通过标准。

---

## 8. 非功能需求

### 8.1 数值精度

- 全程使用 `float`（32 位单精度）
- 数学函数使用 `expf()` 等 float 版
- 与现有模块保持一致

### 8.2 C 语言标准

C11（`-std=c11`），通过 CMake 的 `CMAKE_C_STANDARD` 设置。与现有项目一致。

### 8.3 注释规范

- **注释语言：中文**
- 每个 `.c` 和 `.h` 文件头部包含文件职责说明
- 每个公开函数前注释说明功能、参数含义、输入输出形状
- 关键计算步骤处注释对应数学公式
- 矩阵行主序索引映射在涉及位置用注释图示说明
- 变量名和函数名使用英文（snake_case）
- 代码中**不标注优化机会**（与决策 A.18 一致）

### 8.4 头文件保护

使用传统宏保护（`#ifndef` / `#define` / `#endif`），命名规范：`MODULE_NAME_H`

### 8.5 代码风格

- 朴素实现，**不进行任何性能优化**（循环融合、SIMD、缓存分块、内存池等）
- 每个数学步骤独立成段，用空行和注释分隔
- 有意保留可优化空间供后续比赛使用
- 代码中**不标注优化机会**
- 所有新代码与现有代码风格保持一致

### 8.6 与现有代码的关系

- **不修改** `swiglu_ffn.c` / `swiglu_ffn.h`
- **不修改** `silu.c` / `silu.h`
- **不修改** `matvec.c` / `matvec.h`
- **不修改** `test_utils.c` / `test_utils.h`
- **不修改** `test_small.c` / `test_large.c`
- **仅修改** `generate_data.c` / `generate_data.h`（新增函数，不删除、不修改现有函数）
- **仅修改** `CMakeLists.txt`（新增目标，不删除、不修改现有目标）

---

## 附录 A：决策记录

### A.1 MoE 路由方式

**决策：** Sparse MoE（Top-k 稀疏路由）。
**理由：** 工业界事实标准（Mixtral、DeepSeek-V3），稀疏计算是最有价值的优化比赛场景。

### A.2 专家数量与 Top-k 参数化

**决策：** `num_experts` 和 `top_k` 作为运行时函数参数。
**理由：** 与现有 `d_model` / `d_ff` 设计哲学一致（决策 A.2）；开发阶段使用较小值便于调试，正式环境可切换为 256 专家 × top-8。

### A.3 线性路由器

**决策：** 单个权重矩阵 `W_router` 形状 `(d_model, num_experts)`，无 bias。
**理由：** 最简洁的路由器结构，与现有 `matvec_mul` 基础设施复用度高，教学上清晰展示"路由 = 线性变换 + softmax + top-k"。

### A.4 先 Top-k 再 Softmax

**决策：** 先选出 top-k 个 logits，再对这 k 个值做 softmax 归一化。
**理由：** Mixtral / DeepSeek-V3 的实际做法，权重和为 1，被选中专家的融合权重语义清晰。

### A.5 专家 = 独立 SwiGLU FFN

**决策：** 每个专家是完整的 `swiglu_ffn` 实例，各自拥有独立的 `W_gate`、`W_up`、`W_down`。
**理由：** 与 Mixtral / DeepSeek 一致，复用现有 `swiglu_ffn` 代码不加修改。

### A.6 加权求和输出

**决策：** `final = Σ wᵢ × output[eᵢ]`，用 softmax 归一化后的权重做加权求和。
**理由：** Softmax 权重代表路由器对专家的"信任度"，加权求和是所有 Sparse MoE 的标准做法。

### A.7 权重指针数组

**决策：** 专家权重通过 `float* W_gate[num_experts]` 指针数组传入，每个专家独立分配。
**理由：** 最直观地反映"每个专家是独立个体"，与现有 `swiglu_ffn` 接口完全兼容，直接复用。

### A.8 新增独立 `swiglu_moe` 函数

**决策：** 不修改 `swiglu_ffn`，新增 `swiglu_moe` 作为 MoE 层函数。
**理由：** 保持 `swiglu_ffn` 不变，两个 test 程序不受影响。MoE 作为上层模块，清晰展示"从单 FFN 到 MoE"的架构层次。

### A.9 用完即释 + 专家输出统一释放

**决策：** `swiglu_moe` 内部临时向量（logits 等）用完立即 `free`；top_k 个专家输出在加权求和后统一释放。
**理由：** 单次使用的临时向量立即释放与现有风格一致；k 个专家输出需要在加权求和阶段同时持有（"全部分配→全部计算→加权求和→统一释放"），这比交替式的"分配→计算→累加→释放→下一个"更清晰地展示 MoE 的"并行专家计算"语义。

### A.10 新增 Softmax 独立模块

**决策：** `include/softmax.h` + `src/softmax.c`，作为独立静态库。
**理由：** Softmax 与 SiLU 一样是基础数学工具，独立模块化有利于教学展示"softmax → SiLU → MoE"的层次。

### A.11 新增 Top-k 独立模块

**决策：** `include/topk.h` + `src/topk.c`，作为独立静态库，使用全排序后取前 k。
**理由：** Top-k 是稀疏路由的核心原语，独立模块便于教学和后续优化。全排序算法最直观——"排序取前 k"比"k 次扫描"更易理解。

### A.12 新增 MoE 测试程序

**决策：** 新增 `test_moe_small` / `test_moe_large`，保持现有 `test_small` / `test_large` 不变。
**理由：** 现有测试是"单 FFN 教学基线"的 demo，应保持完整可用；MoE 测试独立验证新功能。

### A.13 MoE 测试仅打印输入输出

**决策：** MoE 测试仅打印输入向量和输出向量（与现有测试输出风格一致），不打印路由 logits、选中专家索引/权重、中间专家输出。
**理由：** 学生通过代码注释学习 MoE 内部机制；输出保持简洁便于肉眼检查结果正确性（与决策 A.16 / A.17 一致）。

### A.14 MoE 测试维度配置

**决策：** `test_moe_small`：d_model=4, d_ff=8, num_experts=4, top_k=2；`test_moe_large`：d_model=2048, d_ff=7168, num_experts=8, top_k=2。
**理由：** Small 场景 4 选 2 稀疏比例 50%，demo 效果直观。Large 场景与开发阶段 8 专家配置一致。

### A.15 新模块各独立静态库

**决策：** `softmax`、`topk`、`swiglu_moe` 各为独立 CMake 静态库目标。
**理由：** 每个模块职责单一，CMake 依赖关系清晰反映模块层次——`swiglu_ffn` 是基础件，`softmax` / `topk` 是路由原语，`swiglu_moe` 是上层编排者。

### A.16 不做正确性断言

**决策：** MoE 测试程序不做数值比较，以不崩溃/无 NaN 为通过标准。
**理由：** 与现有测试哲学一致（决策 A.22）；无预计算正确答案；教学基线定位下视觉检查输出合理性即可。

### A.17 不修改现有模块

**决策：** 除 `generate_data`（新增函数）和 `CMakeLists.txt`（新增目标）外，不修改任何现有源文件。
**理由：** 保持"单 FFN 教学基线"完整可用；MoE 作为干净的功能扩展，而非对现有代码的重构。

### A.18 Softmax 不做数值稳定优化

**决策：** Softmax 朴素实现，不减去 max 做数值稳定。
**理由：** 与 SiLU、MatVec 的朴素实现风格一致，有意保留可优化空间。对于 `[-1, 1)` 均匀分布的 logits，数值范围安全。

### A.19 Top-k 使用全排序

**决策：** 构造 n 个 (value, index) 对，按 value 降序排列，取前 k 个。使用插入排序或冒泡排序。
**理由：** "排序取前 k"比"k 次扫描"在教学上更直观。O(n²) 对于 8 或 256 个专家都可接受，且排序算法本身也是优化目标。

### A.20 专家输出独立缓冲区

**决策：** 为每个选中的专家分配独立的 `expert_out` 缓冲区（大小 `d_model`），全部计算完成后统一加权求和，再统一释放。
**理由：** 展示 MoE "并行专家计算"的语义——所有专家的输出在概念上是同时产生的。相比交替式的"分配→计算→累加→释放→下一个"，这个模式更清晰地体现了 MoE 的架构：路由决策 → 专家并行计算 → 结果合成。
