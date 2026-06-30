# SwiGLU FFN — 需求规格文档

---

## 1. 项目概述

### 1.1 项目定位

使用 C 语言实现一个 **CPU 运行的 SwiGLU FFN（Feed-Forward Network）函数**，作为 **优化比赛的基线代码**。

代码的首要目标是 **"教会行外人理解 SwiGLU FFN 的计算过程"**，因此：
- 逻辑清晰优先于运行效率
- 注释规范完整
- 每步计算都显式保留可读的中间结果，**有意留出优化空间**（不进行循环融合、不消除中间分配、不引入 SIMD 等）

### 1.2 项目名称

`swiglu_ffn`

---

## 2. 核心功能需求

### 2.1 SwiGLU FFN 数学定义

采用 **LLaMA 风格的无 bias SwiGLU**：

```
output = silu(input @ W_gate) ⊙ (input @ W_up) @ W_down
```

**维度变化：** `d_model → d_ff → d_ff → d_model`

**计算步骤：**
1. `gate_proj = input @ W_gate` — 升维投影，`(d_model,) → (d_ff,)`
2. `up_proj = input @ W_up` — 升维投影，`(d_model,) → (d_ff,)`
3. `gate_activated[i] = silu(gate_proj[i])` — 逐元素门控激活
4. `gated[i] = gate_activated[i] * up_proj[i]` — 逐元素门控乘法
5. `output = gated @ W_down` — 降维投影，`(d_ff,) → (d_model,)`

无 bias 项。无 dropout。无归一化。

### 2.2 输入输出接口

**`swiglu_ffn` 函数签名：**

```c
void swiglu_ffn(const float* input, const float* W_gate, const float* W_up,
                const float* W_down, int d_model, int d_ff, float* output);
```

- `input`：形状 `(d_model,)`，只读，由调用者分配并传入
- `W_gate`、`W_up`：形状 `(d_model, d_ff)`，只读，行主序存储，由调用者分配并传入
- `W_down`：形状 `(d_ff, d_model)`，只读，行主序存储，由调用者分配并传入
- `d_model`、`d_ff`：维度参数，运行时指定
- `output`：形状 `(d_model,)`，只写，由调用者分配并传入

函数不分配任何由调用者负责释放的内存。Input 和 output 指针允许指向同一地址（别名安全）。

### 2.3 维度参数化

维度完全参数化，通过函数参数传入，不使用宏或全局常量。支持从极小维度（如 2×3）到大维度（如 2048×7168）。

---

## 3. 子模块需求

### 3.1 SiLU 激活函数

**文件：** `include/silu.h` + `src/silu.c`

**接口：**

```c
float silu(float z);
```

**数学定义：**

$$\text{silu}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**实现约束：**
- 使用 `<math.h>` 中的 `expf()` 计算指数
- 对调用者传入的每个标量值独立计算（不提供向量化版本）
- 该模块**不依赖项目中其他任何模块**

### 3.2 矩阵-向量乘法

**文件：** `include/matvec.h` + `src/matvec.c`

**接口：**

```c
void matvec_mul(const float* x, const float* W, int in_dim, int out_dim, float* y);
```

**计算：** `y = x @ W`，其中 `x` 形状 `(in_dim,)`，`W` 形状 `(in_dim, out_dim)`（行主序），`y` 形状 `(out_dim,)`。

- `y` 由调用者分配并传入
- 函数内部不分配任何内存
- 使用朴素的双层循环实现（不进行任何访存优化）

### 3.3 测试数据生成

**文件：** `include/generate_data.h` + `src/generate_data.c`

**四个独立函数，各返回 `float*`（内部分配，调用者负责 `free`）：**

```c
float* generate_input(int d_model, int seed);
float* generate_W_gate(int d_model, int d_ff, int seed);
float* generate_W_up(int d_model, int d_ff, int seed);
float* generate_W_down(int d_model, int d_ff, int seed);
```

**生成规则：**
- 所有值服从 `[-1, 1)` 均匀分布
- 使用 `(float)rand() / RAND_MAX * 2.0f - 1.0f` 生成
- 每个函数内部调用 `srand(derived_seed)` 然后逐元素 `rand()`
- `derived_seed` 由 `base_seed` 与函数标签（如 `"input"`, `"W_gate"` 等）拼接后经 MD5 哈希派生，确保四个函数生成不同但可复现的数据

**种子派生函数（模块内部 static，不对外暴露）：**

```c
static int derive_seed(int base_seed, const char* tag);
```

- 使用 OpenSSL（libcrypto）的 `MD5()` 函数
- 输入为 `sprintf(buf, "%d_%s", base_seed, tag)` 构造的字符串
- 取 MD5 digest 前 4 字节转为 `int`，作为 `srand` 的种子

### 3.4 测试工具

**文件：** `include/test_utils.h` + `src/test_utils.c`

**功能约束：**
- 打印向量：格式化输出向量内容，支持指定打印前 N 个和后 N 个元素
- 统计摘要：计算向量的 min / max / mean

具体接口由实现者根据上述功能约束自行设计。

---

## 4. 内存管理需求

### 4.1 分配策略

| 数据类别 | 分配方 | 释放方 | 位置 |
|----------|--------|--------|------|
| `input` / `output` | 调用者 | 调用者 | 调用者决定 |
| 权重矩阵 `W_gate` `W_up` `W_down` | 调用者 | 调用者 | 调用者决定 |
| `matvec_mul` 输出 `y` | 调用者 | 调用者 | 调用者决定 |
| 生成函数返回的数据 | 生成函数内 `malloc` | 调用者 `free` | 堆 |
| `swiglu_ffn` 内部临时向量 | `swiglu_ffn` 内 `malloc` | `swiglu_ffn` 内 `free` | 堆 |

### 4.2 swiglu_ffn 内部分配

内部需 `malloc` 的临时向量（全部在堆上分配）：

| 向量 | 大小 | 生命周期 |
|------|------|----------|
| `gate_proj` | `d_ff` | 用完立即 `free` |
| `up_proj` | `d_ff` | 用完立即 `free` |
| `gate_activated` | `d_ff` | 用完立即 `free` |
| `gated` | `d_ff` | 用完立即 `free` |

每个临时向量在不再需要时立即释放，不在函数末尾统一释放。

**禁止栈上分配矩阵级别的空间**（输入/输出/权重等由调用者管理的不受此限）。

### 4.3 分配失败处理

所有 `malloc` 调用后使用 `assert(ptr != NULL)` 检查。不实现 `goto cleanup` 等错误恢复路径。

---

## 5. 非功能需求

### 5.1 数值精度

- 全程使用 `float`（32 位单精度）
- 数学函数使用 `expf()` 等 float 版
- 不考虑 double 精度

### 5.2 C 语言标准

C11（`-std=c11`），通过 CMake 的 `CMAKE_C_STANDARD` 设置。

### 5.3 注释规范

- **注释语言：中文**
- 每个 `.c` 和 `.h` 文件头部包含文件职责说明
- 每个公开函数前注释说明功能、参数含义、输入输出形状
- 关键计算步骤处注释对应数学公式
- 矩阵行主序索引映射在涉及位置用注释图示说明
- 变量名和函数名使用英文（snake_case）

### 5.4 头文件保护

使用传统宏保护（`#ifndef` / `#define` / `#endif`），命名规范：`MODULE_NAME_H`

### 5.5 代码风格

- 朴素实现，**不进行任何性能优化**（循环融合、SIMD、缓存分块、内存池等）
- 每个数学步骤独立成段，用空行和注释分隔
- 有意保留可优化空间供后续比赛使用
- 代码中**不标注优化机会**

---

## 6. 构建系统需求

### 6.1 CMake 结构

使用 CMake 构建，最小 CMake 版本为 3.10。

**构建目标：**

| 目标 | 类型 | 源文件 |
|------|------|--------|
| `swiglu_ffn` | 静态库 | `src/silu.c` `src/matvec.c` `src/swiglu_ffn.c` |
| `generate_data` | 静态库 | `src/generate_data.c` |
| `test_utils` | 静态库 | `src/test_utils.c` |
| `test_small` | 可执行文件 | `test/test_small.c`（链接上述三个库 + OpenSSL::Crypto） |
| `test_large` | 可执行文件 | `test/test_large.c`（链接上述三个库 + OpenSSL::Crypto） |

### 6.2 外部依赖

- **OpenSSL（libcrypto）**：通过 `find_package(OpenSSL REQUIRED)` 引入，用于 MD5 哈希
- 其余依赖仅 C 标准库和 `<math.h>`

### 6.3 目录结构

```
swiglu_ffn/
├── CMakeLists.txt
├── include/
│   ├── silu.h
│   ├── matvec.h
│   ├── swiglu_ffn.h
│   ├── generate_data.h
│   └── test_utils.h
├── src/
│   ├── silu.c
│   ├── matvec.c
│   ├── swiglu_ffn.c
│   ├── generate_data.c
│   └── test_utils.c
└── test/
    ├── test_small.c
    └── test_large.c
```

---

## 7. 测试需求

### 7.1 test_small（维度 4→8）

- `d_model = 4`，`d_ff = 8`
- 使用生成函数创建 input、W_gate、W_up、W_down（`seed = 42`）
- 调用 `swiglu_ffn` 计算输出
- **打印内容：** 输入向量全部 4 个值 + 输出向量全部 4 个值（不打印中间向量）
- 程序退出前释放所有分配的资源

### 7.2 test_large（维度 2048→7168）

- `d_model = 2048`，`d_ff = 7168`
- 使用生成函数创建 input、W_gate、W_up、W_down（`seed = 42`）
- 调用 `swiglu_ffn` 计算输出
- **打印内容：**
  - 输出向量前 5 个值和后 5 个值（用 `...` 省略中间）
  - 输出向量的统计摘要：min、max、mean
  - 不打印输入向量、不打印中间向量
- 程序退出前释放所有分配的资源

### 7.3 正确性验证

不做数值正确性断言。测试程序以不崩溃、无 NaN、正常退出为通过标准。

---

## 附录 A：决策记录

### A.1 数学公式选择

**决策：** LLaMA 风格无 bias SwiGLU。
**理由：** 工业界最广泛使用，三步结构清晰对应教学叙事。

### A.2 维度参数化

**决策：** 参数化维度 + test_small/test_large 两个固定规模的 demo。
**理由：** 参数化体现"维度可变的设计选择"；small 方便理解，large 验证正确性。

### A.3 单向量接口

**决策：** `(d_model,) → (d_model,)` 单个向量接口，不处理 batch/sequence 维度。
**理由：** 教学关注核心计算，batch 维度可通过循环自然扩展。

### A.4 权重值来源

**决策：** 随机生成后固定 seed 确保可复现。
**理由：** 无需手算验证（教学不要求纸笔对答案），可复现性由固定 seed 保证。

### A.5 SiLU 独立成函数

**决策：** `float silu(float z)` 标量函数，独立模块。
**理由：** SiLU 是独立激活函数概念，分离有利于模块化教学；标量形式强调"逐元素"概念。

### A.6 内存管理策略

**决策：** Input/output 由调用者传入；内部临时向量堆分配；用完立即释放。
**理由：** 与工程实践一致；"用完即释"展示局部资源管理。

### A.7 权重矩阵传递方式

**决策：** `const float*` 指针由调用者传入。
**理由：** 与 input/output 设计哲学统一——计算函数不拥有数据。

### A.8 行主序存储

**决策：** 行主序 + 注释图示。
**理由：** 遵循 C 语言原生内存模型，注释弥补访问模式的可视化。

### A.9 错误处理为 assert

**决策：** 使用 `assert()` 而非错误码返回。
**理由：** 教学基线代码——简化控制流，避免错误处理分支淹没主线计算。

### A.10 CMake 多目标结构

**决策：** 源码编译为静态库，test 编译为可执行文件链接库。
**理由：** 展示模块化构建；test 和库的分离方便后续比赛场景中替换优化实现。

### A.11 float 精度

**决策：** 全程 float + `expf()`。
**理由：** 实际 LLM 推理主流精度，避免隐式转换。

### A.12 C11 标准

**决策：** CMAKE_C_STANDARD 11。
**理由：** 主流编译器完整支持，提供 `//` 注释和混合声明。

### A.13 中文注释

**决策：** 注释用中文，标识符用英文 snake_case。
**理由：** 降低行外人理解门槛，同时保持代码国际化。

### A.14 种子派生 — OpenSSL MD5

**决策：** `"base_seed_tag"` 字符串经 MD5 哈希派生独立种子。
**理由：** 种子派生可靠且可复现；OpenSSL 同时满足项目中其他模块对哈希的需求。

### A.15 生成函数返回 float*

**决策：** 生成函数内部分配内存并返回指针，调用者负责 free。
**理由：** 生成语义 = "创造并交付"，分配与填充为原子操作。

### A.16 test_small 仅打印输入输出

**决策：** 4→8 测试只打印输入和输出的 4 个值。
**理由：** 保持输出简洁，聚焦"黑盒演示"；想深入理解的读者可通过 seed 重现中间值。

### A.17 test_large 打印头尾 + 统计

**决策：** 2048→7168 测试打印头尾各 5 个值 + min/max/mean。
**理由：** 全量打印刷屏不可读；统计摘要提供直观的正确性感受。

### A.18 代码不标注优化点

**决策：** 代码中不对可优化位置做任何标注。
**理由：** 优化空间留给比赛参赛者自行发现，评分更公平。

### A.19 SiLU 结果单独存储（gate_activated）

**决策：** 新增一个 `gate_activated` 数组存储逐元素 silu 结果，再与 `up_proj` 逐元素乘。
**理由：** 保留可优化空间（内存合并、循环融合等），故意不内联。

### A.20 matvec_mul 独立模块

**决策：** 单独 `matvec.h/c` 文件，通用向量-矩阵乘。
**理由：** 独立于 SwiGLU 的工具函数，模块化示范。

### A.21 test/ 目录分离

**决策：** 测试文件放入独立 `test/` 目录。
**理由：** 源码与测试职责分明，C 项目常见惯例。

### A.22 不做正确性断言

**决策：** 测试程序不做数值比较，以不崩溃/无 NaN 为通过标准。
**理由：** 无预计算正确答案；教学 + 基线定位下，视觉检查输出合理性即可。

### A.23 临时向量用后即释放

**决策：** 每个临时向量不再需要时立即 `free`，不集中延后。
**理由：** 展示局部资源管理；同时为后续优化（如合并分配）留出空间。
