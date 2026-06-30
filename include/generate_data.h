/*
 * generate_data.h — 测试数据生成模块
 *
 * 为 SwiGLU FFN 生成可复现的随机测试数据。
 * 所有值服从 [-1, 1) 均匀分布。
 * 通过 OpenSSL MD5 从 base_seed + 标签派生独立种子，
 * 确保四个生成函数生成不同但可复现的数据。
 */

#ifndef GENERATE_DATA_H
#define GENERATE_DATA_H

/*
 * 生成输入向量
 *
 * 参数：
 *   d_model — 输入向量维度
 *   seed    — 基础随机种子
 *
 * 返回：
 *   指向新分配的输入向量的指针（调用者负责 free）
 *   值服从 [-1, 1) 均匀分布
 */
float* generate_input(int d_model, int seed);

/*
 * 生成门控投影权重矩阵 W_gate
 *
 * 参数：
 *   d_model — 输入维度
 *   d_ff    — 输出维度（前馈网络中间维度）
 *   seed    — 基础随机种子
 *
 * 返回：
 *   指向新分配的权重矩阵的指针，形状 (d_model, d_ff)，行主序存储
 *   （调用者负责 free）
 *   值服从 [-1, 1) 均匀分布
 */
float* generate_W_gate(int d_model, int d_ff, int seed);

/*
 * 生成上投影权重矩阵 W_up
 *
 * 参数：
 *   d_model — 输入维度
 *   d_ff    — 输出维度（前馈网络中间维度）
 *   seed    — 基础随机种子
 *
 * 返回：
 *   指向新分配的权重矩阵的指针，形状 (d_model, d_ff)，行主序存储
 *   （调用者负责 free）
 *   值服从 [-1, 1) 均匀分布
 */
float* generate_W_up(int d_model, int d_ff, int seed);

/*
 * 生成下投影权重矩阵 W_down
 *
 * 参数：
 *   d_model — 输出维度
 *   d_ff    — 输入维度（前馈网络中间维度）
 *   seed    — 基础随机种子
 *
 * 返回：
 *   指向新分配的权重矩阵的指针，形状 (d_ff, d_model)，行主序存储
 *   （调用者负责 free）
 *   值服从 [-1, 1) 均匀分布
 */
float* generate_W_down(int d_model, int d_ff, int seed);

#endif /* GENERATE_DATA_H */
