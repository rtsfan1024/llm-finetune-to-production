# encoding=utf-8

import math

"""
采用6ND方法预估总计算量：
6是预估系数，代表每个参数和每个Token都需要大约6次的乘法和加法操作；
N是Number of Parameters，代表模型的参数量;
D是Data Tokens，代表训练数据的Token数量。
"""

# 系数
coefficient = 6

# 模型参数量（单位Billion）
number_of_parameters = 35
# 参数量转化为个位数
N = number_of_parameters * 10 ** 9

# Token数量（单位Billion）
data_tokens = 500
# 参数量转化为个位数
D = data_tokens * 10 ** 9

# 总算力需求（Based on 6ND）
P = coefficient * N * D

"""
GPU卡的数量 = P / ( 期望的训练时长 * 集群算力效率 * 单块GPU卡的性能 )
"""

# 期望的训练天数（单位Day）
expected_training_days = 30
# 训练天数转化为训练秒数（单位Second）
training_duration = expected_training_days * 24 * 60 * 60

# GPU使用率
"""
集群算力达到100%的利用率是非常困难的，主要由于以下三个原因：
1、通信开销：不同的GPU需要通过网络同步参数和梯度信息，这些信息需要时间。
2、负载不均衡：数据集不能被均匀地分配到每个GPU，或者不同的GPU处理速度不一样。
3、硬件和软件的限制。
"""
use_ratio = 0.5

# 单块GPU卡的性能（单位TFlops）
"""
T是tera, 代表"万亿";
Flops是Floating Point Operations Per Second, 代表每秒钟可以执行的浮点运算次数。
1TFlops表示每秒钟可以执行一万亿次的浮点运算。

# 接口说明
PCIE: Peripheral Component Interconnect Express, 一种高速串行计算机扩展总线标准, PCIe是一种通用接口.
SXM: 一种NVIDIA的专有接口, 被称为"Mezzanine"接口, 提供了更高的带宽和更低的延迟.
NVL: NVIDIA的NVLink接口, NVLink是NVIDIA开发的一种高速直接GPU到GPU的互连技术, 允许高速数据交换, 提高多GPU系统的性能和效率.
# 优化说明
Tensor Core是NVIDIA在最新的GPU（如Volta，Turing，Ampere系列）中引入的一种新的硬件单元.
Tensor Core专门设计用于执行深度学习中的张量运算, 特别是矩阵乘法和累加.
"""
A100_80G_PCIe_FP64 = 9.7
A100_80G_PCIe_FP64_TensorCore = 19.5
A100_80G_PCIe_FP32 = 19.5
A100_80G_PCIe_TF32 = 156
A100_80G_PCIe_FP16_TensorCore = 312
A100_80G_PCIe_INT8_TensorCore = 624
A100_80G_SXM_FP64 = 9.7
A100_80G_SXM_FP64_TensorCore = 19.5
A100_80G_SXM_FP32 = 19.5
A100_80G_SXM_TF32 = 312
A100_80G_SXM_FP16_TensorCore = 624
A100_80G_SXM_INT8_TensorCore = 1248
A800_40G_PCIe_FP64 = 9.7
A800_40G_PCIe_FP64_TensorCore = 19.5
A800_40G_PCIe_FP32 = 19.5
A800_40G_PCIe_TF32 = 156
A800_40G_PCIe_FP16_TensorCore = 312
A800_40G_PCIe_INT8_TensorCore = 624
A800_80G_PCIe_FP64 = 9.7
A800_80G_PCIe_FP64_TensorCore = 19.5
A800_80G_PCIe_FP32 = 19.5
A800_80G_PCIe_TF32 = 312
A800_80G_PCIe_FP16_TensorCore = 624
A800_80G_PCIe_INT8_TensorCore = 1248
A800_80G_SXM_FP64 = 9.7
A800_80G_SXM_FP64_TensorCore = 19.5
A800_80G_SXM_FP32 = 19.5
A800_80G_SXM_TF32 = 312
A800_80G_SXM_FP16_TensorCore = 624
A800_80G_SXM_INT8_TensorCore = 1248
H100_80G_SXM_FP64 = 34
H100_80G_SXM_FP64_TensorCore = 67
H100_80G_SXM_FP32 = 67
H100_80G_SXM_TF32_TensorCore = 989
H100_80G_SXM_FP16_TensorCore = 1979
H100_80G_SXM_FP8_TensorCore = 3958
H100_80G_SXM_INT8_TensorCore = 3958
H100_80G_PCIe_FP64 = 26
H100_80G_PCIe_FP64_TensorCore = 51
H100_80G_PCIe_FP32 = 51
H100_80G_PCIe_TF32_TensorCore = 756
H100_80G_PCIe_FP16_TensorCore = 1513
H100_80G_PCIe_FP8_TensorCore = 3026
H100_80G_PCIe_INT8_TensorCore = 3026
H100_188G_NVL_FP64 = 68
H100_188G_NVL_FP64_TensorCore = 134
H100_188G_NVL_FP32 = 134
H100_188G_NVL_TF32_TensorCore = 1979
H100_188G_NVL_FP16_TensorCore = 3958
H100_188G_NVL_FP8_TensorCore = 7916
H100_188G_NVL_INT8_TensorCore = 7916
H800_80G_SXM_FP64 = 1
H800_80G_SXM_FP64_TensorCore = 1
H800_80G_SXM_FP32 = 989
H800_80G_SXM_TF32_TensorCore = 67
H800_80G_SXM_FP16_TensorCore = 1979
H800_80G_SXM_FP8_TensorCore = 3958
H800_80G_SXM_INT8_TensorCore = 3958
H800_80G_PCIe_FP64 = 0.8
H800_80G_PCIe_FP64_TensorCore = 0.8
H800_80G_PCIe_TF32 = 51
H800_80G_PCIe_FP32_TensorCore = 756
H800_80G_PCIe_FP16_TensorCore = 1513
H800_80G_PCIe_FP8_TensorCore = 3026
H800_80G_PCIe_INT8_TensorCore = 3026

# 以H800_40G_SXM_FP32进行评估，所以除2
single_gpu = H800_80G_SXM_FP32 / 2

how_many_gpu = round((P / (training_duration * use_ratio * single_gpu)) / 10 ** 12, 2)

print("理论需要", how_many_gpu, "张H800_40G卡")
print("实际采购", math.ceil(how_many_gpu), "张H800_40G卡")

print("理论需要", (math.ceil(how_many_gpu)) / 8, "台GPU服务器")
print("实际采购", math.ceil((math.ceil(how_many_gpu)) / 8), "台GPU服务器")



