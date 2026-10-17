# GPU 推理主机的 CPU 开销

[English](cpu-tuning.md) | 简体中文

本分支基于上游 vLLM，以及 `github.com/weicj` 的 vLLM 2080 Ti Definitive
Edition。即使推理计算在 GPU 上，HTTP、分词、调度、元数据准备、内核提交和
流式输出仍需要 CPU。更多核心能让不同进程同时运行，但不能让一个 Python
调度或提交循环同时使用全部核心。

## 运行时修改

编译后的 FULL 图元数据刷新会跳过连续出现的同一源和目标元数据引用。
同一注意力组内的层共享元数据，因此无需按层重复遍历和复制。只跳过连续
别名；中间出现其他写入时仍保留原顺序。两条图路径都跳过张量自复制。
V2 对内置标量叶节点跳过拓扑和容器检查；父对象的拓扑校验，以及张量的
形状、类型和设备校验仍然有效。

批次规划还会在零或一个请求时跳过重排；一次统计四种请求区域，替代分别
归约四个掩码。最多 16 个请求的小批次只读取一次 CPU 偏移量，通过 Python
整数比较确定解码与预填充边界；更大批次及显式预填充掩码仍走原向量化路径。
填充 token 数和统一解码规则保持一致。重排快捷路径仅作用于调用共享重排
工具的运行器，不会给自行处理顺序的运行器新增该调用。

这些修改减少 CPU 暂存开销，不引入新的图模式或内核策略。真实 CPU 张量
回归测试覆盖缓冲区地址、共享及冲突别名，以及拓扑、张量和容器错误。
在宣称服务提速或将修改作为已验证部署版本发布之前，仍需 RTX 2080 Ti
端到端验证。

## Xeon E5 v4 主机设置

- 除非有意覆盖 vLLM，否则不要设置 `OMP_NUM_THREADS`。本版本已经在启动
  阶段使用 CPU 并行，在稳定 GPU 服务阶段将 Torch 算子线程数降到一。
  为微小 CPU 操作强制设置 40 或 80 个线程可能增加争用。如果其他服务配置
  强制了更大的值，可对比测试 `OMP_NUM_THREADS=1`；它也会限制启动并行度。
- TP2 至少保留四个物理核心给 API、引擎和两个 GPU 工作进程，并为系统及
  后台任务留余量。不要把所有进程固定到一个核心或同核的两个超线程。
  更多 API 工作进程有助于并发输入处理，不会并行化单次聊天的解码循环。
- 多路主机先检查 `nvidia-smi topo -m`、`numactl --hardware` 和
  `lscpu -e=CPU,NODE,SOCKET,CORE`。安装 `numactl` 后，可在现有直接调用的
  `vllm serve` 命令中加入 `--numa-bind` 做对比；vLLM 会自动检测 GPU
  局部性。不要猜核心或节点编号；手动设置 `--numa-bind-cpus` 和
  `--numa-bind-nodes` 必须依据实际拓扑。避免 API 和后台任务挤占工作进程。
  NUMA 参数属于 vLLM CLI，不能写入路由 profile。参见
  [上游 NUMA 指南](https://docs.vllm.ai/en/latest/configuration/optimization/#numa-binding-for-multi-socket-gpu-nodes)。
- 若流式输出或 HTTP 的 CPU 开销明显，可对比测试 `--stream-interval 4`。
  它批量发送 token，可能减少主机开销，但输出颗粒度会变粗。同时测量客户端
  延迟和吞吐；需要逐 token 即时输出时保留默认值一。
- 保留路由已验证的图模式和推测解码设置。不要为了降低 CPU 使用率强制
  eager 执行或开启不安全的同步模式。计时期间避免请求、输出日志及追踪。

这些设置属于实验，不保证 E5-2698 v4 的具体性能。仅凭 GPU 利用率低不能
确定 CPU 是瓶颈。

## 可复现检查

在仓库目录中，使用已安装 Torch 的 Python 环境运行：

```bash
python -B tests/standalone_tests/test_metadata_cpu.py
python -B tests/standalone_tests/test_attention_cpu.py
python -B tools/benchmark_metadata_cpu.py --baseline-ref 93c9013e37
python -B tools/benchmark_metadata_cpu.py --suite attention --baseline-ref 93c9013e37
```

微基准直接加载源码中的真实辅助函数，不导入 vLLM GPU 扩展。它使用真实
CPU 张量、一个 Torch 线程、48 个合成层别名，预热后对七轮各 1,000 次调用
取中位数。同时报告共享层和独立元数据场景，以显示遍历成本。
它不是模型、Xeon 或 CUDA 基准，不包含 GPU 传输、内核、调度和 HTTP 开销。
比较后续修改时请显式选择其他 Git 基线。

attention 测试组测量单请求及多请求重排、推测解码分割、普通解码快捷返回，
以及混合预填充和解码批次。重排会修改批次，所以未排序场景在两个版本的
计时中都包含相同的测试数据重置成本；该测试组同样仅测量 CPU 开销。

在目标主机上，保持修改前后的模型、profile、模式、KV 精度、推测解码配置、
提示词、token 数和 GPU 时钟一致。遵循 `profiles/2x2080Ti/README.md`
中的参考口径：排除预热，参考测量时关闭前缀缓存，执行三次 4K/128 和一次
完整的 32K/512，并核对回答及准确 token 数。原始结果放在受版本管理源码
之外。使用 [Nsight Systems](https://docs.vllm.ai/en/latest/contributing/profiling/)
区分 CPU 提交间隙与 GPU 计算或通信耗时，不要把追踪开销当作生产延迟。
