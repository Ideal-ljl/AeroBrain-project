# AeroBrain — v5 MSP + MAM

从 AirBrain 当前代码中提取的独立模型包，仅支持最终 v5 MSP（模板运动先验）+ MAM。保留 Qwen3.5 VLM、Action Expert、MAM 编码器、运动预测头、模板先验以及完整 checkpoint 的加载代码。模型参数名称保持兼容。

不包含训练入口、数据集、评测/仿真、RL、实验脚本、旧版运动先验或模型权重。`openfly_his`、`airbrain` 的包名保留，避免改变现有代码和 checkpoint 的引用。

## 加载

在已有 PyTorch 环境中安装：

```bash
cd /mnt/petrelfs/liujunli/aerobrain
pip install -e .
python load_model.py /absolute/path/to/v5_checkpoint
# checkpoint 未包含 processor 文件时：
python load_model.py /absolute/path/to/v5_checkpoint --no-processor
```

```python
import torch
from load_model import load_model

model, processor = load_model(
    "/absolute/path/to/v5_checkpoint",
    device="cuda",
    dtype=torch.bfloat16,
    load_processor=False,  # 有完整 processor 文件时改为 True
)

# inputs 为已经预处理并移动到模型设备的输入张量。
# states 使用训练时相同的归一化，input_ids 每条样本包含一个可见的 history_token_id。
# 有历史时需传入 mam_history_actions / mam_history_mask、anchor 图像及事件分段张量。
# 完整参数见 OpenFLYModel.sample_actions。
environment = model.motion_prior.environment_ids(["env_airsim_18"], device=model.device)
actions = model.sample_actions(**inputs, motion_environment=environment, denoise_steps=5)
```

环境名称必须存在于 `model.motion_prior.spec['environments']`，示例名称请按实际 checkpoint 修改。返回动作处于 checkpoint 的 `legacy_env` 归一化坐标，实际执行前需由调用方反归一化。

checkpoint 必须包含 `config.json` 及完整模型权重（支持 safetensors 分片），并满足 `gmm_prior_type=motion_conditional`、`use_gmm=False`、`use_mam_memory=True`、`motion_prior_config.version=5`。MSP 预测头、模板、各环境归一化参数和协方差已嵌入 checkpoint，加载时不依赖原 probe 目录，不重新导入外部预测头。加载器检查缺失、多余和不匹配的权重，拒绝部分初始化。

`sample_actions` 允许空历史（不生成 MAM 前缀），但不会自动从原始图像/动作构建历史事件。本包保留模型侧 MAM 事件检测与编码组件，数据预处理和历史管理由调用方提供。

## 文件

- `load_model.py`：本地完整 checkpoint 加载入口，默认 SDPA。
- `openfly_his/`：模型、配置、Expert、v5 模板先验和预测头。
- `airbrain/memory/mam/`：MAM 事件检测、片段编码和记忆编码。

验证：使用微型 Qwen3.5 模型和真实 MAM anchor 图像张量，对比原实现和提取实现的参数键、权重及固定随机种子的完整动作推理，并执行 save_pretrained → load_model 回环。此验证不代表已加载生产规模权重。
