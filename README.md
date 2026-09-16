# E-commerce QA Agent

一个面向游戏商品问答的 **LLM Post-Training 项目**：对 Qwen2.5 依次做 SFT → DPO → RLHF(GRPO)，
得到一个会调用工具（搜索 / 详情 / 评论）查询、并给出有依据（grounded）答案的多轮对话 Agent，
最后用 Base / SFT / DPO / RLHF 四模型对比评测，量化每一步后训练带来的提升。

## 流程

```
Synthetic Catalog ──> Task Data ──> SFT ──> DPO ──> RLHF(GRPO) ──> Agent Loop ──> Evaluation
```

| 阶段 | 作用 | 代码 |
|---|---|---|
| **SFT** | 教师轨迹监督微调（completion-only masking），教会工具调用格式 | [train/sft.py](train/sft.py) |
| **DPO** | 偏好对直接优化策略，离线提升回答质量 | [train/dpo.py](train/dpo.py) |
| **RLHF/GRPO** | 奖励模型 + 在线组采样策略梯度（DeepSeek-R1 路线） | [train/reward_model.py](train/reward_model.py) + [train/grpo.py](train/grpo.py) |
| **Agent** | 多轮工具调用 loop（`<tool_call>` 解析 / 执行 / 上限保护） | [agent/agent.py](agent/agent.py) |
| **Evaluation** | Base/SFT/DPO/RLHF 五指标对比 | [evaluation/evaluate.py](evaluation/evaluate.py) |

## 技术要点

- **LoRA / QLoRA**：r=8 低秩适配，仅训练约 0.22% 参数；云端 7B 用 4-bit QLoRA
- **Completion-only masking**：SFT 只对回复 token 计算 loss，忽略 prompt 部分
- **DPO**：手写偏好损失，β 控制相对参考模型的偏离
- **GRPO**：组采样 → 奖励模型打分 → 组内归一化优势 → 策略梯度 + KL 正则（k3 估计器）
- **原生工具调用**：Qwen2.5 function-calling（`<tool_call>` + JSON）

## 目录结构

```
config.py                  全局配置（模型 ID / system prompt / 路径）
data/
  generate_data.py         合成目录（30 商品 / 150 评论）
  task_templates.py        5 类任务模板
  generate_trajectories.py 教师轨迹蒸馏（SFT 数据）
  generate_dpo_pairs.py    DPO 偏好对
  raw/ processed/          数据文件（由脚本生成，不随仓库提交）
tools/                     检索工具 + 注册表（search / details / reviews）
agent/
  agent.py                 Agent loop
  model_loader.py          base + LoRA 加载（推理 / 训练两用）
  tool_executor.py         <tool_call> 解析执行
train/
  sft.py dpo.py reward_model.py grpo.py
evaluation/
  evaluate.py              四模型对比评测
```

## 快速开始

依赖：`transformers`、`peft`、`torch`、`openai`。教师蒸馏与 LLM 裁判需要在 `.env`
中配置 `DEEPSEEK_API_KEY`。

### 本地（0.5B，RTX 3050 4GB 调试）

```bash
# 数据（由脚本生成）
python -m data.generate_trajectories --n-per-type 40      # SFT 轨迹
python -m data.generate_dpo_pairs --n-per-type 6 --sft-adapter models/sft_lora   # DPO 对

# 训练
python -m train.sft                                        # SFT
python -m train.dpo --sft-adapter models/sft_lora          # DPO
python -m train.reward_model --sft-adapter models/sft_lora # 奖励模型
python -m train.grpo --sft-adapter models/sft_lora --reward models/reward_model  # GRPO

# 演示 & 评测
python -m agent.agent "推荐一款 40 美元以内的新手 RPG"
python -m evaluation.evaluate --sft-adapter models/sft_lora --dpo-adapter models/dpo_lora
```

> 4GB 卡跑完整 SFT 会因显存碎片化而明显变慢，建议本地用 `--debug` 冒烟验证，正式训练上云。

### 云端（7B）

把 [config.py](config.py) 的 `MODEL_ID` 改为 `Qwen/Qwen2.5-7B-Instruct`：

```bash
python -m train.sft --qlora                                   # 4-bit QLoRA
python -m data.generate_dpo_pairs --n-per-type 40 --sft-adapter models/sft_lora
python -m train.dpo --sft-adapter models/sft_lora
python -m train.reward_model --sft-adapter models/sft_lora
python -m train.grpo --sft-adapter models/sft_lora --reward models/reward_model --steps 500
python -m evaluation.evaluate --n-per-type 8
```

## 评测

对每个模型统计 5 类任务上的四个指标：

- **tool_usage**：是否至少调用一次工具
- **grounded**：回答中提到的产品名是否都出现在自己的工具结果里（无幻觉）
- **task_success**：grounded 且 LLM 裁判 ≥4 分（端到端成功率）
- **avg_judge**：DeepSeek 裁判 1–5 分

期望结果是 `Base < SFT < DPO < RLHF` 逐级上升，即本项目要验证的核心结论。

## 状态

全部阶段（SFT / DPO / RLHF / Agent / Evaluation）代码已完成，数据为合成小规模数据
（30 商品 / 150 评论），正式训练建议在云端完成。
