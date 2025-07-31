Qwen 高效微调项目
使用 transformers 和 peft 库对 Qwen 大模型进行参数高效微调（LoRA/Prompt Tuning），支持半精度（FP16）训练以降低显存占用。

📌 功能特性
✅ 参数高效微调：支持 LoRA 和 Prompt Tuning，仅优化少量参数。

🚀 半精度训练：FP16 混合精度加速，显存占用减少 50%。

🤖 多任务适配：支持文本生成、分类等下游任务。

💾 轻量存储：仅保存适配器权重（LoRA/Prompt），无需全模型参数。

🛠️ 环境安装
bash
pip install torch transformers peft accelerate datasets bitsandbytes
推荐 Python 3.8+ 和 CUDA 11.7+。

需 NVIDIA GPU（支持 FP16 计算）。

🚀 快速开始
1. 数据准备
格式：JSON 文件，包含 instruction/input/output 字段（示例见 data/ 目录）。

示例数据：

json
[{"instruction": "翻译为英文", "input": "你好", "output": "Hello"}]
2. LoRA 微调
bash
python train_lora.py \
    --model_path path/to/qwen \
    --data_path data/train.json \
    --output_dir outputs/lora \
    --fp16 True  # 启用半精度
关键参数：

--lora_r: LoRA 秩（默认 8）

--lora_alpha: 缩放系数（默认 32）

--target_modules: 目标模块（默认 q_proj,k_proj,v_proj）

3. Prompt Tuning 微调
bash
python train_prompt_tuning.py \
    --model_path path/to/qwen \
    --data_path data/train.json \
    --num_virtual_tokens 20 \
    --prompt_init_text "分类任务：" 
📂 代码结构
text
.
├── train_lora.py            # LoRA 微调脚本
├── train_prompt_tuning.py   # Prompt Tuning 脚本
├── inference.py             # 加载适配器推理
├── data/                    # 示例数据
│   ├── train.json
│   └── test.json
└── outputs/                 # 保存适配器权重
⚙️ 参数配置
通用训练参数（TrainingArguments）
参数名	说明
fp16	启用半精度训练（默认 True）
per_device_train_batch_size	批次大小（根据显存调整）
learning_rate	学习率（LoRA 建议 3e-4，Prompt Tuning 建议 3e-2）
LoRA 专用参数
python
LoraConfig(
    r=8,                         # 秩
    lora_alpha=32,               # 缩放系数
    target_modules=["q_proj"],   # 目标模块
    lora_dropout=0.05
)
Prompt Tuning 专用参数
python
PromptTuningConfig(
    num_virtual_tokens=20,       # 软提示长度
    prompt_tuning_init="TEXT"    # 初始化方式
)
🧠 推理示例
加载微调后的适配器进行预测：

python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("path/to/qwen")
model = PeftModel.from_pretrained(model, "outputs/lora")  # 加载 LoRA
model.half()  # 切换到 FP16

inputs = tokenizer("Instruction: 翻译为英文\nInput: 早安", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
📊 性能对比
方法	可训练参数量	显存占用（Qwen-7B）	训练速度
全参数微调	7B	80GB+	1x
LoRA (FP16)	0.1%	~12GB	1.5x
Prompt Tuning	0.01%	~8GB	2x
❓ 常见问题
显存不足：

启用梯度检查点：gradient_checkpointing=True

使用 4-bit 量化：在 from_pretrained 中添加 BitsAndBytesConfig。

NaN 损失：

降低学习率或启用梯度缩放：fp16_full_eval=True。

如何适配其他任务？

修改 data.json 中的字段和 preprocess_function。

