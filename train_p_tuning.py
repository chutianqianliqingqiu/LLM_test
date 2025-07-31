import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType
)
from datasets import load_dataset

# 配置参数
model_path = "../local/qwen"  # 本地 Qwen 模型路径
data_path = "data/train.json"          # 训练数据路径
output_dir = "outputs/p_tuning"        # 输出目录
prompt_length = 20                     # 连续提示长度
prompt_init_text = "完成以下任务："     # 提示初始化文本（可选）

# 加载模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 半精度加载
    device_map="auto"
)

# 配置 P-Tuning (基于 peft 的 PromptTuning)
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT if prompt_init_text else PromptTuningInit.RANDOM,
    num_virtual_tokens=prompt_length,
    prompt_tuning_init_text=prompt_init_text,
    tokenizer_name_or_path=model_path,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数量

# 数据预处理
def preprocess_function(examples):
    inputs = [f"{prompt_init_text}{example['input']}" for example in examples]
    targets = [example["output"] for example in examples]
    
    model_inputs = tokenizer(inputs, truncation=True, max_length=512)
    labels = tokenizer(targets, truncation=True, max_length=512).input_ids
    model_inputs["labels"] = labels
    return model_inputs

dataset = load_dataset("json", data_files=data_path)
tokenized_data = dataset.map(preprocess_function, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    learning_rate=3e-2,          # P-Tuning 需要较高学习率
    num_train_epochs=5,
    fp16=True,                   # 半精度训练
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

# 开始训练
trainer.train()
model.save_pretrained(output_dir)  # 保存适配器权重
