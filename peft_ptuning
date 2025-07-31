from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel
model_path = "../local/qwen"  # 本地 Qwen 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)


prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",      # 因果语言模型
    num_virtual_tokens=20,      # 软提示的 token 数量（通常 10-100）
    prompt_tuning_init="TEXT",   # 初始化方式（"TEXT" 或 "RANDOM"）
    prompt_tuning_init_text="将以下文本分类为正面或负面情感：",  # 初始化文本（可选）
    tokenizer_name_or_path=model_path,
)

model = get_peft_model(model, prompt_config)
model.print_trainable_parameters()  # 检查可训练参数量（应仅为 prompts 参数）


data = load_dataset("json", data_files="data.json")

def preprocess_function(examples):
    # 将标签转换为文本（适配 Prompt Tuning）
    label_mapping = {"positive": "正面情感", "negative": "负面情感"}
    inputs = [f"文本：{text}\n情感：" for text in examples["text"]]
    labels = [label_mapping[label] for label in examples["label"]]
    
    # 编码输入和标签
    model_inputs = tokenizer(inputs, truncation=True, max_length=512)
    labels = tokenizer(labels, truncation=True, max_length=10).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./prompt_tuning_output",
    per_device_train_batch_size=4,
    learning_rate=3e-2,          # Prompt Tuning 通常需要更高学习率
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    fp16=True,                   # 混合精度训练
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

trainer.train()
model.save_pretrained("./prompt_tuning_adapter")  # 仅保存 prompts 参数


# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# 加载 Prompt Tuning 适配器
model = PeftModel.from_pretrained(base_model, "./prompt_tuning_adapter")

# 输入示例
input_text = "文本：这个餐厅的菜品非常美味\n情感："
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# 预期输出：正面情感
