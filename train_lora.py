from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from peft import PeftModel

model_path = "../local/qwen"  # 本地 Qwen 模型目录
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 可选：4-bit 量化（降低显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # 去掉此行禁用量化
    device_map="auto",
    trust_remote_code=True
)


lora_config = LoraConfig(
    r=8,                  # LoRA 秩
    lora_alpha=32,        # 缩放系数
    target_modules=["q_proj", "k_proj", "v_proj"],  # 目标模块（Qwen 通常为注意力层的 Q/K/V）
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数量（应远小于总参数量）


data = load_dataset("json", data_files="data.json")
def preprocess_function(examples):
    inputs = [f"Instruction: {i}\nInput: {inp}\nOutput: " for i, inp in zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(inputs, truncation=True, max_length=512)
    labels = tokenizer(examples["output"], truncation=True, max_length=512).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,               # 混合精度训练
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

trainer.train()
model.save_pretrained("./lora_adapter")  # 仅保存 LoRA 权重（轻量）


# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

input_text = "Instruction: 翻译为英文\nInput: 你好吗\nOutput: "
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
