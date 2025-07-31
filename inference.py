import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 配置参数
MODEL_PATH = "../local/qwen"  # 本地 Qwen 模型路径
ADAPTER_PATH = "outputs/lora"          # 适配器路径（LoRA 或 Prompt Tuning）
FP16 = True                            # 是否启用半精度推理

# 加载基础模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    device_map="auto"
)

# 加载适配器
if "lora" in ADAPTER_PATH.lower():
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
elif "prompt" in ADAPTER_PATH.lower():
    from peft import PromptTuningConfig
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# 推理函数
def generate_response(instruction, input_text):
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Output: ")[-1]  # 提取生成部分

# 交互式测试
if __name__ == "__main__":
    print("===== 输入 'exit' 退出 =====")
    while True:
        instruction = input("\nInstruction (e.g. '翻译为英文'): ").strip()
        if instruction.lower() == "exit":
            break
        input_text = input("Input: ").strip()
        
        output = generate_response(instruction, input_text)
        print(f"\nOutput: {output}")
