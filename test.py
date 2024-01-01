from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Q: What is the largest animal?\nA:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

print("GENERATING...")

generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)

print("GENERATED!")
print(tokenizer.decode(generation_output[0]))
