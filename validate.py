from transformers import AutoModelForCausalLM, AutoTokenizer

pretrained_model_path = "./proper-to-gamerspeak-model-2/checkpoint-500"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_path, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

from transformers import StoppingCriteria


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt, "")
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        # max_length=100,
        # min_length=50,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        stopping_criteria=MyStoppingCriteria("\n", prompt),
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


output = generate("Where do I download the fabric client?\n")
print(output)
