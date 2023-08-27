import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
from conversation import Conversation

# name = 'IlyaGusev/saiga2_7b_lora'
name = 'IlyaGusev/saiga2_13b_lora'

# Sample texts
texts = [
    "Александр Сергеевич Пушкин родился в ",
    "Alexander Sergeevich Pushkin was born in ",
    "Олександр Сергійович Пушкін народився в ",
    "Аляксандр Сяргеевіч Пушкін нарадзіўся ў ",
    "Александр Сергеевич Пушкин шәһәрендә туган",
    "Александр Сергеевич Пушкин дүниеге келген",
    "亚历山大*谢尔盖耶维奇*普希金出生于",
    "알렉산더 세르게 비치 푸쉬킨은 ",
    "Alexander Sergeevich Pushkinはで生まれました",
    "Александр Сергеевич Пушкин ҫуралнӑ ",
    "Александр Сергеевич Пушкин төрөөбүтэ ",
    "Alexander Puschkin wurde in ",
    "Alexandre Sergueïevitch Pouchkine est né le à ",
]

# Start model load time
start_time = time.time()

# Download config
config = PeftConfig.from_pretrained(name)
dtype = torch.float16

# Download model source and weights
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    # torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    name,
    legacy=True,
    torch_dtype=torch.float16
)
model.eval()

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
generation_config = GenerationConfig.from_pretrained(name)
# print(generation_config)

# Model load time
model_load_time = time.time() - start_time
print(f"Model loading time: {model_load_time:.2f} seconds")

# For calculating the average generation time and tokens per second
total_generation_time = 0
total_tokens = 0


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        # generation_config=generation_config,
        max_new_tokens=1024,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        # seed=42,
        do_sample=True,
        use_cache=False
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


# Generating text for each sample
for text in texts:
    print(f"\nSample: {text}")
    generation_start_time = time.time()  # Start generation time
    conversation = Conversation()
    conversation.add_user_message(text)
    prompt = conversation.get_prompt(tokenizer)
    # print(prompt)
    output = generate(model, tokenizer, prompt, generation_config)
    print(output)

    # Generation time
    generation_time = time.time() - generation_start_time
    # Accumulate total generation time
    total_generation_time += generation_time

    # Tokens per second
    tokens_for_this_sample = len(tokenizer.encode(output))
    tokens_per_second = tokens_for_this_sample / generation_time
    total_tokens += tokens_for_this_sample

    # Print results for this sample
    print(f"Generation time for this sample: {generation_time:.2f} seconds")
    print(f"Tokens for this sample: {tokens_for_this_sample}")
    print(f"Tokens per second: {tokens_per_second:.1f} t/s\n")

# Calculate average values
average_generation_time = total_generation_time / len(texts)
average_tokens = total_tokens / len(texts)
average_tokens_per_second = total_tokens / total_generation_time

# Print average results
print(f"\nAverage generation time: {average_generation_time:.2f} seconds")
print(f"Average tokens: {average_tokens:.1f}")
print(f"Average tokens per second: {average_tokens_per_second:.1f} t/s")
