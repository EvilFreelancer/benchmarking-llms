from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import time

name = 'Gaivoronsky/ruGPT-3.5-13B-8bit'

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

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoGPTQForCausalLM.from_quantized(
    name,
    # use_safetensors=True,
    trust_remote_code=True,
    device="cuda:0",
    use_triton=False,
    quantize_config=None
)

# Model load time
model_load_time = time.time() - start_time
print(f"Model loading time: {model_load_time:.2f} seconds")

# For calculating the average generation time and tokens per second
total_generation_time = 0
total_tokens = 0

for text in texts:
    print(f"\nSample: {text}")
    generation_start_time = time.time()  # Start generation time

    encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
    output = model.generate(
        **encoded_input,
        num_beams=1,
        max_new_tokens=1024,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
        # seed=42,
        do_sample=True,
        use_cache=False
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
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
    print(f"Tokens per second: {tokens_per_second:.2f} t/s\n")

# Calculate average values
average_generation_time = total_generation_time / len(texts)
average_tokens = total_tokens / len(texts)
average_tokens_per_second = total_tokens / total_generation_time

# Print average results
print(f"\nAverage generation time: {average_generation_time:.2f} seconds")
print(f"Average tokens: {average_tokens:.2f}")
print(f"Average tokens per second: {average_tokens_per_second:.2f} t/s")
