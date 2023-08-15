# Benchmarking Large Language Models (LLMs)

This comparison evaluates various large language models (LLMs) based on their hardware usage,
number of parameters, and context size.

Test Environment:

* Graphics Card: RTX 4090 24Gb
* CUDA Version: 11.7 (for ruGPT3 family) and 11.8 (for other models)
* Python Version: 3.11.4

Note:

* I was unable to test the 13B models due to GPU memory limitations.
* I have not been granted access to test LLaMA 2 yet.

## Testing Prompts

For my tests, I evaluated how models responded to prompts about the birthdate of the famous poet, Alexander Sergeevich
Pushkin. I employed diverse prompts in various languages and transliterations to ensure a comprehensive evaluation. This
method was inspired by the model testing approach for mGPT 1.3B, as demonstrated in
this [example notebook](https://github.com/ai-forever/mgpt/blob/main/notebooks/mgpt_huggingface_generation_example.ipynb).

## Evaluation Parameters

To maintain consistency in my evaluations, I used the following generation parameters:

* Maximum new tokens: 1024
* Top-k: 20
* Top-p: 0.9
* Repetition Penalty: 1.1
* Sampling: Enabled
* Caching: Disabled

I chose these parameters to:

* Determine the model's verbosity.
* Measure its generation speed.
* Most crucially, understand its memory requirements.

Through my testing, I discovered that performing CUDA cache clearance `torch.cuda.empty_cache()` results
in a reduction of generation speed, averaging between 15-25%.

## Results

The table provides a detailed comparison and performance metrics of various large language models (LLMs).

| Name                                                                           | Size           | Context | VRAM (Gb)      | MAX Init RAM (Gb) | AVG GenTime (s) | AVG Tokens | AVG t/s |
|--------------------------------------------------------------------------------|----------------|---------|----------------|-------------------|-----------------|------------|---------|
| [StableBeluga 7b](https://huggingface.co/stabilityai/StableBeluga-7B)          | 7b             | 4096    | ~22.5          | ~22.7             | ~31.25          | ~529.7     | ~16.9   |
| [LLaMA 7b](https://huggingface.co/huggyllama/llama-7b)                         | 7b             | 4096    | ~22.47         | ~22.7             | ~34.52          | ~545.5     | ~15.8   |
| [LLaMA 2 7b](https://huggingface.co/meta-llama/Llama-2-7b)                     | 7b             | 4096    |                |                   |                 |            |         | 
| [LLaMA 2 7b 32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K)       | 7b-32k         | 32768   | ~21.5          | ~22.7             | ~56.63          | ~868.5     | ~15.3   | 
| [MosaicML 7b](https://huggingface.co/mosaicml/mpt-7b)                          | 7b             | 8192    | ~22.6 (~13.7)  | ~9.8              | ~87.27          | ~1046.2    | ~12.0   |
| [MosaicML 7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)  | 7b-storywriter | 65536   | ~22.9          | ~10.4             | ~109.12         | ~1048.2    | ~9.6    |
| [MosaicML 7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)        | 7b-instruct    | 4096    | ~22.93         | ~9.8              | ~110.47         | ~1045.2    | ~9.5    |
| [MosaicML 7b-instruct-8k](https://huggingface.co/mosaicml/mpt-7b-instruct-8k)  | 7b-instruct-8k | 8192    | ~22.66         | ~10.5             | ~84.32          | ~1045.5    | ~12.4   |
| [ruGPT 3 small](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)   | 125m           | 2048    | ~6.18          | ~1.3              | ~6.4            | ~1041.8    | ~162.7  |
| [ruGPT 3 medium](https://huggingface.co/ai-forever/rugpt3medium_based_on_gpt2) | 410m           | 2048    | ~6.66          | ~2.6              | ~12.74          | ~1044.3    | ~82.0   |
| [ruGPT 3 large](https://huggingface.co/ai-forever/rugpt3large_based_on_gpt2)   | 750m           | 2048    | ~7.48          | ~5.2              | ~15.19          | ~1045.5    | ~68.8   |
| [ruGPT 3 xl](https://huggingface.co/ai-forever/rugpt3xl)                       | 1.3B           | 2048    | ~13.76         | ~4.7              | ~13.38          | ~567.1     | ~42.4   |
| [ruGPT 3.5 13b](https://huggingface.co/ai-forever/ruGPT-3.5-13B)               | 13b            | 2048    |                |                   |                 |            |         |
| [mGPT](https://huggingface.co/ai-forever/mGPT)                                 | 1.3b           | 2048    | ~22.96 (~4.11) | ~7.01             | ~24.72          | ~1046.8    | ~42.3   |
| [mGPT 13b](https://huggingface.co/ai-forever/mGPT-13B)                         | 13b            | 2048    |                |                   |                 |            |         |

* **Name** - The name of the large language model (LLM), often hyperlinked to its source or documentation.
* **Size** - The number of parameters the model has, typically represented in billions (b) or other units.
* **Context** - The maximum number of tokens the model can consider from previous inputs in a conversation or text
  sequence. |
* **VRAM (Gb)** - The amount of Video RAM (in gigabytes) required to run the model.
* **MAX Init RAM (Gb)** - The maximum amount of system RAM (in gigabytes) used during the model's initialization.
* **AVG GenTime (s)** - The average time (in seconds) it takes for the model to generate a response or complete a given
  task.
* **AVG Tokens** - The average number of tokens generated by the model in its responses or outputs.
* **AVG t/s** - The average number of tokens generated by the model per second.

## Scripts

* **llama.py** - A script to test LLaMA and LLaMA 2 models and model based on them.
* **mpt.py** - A script to test MosaicML models.
* **rugpt.py** - A script to test ruGPT3small, ruGPT3medium, ruGPT3large and mGPT.
* **rugpt3xl.py** - A script to test ruGPT3XL only.
    * Dockerfile - A Dockerfile to run rugpt3xl.py in a container.
    * docker-compose.yml - A docker-compose file to run rugpt3xl.py in a container.
    * requirements-xl.txt - A list of Python packages required to run rugpt3xl.py in a container.

# Links

* My Telegram channel: https://t.me/evilfreelancer
* Salute AI Community Telegram channel: https://t.me/SaluteTechGroup
