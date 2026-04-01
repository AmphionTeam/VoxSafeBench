<div align="center">

# 🎙️ VoxSafeBench

**Not Just *What* Is Said, but *Who*, *How*, and *Where***

[![Demo Page](https://img.shields.io/badge/Project%20Page-blue?logo=internet-explorer&logoColor=white)](https://jensenyx.github.io/VoxAlign.github.io/)
[![arXiv Paper](https://img.shields.io/badge/arXiv%20Paper-b31b1b?logo=arxiv&logoColor=white)](#)
[![Dataset](https://img.shields.io/badge/Dataset-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/YuxiangW/VoxSafeBench)
<br>

![Benchmark Safety](https://img.shields.io/badge/Benchmark-Safety-red?logo=security&logoColor=white)
![Benchmark Fairness](https://img.shields.io/badge/Benchmark-Fairness-ff69b4?logo=scales&logoColor=white)
![Benchmark Privacy](https://img.shields.io/badge/Benchmark-Privacy-success?logo=privacy&logoColor=white)

</div>

---

**VoxSafeBench** is a comprehensive benchmark designed to assess the social alignment of Speech Language Models (SLMs) built around three core pillars: **Safety**, **Fairness**, and **Privacy**. VoxSafeBench adopts a unique Two-Tier design: Tier 1 evaluates content-centric risks with matched text and audio inputs, while Tier 2 evaluates audio-conditioned risks in which the transcript is benign but the correct response depends on who is speaking, how they speak, or where they speak.

<div align="center">

![VoxSafeBench Benchmark Overview](./bigpic.png)

</div>

---


<details open>
<summary><b>📑 Table of Contents</b></summary>



- [🎙️ VoxSafeBench](#️-voxsafebench)
  - [⚙️ Environment Setup](#️-environment-setup)
  - [📥 Dataset download](#-dataset-download)
  - [🤖 Models download](#-models-download)
    - [Qwen3-Omni](#qwen3-omni)
    - [Qwen3-Omni-thinking](#qwen3-omni-thinking)
    - [Mimo-audio/Mimo-audio-thinking](#mimo-audiomimo-audio-thinking)
    - [Kimi-audio](#kimi-audio)
    - [Gemini-3 \& GPT-4o-audio](#gemini-3--gpt-4o-audio)
  - [🚀 Model Inference (Unified Runner)](#-model-inference-unified-runner)
    - [Basic Usage](#basic-usage)
    - [Advanced Options](#advanced-options)
    - [Output Location](#output-location)
  - [📈 Final Evaluation](#-final-evaluation)
    - [Prerequisite](#prerequisite)
    - [Basic Usage](#basic-usage-1)
    - [Advanced Options](#advanced-options-1)
    - [Output Location](#output-location-1)
  - [⚠️ Note](#️-note)

</details>



## ⚙️ Environment Setup

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/AmphionTeam/VoxSafeBench.git
cd VoxSafeBench
```

Evaluating different models requires configuring their respective environments under each model's code repository. 

For **closed-source models** (e.g., `gemini_3_flash`, `gemini_3_pro` and `gpt_4o_audio`) and the overall evaluation scripts (`run_inference.py`, `run_evaluation.py`), you need to install the following dependencies:

```bash
pip install openai google-genai python-dotenv tqdm
```

For **open-source models**, please refer to the specific setup instructions in their respective official repositories:
- **Qwen3-omni:** [QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)
- **Mimo-audio:** [XiaomiMiMo/MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio)
- **Kimi-audio:** [MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio)

---

## 📥 Dataset download

```bash
export HF_ENDPOINT=https://hf-mirror.com

HF_HUB_DOWNLOAD_TIMEOUT=240 huggingface-cli download --repo-type dataset --resume-download YuxiangW/VoxSafeBench --local-dir ./datasets --max-workers 32
```

If the error is caused by network issues, you can try a few more times.

## 🤖 Models download

### Qwen3-Omni
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ./model_warehouse/Qwen3_omni
```

### Qwen3-Omni-thinking
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download Qwen/Qwen3-Omni-30B-A3B-Thinking --local-dir ./model_warehouse/Qwen3_omni_thinking
```

### Mimo-audio/Mimo-audio-thinking
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download XiaomiMiMo/MiMo-Audio-7B-Instruct --local-dir ./model_warehouse/Mimo_audio/MiMo-Audio-7B-Instruct

huggingface-cli download --resume-download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir ./model_warehouse/Mimo_audio/MiMo-Audio-Tokenizer
```

### Kimi-audio
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download moonshotai/Kimi-Audio-7B-Instruct --local-dir ./model_warehouse/Kimi_audio
```
> ⚠️ **Please replace** `model_warehouse/Kimi_audio/modeling_moonshot_kimia.py` with `utils/modeling_moonshot_kimia.py`.  
> ⚠️ If you don't make this replacement, Kimi-audio cannot perform pure text input.

### Gemini-3 & GPT-4o-audio
Please put your `gemini_key` and `Openai_key` in the `.env` file.

---

## 🚀 Model Inference (Unified Runner)

We provide a unified inference script `run_inference.py` to generate responses from any model across various tasks seamlessly.

### Basic Usage

You can use the unified runner by specifying the model and the task:

```bash
# Run a specific task for a model
python run_inference.py --model Qwen3_omni --task Safety-tier2/Emotion

# Run all tasks for a model
python run_inference.py --model Qwen3_omni --all
```

**Available Models:**
- `Qwen3_omni`
- `Qwen3_omni_thinking`
- `Mimo_audio`
- `Mimo_audio_thinking`
- `Kimi_audio`
- `gemini_3_flash`
- `gemini_3_pro`
- `gpt_4o_audio`

*(You can see all available models dynamically via `python run_inference.py --help`)*

**Available Tasks:**

<details>
<summary><b>Fairness Tasks</b></summary>

- `Fairness-tier1/test`
- `Fairness-tier2/Bias_analysis`
- `Fairness-tier2/test`

</details>

<details>
<summary><b>Privacy Tasks</b></summary>

- `Privacy-tier1/Hard_privacy`
- `Privacy-tier1/Soft_privacy`
- `Privacy-tier2/Audio_conditioned_privacy`
- `Privacy-tier2/Interactional_privacy`

</details>

<details>
<summary><b>Safety Tasks (Tier 1)</b></summary>

- `Safety-tier1/Agentic_Action_Risks`
- `Safety-tier1/Multiturn_jailbreak`
- `Safety-tier1/No_jailbreak`
- `Safety-tier1/Singleturn_jailbreak`

</details>

<details>
<summary><b>Safety Tasks (Tier 2)</b></summary>

- `Safety-tier2/Child_presence`
- `Safety-tier2/Child_voice`
- `Safety-tier2/Emotion`
- `Safety-tier2/Impaired_capacity`
- `Safety-tier2/Overlap_instruction_injection`
- `Safety-tier2/Symbolic_background`
- `Safety-tier2/Unsafe_ambient`

</details>

### Advanced Options

You can override default settings per run:

```bash
python run_inference.py --model gpt_4o_audio --all --max-workers 8 --save-interval 20
```

- `--max-workers`: Set the number of concurrent workers (useful for API models like GPT-4o or Gemini).
- `--save-interval`: How often to save results to disk.
- `--model-name`: Override the internal model name string passed to the API or model loader.

### Output Location

Results are written automatically to:

```text
results/<model_name>/<task_name>/results.jsonl
```

The runner automatically:
- Reads input data from the appropriate `datasets/**/metadata.jsonl`.
- Resumes from existing `results.jsonl` (skips completed items, retries items that failed with `ERROR:`).
- Merges the model's outputs into the existing dataset fields.

---

## 📈 Final Evaluation

After running model inference, you can evaluate the generated responses and compute metrics using the `run_evaluation.py` script. The evaluation uses automated judges (e.g., DeepSeek, OpenAI Moderation) and rule-based evaluators to score safety, privacy, and fairness.

### Prerequisite

Ensure you have the required API keys configured in your `.env` file, as the script uses these external services for evaluation.

### Basic Usage

You can run the evaluation script with different levels of granularity:

```bash
# Evaluate ALL models and ALL tasks (found in the `results/` directory)
python run_evaluation.py

# Evaluate all tasks for a specific model
python run_evaluation.py --model Qwen3_omni

# Evaluate a specific task for a specific model
python run_evaluation.py --model Qwen3_omni --task Safety-tier1/No_jailbreak
```

### Advanced Options

You can specify the number of concurrent threads used during evaluation:

```bash
python run_evaluation.py --model gpt_4o_audio --threads 16
```

- `--threads`: Set the maximum number of worker threads (default is 8) to speed up API-based evaluations.

### Output Location

The evaluation script reads from the `results/` directory and writes the final evaluated data and metrics to the `final_eval_results/` directory:

```text
final_eval_results/<model_name>/<task_name>/
├── results.jsonl   # Data with evaluation scores and judgments appended
└── log.txt         # Computed metrics and summary statistics
```

---

## ⚠️ Note

- For the **Privacy-tier2/Inferential_privacy** task, please use the [HearSay Benchmark](https://github.com/JinWang79/HearSay_Benchmark).
- All experiments were conducted on Nvidia A800 GPUs.
