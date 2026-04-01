import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    from .task_configs import TASK_CONFIGS, TASK_IDS, TaskConfig
except ImportError:
    from task_configs import TASK_CONFIGS, TASK_IDS, TaskConfig

import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeProcessor
except ImportError:
    process_mm_info = None
    Qwen3OmniMoeProcessor = None

MODEL_NAME_DEFAULT = "Qwen3_omni"
SAVE_INTERVAL_DEFAULT = 10
MAX_WORKERS_DEFAULT = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_root() -> Path:
    return repo_root() / "models" / "Qwen3_omni"


def results_root() -> Path:
    return repo_root() / "results" / "Qwen3_omni"


MODEL_PATH = "model_warehouse/Qwen3_omni"

USE_TRANSFORMERS = False
TRANSFORMERS_USE_FLASH_ATTN2 = True
USE_AUDIO_IN_VIDEO = True
RETURN_AUDIO = False


_MODEL_CACHE = None

def _load_model_processor():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
        
    if USE_TRANSFORMERS:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        if TRANSFORMERS_USE_FLASH_ATTN2:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                dtype='auto',
                attn_implementation='flash_attention_2',
                device_map="auto",
            )
        else:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_PATH, device_map="auto", dtype='auto',
            )
    else:
        from vllm import LLM
        os.environ['VLLM_USE_V1'] = '0'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        model = LLM(
            model=MODEL_PATH, 
            trust_remote_code=True, 
            gpu_memory_utilization=0.95,
            tensor_parallel_size=1,
            limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_PATH,
    )
    _MODEL_CACHE = (model, processor)
    return model, processor


def run_model(model, processor, messages, return_audio, use_audio_in_video):
    if USE_TRANSFORMERS:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids, audio = model.generate(
            **inputs, 
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=8192, 
            thinker_do_sample=False,
            speaker="Ethan", 
            use_audio_in_video=use_audio_in_video,
            return_audio=return_audio
        )
        response = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        if audio is not None:
            audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
        return response, audio
    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = {
            'prompt': text, 
            'multi_modal_data': {}, 
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video}
        }
        if images is not None: inputs['multi_modal_data']['image'] = images
        if videos is not None: inputs['multi_modal_data']['video'] = videos
        if audios is not None: inputs['multi_modal_data']['audio'] = audios
        outputs = model.generate(inputs, sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        return response, None


def resolve_audio_path(metadata_path: Path, raw_path: str) -> Path:
    raw = str(raw_path).strip()
    if not raw:
        raise ValueError("Empty audio path")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (metadata_path.parent / candidate).resolve()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_system_prompt(sample: Dict) -> str:
    return normalize_text(sample.get("system_prompt", sample.get("system", "")))


def extract_query(sample: Dict) -> str:
    return normalize_text(sample.get("query", ""))


def run_single_inference(model, processor, sample: Dict, audio_path: Path) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    system_prompt = extract_system_prompt(sample)
    query = extract_query(sample)
    
    content = [{"type": "audio", "audio": str(audio_path)}]
    if query:
        content.append({"type": "text", "text": query})
        
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    
    response, _ = run_model(model, processor, messages, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    return response


def run_multiturn_inference(
    model, processor,
    sample: Dict,
    metadata_path: Path,
    turn_audio_fields: List[str],
) -> str:
    system_prompt = extract_system_prompt(sample)
    query = extract_query(sample)
    
    audio_paths = []
    for field in turn_audio_fields:
        raw_audio = normalize_text(sample.get(field, ""))
        if not raw_audio:
            raise ValueError(f"Missing audio field in multiturn sample: {field}")
        audio_path = resolve_audio_path(metadata_path, raw_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_paths.append(str(audio_path))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    last_reply = ""
    for i, ap in enumerate(audio_paths):
        content = [{"type": "audio", "audio": ap}]
        if query:
            content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})
        
        last_reply, _ = run_model(model, processor, messages, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        if i < len(audio_paths) - 1:
            messages.append({"role": "assistant", "content": last_reply})
            
    return last_reply


def run_conversations_inference(model, processor, sample: Dict, metadata_path: Path) -> str:
    system_prompt = extract_system_prompt(sample)
    convs = sample.get("conversations", [])
    if not isinstance(convs, list) or not convs:
        raise ValueError("Invalid or empty conversations field")

    last_assistant_idx = -1
    for i in range(len(convs) - 1, -1, -1):
        if isinstance(convs[i], dict):
            role = normalize_text(convs[i].get("from", convs[i].get("role", ""))).lower()
            if role == "assistant":
                last_assistant_idx = i
                break

    audio_path: Optional[Path] = None
    for conv in convs:
        if not isinstance(conv, dict):
            continue
        role = normalize_text(conv.get("from", conv.get("role", ""))).lower()
        raw_audio = normalize_text(conv.get("audio_path", ""))
        if role == "user" and raw_audio:
            audio_path = resolve_audio_path(metadata_path, raw_audio)
            break

    if audio_path is None:
        raise ValueError("No user audio turn found in conversations")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    stop_idx = last_assistant_idx if last_assistant_idx >= 0 else len(convs)
    user_audio_used = False
    
    for conv in convs[:stop_idx]:
        if not isinstance(conv, dict):
            continue
        role = normalize_text(conv.get("from", conv.get("role", ""))).lower()
        value = normalize_text(conv.get("value", conv.get("content", "")))
        
        if role == "user":
            if not user_audio_used:
                content = [{"type": "audio", "audio": str(audio_path)}]
                messages.append({"role": "user", "content": content})
                user_audio_used = True
            else:
                messages.append({"role": "user", "content": value})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": value})
        elif role == "tool":
            messages.append({"role": "user", "content": value})
            
    response, _ = run_model(model, processor, messages, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    return response


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_output_path(task_id: str) -> Path:
    return results_root() / task_id / "results.jsonl"


def sample_done(sample: Dict, output_fields: List[str]) -> bool:
    for field in output_fields:
        value = normalize_text(sample.get(field, ""))
        if not value or value.startswith("ERROR:"):
            return False
    return True


def merge_existing_results(data: List[Dict], existing: List[Dict], output_fields: List[str]) -> None:
    for idx, row in enumerate(data):
        if idx >= len(existing):
            break
        old = existing[idx]
        for field in output_fields:
            if field in old:
                row[field] = old[field]


def process_sample(
    model, processor,
    config: TaskConfig,
    sample: Dict,
    metadata_path: Path,
) -> Dict[str, str]:
    if config.mode == "single":
        assert config.audio_field
        audio_path = resolve_audio_path(metadata_path, normalize_text(sample.get(config.audio_field, "")))
        response = run_single_inference(model, processor, sample, audio_path)
        return {config.output_fields[0]: response}

    if config.mode == "pair":
        assert config.audio_to_output
        outputs: Dict[str, str] = {}
        for audio_field, output_field in config.audio_to_output.items():
            audio_path = resolve_audio_path(metadata_path, normalize_text(sample.get(audio_field, "")))
            outputs[output_field] = run_single_inference(model, processor, sample, audio_path)
        return outputs

    if config.mode == "multiturn":
        assert config.turn_audio_fields
        response = run_multiturn_inference(
            model=model,
            processor=processor,
            sample=sample,
            metadata_path=metadata_path,
            turn_audio_fields=config.turn_audio_fields,
        )
        return {config.output_fields[0]: response}

    if config.mode == "conversations":
        response = run_conversations_inference(
            model=model,
            processor=processor,
            sample=sample,
            metadata_path=metadata_path,
        )
        return {config.output_fields[0]: response}

    raise ValueError(f"Unsupported mode: {config.mode}")


def run_task(
    task_id: str,
    max_workers: int = MAX_WORKERS_DEFAULT,
    save_interval: int = SAVE_INTERVAL_DEFAULT,
    model_name: str = MODEL_NAME_DEFAULT,
) -> Dict[str, int]:
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task id: {task_id}")

    config = TASK_CONFIGS[task_id]
    metadata_path = repo_root() / config.dataset_relpath
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    output_path = build_output_path(task_id)
    data = read_jsonl(metadata_path)
    if output_path.exists():
        existing = read_jsonl(output_path)
        merge_existing_results(data, existing, config.output_fields or [])

    to_process = [idx for idx, item in enumerate(data) if not sample_done(item, config.output_fields or [])]
    total = len(data)
    done = total - len(to_process)

    print(f"\n{'=' * 80}")
    print(f"Task: {task_id}")
    print(f"Input: {metadata_path}")
    print(f"Output: {output_path}")
    print(f"Need processing: {len(to_process)}/{total}")
    print(f"{'=' * 80}\n")

    if not to_process:
        return {"total": total, "processed": 0, "done_before": done, "errors": 0}

    completed = 0
    errors = 0

    # Load model once, using global cache if possible
    model, processor = _load_model_processor()

    def _save() -> None:
        write_jsonl(output_path, data)

    # Process sequentially for vllm/transformers
    pbar = tqdm(total=len(to_process), desc=f"Processing {task_id}")
    for idx in to_process:
        try:
            outputs = process_sample(model, processor, config, data[idx], metadata_path)
            for key, value in outputs.items():
                data[idx][key] = value
        except Exception as exc:
            errors += 1
            for field in config.output_fields or []:
                current = normalize_text(data[idx].get(field, ""))
                if not current or current.startswith("ERROR:"):
                    data[idx][field] = f"ERROR: {exc}"
        finally:
            completed += 1
            pbar.update(1)
            if completed % save_interval == 0:
                _save()
    pbar.close()

    _save()
    return {
        "total": total,
        "processed": completed,
        "done_before": done,
        "errors": errors,
    }
