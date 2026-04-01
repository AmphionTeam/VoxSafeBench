import json
import os
import sys
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

MODEL_NAME_DEFAULT = "Mimo_audio"
SAVE_INTERVAL_DEFAULT = 10
MAX_WORKERS_DEFAULT = 1

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def model_root() -> Path:
    return repo_root() / "models" / "Mimo_audio"

def results_root() -> Path:
    return repo_root() / "results" / "Mimo_audio"

# Hardcoded model/tokenizer paths
MODEL_PATH = str(repo_root() / "model_warehouse" / "Mimo_audio" / "MiMo-Audio-7B-Instruct")
TOKENIZER_PATH = str(repo_root() / "model_warehouse" / "Mimo_audio" / "MiMo-Audio-Tokenizer")

_MODEL_CACHE = None

def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    repo_dir = str(repo_root())
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
        
        
    mimo_src_dir = str(repo_root() / "model_warehouse" / "Mimo_audio")
    if mimo_src_dir not in sys.path:
        sys.path.append(mimo_src_dir)
        
    from utils.src.mimo_audio.mimo_audio import MimoAudio
    
    print(f"Loading MimoAudio model from {MODEL_PATH}")
    model = MimoAudio(MODEL_PATH, TOKENIZER_PATH)
    _MODEL_CACHE = model
    return model


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


def run_single_inference(model, sample: Dict, audio_path: Path) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    query = extract_query(sample)
    system_prompt = extract_system_prompt(sample)

    prompt_text = query
    if system_prompt:
        prompt_text = f"{system_prompt}\n\n{query}" if query else system_prompt
        
    if prompt_text:
        return model.audio_understanding_sft(str(audio_path), prompt_text, thinking=False)
    else:
        return model.speech2text_dialogue_sft_multiturn([{"role": "user", "content": str(audio_path)}], thinking=False)


def run_multiturn_inference(
    model,
    sample: Dict,
    metadata_path: Path,
    turn_audio_fields: List[str],
) -> str:
    from utils.src.mimo_audio.process_speechdata import InputSegment
    from utils.src.mimo_audio.modeling_mimo_audio import MiMoStopper

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

    lm_prompt = []
    if system_prompt:
        lm_prompt += [
            InputSegment(text="<|im_start|>system\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token),
            InputSegment(text=system_prompt, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token),
            InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token)
        ]

    last_reply = ""
    for i, ap in enumerate(audio_paths):
        # 1) Append user turn
        lm_prompt.append(InputSegment(text="<|im_start|>user\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
        lm_prompt.append(InputSegment(audio=model.preprocess_input(ap), speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
        if query:
            lm_prompt.append(InputSegment(text=query, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
        lm_prompt.append(InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
        
        # 2) Append assistant trigger for generation
        lm_prompt.append(InputSegment(text="<|im_start|>assistant\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
        lm_prompt.append(InputSegment(text="<think>\n\n</think>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
        
        # 3) Generate response for the current context
        input_ids = model.get_input_ids(lm_prompt)
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[model.tokenizer.eos_token_id, model.im_end_idx],
                group_size=model.group_size,
                audio_channels=model.audio_channels,
            )
        ]
        last_reply = model.forward(input_ids, stopping_criteria=stopping_criteria, task_name="spoken_dialogue")
        
        # 4) Remove the generation trigger ("<|im_start|>assistant\n", "<think>...") 
        # and instead explicitly append the generated text, so we can build upon it for the next loop.
        lm_prompt = lm_prompt[:-2]
        if i < len(audio_paths) - 1:
            lm_prompt.append(InputSegment(text="<|im_start|>assistant\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            lm_prompt.append(InputSegment(text=last_reply, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            lm_prompt.append(InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            
    return last_reply


def run_conversations_inference(model, sample: Dict, metadata_path: Path) -> str:
    from utils.src.mimo_audio.process_speechdata import InputSegment
    from utils.src.mimo_audio.modeling_mimo_audio import MiMoStopper

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

    stop_idx = last_assistant_idx if last_assistant_idx >= 0 else len(convs)

    lm_prompt = []
    if system_prompt:
        lm_prompt += [
            InputSegment(text="<|im_start|>system\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token),
            InputSegment(text=system_prompt, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token),
            InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token)
        ]

    is_first_user_turn = True
    for conv in convs[:stop_idx]:
        if not isinstance(conv, dict):
            continue
        role = normalize_text(conv.get("from", conv.get("role", ""))).lower()
        value = normalize_text(conv.get("value", conv.get("content", "")))
        
        if role == "user":
            raw_audio = normalize_text(conv.get("audio_path", ""))
            lm_prompt.append(InputSegment(text="<|im_start|>user\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            
            if raw_audio:
                audio_path = resolve_audio_path(metadata_path, raw_audio)
                lm_prompt.append(InputSegment(audio=model.preprocess_input(str(audio_path)), speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            
            if value and not is_first_user_turn:
                lm_prompt.append(InputSegment(text=value, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
                
            lm_prompt.append(InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            is_first_user_turn = False
            
        elif role == "assistant":
            lm_prompt.append(InputSegment(text="<|im_start|>assistant\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            lm_prompt.append(InputSegment(text=value, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            lm_prompt.append(InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            
        elif role == "tool":
            lm_prompt.append(InputSegment(text="<|im_start|>user\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            lm_prompt.append(InputSegment(text=value, speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))
            lm_prompt.append(InputSegment(text="<|im_end|>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token))

    # Append generation trigger for the target assistant reply
    lm_prompt += [
        InputSegment(text="<|im_start|>assistant\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token),
        InputSegment(text="<think>\n\n</think>\n", speech_zeroemb_idx=model.speech_zeroemb_idx, text_zeroemb_idx=model.empty_token)
    ]
    
    input_ids = model.get_input_ids(lm_prompt)
    stopping_criteria = [
        MiMoStopper(
            stop_tokens=[model.tokenizer.eos_token_id, model.im_end_idx],
            group_size=model.group_size,
            audio_channels=model.audio_channels,
        )
    ]
    response = model.forward(input_ids, stopping_criteria=stopping_criteria, task_name="spoken_dialogue")
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
    model,
    config: TaskConfig,
    sample: Dict,
    metadata_path: Path,
) -> Dict[str, str]:
    if config.mode == "single":
        assert config.audio_field
        audio_path = resolve_audio_path(metadata_path, normalize_text(sample.get(config.audio_field, "")))
        response = run_single_inference(model, sample, audio_path)
        return {config.output_fields[0]: response}

    if config.mode == "pair":
        assert config.audio_to_output
        outputs: Dict[str, str] = {}
        for audio_field, output_field in config.audio_to_output.items():
            audio_path = resolve_audio_path(metadata_path, normalize_text(sample.get(audio_field, "")))
            outputs[output_field] = run_single_inference(model, sample, audio_path)
        return outputs

    if config.mode == "multiturn":
        assert config.turn_audio_fields
        response = run_multiturn_inference(
            model=model,
            sample=sample,
            metadata_path=metadata_path,
            turn_audio_fields=config.turn_audio_fields,
        )
        return {config.output_fields[0]: response}

    if config.mode == "conversations":
        response = run_conversations_inference(
            model=model,
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

    model = _load_model()

    def _save() -> None:
        write_jsonl(output_path, data)

    pbar = tqdm(total=len(to_process), desc=f"Processing {task_id}")
    for idx in to_process:
        try:
            outputs = process_sample(model, config, data[idx], metadata_path)
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
