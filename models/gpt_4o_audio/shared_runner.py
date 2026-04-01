import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    from .task_configs import TASK_CONFIGS, TASK_IDS, TaskConfig
except ImportError:
    from task_configs import TASK_CONFIGS, TASK_IDS, TaskConfig

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


MODEL_NAME_DEFAULT = "gpt-4o-audio-preview-2025-06-03"
SAVE_INTERVAL_DEFAULT = 10
MAX_WORKERS_DEFAULT = 4


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_root() -> Path:
    return repo_root() / "models" / "gpt_4o_audio"


def results_root() -> Path:
    return repo_root() / "results" / "gpt_4o_audio"


def load_dotenv_if_present() -> None:
    env_path = repo_root() / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_api_key() -> None:
    load_dotenv_if_present()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is missing. Please set it in .env or environment variables.")


def ensure_openai_package() -> None:
    if OpenAI is None:
        raise ImportError("openai package is missing. Install it with: pip install openai")


def detect_audio_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".wav":
        return "wav"
    if ext == ".mp3":
        return "mp3"
    raise ValueError(f"Unsupported audio format: {ext}")


def b64encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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


def make_user_content(audio_path: Path, query: str) -> List[Dict]:
    audio_format = detect_audio_format(audio_path)
    audio_b64 = b64encode_file(audio_path)
    content: List[Dict] = []
    if query:
        content.append({"type": "text", "text": query})
    content.append(
        {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": audio_format},
        }
    )
    return content


def get_text_from_response(response) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join([p for p in parts if p]).strip()
    if content is None:
        return ""
    return str(content).strip()


def run_single_inference(client: OpenAI, sample: Dict, audio_path: Path, model_name: str) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    system_prompt = extract_system_prompt(sample)
    query = extract_query(sample)
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": make_user_content(audio_path, query)})
    response = client.chat.completions.create(
        model=model_name,
        modalities=["text"],
        messages=messages,
    )
    return get_text_from_response(response)


def run_multiturn_inference(
    client: OpenAI,
    sample: Dict,
    metadata_path: Path,
    turn_audio_fields: List[str],
    model_name: str,
) -> str:
    system_prompt = extract_system_prompt(sample)
    query = extract_query(sample)
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    last_reply = ""
    for field in turn_audio_fields:
        raw_audio = normalize_text(sample.get(field, ""))
        if not raw_audio:
            raise ValueError(f"Missing audio field in multiturn sample: {field}")
        audio_path = resolve_audio_path(metadata_path, raw_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        messages.append({"role": "user", "content": make_user_content(audio_path, query)})
        response = client.chat.completions.create(
            model=model_name,
            modalities=["text"],
            messages=messages,
        )
        last_reply = get_text_from_response(response)
        messages.append({"role": "assistant", "content": last_reply})
    return last_reply


def run_conversations_inference(client: OpenAI, sample: Dict, metadata_path: Path, model_name: str) -> str:
    """Handle both direct harm (single-turn) and indirect harm (multi-turn with tool history).

    Direct harm:   conversations = [user(audio), assistant(target)]
    Indirect harm: conversations = [user(audio), assistant, tool, assistant, ..., assistant(target)]

    In both cases we find the last assistant turn (the response to generate), replay all
    preceding turns as context, then call the model once to produce that final reply.
    User turns use audio; intermediate assistant turns use their text value; tool turns
    are forwarded as user-role text messages (matching the reference implementation).
    """
    system_prompt = extract_system_prompt(sample)
    convs = sample.get("conversations", [])
    if not isinstance(convs, list):
        raise ValueError("Invalid conversations field")
    if not convs:
        raise ValueError("Empty conversations field")

    # Find the last assistant turn — this is what we ask the model to generate.
    last_assistant_idx = -1
    for i in range(len(convs) - 1, -1, -1):
        if isinstance(convs[i], dict):
            role = normalize_text(convs[i].get("from", convs[i].get("role", ""))).lower()
            if role == "assistant":
                last_assistant_idx = i
                break

    # Resolve audio path from the first user turn.
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

    query = extract_query(sample)

    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Replay conversation history up to (but not including) the last assistant turn.
    stop_idx = last_assistant_idx if last_assistant_idx >= 0 else len(convs)
    user_audio_used = False
    for conv in convs[:stop_idx]:
        if not isinstance(conv, dict):
            continue
        role = normalize_text(conv.get("from", conv.get("role", ""))).lower()
        value = normalize_text(conv.get("value", conv.get("content", "")))

        if role == "user":
            # The first user turn carries the audio; subsequent user turns (rare) use text.
            if not user_audio_used:
                messages.append({"role": "user", "content": make_user_content(audio_path, "")})
                user_audio_used = True
            else:
                messages.append({"role": "user", "content": value})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": value})
        elif role == "tool":
            # Tool results are forwarded as user-role messages.
            messages.append({"role": "user", "content": value})

    response = client.chat.completions.create(
        model=model_name,
        modalities=["text"],
        messages=messages,
    )
    return get_text_from_response(response)


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
    client: OpenAI,
    config: TaskConfig,
    sample: Dict,
    metadata_path: Path,
    model_name: str,
) -> Dict[str, str]:
    if config.mode == "single":
        assert config.audio_field
        audio_path = resolve_audio_path(metadata_path, normalize_text(sample.get(config.audio_field, "")))
        response = run_single_inference(client, sample, audio_path, model_name)
        return {config.output_fields[0]: response}

    if config.mode == "pair":
        assert config.audio_to_output
        outputs: Dict[str, str] = {}
        for audio_field, output_field in config.audio_to_output.items():
            audio_path = resolve_audio_path(metadata_path, normalize_text(sample.get(audio_field, "")))
            outputs[output_field] = run_single_inference(client, sample, audio_path, model_name)
        return outputs

    if config.mode == "multiturn":
        assert config.turn_audio_fields
        response = run_multiturn_inference(
            client=client,
            sample=sample,
            metadata_path=metadata_path,
            turn_audio_fields=config.turn_audio_fields,
            model_name=model_name,
        )
        return {config.output_fields[0]: response}

    if config.mode == "conversations":
        response = run_conversations_inference(
            client=client,
            sample=sample,
            metadata_path=metadata_path,
            model_name=model_name,
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
    ensure_openai_package()
    ensure_api_key()

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

    lock = Lock()
    completed = 0
    errors = 0
    client = OpenAI()

    def _save() -> None:
        with lock:
            write_jsonl(output_path, data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_sample, client, config, data[idx], metadata_path, model_name): idx
            for idx in to_process
        }
        pbar = tqdm(total=len(to_process), desc=f"Processing {task_id}")
        for future in as_completed(futures):
            idx = futures[future]
            try:
                outputs = future.result()
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
