import json
import os
import time
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
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]


MODEL_NAME_DEFAULT = "gemini-3-flash-preview"
SAVE_INTERVAL_DEFAULT = 10
MAX_WORKERS_DEFAULT = 4


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_root() -> Path:
    return repo_root() / "models" / "gemini-3"


def results_root() -> Path:
    return repo_root() / "results" / "gemini_3_flash"


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
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise ValueError("GEMINI_API_KEY is missing. Please set it in .env or environment variables.")


def ensure_genai_package() -> None:
    if genai is None:
        raise ImportError("google-genai package is missing. Install it with: pip install google-genai")


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


def upload_audio_file(client: genai.Client, audio_path: Path) -> object:
    with open(audio_path, "rb") as f:
        audio_file = client.files.upload(file=f, config={'mime_type': 'audio/mp3'})
    while audio_file.state.name == "PROCESSING":
        time.sleep(1)
        audio_file = client.files.get(name=audio_file.name)
    if audio_file.state.name == "FAILED":
        raise Exception(f"Audio processing failed for {audio_path}")
    return audio_file


def cleanup_uploaded_files(client: genai.Client, uploaded_files: List[object]) -> None:
    for uploaded_file in uploaded_files:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass


def get_text_from_response(response) -> str:
    if not response or not hasattr(response, 'text'):
        return ""
    return str(response.text).strip()


def run_single_inference(client: genai.Client, sample: Dict, audio_path: Path, model_name: str) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    system_prompt = extract_system_prompt(sample)
    query = extract_query(sample)
    uploaded_files = []
    
    try:
        audio_file = upload_audio_file(client, audio_path)
        uploaded_files.append(audio_file)
        
        generate_content_config = types.GenerateContentConfig()
        if system_prompt:
            generate_content_config.system_instruction = [
                types.Part.from_text(text=system_prompt)
            ]
            
        parts = [types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type)]
        if query:
            parts.append(types.Part.from_text(text=query))
            
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=parts)],
            config=generate_content_config,
        )
        return get_text_from_response(response)
    finally:
        cleanup_uploaded_files(client, uploaded_files)


def run_multiturn_inference(
    client: genai.Client,
    sample: Dict,
    metadata_path: Path,
    turn_audio_fields: List[str],
    model_name: str,
) -> str:
    system_prompt = extract_system_prompt(sample)
    query = extract_query(sample)
    uploaded_files = []
    
    try:
        audio_files = []
        for field in turn_audio_fields:
            raw_audio = normalize_text(sample.get(field, ""))
            if not raw_audio:
                raise ValueError(f"Missing audio field in multiturn sample: {field}")
            audio_path = resolve_audio_path(metadata_path, raw_audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            af = upload_audio_file(client, audio_path)
            uploaded_files.append(af)
            audio_files.append(af)
            
        generate_content_config = types.GenerateContentConfig()
        if system_prompt:
            generate_content_config.system_instruction = [
                types.Part.from_text(text=system_prompt)
            ]
            
        chat = client.chats.create(
            model=model_name,
            config=generate_content_config
        )
        
        last_reply = ""
        for i, af in enumerate(audio_files):
            parts = [types.Part.from_uri(file_uri=af.uri, mime_type=af.mime_type)]
            # optionally append query if present (or just for the first turn if preferred, 
            # here we follow original which seemed to attach query to all user messages)
            if query:
                parts.append(types.Part.from_text(text=query))
                
            response = chat.send_message(parts)
            last_reply = get_text_from_response(response)
            
        return last_reply
    finally:
        cleanup_uploaded_files(client, uploaded_files)


def run_conversations_inference(client: genai.Client, sample: Dict, metadata_path: Path, model_name: str) -> str:
    system_prompt = extract_system_prompt(sample)
    convs = sample.get("conversations", [])
    if not isinstance(convs, list):
        raise ValueError("Invalid conversations field")
    if not convs:
        raise ValueError("Empty conversations field")

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

    uploaded_files = []
    contents = []
    
    try:
        audio_file = upload_audio_file(client, audio_path)
        uploaded_files.append(audio_file)
        
        stop_idx = last_assistant_idx if last_assistant_idx >= 0 else len(convs)
        user_audio_used = False
        
        for conv in convs[:stop_idx]:
            if not isinstance(conv, dict):
                continue
            role = normalize_text(conv.get("from", conv.get("role", ""))).lower()
            value = normalize_text(conv.get("value", conv.get("content", "")))
            
            if role == "user":
                if not user_audio_used:
                    parts = [types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type)]
                    contents.append(types.Content(role="user", parts=parts))
                    user_audio_used = True
                else:
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=value)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=value)]))
            elif role == "tool":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=value)]))
                
        generate_content_config = types.GenerateContentConfig()
        if system_prompt:
            generate_content_config.system_instruction = [
                types.Part.from_text(text=system_prompt)
            ]
            
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        return get_text_from_response(response)
    finally:
        cleanup_uploaded_files(client, uploaded_files)


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
    client: genai.Client,
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
    ensure_genai_package()
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
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
