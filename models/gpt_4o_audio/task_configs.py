from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


TaskMode = Literal["single", "pair", "multiturn", "conversations"]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    dataset_relpath: str
    mode: TaskMode
    audio_field: Optional[str] = None
    audio_to_output: Optional[Dict[str, str]] = None
    turn_audio_fields: Optional[List[str]] = None
    output_fields: Optional[List[str]] = None


TASK_CONFIGS: Dict[str, TaskConfig] = {
    # Safety tier2
    "Safety-tier2/Emotion": TaskConfig(
        task_id="Safety-tier2/Emotion",
        dataset_relpath="datasets/Safety-tier2/Emotion/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier2/Unsafe_ambient": TaskConfig(
        task_id="Safety-tier2/Unsafe_ambient",
        dataset_relpath="datasets/Safety-tier2/Unsafe_ambient/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier2/Symbolic_background": TaskConfig(
        task_id="Safety-tier2/Symbolic_background",
        dataset_relpath="datasets/Safety-tier2/Symbolic_background/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier2/Overlap_instruction_injection": TaskConfig(
        task_id="Safety-tier2/Overlap_instruction_injection",
        dataset_relpath="datasets/Safety-tier2/Overlap_instruction_injection/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier2/Impaired_capacity": TaskConfig(
        task_id="Safety-tier2/Impaired_capacity",
        dataset_relpath="datasets/Safety-tier2/Impaired_capacity/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier2/Child_voice": TaskConfig(
        task_id="Safety-tier2/Child_voice",
        dataset_relpath="datasets/Safety-tier2/Child_voice/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier2/Child_presence": TaskConfig(
        task_id="Safety-tier2/Child_presence",
        dataset_relpath="datasets/Safety-tier2/Child_presence/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    # Safety tier1
    "Safety-tier1/Agentic_Action_Risks": TaskConfig(
        task_id="Safety-tier1/Agentic_Action_Risks",
        dataset_relpath="datasets/Safety-tier1/Agentic_Action_Risks/metadata.jsonl",
        mode="conversations",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier1/Singleturn_jailbreak": TaskConfig(
        task_id="Safety-tier1/Singleturn_jailbreak",
        dataset_relpath="datasets/Safety-tier1/Singleturn_jailbreak/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier1/Multiturn_jailbreak": TaskConfig(
        task_id="Safety-tier1/Multiturn_jailbreak",
        dataset_relpath="datasets/Safety-tier1/Multiturn_jailbreak/metadata.jsonl",
        mode="multiturn",
        turn_audio_fields=[
            "turn1_audio_file_name",
            "turn2_audio_file_name",
            "turn3_audio_file_name",
        ],
        output_fields=["gpt_4o_audio"],
    ),
    "Safety-tier1/No_jailbreak": TaskConfig(
        task_id="Safety-tier1/No_jailbreak",
        dataset_relpath="datasets/Safety-tier1/No_jailbreak/metadata.jsonl",
        mode="pair",
        audio_to_output={
            "clean_audio_file_name": "gpt_4o_audio_clean",
            "diverse_audio_file_name": "gpt_4o_audio_diverse",
        },
        output_fields=["gpt_4o_audio_clean", "gpt_4o_audio_diverse"],
    ),
    # Fairness
    "Fairness-tier1/test": TaskConfig(
        task_id="Fairness-tier1/test",
        dataset_relpath="datasets/Fairness-tier1/test/metadata.jsonl",
        mode="pair",
        audio_to_output={
            "clean_audio_file_name": "gpt_4o_audio_clean",
            "diverse_audio_file_name": "gpt_4o_audio_diverse",
        },
        output_fields=["gpt_4o_audio_clean", "gpt_4o_audio_diverse"],
    ),
    "Fairness-tier2/test": TaskConfig(
        task_id="Fairness-tier2/test",
        dataset_relpath="datasets/Fairness-tier2/test/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
    "Fairness-tier2/Bias_analysis": TaskConfig(
        task_id="Fairness-tier2/Bias_analysis",
        dataset_relpath="datasets/Fairness-tier2/Bias_analysis/metadata.jsonl",
        mode="pair",
        audio_to_output={
            "audio_file_name": "gpt_4o_audio",
            "flipped_audio_file_name": "gpt_4o_audio_flipped",
        },
        output_fields=["gpt_4o_audio", "gpt_4o_audio_flipped"],
    ),
    # Privacy
    "Privacy-tier1/Hard_privacy": TaskConfig(
        task_id="Privacy-tier1/Hard_privacy",
        dataset_relpath="datasets/Privacy-tier1/Hard_privacy/metadata.jsonl",
        mode="pair",
        audio_to_output={
            "clean_audio_file_name": "gpt_4o_audio_clean",
            "diverse_audio_file_name": "gpt_4o_audio_diverse",
        },
        output_fields=["gpt_4o_audio_clean", "gpt_4o_audio_diverse"],
    ),
    "Privacy-tier1/Soft_privacy": TaskConfig(
        task_id="Privacy-tier1/Soft_privacy",
        dataset_relpath="datasets/Privacy-tier1/Soft_privacy/metadata.jsonl",
        mode="pair",
        audio_to_output={
            "clean_audio_file_name": "gpt_4o_audio_clean",
            "diverse_audio_file_name": "gpt_4o_audio_diverse",
        },
        output_fields=["gpt_4o_audio_clean", "gpt_4o_audio_diverse"],
    ),
    "Privacy-tier2/Interactional_privacy": TaskConfig(
        task_id="Privacy-tier2/Interactional_privacy",
        dataset_relpath="datasets/Privacy-tier2/Interactional_privacy/metadata.jsonl",
        mode="multiturn",
        turn_audio_fields=[
            "turn1_audio_file_name",
            "turn2_audio_file_name",
            "turn3_audio_file_name",
        ],
        output_fields=["gpt_4o_audio"],
    ),
    "Privacy-tier2/Audio_conditioned_privacy": TaskConfig(
        task_id="Privacy-tier2/Audio_conditioned_privacy",
        dataset_relpath="datasets/Privacy-tier2/Audio_conditioned_privacy/metadata.jsonl",
        mode="single",
        audio_field="audio_file_name",
        output_fields=["gpt_4o_audio"],
    ),
}


TASK_IDS: List[str] = sorted(TASK_CONFIGS.keys())
