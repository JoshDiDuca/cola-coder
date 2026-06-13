"""Knowledge-distillation support for cola-coder.

Lets a stronger TEACHER model (local Qwen/DeepSeek via Ollama/llama.cpp/vLLM, or a
cloud API like DeepSeek/OpenAI/OpenRouter) generate completions that the small
cola-coder STUDENT is then trained on (black-box / sequence-level distillation,
a.k.a. SeqKD).

Why black-box (text outputs) rather than white-box (logits)? The student uses a
custom BPE tokenizer that differs from any teacher's, so logit-level KD doesn't
line up token-for-token. Collecting the teacher's TEXT and SFT-ing the student on
it is tokenizer-agnostic and GPU-light — a cloud teacher uses ZERO local GPU, and a
local teacher runs on a spare GPU / CPU off the training GPU. This is the practical
enabler for on-policy distillation (backlog MODEL-024) on limited hardware.

Security: a teacher's generated code is UNTRUSTED — sandbox it (SandboxedRunner /
TscRunner) before any execution-based verification. When sending prompts to a CLOUD
teacher, secrets are redacted first (CredentialScanner) so the user's code can't
leak to an external API.
"""

from .teacher import (
    OpenAICompatibleTeacher,
    Teacher,
    TeacherError,
    build_teacher,
)

__all__ = [
    "Teacher",
    "TeacherError",
    "OpenAICompatibleTeacher",
    "build_teacher",
]
