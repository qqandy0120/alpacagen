from pathlib import Path

PROMPT_DIR = Path(__file__).parent

def get_prompt(language: str) -> str:
    """Load prompt template for specified language."""
    prompt_path = PROMPT_DIR / f"prompt_gen_{language}.txt"
    if not prompt_path.exists():
        raise ValueError(f"No prompt template found for language: {language}")
    return prompt_path.read_text(encoding='utf-8')

__all__ = ['get_prompt']