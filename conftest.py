"""Keep tests hermetic: no model downloads, no real LLM."""
import os

os.environ.setdefault("SIMILARITY_BACKEND", "lexical")
for k in ("GEMINI_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "TESTSET_PATH"):
    os.environ.pop(k, None)
