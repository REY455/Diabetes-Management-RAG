import os
import requests
from fastapi import HTTPException
from dotenv import load_dotenv

# ==============================
# 🔹 LOAD ENV
# ==============================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip()
FALLBACK_MODEL = "openai/gpt-4o-mini"

CHEAP_MODE = os.getenv("CHEAP_MODE", "false").lower() == "true"


# ==============================
# 🔥 MODEL ROUTER
# ==============================
def choose_model(query: str = "", context: str = ""):
    q = query.lower()

    if CHEAP_MODE:
        return FALLBACK_MODEL

    if any(word in q for word in [
        "analyze", "compare", "evaluate", "compliance",
        "legal", "risk", "difference", "pros and cons"
    ]):
        return "anthropic/claude-3.5-sonnet"

    if len(query) > 120 or len(context) > 1500:
        return "openai/gpt-4o"

    return DEFAULT_MODEL


# ==============================
# 🔥 CORE CALL (WITH FALLBACK)
# ==============================
def _call_openrouter(model_name, prompt):
    return requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI assistant. Answer clearly and accurately."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500
        },
        timeout=30
    )


# ==============================
# 🔥 MAIN FUNCTION
# ==============================
def ask_llm(
    prompt: str,
    query: str = "",
    context: str = "",
    model: str | None = None
):
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "Missing OPENROUTER_API_KEY")

    # 🔥 decide model
    model_name = model.strip() if model else choose_model(query, context)

    # 🔥 TRY PRIMARY MODEL
    try:
        response = _call_openrouter(model_name, prompt)
    except requests.exceptions.Timeout:
        raise HTTPException(500, "LLM request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(500, f"Request failed: {str(e)}")

    # 🔥 FALLBACK IF FAILED
    if response.status_code != 200:
        print(f"⚠️ Model failed: {model_name} → falling back")

        try:
            response = _call_openrouter(FALLBACK_MODEL, prompt)
            model_name = FALLBACK_MODEL
        except Exception as e:
            raise HTTPException(500, f"Fallback failed: {str(e)}")

        if response.status_code != 200:
            raise HTTPException(500, f"OpenRouter Error: {response.text}")

    # 🔥 PARSE JSON
    try:
        data = response.json()
    except Exception:
        raise HTTPException(500, "Invalid JSON response from LLM")

    # 🔥 SAFE CONTENT EXTRACTION
    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(500, f"Empty LLM response: {data}")

    result = choices[0]["message"].get("content", "").strip()

    if not result:
        result = "⚠️ Model returned empty response."

    # 🔥 TOKEN USAGE
    usage = data.get("usage", {})

    return {
        "response": result,
        "model": model_name,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }