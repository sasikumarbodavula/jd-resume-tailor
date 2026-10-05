"""Thin Gemini wrapper used by the two LLM nodes (parse_jd, rewrite_content)."""
import os
from typing import Optional, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app import config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce schema-valid output after the allowed retry."""


def _friendly_api_error(message: str) -> Optional[str]:
    low = message.lower()
    if "429" in low or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return ("Gemini rate limit / quota reached (free tier allows only a few requests per minute). "
                "Wait about a minute and try again, or set a different GEMINI_MODEL in .env.")
    if "api key not valid" in low or "api_key_invalid" in low:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env."
    if ("404" in low or "not found" in low) and "model" in low:
        return "Gemini model not found. Set a valid GEMINI_MODEL in .env (see ai.google.dev/gemini-api/docs/models)."
    return None


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_google_genai import ChatGoogleGenerativeAI  # imported lazily so tests need no key

    return ChatGoogleGenerativeAI(model=config.gemini_model(), temperature=0, google_api_key=api_key)


def invoke_structured(schema: Type[T], prompt: str, system: Optional[str] = None) -> T:
    """Call Gemini with `with_structured_output(schema)`.

    Pydantic validates the response. On failure, retry ONCE with the error appended to the prompt;
    on a second failure raise LLMError (never continue silently).
    """
    structured_llm = get_llm().with_structured_output(schema)
    current_prompt = prompt
    last_error: Optional[Exception] = None
    for _attempt in range(2):
        messages = ([SystemMessage(content=system)] if system else []) + [HumanMessage(content=current_prompt)]
        try:
            result = structured_llm.invoke(messages)
            if result is None:
                raise ValueError("model returned no structured output")
            if isinstance(result, dict):
                result = schema.model_validate(result)
            return result
        except Exception as exc:  # validation / parsing / empty output
            friendly = _friendly_api_error(str(exc))
            if friendly:  # quota / bad key / unknown model: retrying cannot help
                raise LLMError(friendly) from exc
            last_error = exc
            current_prompt = (f"{prompt}\n\nYour previous response was invalid: {exc}\n"
                              "Return only output that matches the schema exactly.")
    raise LLMError(f"Gemini did not return valid {schema.__name__} after 2 attempts: {last_error}")
