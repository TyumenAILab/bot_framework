from __future__ import annotations

import math
import os
import ssl
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(default="", repr=False)
    mode: str = "chat"
    timeout: float = 300.0
    max_retries: int = 2
    poll_interval: float = 1.0
    verify_ssl: bool = True
    ca_bundle: str | None = None
    trust_env: bool = True
    request_interval: float = 0.0
    supports_streaming: bool = True

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials")
        if url.query or url.fragment:
            raise ValueError("base_url must not contain query parameters or fragments")
        if not self.model.strip():
            raise ValueError("A served model name is required")
        if self.mode not in {"chat", "async"} or (self.mode == "async" and self.provider != "302ai"):
            raise ValueError("async mode is supported only for 302ai")
        if self.provider == "302ai" and not self.api_key:
            raise ValueError("API_302AI_KEY is required for 302ai")
        for name in ("timeout", "poll_interval", "request_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name != "request_interval" and value == 0):
                raise ValueError(f"Invalid {name}")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if any(c in self.api_key for c in "\r\n"):
            raise ValueError("API key must not contain line breaks")

    @classmethod
    def from_env(cls, provider: str | None = None) -> LLMConfig:
        name = (provider or os.getenv("LLM_PROVIDER", "vllm")).lower()
        name = {"302": "302ai", "local_hub": "hub", "vlm": "vllm"}.get(name, name)
        if name == "302ai":
            model = os.getenv("A302_MODEL_NAME", "gpt-4o")
            base = os.getenv("A302_API_BASE", "https://api.302.ai/v1")
            key = os.getenv("API_302AI_KEY", "")
        elif name == "hub":
            model = os.getenv("LOCAL_HUB_MODEL_NAME", "")
            base = os.getenv("LOCAL_HUB_API_BASE", "")
            key = os.getenv("LOCAL_HUB_API_KEY", "")
        elif name in {"vllm", "openai_compatible"}:
            model = os.getenv("VLLM_MODEL") or os.getenv("LOCAL_HUB_MODEL_NAME", "")
            base = os.getenv("VLLM_BASE_URL") or os.getenv("LOCAL_HUB_API_BASE", "http://localhost:8000/v1")
            key = os.getenv("VLLM_API_KEY") or os.getenv("LOCAL_HUB_API_KEY", "")
        else:
            raise ValueError(f"Unsupported LLM provider: {name}")
        # Bare hosts from the old bots are supported; explicit custom paths are preserved.
        base = base.rstrip("/")
        if not urlsplit(base).path:
            base += "/v1"
        return cls(
            provider=name,
            model=model,
            base_url=base,
            api_key=key,
            mode=os.getenv("A302_MODE", "chat") if name == "302ai" else "chat",
            timeout=float(os.getenv("LLM_TIMEOUT", "300")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            verify_ssl=_env_bool("LOCAL_HUB_VERIFY_SSL" if name == "hub" else "LLM_VERIFY_SSL", True),
            ca_bundle=os.getenv("LOCAL_HUB_CA_BUNDLE" if name == "hub" else "LLM_CA_BUNDLE") or None,
            trust_env=_env_bool("LLM_TRUST_ENV", True),
            request_interval=float(
                os.getenv(
                    "LOCAL_HUB_REQUEST_INTERVAL" if name == "hub" else "LLM_REQUEST_INTERVAL",
                    "2" if name == "hub" else "0",
                )
            ),
            supports_streaming=_env_bool(
                "LOCAL_HUB_STREAMING" if name == "hub" else "LLM_STREAMING", name != "hub"
            ),
        )

    def tls_context(self) -> bool | ssl.SSLContext:
        if self.ca_bundle:
            return ssl.create_default_context(cafile=self.ca_bundle)
        return self.verify_ssl


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")
