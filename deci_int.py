# Author: Amitesh Jha | iSoft | 2025-10-07 (Refactored: Gemini)
# Streamlit + LangChain RAG app — CPU-safe embeddings + Anthropic proxies-proof init.

from __future__ import annotations
import os, glob, time, base64, hashlib, logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
import pandas as pd

# --- Torch / device hygiene to avoid meta-tensor issues ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # force no CUDA
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Quiet Chroma / disable OTel
os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "none")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
logging.getLogger("chromadb").setLevel(logging.WARNING)

# LangChain bits
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    from langchain_community.llms import Ollama as ChatOllama

# Anthropic SDK (new & old)
try:
    from anthropic import Anthropic as _AnthropicClientNew
except Exception:
    _AnthropicClientNew = None
try:
    from anthropic import Client as _AnthropicClientOld
except Exception:
    _AnthropicClientOld = None

# Constants & Settings
DEFAULT_OLLAMA = "llama3.2"
DEFAULT_CLAUDE = "claude-sonnet-4-5"
_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_MODEL_KW = {
    "device": "cpu",
    "trust_remote_code": False,
}
_ENCODE_KW = {
    "normalize_embeddings": True,
}
TEXT_EXTS = {".txt", ".md", ".rtf", ".html", ".htm", ".json", ".xml"}
DOC_EXTS  = {".pdf", ".docx", ".csv", ".tsv", ".xlsx", ".xlsm", ".xltx", ".pptx"}
SUPPORTED_EXTS = TEXT_EXTS | DOC_EXTS

GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|hola|namaste|hiya|hi there|hello there|"
    r"good\s+(morning|afternoon|evening))[\s!,.?]*$",
    re.IGNORECASE,
)

# --------------------- Minimal direct Claude model (bypass proxies kw path) ---------------------

class ClaudeDirect(BaseChatModel):
    model: str = DEFAULT_CLAUDE     # valid model id
    temperature: float = 0.2
    max_tokens: int = 800
    _client: object = None  # Anthropic client set at init

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_client", client)

    @property
    def _llm_type(self) -> str:
        return "claude_direct"

    def _convert_msgs(self, messages: list[BaseMessage]):
        out = []
        for m in messages:
            role = "user" if m.type == "human" else ("assistant" if m.type == "ai" else "user")
            if isinstance(m.content, str):
                text = m.content
            else:
                parts = m.content or []
                text = "".join(p.get("text","") if isinstance(p, dict) else str(p) for p in parts)
            out.append({"role": role, "content": [{"type": "text", "text": text}]})
        return out

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        amsgs = self._convert_msgs(messages)
        resp = self._client.messages.create(
            model=self.model,
            messages=amsgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = ""
        content = getattr(resp, "content", []) or []
        for blk in content:
            if getattr(blk, "type", None) == "text":
                text += getattr(blk, "text", "") or ""
            elif isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "") or ""
        ai = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai)])


def build_citation_block(source_docs: List[Document], kb_root: str | None = None) -> str:
    names = []
    from collections import Counter

    for d in source_docs or []:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "unknown")

        try:
            if kb_root:
                rel = Path(src).resolve().relative_to(Path(kb_root).resolve())
                display = str(rel)
            else:
                display = Path(src).name
        except Exception:
            display = Path(src).name

        names.append(display)

    if not names:
        return ""

    counts = Counter(names)
    lines = [f"- {name}" + (f" ×{n}" if n > 1 else "") for name, n in counts.items()]
    return "\n\n**Sources**\n" + "\n".join(lines)

# --------------------- UI / THEME (no changes needed) ---------------------
st.set_page_config(page_title="LLM Chat • LangChain RAG", page_icon="💬", layout="wide")

def _resolve_logo_path() -> Optional[Path]:
    env_logo = os.getenv("ISOFT_LOGO_PATH")
    candidates = [Path.cwd() / "assets" / "isoft_logo.png",
                  Path(env_logo).expanduser().resolve() if env_logo else None]
    for p in candidates:
        if p and p.exists():
            return p
    return None

def _resolve_avatar_paths() -> Tuple[Optional[Path], Optional[Path]]:
    env_user = os.getenv("USER_AVATAR_PATH")
    env_asst = os.getenv("ASSISTANT_AVATAR_PATH")
    user_candidates = [Path.cwd() / "assets" / "avatar.png",
                       Path(env_user).expanduser().resolve() if env_user else None]
    asst_candidates = [Path.cwd() / "assets" / "llm.png",
                       Path(env_asst).expanduser().resolve() if env_asst else None]
    user = next((p for p in user_candidates if p and p.exists()), None)
    asst = next((p for p in asst_candidates if p and p.exists()), None)
    return user, asst

def _img_to_data_uri(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = (path.suffix.lower().lstrip(".") or "png")
    mime = "image/png" if ext in ("png", "apng") else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/svg+xml")
    return f"data:{mime};base64,{b64}"

USER_AVATAR_PATH, ASSIST_AVATAR_PATH = _resolve_avatar_paths()
USER_AVATAR_URI = _img_to_data_uri(USER_AVATAR_PATH)
ASSIST_AVATAR_URI = _img_to_data_uri(ASSIST_AVATAR_PATH)

user_bg  = f"background-image:url('{USER_AVATAR_URI}');" if USER_AVATAR_URI else ""
asst_bg  = f"background-image:url('{ASSIST_AVATAR_URI}');" if ASSIST_AVATAR_URI else ""

css = """
<style>
:root{{ 
  --bg:#f7f8fb; --sidebar-bg:#f5f7fb; --panel:#fff; --text:#0b1220;
  --muted:#5d6b82; --accent:#2563eb; --border:#e7eaf2;
  --bubble-user:#eef4ff; --bubble-assist:#f6f7fb;
}}
html, body, [data-testid="stAppViewContainer"]{{ background:var(--bg); color:var(--text); }}
section[data-testid="stSidebar"]{{ background:var(--sidebar-bg); border-right:1px solid var(--border); }}
main .block-container{{ padding-top:.6rem; }}
.container-narrow{{ max-width:1080px; margin:0 auto; }}
.chat-card{{ background:var(--panel); border:1px solid var(--border); border-radius:14px; box-shadow:0 6px 16px rgba(16,24,40,.05); overflow:hidden; }}
.chat-scroll{{ max-height: 75vh; overflow:auto; padding:.65rem .9rem; }}
.msg{{ display:flex; align-items:flex-start; gap:.65rem; margin:.45rem 0; }}
.avatar{{ width:32px; height:32px; border-radius:50%; border:1px solid var(--border); background-size:cover; background-position:center; background-repeat:no-repeat; flex:0 0 32px; }}
.avatar.user {{
  {user_bg}
}}
.avatar.assistant {{
  {asst_bg}
}}
.bubble{{ border:1px solid var(--border); background:var(--bubble-assist); padding:.8rem .95rem; border-radius:12px; max-width:860px; white-space:pre-wrap; line-height:1.45; }}
.msg.user .bubble{{ background:var(--bubble-user); }}
.composer{{ padding:.6rem .75rem; border-top:1px solid var(--border); background:#fff; position:sticky; bottom:0; z-index:2; }}
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px; padding:.5rem .
