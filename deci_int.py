# Author: Amitesh Jha | iSoft | 2025-10-07 (Refactored: Gemini)
# Streamlit + LangChain RAG app — CPU-safe embeddings + Anthropic proxies-proof init.

from __future__ import annotations
import os, glob, time, base64, hashlib, logging
import re # Added for greeting short-circuit
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
    # ... (rest of ClaudeDirect class remains the same)
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
                # Simple handling for content list (assuming text parts)
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
            # new SDK returns objects with .type/.text; some environments return dicts
            if getattr(blk, "type", None) == "text":
                text += getattr(blk, "text", "") or ""
            elif isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "") or ""
        ai = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai)])


def build_citation_block(source_docs: List[Document], kb_root: str | None = None) -> str:
    # ... (rest of build_citation_block remains the same)
    names = []
    from collections import Counter # Moved import inside function for scope

    for d in source_docs or []:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "unknown")

        # Prefer path relative to KB; otherwise, just the filename
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

# (Avatar/CSS functions remain here)
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

# Resolve avatars before CSS
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
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px; padding:.5rem .7rem; font-size:.9rem; color:#111827; margin:.5rem 0 .8rem; }}
.smallcaps{{ font-variant: all-small-caps; color:#475569; }}
</style>
""".format(user_bg=user_bg, asst_bg=asst_bg)

st.markdown(css, unsafe_allow_html=True)

# --------------------- Helpers (minimal changes) ---------------------

def get_kb_dir() -> str:
    kb = os.path.abspath(os.path.join(".", "KB"))
    os.makedirs(kb, exist_ok=True)
    return kb

def human_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.2f} s"

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def iter_files(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(list(set(paths)))

def compute_kb_signature(folder: str) -> Tuple[str, int]:
    files = iter_files(folder)
    lines = []
    base = os.path.abspath(folder)
    for p in files:
        try:
            stt = os.stat(p)
            rel = os.path.relpath(os.path.abspath(p), base)
            lines.append(f"{rel}|{stt.st_size}|{int(stt.st_mtime)}")
        except Exception:
            continue
    lines.sort()
    raw = "\n".join(lines)
    return stable_hash(raw if raw else f"EMPTY-{time.time()}"), len(files)

# --------------------- Loading ---------------------
def _fallback_read(path: str) -> str:
    # ... (rest of _fallback_read remains the same)
    try:
        if path.lower().endswith((".xlsx", ".xlsm", ".xltx")):
            df = pd.read_excel(path).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"{header}\n{body}"
        if path.lower().endswith((".csv", ".tsv")):
            df = pd.read_csv(path, sep="\t" if path.lower().endswith(".tsv") else ",").astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"{header}\n{body}"
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return Path(path).read_bytes().decode("utf-8", errors="ignore")
        except Exception:
            return ""

def load_one(path: str) -> List[Document]:
    # ... (rest of load_one remains the same)
    p = path.lower()
    try:
        if p.endswith(".pdf"):
            return PyPDFLoader(path).load()
        if p.endswith((".html", ".htm")):
            return BSHTMLLoader(path).load()
        if p.endswith(".docx"):
            return Docx2txtLoader(path).load()
        if p.endswith(".csv"):
            return CSVLoader(path).load()
        if p.endswith(".tsv"):
            return CSVLoader(path, csv_args={"delimiter": "\t"}).load()
        if p.endswith((".txt", ".md", ".json", ".xml", ".rtf", ".pptx", ".xlsx", ".xlsm", ".xltx")):
            txt = _fallback_read(path)
            return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
        txt = _fallback_read(path)
        return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
    except Exception:
        txt = _fallback_read(path)
        return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []

def load_documents(folder: str) -> List[Document]:
    docs: List[Document] = []
    for path in iter_files(folder):
        docs.extend(load_one(path))
    return docs

# --------------------- Indexing ---------------------
@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200

def _make_embeddings():
    # ... (rest of _make_embeddings remains the same)
    return HuggingFaceEmbeddings(
        model_name=_EMB_MODEL,
        model_kwargs=_EMB_MODEL_KW,
        encode_kwargs=_ENCODE_KW,
    )

def index_folder_langchain(folder: str, persist_dir: str, collection_name: str, emb_model: str, chunk_cfg: ChunkingConfig) -> Tuple[int, int]:
    # ... (rest of index_folder_langchain remains the same)
    raw_docs = load_documents(folder)
    if not raw_docs:
        return (0, 0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_cfg.chunk_size, chunk_overlap=chunk_cfg.chunk_overlap, separators=["\n\n", "\n", ". ", " "])
    splat = splitter.split_documents(raw_docs)
    embeddings = _make_embeddings()
    _ = Chroma.from_documents(documents=splat, embedding=embeddings, collection_name=collection_name, persist_directory=persist_dir).persist()
    return (len(raw_docs), len(splat))

def get_vectorstore(persist_dir: str, collection_name: str, emb_model: str) -> Chroma:
    # ... (rest of get_vectorstore remains the same)
    key = f"_vs::{persist_dir}::{collection_name}::{emb_model}"
    if key in st.session_state:
        return st.session_state[key]
    embeddings = _make_embeddings()
    vs = Chroma(collection_name=collection_name, persist_directory=persist_dir, embedding_function=embeddings)
    st.session_state[key] = vs
    return vs

# --------------------- Anthropic init helpers ---------------------
def _strip_proxy_env() -> None:
    # ... (rest of _strip_proxy_env remains the same)
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(v, None)

def _get_secret_api_key() -> Optional[str]:
    # ... (rest of _get_secret_api_key remains the same)
    try:
        s = st.secrets
    except Exception:
        s = None

    if s:
        for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
            v = s.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for parent in ("anthropic","claude","secrets"):
            if parent in s and isinstance(s[parent], dict):
                ns = s[parent]
                for k in ("api_key","ANTHROPIC_API_KEY","key","token"):
                    v = ns.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()

    for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _anthropic_client_from_secrets():
    # ... (rest of _anthropic_client_from_secrets remains the same)
    _strip_proxy_env()
    api_key = _get_secret_api_key()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY (set in Streamlit secrets or env).")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if _AnthropicClientNew is not None:
        return _AnthropicClientNew(api_key=api_key)
    if _AnthropicClientOld is not None:
        return _AnthropicClientOld(api_key=api_key)
    raise RuntimeError("Anthropic SDK not installed correctly.")

# --------------------- Chain builders ---------------------

def make_llm(backend: str, model_name: str, temperature: float):
    # ... (rest of make_llm remains the same)
    if backend.startswith("Claude"):
        # Always bypass ChatAnthropic to avoid any proxies kw path
        client = _anthropic_client_from_secrets()
        return ClaudeDirect(
            client=client,
            model=model_name or DEFAULT_CLAUDE,
            temperature=temperature,
            max_tokens=800,
        )
    return ChatOllama(model=model_name or DEFAULT_OLLAMA, temperature=temperature)

def make_chain(vs: Chroma, llm: BaseChatModel, k: int):
    # ... (rest of make_chain remains the same)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    # Tell memory which output field to capture
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",        # <-- key fix
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )


# --------------------- Defaults + auto-index ---------------------
def settings_defaults() -> Dict[str, Any]:
    # ... (rest of settings_defaults remains the same)
    kb_dir = get_kb_dir()
    return {
        "persist_dir": ".chroma",
        "collection_name": f"kb-{stable_hash(kb_dir)}",
        "base_folder": kb_dir,
        "emb_model": _EMB_MODEL,
        "chunk_cfg": ChunkingConfig(),
        "backend": "Claude (Anthropic)",
        "ollama_model": DEFAULT_OLLAMA,
        "claude_model": DEFAULT_CLAUDE,
        "temperature": 0.2,
        "top_k": 5,
        "auto_index_min_interval_sec": 8,
    }

def auto_index_if_needed(status_placeholder: Optional[object] = None) -> Optional[Chroma]:
    # ... (rest of auto_index_if_needed remains the same)
    folder = st.session_state.get("base_folder")
    persist = st.session_state.get("persist_dir")
    colname = st.session_state.get("collection_name")
    emb_model = st.session_state.get("emb_model")
    min_gap = int(st.session_state.get("auto_index_min_interval_sec", 8))

    sig_now, file_count = compute_kb_signature(folder)
    last_sig = st.session_state.get("_kb_last_sig")
    last_time = float(st.session_state.get("_kb_last_index_ts", 0.0))
    now = time.time()

    need_index = (last_sig != sig_now) or (last_sig is None)
    throttled = (now - last_time) < min_gap
    target = status_placeholder if status_placeholder is not None else st

    if need_index and not throttled:
        try:
            target.markdown('<div class="status-inline">Indexing…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_langchain(folder, persist, colname, emb_model, st.session_state.get("chunk_cfg", ChunkingConfig()))
            st.session_state["_kb_last_sig"] = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"] = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            label = f"Indexed: <b>{n_docs}</b> files processed, <b>{n_chunks}</b> chunks"
        except Exception as e:
            label = f"Auto-index failed: <b>{e}</b>"
        target.markdown(f'<div class="status-inline">{label}</div>', unsafe_allow_html=True)
    else:
        ts = st.session_state.get("_kb_last_index_ts")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"
        target.markdown(f'<div class="status-inline">Auto-index is <b>ON</b> · Files: <b>{file_count}</b> · Last indexed: <b>{when}</b> · Collection: <code>{colname}</code></div>', unsafe_allow_html=True)

    try:
        return get_vectorstore(persist, colname, emb_model)
    except Exception:
        return None

# --------------------- UI Functions ---------------------

def render_sidebar():
    """Renders the entire Streamlit sidebar with settings."""
    with st.sidebar:
        lp = _resolve_logo_path()
        if lp and Path(lp).exists():
            try:
                st.image(str(lp), caption="iSOFT ANZ Pvt Ltd", width=240)
            except Exception:
                pass
        else:
            st.info("Add assets/isoft_logo.png for branding.")

        st.subheader("⚙️ Settings")
        st.caption("Auto-index is enabled. Edit paths/models below if needed.")

        # --- KB Settings ---
        st.session_state["base_folder"] = st.text_input("Knowledge Base Folder", value=st.session_state["base_folder"])
        st.session_state["persist_dir"] = st.text_input("Chroma Persist Directory", value=st.session_state["persist_dir"])
        st.session_state["collection_name"] = st.text_input("Collection Name", value=st.session_state["collection_name"])

        st.divider()

        # --- LLM Settings ---
        st.session_state["backend"] = st.radio("LLM Backend", ["Claude (Anthropic)", "Ollama (local)"], index=0, key="llm_backend_radio")
        if st.session_state["backend"].startswith("Claude"):
            st.session_state["claude_model"] = st.text_input("Claude Model Name", value=st.session_state["claude_model"])
        else:
            st.session_state["ollama_model"] = st.text_input("Ollama Model Name", value=st.session_state["ollama_model"])

        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state["top_k"] = st.slider("Top-K (Retrieval)", 1, 15, 5)
        st.session_state["auto_index_min_interval_sec"] = st.number_input(
            "Auto-index min interval (sec)", min_value=1, max_value=300, value=8, step=1
        )
        
        # Anthropic status
        try:
            import anthropic as _anth
            st.caption(
                f"anthropic=={getattr(_anth, '__version__', 'unknown')} • direct client mode"
            )
        except Exception:
            st.caption("anthropic not importable")

def render_chat_history():
    """Renders the chat history using st.chat_message."""
    # This is a cleaner way than the manual HTML you had
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def render_prompt_options():
    """Renders suggested prompt buttons."""
    
    st.markdown('<div class="smallcaps">Suggested Prompts</div>', unsafe_allow_html=True)
    
    # Example prompts (customize these based on expected KB content)
    prompts = [
        "What is the policy on annual leave?",
        "Summarize the Q3 financial results.",
        "Who is the current project lead for the 'Phoenix' initiative?",
        "Can you explain the key features of the new system?",
    ]
    
    # Display buttons in a row
    cols = st.columns(len(prompts))
    for i, prompt in enumerate(prompts):
        if cols[i].button(prompt, key=f"prompt_option_{i}"):
            # When a button is clicked, set the chat input and trigger handler
            st.session_state["user_prompt_input"] = prompt
            st.session_state["_trigger_send"] = True
            st.rerun()

# --------------------- Main Execution Logic ---------------------

def handle_user_input(query: str, vs: Optional[Chroma]):
    """Processes the user query, updates history, and runs the RAG chain."""
    
    # 1. Append user message
    st.session_state["messages"].append({"role": "user", "content": query})

    # 2. Short-circuit for simple greetings (improves latency)
    if len(query) <= 40 and GREETING_RE.match(query):
        st.session_state["messages"].append(
            {"role": "assistant", "content": "Hello! How can I help you today with your Knowledge Base?"}
        )
        # Re-run to update the chat UI
        st.rerun() 
        return

    # 3. Check Vector Store
    if vs is None:
        st.session_state["messages"].append(
            {"role": "assistant", "content": "Vector store unavailable. Check your settings and try again."}
        )
        st.rerun()
        return

    # 4. Initialize LLM
    backend = st.session_state["backend"]
    model_name = (
        st.session_state["claude_model"]
        if backend.startswith("Claude")
        else st.session_state["ollama_model"]
    )
    try:
        llm = make_llm(backend, model_name, float(st.session_state["temperature"]))
    except Exception as e:
        st.session_state["messages"].append({"role": "assistant", "content": f"LLM init error: {e}"})
        st.rerun()
        return

    # 5. Run RAG Chain
    chain = make_chain(vs, llm, int(st.session_state["top_k"]))

    t0 = time.time()
    try:
        # NOTE: Using st.spinner is a nicer UX for long operations
        with st.spinner(f"Querying {backend} with RAG..."):
            result = chain.invoke({"question": query})
            answer = result.get("answer", "").strip() or "I could not find an answer in the Knowledge Base."
            sources = result.get("source_documents", []) or []

        # Build citation block only if sources exist
        citation_block = build_citation_block(
            sources, kb_root=st.session_state.get("base_folder")
        )
        
        # In a cleaner UI, use the standard Streamlit chat message for the main answer
        # and append sources separately if needed. I'll include them in the content for now.
        msg = f"{answer}{citation_block}\n\n_(Answered in {human_time((time.time()-t0)*1000)})_"
        
    except Exception as e:
        msg = f"RAG error: {e}"

    # 6. Append assistant message
    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.rerun()


def main():
    # 1. Initialize session state defaults
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)
    
    # Initialize a temporary state for the "prompt options" click
    st.session_state.setdefault("_trigger_send", False)

    # 2. Render UI
    render_sidebar()

    # Title & Index Status
    st.markdown("### 💬 Chat with your Knowledge Base (LangChain RAG)")
    hero_status = st.container()
    vs = auto_index_if_needed(status_placeholder=hero_status)
    
    # Initial Chat History
    st.session_state.setdefault(
        "messages",
        [{"role": "assistant", "content": "Hi! Ask anything about your Knowledge Base, or click on a suggested prompt below."}],
    )

    # Chat UI container
    # Replaced manual HTML rendering with standard Streamlit chat elements for simplicity/better integration
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    render_chat_history()
    st.markdown("</div>", unsafe_allow_html=True) # End chat-scroll

    # New Composer with Prompt Options
    st.markdown('<div class="composer">', unsafe_allow_html=True)
    render_prompt_options()

    # Use the native st.chat_input which handles the input and submit better
    user_text = st.chat_input(
        "Type your question...",
        key="user_prompt_input" # Use a consistent key
    )
    
    # Check if a prompt option was clicked
    if st.session_state.get("_trigger_send", False):
        user_text = st.session_state.pop("user_prompt_input", None)
        st.session_state.pop("_trigger_send")
    
    st.markdown("</div>", unsafe_allow_html=True) # End composer
    st.markdown("</div>", unsafe_allow_html=True) # End chat-card

    # 3. Handle User Input
    if user_text and user_text.strip():
        handle_user_input(user_text.strip(), vs)

if __name__ == "__main__":
    main()
