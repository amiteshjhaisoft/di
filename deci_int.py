# Author: Amitesh Jha | iSoft | 2025-10-07
# Streamlit + LangChain RAG app — hardened Anthropic init and CPU-safe embeddings.

from __future__ import annotations
import os, glob, time, base64, hashlib, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd

# --- Torch / device hygiene to avoid meta-tensor issues ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # force no CUDA
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

# LLMs
from langchain_anthropic import ChatAnthropic

# Prefer Anthropic(new) but fallback to Client(old) if needed
try:
    from anthropic import Anthropic as _AnthropicClientNew
except Exception:
    _AnthropicClientNew = None
try:
    from anthropic import Client as _AnthropicClientOld
except Exception:
    _AnthropicClientOld = None

try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    from langchain_community.llms import Ollama as ChatOllama

from langchain_community.document_loaders import (
    PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader
)

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
    asst_candidates = [Path.cwd() / "assets" / "Forecast360.png",
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

# Build dynamic background-image declarations
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
.chat-scroll{{ max-height: 58vh; overflow:auto; padding:.65rem .9rem; }}
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


# --------------------- Helpers ---------------------
TEXT_EXTS = {".txt", ".md", ".rtf", ".html", ".htm", ".json", ".xml"}
DOC_EXTS  = {".pdf", ".docx", ".csv", ".tsv", ".xlsx", ".xlsm", ".xltx", ".pptx"}
SUPPORTED_EXTS = TEXT_EXTS | DOC_EXTS

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
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader
from langchain.schema import Document

def _fallback_read(path: str) -> str:
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

_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_MODEL_KW = {
    "device": "cpu",
    "trust_remote_code": False,
    "torch_dtype": "float32",
}
_ENCODE_KW = {"normalize_embeddings": True}

def _make_embeddings():
    # Force CPU + float32 so we never move from meta tensors
    return HuggingFaceEmbeddings(model_name=_EMB_MODEL, model_kwargs=_EMB_MODEL_KW, encode_kwargs=_ENCODE_KW)

def index_folder_langchain(folder: str, persist_dir: str, collection_name: str, emb_model: str, chunk_cfg: ChunkingConfig) -> Tuple[int, int]:
    raw_docs = load_documents(folder)
    if not raw_docs:
        return (0, 0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_cfg.chunk_size, chunk_overlap=chunk_cfg.chunk_overlap, separators=["\n\n", "\n", ". ", " "])
    splat = splitter.split_documents(raw_docs)
    embeddings = _make_embeddings()
    _ = Chroma.from_documents(documents=splat, embedding=embeddings, collection_name=collection_name, persist_directory=persist_dir).persist()
    return (len(raw_docs), len(splat))

def get_vectorstore(persist_dir: str, collection_name: str, emb_model: str) -> Chroma:
    key = f"_vs::{persist_dir}::{collection_name}::{emb_model}"
    if key in st.session_state:
        return st.session_state[key]
    embeddings = _make_embeddings()
    vs = Chroma(collection_name=collection_name, persist_directory=persist_dir, embedding_function=embeddings)
    st.session_state[key] = vs
    return vs

# --------------------- Anthropic init helpers ---------------------
def _strip_proxy_env() -> None:
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(v, None)

def _get_secret_api_key() -> Optional[str]:
    for k in ("ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key"):
        try:
            if k in st.secrets:
                return st.secrets[k]
        except Exception:
            pass
    return os.getenv("ANTHROPIC_API_KEY")

def _anthropic_client_from_secrets():
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
DEFAULT_OLLAMA = "llama3.2"
DEFAULT_CLAUDE = "claude-sonnet-4-5"

def make_llm(backend: str, model_name: str, temperature: float):
    if backend.startswith("Claude"):
        client = _anthropic_client_from_secrets()
        return ChatAnthropic(client=client, model=model_name, temperature=temperature, max_tokens=800)
    return ChatOllama(model=model_name, temperature=temperature)

def make_chain(vs: Chroma, llm, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False)

# --------------------- Defaults + auto-index ---------------------
def settings_defaults():
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

# --------------------- Main ---------------------
def main():
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)

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

        st.session_state["base_folder"] = st.text_input("Knowledge Base", value=st.session_state["base_folder"])
        st.session_state["persist_dir"] = st.text_input("Chroma persist", value=st.session_state["persist_dir"])
        st.session_state["collection_name"] = st.text_input("Collection", value=st.session_state["collection_name"])

        st.session_state["backend"] = st.radio("LLM", ["Claude (Anthropic)", "Ollama (local)"], index=0)
        if st.session_state["backend"].startswith("Claude"):
            st.session_state["claude_model"] = st.text_input("Claude model", value=st.session_state["claude_model"])
        else:
            st.session_state["ollama_model"] = st.text_input("Ollama model", value=st.session_state["ollama_model"])

        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state["top_k"] = st.slider("Top-K", 1, 15, 5)
        st.session_state["auto_index_min_interval_sec"] = st.number_input("Auto-index min interval (sec)", min_value=1, max_value=300, value=8, step=1)

        try:
            import anthropic as _anth
            st.caption(f"anthropic=={getattr(_anth, '__version__', 'unknown')} • langchain-anthropic active")
        except Exception:
            st.caption("anthropic not importable")

    st.markdown("### 💬 Chat with your Knowledge Base (LangChain RAG)")
    hero_status = st.container()
    vs = auto_index_if_needed(status_placeholder=hero_status)

    st.session_state.setdefault("messages", [{"role": "assistant", "content": "Hi! Ask anything about your Knowledge Base."}])

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for m in st.session_state["messages"]:
        who = "user" if m["role"] == "user" else "assistant"
        st.markdown(f'<div class="msg {who}"><div class="avatar {who}"></div><div class="bubble">{m["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.setdefault("_compose_nonce", 0)
    compose_key = f"compose_input_{st.session_state['_compose_nonce']}"

    st.markdown('<div class="composer">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 0.2])
    with c1:
        user_text = st.text_area("Message", key=compose_key, placeholder="Type your question…", label_visibility="collapsed")
    with c2:
        send = st.button("Send", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if send and user_text and user_text.strip():
        query = user_text.strip()
        st.session_state["messages"].append({"role": "user", "content": query})

        if vs is None:
            st.session_state["messages"].append({"role": "assistant", "content": "Vector store unavailable. Check your settings and try again."})
            st.session_state["_compose_nonce"] += 1
            st.rerun()

        backend = st.session_state["backend"]
        model_name = st.session_state["claude_model"] if backend.startswith("Claude") else st.session_state["ollama_model"]
        try:
            llm = make_llm(backend, model_name, float(st.session_state["temperature"]))
        except Exception as e:
            st.session_state["messages"].append({"role": "assistant", "content": f"LLM init error: {e}"})
            st.session_state["_compose_nonce"] += 1
            st.rerun()

        chain = make_chain(vs, llm, int(st.session_state["top_k"]))

        t0 = time.time()
        try:
            result = chain.invoke({"question": query})
            answer = result.get("answer", "").strip() or "(no answer)"
            sources = result.get("source_documents", []) or []
            cited = []
            for i, d in enumerate(sources, start=1):
                src = (d.metadata or {}).get("source", "unknown")
                cited.append(f"[{i}] {src}")
            citation_block = ("\n\nSources:\n" + "\n".join(cited)) if cited else ""
            msg = f"{answer}{citation_block}\n\n_(Answered in {human_time((time.time()-t0)*1000)})_"
        except Exception as e:
            msg = f"RAG error: {e}"
        st.session_state["messages"].append({"role": "assistant", "content": msg})
        st.session_state["_compose_nonce"] += 1
        st.rerun()

if __name__ == "__main__":
    main()
