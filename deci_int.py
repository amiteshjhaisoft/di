# Author: Amitesh Jha | iSoft | 2025-10-07
# LangChain refactor of your LLM chat + local RAG (Chroma) Streamlit app.
# - Uses: Directory -> Loaders -> TextSplitter -> Embeddings -> Chroma -> ConversationalRetrievalChain
# - LLMs: Claude (Anthropic) or local Ollama
# - Robust to Chroma 0.5+ (persistent store, telemetry disabled, single instance)
#
# Install:
#   pip install -r requirements.txt
#
# Run:
#   streamlit run deci_int_langchain.py
#
# Env (optional):
#   ANTHROPIC_API_KEY         -> for Claude
#   ISOFT_LOGO_PATH           -> path to iSOFT logo
#   USER_AVATAR_PATH          -> override user avatar path
#   ASSISTANT_AVATAR_PATH     -> override assistant avatar path
#   (Ollama runs at http://localhost:11434)

from __future__ import annotations

import os, io, re, gc, glob, time, uuid, base64, hashlib, logging, shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------
# Chroma / Telemetry hygiene (quiet logs, disable OTel)
# ---------------------------------------------------------------------
os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "none")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ---------------------------------------------------------------------
# LangChain bits
# ---------------------------------------------------------------------
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# LLMs
from langchain_anthropic import ChatAnthropic
try:
    from langchain_community.chat_models import ChatOllama  # preferred newer import
except Exception:
    from langchain_community.llms import Ollama as ChatOllama

# Loaders (lightweight set; fall back to manual readers where needed)
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader
)

# ======================================================================
# UI / THEME
# ======================================================================
st.set_page_config(page_title="LLM Chat • LangChain RAG", page_icon="💬", layout="wide")

# ---- Branding helpers -------------------------------------------------

def _resolve_logo_path() -> Optional[Path]:
    env_logo = os.getenv("ISOFT_LOGO_PATH")
    candidates = [
        Path.cwd() / "assets" / "isoft_logo.png",
        Path(env_logo).expanduser().resolve() if env_logo else None,
    ]
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

USER_AVATAR_PATH, ASSIST_AVATAR_PATH = _resolve_avatar_paths()
USER_AVATAR_URI = _img_to_data_uri(USER_AVATAR_PATH)
ASSIST_AVATAR_URI = _img_to_data_uri(ASSIST_AVATAR_PATH)

st.markdown(f"""
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
  {"background-image:url('" + USER_AVATAR_URI + "');" if USER_AVATAR_URI else ""}
}}
.avatar.assistant {{
  {"background-image:url('" + ASSIST_AVATAR_URI + "');" if ASSIST_AVATAR_URI else ""}
}}
.bubble{{ border:1px solid var(--border); background:var(--bubble-assist); padding:.8rem .95rem; border-radius:12px; max-width:860px; white-space:pre-wrap; line-height:1.45; }}
.msg.user .bubble{{ background:var(--bubble-user); }}
.composer{{ padding:.6rem .75rem; border-top:1px solid var(--border); background:#fff; position:sticky; bottom:0; z-index:2; }}
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px; padding:.5rem .7rem; font-size:.9rem; color:#111827; margin:.5rem 0 .8rem; }}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# General helpers
# ======================================================================
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

# ======================================================================
# File -> Documents (LangChain loaders + safe fallbacks)
# ======================================================================

def _fallback_read(path: str) -> str:
    # Texty best-effort (avoids heavy unstructured deps)
    try:
        if path.lower().endswith((".xlsx", ".xlsm", ".xltx")):
            df = pd.read_excel(path)
            df = df.astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"{header}\n{body}"
        if path.lower().endswith((".csv", ".tsv")):
            df = pd.read_csv(path, sep="\t" if path.lower().endswith(".tsv") else ",")
            df = df.astype(str).iloc[:1000, :50]
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
            if not txt.strip():
                return []
            return [Document(page_content=txt, metadata={"source": path})]
        txt = _fallback_read(path)
        if not txt.strip():
            return []
        return [Document(page_content=txt, metadata={"source": path})]
    except Exception:
        txt = _fallback_read(path)
        if not txt.strip():
            return []
        return [Document(page_content=txt, metadata={"source": path})]


def load_documents(folder: str) -> List[Document]:
    docs: List[Document] = []
    for path in iter_files(folder):
        docs.extend(load_one(path))
    return docs

# ======================================================================
# Indexing: TextSplitter + Embeddings + Chroma (via LangChain)
# ======================================================================

@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200


def index_folder_langchain(
    folder: str,
    persist_dir: str,
    collection_name: str,
    emb_model: str,
    chunk_cfg: ChunkingConfig,
) -> Tuple[int, int]:
    """
    Returns: (num_source_docs, num_chunks)
    """
    raw_docs = load_documents(folder)
    if not raw_docs:
        return (0, 0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.chunk_size,
        chunk_overlap=chunk_cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    splat = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    # Build / update vectorstore (idempotent)
    _ = Chroma.from_documents(
        documents=splat,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    ).persist()

    gc.collect()
    return (len(raw_docs), len(splat))


def get_vectorstore(
    persist_dir: str,
    collection_name: str,
    emb_model: str,
) -> Chroma:
    # Cache LangChain VectorStore in session (prevents repeated instantiation issues)
    key = f"_vs::{persist_dir}::{collection_name}::{emb_model}"
    if key in st.session_state:
        return st.session_state[key]
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    st.session_state[key] = vs
    return vs

# ======================================================================
# LLM selection + ConversationalRetrievalChain
# ======================================================================
DEFAULT_OLLAMA = "llama3.2"
DEFAULT_CLAUDE = "claude-3-5-sonnet-20240620"


def make_llm(backend: str, model_name: str, temperature: float):
    if backend.startswith("Claude"):
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=800)
    # Ollama (local)
    return ChatOllama(model=model_name, temperature=temperature)


def make_chain(vs: Chroma, llm, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain

# ======================================================================
# Settings defaults + Auto-index-or-refresh (signature-based)
# ======================================================================

def settings_defaults():
    kb_dir = get_kb_dir()
    return {
        "persist_dir": ".chroma",
        "collection_name": f"kb-{stable_hash(kb_dir)}",
        "base_folder": kb_dir,
        "emb_model": "sentence-transformers/all-MiniLM-L6-v2",
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

    vs = None
    if need_index and not throttled:
        try:
            target.markdown('<div class="status-inline">Indexing…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_langchain(
                folder=folder,
                persist_dir=persist,
                collection_name=colname,
                emb_model=emb_model,
                chunk_cfg=st.session_state.get("chunk_cfg", ChunkingConfig()),
            )
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
        target.markdown(
            f'<div class="status-inline">Auto-index is <b>ON</b> · Files: <b>{file_count}</b> · Last indexed: <b>{when}</b> · Collection: <code>{colname}</code></div>',
            unsafe_allow_html=True,
        )

    # Always return a handle (existing or fresh)
    try:
        vs = get_vectorstore(persist, colname, emb_model)
    except Exception:
        vs = None
    return vs

# ======================================================================
# Main App
# ======================================================================

def main():
    # Defaults
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)

    # Sidebar
    with st.sidebar:
        lp = _resolve_logo_path()
        if lp and Path(lp).exists():
            try:
                # use explicit width to avoid unknown kw issues
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

        st.session_state["emb_model"] = st.selectbox(
            "Embedding model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L12-v2",
                "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            ],
            index=0,
        )

        st.session_state["backend"] = st.radio("LLM", ["Claude (Anthropic)", "Ollama (local)"], index=0)
        if st.session_state["backend"].startswith("Claude"):
            st.session_state["claude_model"] = st.text_input("Claude model", value=st.session_state["claude_model"])
        else:
            st.session_state["ollama_model"] = st.text_input("Ollama model", value=st.session_state["ollama_model"])

        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
        st.session_state["top_k"] = st.slider("Top-K", 1, 15, int(st.session_state["top_k"]))
        st.session_state["auto_index_min_interval_sec"] = st.number_input(
            "Auto-index min interval (sec)", min_value=1, max_value=300,
            value=int(st.session_state["auto_index_min_interval_sec"]), step=1
        )

    # Hero / status
    st.markdown("### 💬 Chat with your Knowledge Base (LangChain RAG)")
    hero_status = st.container()
    vs = auto_index_if_needed(status_placeholder=hero_status)

    # Conversation state
    st.session_state.setdefault("messages", [
        {"role": "assistant", "content": "Hi! Ask anything about your Knowledge Base."}
    ])

    # Chat lane
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for m in st.session_state["messages"]:
        who = "user" if m["role"] == "user" else "assistant"
        st.markdown(
            f'<div class="msg {who}"><div class="avatar {who}"></div><div class="bubble">{m["content"]}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- composer UI (Fix A: nonce-based key so we don't mutate widget value) ---
    st.session_state.setdefault("_compose_nonce", 0)
    compose_key = f"compose_input_{st.session_state['_compose_nonce']}"

    st.markdown('<div class="composer">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 0.2])
    with c1:
        user_text = st.text_area(
            "Message",
            key=compose_key,
            placeholder="Type your question…",
            label_visibility="collapsed",
        )
    with c2:
        send = st.button("Send", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- send handling ---
    if send and user_text and user_text.strip():
        query = user_text.strip()
        st.session_state["messages"].append({"role": "user", "content": query})

        if vs is None:
            st.session_state["messages"].append(
                {"role": "assistant", "content": "Vector store unavailable. Check your settings and try again."}
            )
            # Clear by bumping the nonce and rerun
            st.session_state["_compose_nonce"] += 1
            st.rerun()

        # LLM
        backend = st.session_state["backend"]
        model_name = st.session_state["claude_model"] if backend.startswith("Claude") else st.session_state["ollama_model"]
        try:
            llm = make_llm(backend, model_name, float(st.session_state["temperature"]))
        except Exception as e:
            st.session_state["messages"].append({"role": "assistant", "content": f"LLM init error: {e}"})
            st.session_state["_compose_nonce"] += 1
            st.rerun()

        # Chain
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

        # ✅ Clear input by incrementing the nonce (no direct value mutation)
        st.session_state["_compose_nonce"] += 1
        st.rerun()

        # LLM
        backend = st.session_state["backend"]
        model_name = st.session_state["claude_model"] if backend.startswith("Claude") else st.session_state["ollama_model"]
        try:
            llm = make_llm(backend, model_name, float(st.session_state["temperature"]))
        except Exception as e:
            st.session_state["messages"].append({"role": "assistant", "content": f"LLM init error: {e}"})
            st.session_state["compose_input"] = ""
            st.rerun()

        # Chain
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
        st.session_state["compose_input"] = ""
        st.rerun()


if __name__ == "__main__":
    main()
