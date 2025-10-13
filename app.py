# Author: Amitesh Jha | iSOFT

# app.py — Modern Streamlit Chat Interface (UI-focused)
# Author: Your Name | 2025-10-13
# Description: A polished, configurable chat UI ready to plug into a RAG/Claude backend.
# - Safe to run without API keys (demo mode)
# - Reads optional YAML at ./config/app.yaml for UI and RAG toggles
# - Includes sidebar settings, message actions, source/citations panel, and file upload
# - Backend hooks are stubbed: wire your retrieval + Claude call inside `orchestrate_response()`

from __future__ import annotations

import os
import json
import time
import base64
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Avatars (custom images)
AVATAR_USER = "assets/avatar.png"
AVATAR_ASSISTANT = "assets/llm.png"

try:
    import yaml  # type: ignore
except Exception:  # yaml is optional; app still runs
    yaml = None  # type: ignore

# ------------------------------
# Config loading (optional YAML)
# ------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "streamlit": {
        "page_title": "Decision Intelligence Chat",
        "layout": "wide",
        "theme": {
            "primaryColor": "#2563eb",  # Tailwind indigo-600
            "backgroundColor": "#0b1220",
            "secondaryBackgroundColor": "#0e1726",
            "textColor": "#e5e7eb",
        },
    },
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "rag": {
        "top_k": 5,
        "score_threshold": 0.1,
        "show_citations": True,
    },
}


def load_yaml_config() -> Dict[str, Any]:
    """Load ./config/app.yaml if present; merge onto DEFAULT_CONFIG (shallow)."""
    cfg = DEFAULT_CONFIG.copy()
    cfg_path = Path("config/app.yaml")
    if cfg_path.is_file() and yaml is not None:
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            # Shallow merge
            for k, v in y.items():
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to read app.yaml: {e}")
    return cfg


# ------------------------------
# Utilities
# ------------------------------
@dataclass
class Source:
    title: str
    snippet: str
    url: Optional[str] = None
    score: Optional[float] = None
    doc_id: Optional[str] = None


def _img_to_b64(path: str) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return None


def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "config" not in st.session_state:
        st.session_state.config = load_yaml_config()
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = uuid.uuid4().hex[:8]
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True


# ------------------------------
# Backend stubs (plug your logic)
# ------------------------------

def retrieve_documents(query: str, top_k: int = 5) -> List[Source]:
    """Stub: Replace with Milvus search + metadata fetch.
    Return a few dummy sources for demo.
    """
    demo = [
        Source(
            title="Quarterly Sales Plan",
            snippet="Targets and assumptions for Q4, including supply constraints and pricing.",
            url=None,
            score=0.62,
            doc_id="doc_q4_plan",
        ),
        Source(
            title="Ops Note: Inventory Risk",
            snippet="Spike in lead times for APAC vendors; mitigation via safety stock.",
            url=None,
            score=0.58,
            doc_id="doc_ops_risk",
        ),
    ]
    return demo[:top_k]


def call_claude(system_prompt: str, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    """Stub: Wire Anthropic SDK here. Keep UI functional in demo mode."""
    # Example streaming simulation
    canned = (
        "Based on the retrieved context, here are the key insights and a next-step plan."
        "\n\n1) Demand uptick likely in Q4 (holiday promotions)."
        "\n2) Constrain SKUs with long lead times; build safety stock."
        "\n3) Prioritize high-margin bundles in APAC."
    )
    return canned


def orchestrate_response(query: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieval → Prompting → LLM. Return dict with 'answer' and 'sources'."""
    rag_cfg = cfg.get("rag", {})
    llm_cfg = cfg.get("llm", {})

    sources = retrieve_documents(query, top_k=rag_cfg.get("top_k", 5))

    system_prompt = (
        "You are a Decision Intelligence assistant. Use the provided sources to answer."
        " If sources are insufficient, say what else is needed. Always be concise and actionable."
    )

    chat_context = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in {"user", "assistant"}
    ]
    chat_context.append({"role": "user", "content": query})

    answer = call_claude(
        system_prompt=system_prompt,
        messages=chat_context,
        model=llm_cfg.get("model", "claude-3-5-sonnet"),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 1024)),
    )

    return {"answer": answer, "sources": sources}


# ------------------------------
# UI Building Blocks
# ------------------------------

GLOBAL_CSS = """
/* Layout + Colors */
:root{
  --bg:#0b1220; --bg2:#0e1726; --panel:#0f172a; --border:#1f2a44; --text:#e5e7eb; --muted:#a5b4fc;
  --accent:#2563eb; --accent2:#22d3ee; --success:#10b981; --warn:#f59e0b;
}
section.main > div { padding-top: 0 !important; }

/* Chat bubbles */
.chat-wrap{display:flex;gap:12px;margin:8px 0;padding:8px 10px;align-items:flex-start}
.chat-msg{max-width: 1150px; border:1px solid var(--border); background:linear-gradient(180deg,rgba(17,24,39,.75),rgba(17,24,39,.55));
  padding:12px 14px;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
.msg-user{background:linear-gradient(180deg,rgba(37,99,235,.15),rgba(37,99,235,.05));}
.msg-assistant{background:linear-gradient(180deg,rgba(34,211,238,.12),rgba(34,211,238,.05));}
.msg-tool{background:linear-gradient(180deg,rgba(16,185,129,.12),rgba(16,185,129,.05));}

/* Avatar */
.avatar{width:36px;height:36px;border-radius:50%;border:1px solid var(--border);object-fit:cover;box-shadow:0 2px 8px rgba(0,0,0,.2)}

/* Message header */
.msg-head{display:flex;align-items:center;gap:8px;margin-bottom:6px;opacity:.9;font-size:.85rem;color:var(--muted)}
.msg-body{white-space:pre-wrap;line-height:1.55}

/* Sources panel */
.sources-card{border:1px solid var(--border);background:linear-gradient(180deg,rgba(15,23,42,.9),rgba(15,23,42,.7));padding:10px 12px;border-radius:14px}
.source-item{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px dashed rgba(255,255,255,.07)}
.source-item:last-child{border-bottom:0}
.source-score{font-size:.8rem;opacity:.8}

/* Header bar */
.header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;margin:10px 0 4px;
  background:linear-gradient(135deg, rgba(37,99,235,.15), rgba(34,211,238,.10)); border:1px solid var(--border); border-radius:16px}
.header .title{font-weight:800;font-size:1.1rem;letter-spacing:.3px}
.header .badge{font-size:.75rem;opacity:.85;padding:4px 8px;border:1px solid var(--border);border-radius:999px}

/* Input row */
.input-hint{opacity:.7;font-size:.85rem;margin-top:6px}

/* Buttons */
.pill{border-radius:999px;border:1px solid var(--border);padding:6px 12px;background:linear-gradient(180deg,rgba(17,24,39,.85),rgba(17,24,39,.6));}
.pill:hover{filter:brightness(1.1)}

/* File pill */
.file-pill{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;margin-top:6px;border:1px dashed var(--border);border-radius:10px}

/* Small helpers */
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:6px 0}
.kpi .card{border:1px solid var(--border);border-radius:12px;padding:8px 10px;background:linear-gradient(180deg,rgba(2,6,23,.8),rgba(2,6,23,.6))}
"""


def render_header(cfg: Dict[str, Any]):
    col1, col2 = st.columns([1, 1])
    with col1:
        header_html = (
            f"<div class='header'><div class='title'>🧠 Decision Intelligence Chat "
            f"<span class='badge'>Chat ID: {st.session_state.chat_id}</span></div>"
            "<div class='badge'>Claude • RAG • Milvus-ready</div></div>"
        )
        st.markdown(header_html, unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="kpi">'
            ' <div class="card"><div>Model</div><b>' + cfg["llm"]["model"] + '</b></div>'
            ' <div class="card"><div>Top-K</div><b>' + str(cfg["rag"]["top_k"]) + '</b></div>'
            ' <div class="card"><div>Temp</div><b>' + str(cfg["llm"]["temperature"]) + '</b></div>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_sidebar(cfg: Dict[str, Any]):
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.caption("These mirror values from config/app.yaml if present.")

        # LLM
        st.subheader("LLM")
        cfg["llm"]["model"] = st.selectbox(
            "Claude model", ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
            index=["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"].index(cfg["llm"]["model"]) if cfg["llm"]["model"] in ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"] else 0,
            help="Only Claude is wired in this UI; swap later if needed.",
        )
        cfg["llm"]["temperature"] = st.slider("Temperature", 0.0, 1.0, float(cfg["llm"]["temperature"]))
        cfg["llm"]["max_tokens"] = st.slider("Max tokens", 256, 4096, int(cfg["llm"]["max_tokens"]))

        # RAG
        st.subheader("Retrieval")
        cfg["rag"]["top_k"] = st.slider("Top-K", 1, 20, int(cfg["rag"]["top_k"]))
        cfg["rag"]["score_threshold"] = st.slider("Score threshold", 0.0, 1.0, float(cfg["rag"]["score_threshold"]))
        cfg["rag"]["show_citations"] = st.toggle("Show citations", bool(cfg["rag"].get("show_citations", True)))

        # Session controls
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.toast("Chat cleared.")
        with col_b:
            if st.button("📥 Export chat", use_container_width=True, type="secondary"):
                payload = {
                    "chat_id": st.session_state.chat_id,
                    "messages": st.session_state.messages,
                    "config": cfg,
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"chat_{st.session_state.chat_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.divider()
        st.caption("Demo mode: backend calls are stubbed. Wire your Milvus + Claude in orchestrate_response().")


def _role_badge(role: str) -> str:
    return {
        "user": "👤 You",
        "assistant": "🤖 Assistant",
        "tool": "🔧 Tool",
        "system": "🔒 System",
    }.get(role, role)


def render_messages():
    def _avatar_tag(role: str) -> str:
        path = AVATAR_ASSISTANT if role == "assistant" else (AVATAR_USER if role == "user" else None)
        if path:
            b64 = _img_to_b64(path)
            if b64:
                return f"<img class='avatar' src='data:image/png;base64,{b64}' alt='{role}'>"
        # Fallback emoji circle
        emoji = "🤖" if role == "assistant" else ("👤" if role == "user" else "🧩")
        return f"<div class='avatar' style='display:flex;align-items:center;justify-content:center;font-size:18px;'>{emoji}</div>"

    for i, m in enumerate(st.session_state.messages):
        role = m.get("role", "assistant")
        classes = "msg-assistant" if role == "assistant" else ("msg-user" if role == "user" else "msg-tool")
        head = f"<div class='msg-head'>{_role_badge(role)} • {time.strftime('%H:%M')}</div>"
        body = f"<div class='msg-body'>{st._escape_markdown(str(m.get('content','')), unsafe_allow_html=False)}</div>"
        html = (
            "<div class='chat-wrap'>" + _avatar_tag(role) +
            f"<div class='chat-msg {classes}'>{head}{body}</div>" +
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

        # Optional sources under assistant messages
        if role == "assistant" and m.get("sources"):
            with st.expander("Sources / citations", expanded=False):
                for s in m["sources"]:
                    score = f"<span class='source-score'>score: {s.get('score'):.2f}</span>" if s.get("score") is not None else ""
                    title = s.get("title") or s.get("doc_id") or "Document"
                    snippet = s.get("snippet") or ""
                    url = s.get("url")
                    line = f"<div class='source-item'><div><b>{title}</b> {score}<br><span>{snippet}</span>"
                    if url:
                        line += f"<br><a href='{url}' target='_blank'>Open</a>"
                    line += "</div></div>"
                    st.markdown(line, unsafe_allow_html=True)


        # Optional sources under assistant messages
        if role == "assistant" and m.get("sources"):
            with st.expander("Sources / citations", expanded=False):
                for s in m["sources"]:
                    score = f"<span class='source-score'>score: {s.get('score'):.2f}</span>" if s.get("score") is not None else ""
                    title = s.get("title") or s.get("doc_id") or "Document"
                    snippet = s.get("snippet") or ""
                    url = s.get("url")
                    line = f"<div class='source-item'><div><b>{title}</b> {score}<br><span>{snippet}</span>"
                    if url:
                        line += f"<br><a href='{url}' target='_blank'>Open</a>"
                    line += "</div></div>"
                    st.markdown(line, unsafe_allow_html=True)


def render_input_row(cfg: Dict[str, Any]):
    st.markdown("<style>div[data-testid='stChatMessageInput']{bottom:10px;}</style>", unsafe_allow_html=True)

    up_files = st.file_uploader(
        "Attach files (optional) — PDFs, DOCX, CSV, etc.",
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx"],
        accept_multiple_files=True,
        help="These can be routed to your ingestion flow to ground responses.",
    )
    if up_files:
        st.markdown(
            " ".join([f"<span class='file-pill'>📎 {f.name} ({f.size//1024} KB)</span>" for f in up_files]),
            unsafe_allow_html=True,
        )

    user_text = st.chat_input("Ask about your business, KPIs, risks…")
    if user_text:
        # Echo user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        render_messages()  # immediate UI feedback
        st.session_state["pending_query"] = user_text
        st.session_state["pending_files"] = up_files or []
        st.rerun()


# ------------------------------
# App Entrypoint
# ------------------------------

def main():
    cfg = load_yaml_config()

    # Page config
    try:
        st.set_page_config(
            page_title=cfg["streamlit"].get("page_title", "Decision Intelligence Chat"),
            page_icon="🧠",
            layout=cfg["streamlit"].get("layout", "wide"),
        )
    except Exception:
        pass  # set_page_config can only be called once per run

    _init_session_state()

    # Theme-ish CSS (app-local)
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)

    # Header + Sidebar
    render_header(cfg)
    render_sidebar(cfg)

    # Chat history
    render_messages()

    # If there's a pending query (from previous run), answer it now
    if q := st.session_state.pop("pending_query", None):
        with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
            with st.spinner("Thinking…"):
                result = orchestrate_response(q, cfg)
        answer = result.get("answer", "I couldn't produce an answer.")

        # Store assistant message with optional sources for the expander
        sources_payload = [
            {"title": s.title, "snippet": s.snippet, "url": s.url, "score": s.score, "doc_id": s.doc_id}
            for s in result.get("sources", [])
        ]
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_payload if cfg.get("rag", {}).get("show_citations", True) else None,
        })
        st.rerun()

    # Input row at the bottom
    render_input_row(cfg)


if __name__ == "__main__":
    main()