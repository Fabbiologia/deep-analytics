"""
streamlit_app.py - Streamlit UI for the Ecological monitoring Agent

Reproducibility:
- Random seeds are set
- Verbose logging to streamlit_app.log
- Relative paths only
"""

import os
import json
import logging
import random
from datetime import datetime
import base64
import mimetypes
from typing import Any, Dict, List

import streamlit as st
import re
from dotenv import load_dotenv
import os
import tempfile
import glob
import io
from datetime import datetime
import streamlit.components.v1 as components
import time
import pandas as pd

# Load .env BEFORE importing modules that read environment at import-time
load_dotenv(override=True)

# Local project tools (optional extensions)
from tools import (
    python_interpreter_tool,   # callable(code: str, input_data: Optional[dict]) -> dict
    visualization_tool,        # callable
    database_query_tool        # callable or None
)

# Core agent entrypoint from main.py
try:
    import main as main_agent
    from main import run_analysis  # function returning str
    AGENT_AVAILABLE = getattr(main_agent, "agent_available", False)
except Exception as e:
    run_analysis = None
    AGENT_AVAILABLE = False
    logger = logging.getLogger("deep_analysis_streamlit")
    logger.warning(f"Could not import main.run_analysis: {e}")

# -----------------------------------------------------------------------------
# Config & Logging
# -----------------------------------------------------------------------------
load_dotenv()

# Seed RNGs
try:
    import numpy as np
    np.random.seed(123)
except Exception:
    pass
random.seed(123)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='streamlit_app.log'
)
logger = logging.getLogger("deep_analysis_streamlit")

st.set_page_config(page_title="Ecological Monitoring Agent", page_icon="🪸", layout="wide")

# Ensure an artifact directory exists for this session and inform tools where to write
if "artifact_dir" not in st.session_state:
    st.session_state["artifact_dir"] = tempfile.mkdtemp(prefix="eco_agent_")
    os.environ["ECO_AGENT_OUTPUT_DIR"] = st.session_state["artifact_dir"]
else:
    # Keep env in sync if session restored
    os.environ["ECO_AGENT_OUTPUT_DIR"] = st.session_state.get("artifact_dir", "")

# Track session start to filter stale artifacts by default
if "session_start_ts" not in st.session_state:
    st.session_state["session_start_ts"] = time.time()

# Inject custom CSS (assets/custom.css) for refined look-and-feel
try:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as _css_err:
    logger.warning(f"Could not load custom CSS: {_css_err}")

# Header: Powered by (simple, no shaded background)
st.markdown(
    """
    <p class="eco-subtitle" style="margin:0; font-weight:600;">Powered by:</p>
    """,
    unsafe_allow_html=True,
)

# Logos row (loads up to two logos from ./logos), placed directly under header
try:
    logos_dir = os.path.join(os.path.dirname(__file__), "logos")
    if os.path.isdir(logos_dir):
        # Collect common image formats and sort deterministically (filename)
        logo_candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            logo_candidates.extend(glob.glob(os.path.join(logos_dir, ext)))
        logo_candidates.sort(key=lambda p: os.path.basename(p).lower())
        if logo_candidates:
            # Build compact HTML with logos in reverse order (last becomes first)
            def _img_data_uri(path: str) -> str:
                mime, _ = mimetypes.guess_type(path)
                if not mime:
                    mime = "image/png"
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:{mime};base64,{b64}"

            ordered = list(reversed(logo_candidates[:2]))
            src_left = _img_data_uri(ordered[0])
            src_right = _img_data_uri(ordered[1]) if len(ordered) > 1 else None
            html = [
                '<div class="eco-logos">',
                f'  <img class="logo" src="{src_left}" alt="{os.path.basename(ordered[0])}">'
            ]
            if src_right:
                html.append(f'  <img class="logo" src="{src_right}" alt="{os.path.basename(ordered[1])}">')
            html.append('</div>')
            st.markdown("\n".join(html), unsafe_allow_html=True)
            logger.info("Rendered logos (reversed): %s", ", ".join(os.path.basename(p) for p in ordered))
    else:
        logger.info("No logos directory found at %s", logos_dir)
except Exception as _logo_err:
    logger.warning("Could not render logos: %s", _logo_err)

# Instructional prompt with disclaimer below logos
st.markdown(
    """
    <p class="eco-subtitle" style="margin-top:4px;">
     <br/>
      <span style="font-size:0.9em; color:#5a6b72;">[Disclaimer, this is a prototype under development and may make mistakes]</span>
    </p>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Chat input now placed at the bottom (above Clear Figures)

def _format_dataframes_summary(dfs: Dict[str, Dict[str, Any]]) -> str:
    if not dfs:
        return "No dataframes produced."
    lines: List[str] = ["DataFrames created:"]
    for name, meta in dfs.items():
        shape = meta.get("shape")
        columns = meta.get("columns")
        sample = meta.get("sample")
        lines.append(f"- {name}: shape={shape}, columns={columns}")
        if sample:
            lines.append(f"  sample (first 5 rows): {json.dumps(sample)[:2000]}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Modern Chat UI (single page)
# -----------------------------------------------------------------------------

# Subtle subtitle for availability + on-demand reload
if not AGENT_AVAILABLE:
    col_a, col_b = st.columns([3,1])
    with col_a:
        st.warning("Agent is not available. Configure OPENAI_API_KEY and DATABASE_URL, then click Initialize.")
    with col_b:
        if st.button("Initialize", help="Reload agent after setting .env or env vars"):
            try:
                import importlib, sys
                # Re-load .env in case it changed
                load_dotenv(override=True)
                # Import or reload main to re-trigger initialization with new env
                if 'main' in sys.modules:
                    mod = importlib.reload(sys.modules['main'])
                else:
                    mod = importlib.import_module('main')
                # Rebind exports
                run_analysis = getattr(mod, 'run_analysis', None)
                AGENT_AVAILABLE = bool(getattr(mod, 'agent_available', False))
                if AGENT_AVAILABLE and callable(run_analysis):
                    st.success("Agent initialized successfully.")
                else:
                    st.error("Agent still unavailable. Check model support, OPENAI_API_KEY, and DATABASE_URL.")
            except Exception as rex:
                st.error(f"Initialization error: {rex}")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": ""}
    ]

# Initialize a per-session temp artifacts directory for generated figures/maps
if "artifact_dir" not in st.session_state:
    st.session_state["artifact_dir"] = tempfile.mkdtemp(prefix="eco_agent_")
    # Allow tools to discover where to write outputs
    os.environ["ECO_AGENT_OUTPUT_DIR"] = st.session_state["artifact_dir"]
if "last_artifacts" not in st.session_state:
    st.session_state["last_artifacts"] = []

# Also watch the project's 'visualizations/' directory if tools write there
VIS_DIR = os.path.join(os.getcwd(), "visualizations")
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")  # Not used for auto-embedding to avoid stale files

def _snapshot_artifacts() -> set:
    paths = set()
    try:
        paths |= set(glob.glob(os.path.join(st.session_state["artifact_dir"], "**"), recursive=True))
    except Exception:
        pass
    try:
        if os.path.isdir(VIS_DIR):
            paths |= set(glob.glob(os.path.join(VIS_DIR, "**"), recursive=True))
    except Exception:
        pass
    try:
        if os.path.isdir(OUTPUTS_DIR):
            paths |= set(glob.glob(os.path.join(OUTPUTS_DIR, "**"), recursive=True))
    except Exception:
        pass
    return paths

def _render_new_artifacts(before: set, after: set):
    """Embed new artifacts produced since last snapshot.
    Supports: .png/.jpg/.jpeg/.svg images and .html maps (e.g., Folium, Plotly HTML).
    """
    new_files = sorted([p for p in (after - before) if os.path.isfile(p)])
    if not new_files:
        return
    st.markdown("**Generated artifacts:**")
    for p in new_files:
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in {".png", ".jpg", ".jpeg", ".svg"}:
                st.image(p, caption=os.path.basename(p), use_container_width=True)
            elif ext in {".html", ".htm"}:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                # Heuristic height; could be improved via metadata later
                components.html(html, height=700, scrolling=True)
                # Also provide a download button for convenience
                with open(p, "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"Download {os.path.basename(p)}",
                    data=data,
                    file_name=os.path.basename(p)
                )
            else:
                # Show a download link for other file types
                with open(p, "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"Download {os.path.basename(p)}",
                    data=data,
                    file_name=os.path.basename(p)
                )
        except Exception as ex:
            st.info(f"Could not render artifact {os.path.basename(p)}: {ex}")
    # Track last artifacts for quick 'open it' actions
    st.session_state["last_artifacts"] = [
        p for p in new_files
        if os.path.splitext(p)[1].lower() in {".html", ".htm", ".png", ".jpg", ".jpeg", ".svg"}
    ]

# Heuristic: if the user asked for CSV and the assistant returned a simple species list,
# generate a CSV artifact automatically in artifact_dir.
def _maybe_generate_csv_from_answer(user_prompt: str, assistant_answer: str) -> str:
    try:
        if not user_prompt or not assistant_answer:
            return ""
        lp = (user_prompt or "").lower()
        if "csv" not in lp and "export" not in lp:
            return ""
        # Extract species-like two-word binomials from assistant answer, line by line
        lines = [l.strip(" -•\t") for l in assistant_answer.splitlines()]
        # Regex: Capitalized genus + lowercase species, letters only
        pat = re.compile(r"^[A-Z][a-z]+\s+[a-z][a-z\-]*$")
        species = [l for l in lines if pat.match(l)]
        species = sorted(dict.fromkeys(species))  # unique, keep order then sort
        if not species:
            return ""
        # Build filename from prompt cues (region/year if present)
        # Year
        m_year = re.search(r"\b(19|20)\d{2}\b", user_prompt)
        year = m_year.group(0) if m_year else datetime.now().strftime("%Y")
        # Region heuristic: take last capitalized 1-3 word phrase after 'in|at|for'
        m_region = re.search(r"(?:in|at|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", user_prompt)
        region = m_region.group(1).replace(" ", "_") if m_region else "region"
        fname = f"{region}_{year}_species.csv"
        # Ensure artifact dir
        out_dir = st.session_state.get("artifact_dir") or tempfile.mkdtemp(prefix="eco_agent_")
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        # Write CSV
        df = pd.DataFrame({"Species": species})
        df.to_csv(fpath, index=False)
        logging.getLogger("deep_analysis_streamlit").info("Auto-generated CSV: %s (n=%d)", fpath, len(species))
        return fpath
    except Exception as ex:
        logging.getLogger("deep_analysis_streamlit").warning("CSV auto-export failed: %s", ex)
        return ""

def _embed_artifacts_mentioned_in(text: str):
    """Find file names mentioned in text and embed them if found in known dirs.
    Looks into artifact_dir and visualizations/.
    """
    if not isinstance(text, str):
        return
    # simple filename pattern capture
    matches = re.findall(r"([\w\-/ .]+\.(?:png|jpg|jpeg|svg|html|htm))", text, flags=re.IGNORECASE)
    if not matches:
        return
    seen = set()
    embedded_any = False
    for m in matches:
        raw = m.strip()
        name = os.path.basename(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        # Assemble candidates: exact relative path if present, then known dirs
        candidates = []
        rel_path = os.path.normpath(raw)
        if os.path.isfile(rel_path):
            candidates.append(rel_path)
        candidates.extend([
            os.path.join(st.session_state.get("artifact_dir", ""), name),
            os.path.join(VIS_DIR, name),
            os.path.join(OUTPUTS_DIR, name),
        ])
        # Prefer newest existing candidate
        existing = [p for p in candidates if p and os.path.isfile(p)]
        path = None
        if existing:
            try:
                path = max(existing, key=lambda p: os.path.getmtime(p))
            except Exception:
                path = existing[0]
        # If still not found, try a recursive glob search for the basename and prefer newest
        if not path:
            # Try recursive search in artifact_dir, visualizations, and as a last resort, project root (cwd)
            search_dirs = [st.session_state.get("artifact_dir"), VIS_DIR, os.getcwd()]
            for base in filter(None, search_dirs):
                try:
                    hits = [h for h in glob.glob(os.path.join(base, "**", name), recursive=True) if os.path.isfile(h)]
                    if hits:
                        try:
                            path = max(hits, key=lambda p: os.path.getmtime(p))
                        except Exception:
                            path = hits[0]
                        break
                except Exception:
                    continue
        if not path:
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".svg"}:
            st.image(path, caption=name, use_container_width=True)
            embedded_any = True
        elif ext in {".html", ".htm"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            components.html(html, height=700, scrolling=True)
            embedded_any = True
        # PDFs intentionally not auto-embedded to avoid surfacing stale reports
    if not embedded_any:
        st.info("I couldn't auto-embed the referenced files. Check 'Generated artifacts' below or the outputs/ folder.")

def _embed_file(path: str) -> bool:
    try:
        if not os.path.isfile(path):
            return False
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".svg"}:
            st.image(path, caption=os.path.basename(path), use_container_width=True)
            return True
        if ext in {".html", ".htm"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            components.html(html, height=700, scrolling=True)
            return True
    except Exception:
        return False
    return False

def _list_known_artifacts():
    """Always show a consolidated list of known artifacts with inline preview or download.
    Scans artifact_dir, visualizations/, and outputs/ recursively.
    By default shows only artifacts created/modified during this session; toggle to include older files.
    """
    bases = [st.session_state.get("artifact_dir"), VIS_DIR, OUTPUTS_DIR]
    files = []
    for base in filter(None, bases):
        try:
            for p in glob.glob(os.path.join(base, "**/*"), recursive=True):
                if os.path.isfile(p):
                    files.append(p)
        except Exception:
            continue
    if not files:
        return
    # Filter to this-session artifacts by default
    show_all = st.checkbox("Show all artifacts (include older files)", value=False, key="show_all_artifacts")
    session_start = float(st.session_state.get("session_start_ts", 0))
    if not show_all and session_start:
        files = [p for p in files if os.path.getmtime(p) >= session_start]
    # Sort by recency
    files = sorted(files, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    if not files:
        st.info("No artifacts generated in this session yet.")
        return
    with st.container():
        st.markdown("**Generated artifacts** (most recent first):")
        for p in files:
            name = os.path.basename(p)
            ext = os.path.splitext(p)[1].lower()
            meta = f"{datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M:%S')} — {os.path.dirname(p)}"
            try:
                if ext in {".png", ".jpg", ".jpeg", ".svg"}:
                    st.image(p, caption=f"{name}  |  {meta}", use_container_width=True)
                    with open(p, "rb") as f:
                        data = f.read()
                    st.download_button(label=f"Download {name}", data=data, file_name=name)
                elif ext in {".html", ".htm"}:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        html = f.read()
                    components.html(html, height=700, scrolling=True)
                    with open(p, "rb") as f:
                        data = f.read()
                    st.download_button(label=f"Download {name}", data=data, file_name=name)
                else:
                    with open(p, "rb") as f:
                        data = f.read()
                    st.download_button(label=f"Download {name}", data=data, file_name=name)
            except Exception as ex:
                st.info(f"Could not preview {name}: {ex}")

def _try_open_latest() -> bool:
    # Prefer last artifacts recorded this session
    for p in reversed(st.session_state.get("last_artifacts", [])):
        if _embed_file(p):
            return True
    # Fallback: most recent file in watched dirs
    candidates = []
    for base in [st.session_state.get("artifact_dir"), VIS_DIR, OUTPUTS_DIR]:
        if base and os.path.isdir(base):
            for p in glob.glob(os.path.join(base, "**/*"), recursive=True):
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in {".html", ".htm", ".png", ".jpg", ".jpeg", ".svg"}:
                    try:
                        mtime = os.path.getmtime(p)
                    except Exception:
                        mtime = 0
                    candidates.append((mtime, p))
    if candidates:
        candidates.sort()
        return _embed_file(candidates[-1][1])
    return False

# Render history with modern chat components
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at bottom (fixed), above Clear Figures button
prompt = st.chat_input("Ask about long-term rocky reef ecological monitoring in the Gulf of California…")
if prompt:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        # If the user mentions a specific file name, try to embed it immediately
        try:
            _embed_artifacts_mentioned_in(prompt)
        except Exception as _uembed_err:
            logger.warning(f"Could not auto-embed from user prompt: {_uembed_err}")

    # Generate assistant response
    # If the user asks to "open it" or similar, auto-open the latest artifact without asking
    open_intent = any(k in prompt.lower() for k in [
        "open it", "open here", "preview it", "show it here", "open the map", "open map", "preview image"
    ])
    if open_intent:
        with st.chat_message("assistant"):
            if _try_open_latest():
                st.markdown("Opened the latest visualization below.")
            else:
                st.markdown("I couldn't find a recent visualization to open.")
        st.session_state.chat_history.append({"role": "assistant", "content": "Opened the latest visualization."})
    else:
        with st.chat_message("assistant"):
            if run_analysis is None or not AGENT_AVAILABLE:
                st.markdown("Agent unavailable. Please set required secrets and restart.")
            else:
                with st.spinner("Analyzing..."):
                    # Reset last_artifacts so 'open it' and listings only reflect this run
                    st.session_state["last_artifacts"] = []
                    before = _snapshot_artifacts()
                    try:
                        # Nudge the agent to avoid asking for a single reef when the user wants all reefs
                        def _augment_prompt(p: str) -> str:
                            low = (p or "").lower()
                            if any(k in low for k in ["map"]) and any(k in low for k in ["all reefs", "all the reefs", "every reef", "all reefs in the database", "tutti i reef", "todos los arrecifes"]):
                                preface = (
                                    "System: When the user asks for a map of all reefs, DO NOT prompt for a single reef name. "
                                    "Instead, query the database for all unique reefs with coordinates and render them on a single map. "
                                    "Try columns in order: (Latitude, Longitude), (latitude, longitude), (lat, lon). Use distinct Reef name column (try 'Reef', 'reef'). "
                                    "Center and zoom to fit all points. Save an HTML artifact (folium) named 'all_reefs_map.html' and a PNG snapshot 'all_reefs_map.png'.\n\n"
                                )
                                return preface + p
                            return p
                        # Pass recent chat history for context-aware follow-ups
                        answer = run_analysis(_augment_prompt(prompt), chat_history=st.session_state.chat_history)
                    except Exception as e:
                        answer = f"Error during analysis: {e}"
                    # Auto-generate CSV if requested and answer contains a species list
                    _maybe_generate_csv_from_answer(prompt, answer)
                    # Take snapshot after potential CSV creation so it appears under artifacts
                    after = _snapshot_artifacts()
                # Nicely format numbered lists if they come as one long line
                def _prettify_lists(text: str) -> str:
                    try:
                        if not isinstance(text, str):
                            return text
                        out = text
                        # Ensure a newline before list numbers like ' 1. ', ' 2. ', ...
                        out = re.sub(r"\s(\d{1,2}\.)\s", r"\n\1 ", out)
                        # Ensure hyphen bullets render as separate list items
                        out = re.sub(r"(?<!\n)\s-\s", "\n- ", out)
                        # Ensure asterisk bullets render as separate list items
                        out = re.sub(r"(?<!\n)\s\*\s", "\n* ", out)
                        # Make 'Notes and next steps' a proper section break
                        out = re.sub(r"\s*Notes and next steps\s*", "\n\n**Notes and next steps**\n\n", out, flags=re.IGNORECASE)
                        return out
                    except Exception:
                        return text
                st.markdown(_prettify_lists(answer))
                # If the assistant referenced specific artifact filenames, embed them inline
                try:
                    _embed_artifacts_mentioned_in(answer)
                except Exception as ex:
                    logger.warning(f"Could not auto-embed referenced artifacts: {ex}")
                _render_new_artifacts(before, after)
                # Always show consolidated artifacts panel (with download buttons)
                try:
                    _list_known_artifacts()
                except Exception as _list_err:
                    logger.warning(f"Could not list artifacts: {_list_err}")
            # Save to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer if 'answer' in locals() else ""})

# Clear Figures button
if st.button("Clear Figures"):
    import shutil
    shutil.rmtree(st.session_state["artifact_dir"])
    st.session_state["artifact_dir"] = tempfile.mkdtemp(prefix="eco_agent_")
    os.environ["ECO_AGENT_OUTPUT_DIR"] = st.session_state["artifact_dir"]
    st.success("Figures cleared.")

# Always show consolidated artifacts (regardless of whether a run just happened)
try:
    _list_known_artifacts()
except Exception as _final_list_err:
    logger.warning(f"Could not list artifacts at page end: {_final_list_err}")
