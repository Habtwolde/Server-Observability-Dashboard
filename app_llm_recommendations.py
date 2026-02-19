import os
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql as dbsql  # StatementState enum


# =========================
# UI
# =========================
st.set_page_config(page_title="Server Observability Dashboard", layout="wide")

# Light, modern CSS polish (no external deps)
st.markdown(
    """
<style>
/* Tighten page gutters a bit */
.block-container { padding-top: 2.0rem; padding-bottom: 2.0rem; }

/* Title */
h1 { margin-bottom: 0.5rem; }

/* Make metric value slightly larger */
[data-testid="stMetricValue"] { font-size: 2.2rem; }

/* Reduce vertical whitespace between elements */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"] > h3) {
  margin-top: 0.25rem;
}

/* Softer dataframe header */
div[data-testid="stDataFrame"] thead tr th {
  background: rgba(0,0,0,0.03);
}

/* Input alignment: keep labels compact */
label { font-size: 0.9rem !important; }

</style>
""",
    unsafe_allow_html=True,
)

st.title("Server Observability Dashboard")

# =========================
# Client (App identity)
# =========================
@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


w = get_workspace_client()


# =========================
# Config
# =========================
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip() or "47bde9279fec4222"

CATALOG = os.getenv("OBS_CATALOG", "").strip() or "btris_dbx"
SCHEMA = os.getenv("OBS_SCHEMA", "").strip() or "observability"

SERVERS_VIEW = os.getenv("OBS_SERVERS_VIEW", "").strip() or "v_servers"
PERF_TABLE = os.getenv("OBS_PERF_TABLE", "").strip() or "perfmon_metrics_delta"
EVENTS_TABLE = os.getenv("OBS_EVENTS_TABLE", "").strip() or "windows_events_delta"


# =========================
# LLM Endpoint (Databricks Model Serving)
# =========================
# Uses the same env var convention as your Triennial app, but defaults to a safe, known endpoint name.
ENDPOINT = (os.environ.get("TRIENNIAL_ENDPOINT") or "databricks-meta-llama-3-3-70b-instruct").strip()


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None


@st.cache_data(ttl=300)
def generate_llm_recommendations(
    endpoint_name: str,
    *,
    server: str,
    cpu_val: Optional[float],
    mem_val: Optional[float],
    cache_hit_val: Optional[float],
    top_waits_df: Optional[pd.DataFrame],
    meq_df: Optional[pd.DataFrame],
) -> str:
    """
    Generate recommendations using a Databricks Model Serving endpoint (LLM).
    - No extra environment configuration: relies on Databricks App identity via WorkspaceClient.
    - Uses the low-level invocations API to avoid SDK signature drift.
    """
    # Build compact context (avoid sending huge tables)
    waits_preview = []
    if isinstance(top_waits_df, pd.DataFrame) and not top_waits_df.empty:
        cols = [c for c in ["wait_type", "total_wait_ms", "samples"] if c in top_waits_df.columns]
        waits_preview = top_waits_df[cols].head(5).to_dict(orient="records")

    meq_preview = []
    if isinstance(meq_df, pd.DataFrame) and not meq_df.empty:
        cols = [c for c in ["cpu_query", "physical_reads_query", "datetime"] if c in meq_df.columns]
        meq_preview = meq_df[cols].head(3).to_dict(orient="records")

    context = {
        "server": server,
        "snapshot": {
            "max_cpu_utilization_pct": _safe_float(cpu_val),
            "max_memory_utilization_pct": _safe_float(mem_val),
            "cache_hit_ratio_pct": _safe_float(cache_hit_val),
        },
        "top_waits": waits_preview,
        "most_expensive_queries_examples": meq_preview,
    }

    system_msg = (
        "You are a senior database performance engineer (SQL Server / Windows workloads). "
        "Generate concise, actionable recommendations strictly grounded in the provided telemetry snapshot. "
        "If data is missing or N/A, acknowledge uncertainty rather than guessing. "
        "Do NOT invent query text, indexes, wait statistics, or root causes that are not provided."
    )

    user_msg = (
        "Given this telemetry snapshot, provide 3-7 bullet recommendations. "
        "Each bullet must follow: 'Issue → Action'. "
        "Focus on: memory pressure, CPU pressure, waits, caching/buffer pool, and query tuning next steps. "
        "Keep it practical and operational.

"
        f"Telemetry (JSON): {context}"
    )

    body = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }

    try:
        resp = w.api_client.do(
            "POST",
            f"/api/2.0/serving-endpoints/{endpoint_name}/invocations",
            body=body,
        )
    except Exception as e:
        # Bubble a readable error to the caller; UI will fall back.
        raise RuntimeError(f"LLM endpoint call failed: {e}")

    # Parse common response shapes
    content = None
    if isinstance(resp, dict):
        # Chat-completions style
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                content = msg.get("content")
            if content is None:
                content = choices[0].get("text") if isinstance(choices[0], dict) else None
        # Some endpoints return { "predictions": [ "..." ] }
        if content is None:
            preds = resp.get("predictions")
            if isinstance(preds, list) and preds:
                content = preds[0] if isinstance(preds[0], str) else None

    if not content or not isinstance(content, str):
        raise RuntimeError("LLM returned an unexpected response shape (no text content).")

    # Normalize to bullet list in the text area
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        if ln.startswith(("-", "•")):
            bullets.append(ln.lstrip("•").strip())
        else:
            bullets.append(ln)

    # If the model responded as paragraphs, keep it but add '-' prefix for readability
    if bullets and not bullets[0].startswith("-"):
        bullets = [("- " + b) if not b.startswith("-") else b for b in bullets]

    return "\n".join(bullets)


# =========================
# Helpers
# =========================
def q_ident(name: str) -> str:
    """Quote an identifier defensively using backticks (Databricks SQL)."""
    if name is None:
        return ""
    return f"`{name.replace('`', '``')}`"


def fqtn(catalog: str, schema: str, obj: str) -> str:
    """Fully qualified table/view name with safe quoting."""
    return f"{q_ident(catalog)}.{q_ident(schema)}.{q_ident(obj)}"


# =========================
# SQL runner (Statement Execution)
# =========================
@st.cache_data(ttl=60)
def run_query(query: str) -> pd.DataFrame:
    """
    Execute a SQL statement via Databricks Statement Execution API and return a DataFrame.
    Uses manifest schema when available (more reliable than result schema).
    """
    try:
        resp = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=query,
            wait_timeout="30s",
        )

        state = resp.status.state if resp.status else None

        if state in (dbsql.StatementState.PENDING, dbsql.StatementState.RUNNING):
            return pd.DataFrame()

        if state != dbsql.StatementState.SUCCEEDED:
            err_msg = None
            err_code = None
            if resp.status and resp.status.error:
                err_msg = getattr(resp.status.error, "message", None) or str(resp.status.error)
                err_code = getattr(resp.status.error, "error_code", None)
            raise RuntimeError(f"State={state} Code={err_code} Error={err_msg}")

        rows = resp.result.data_array if (resp.result and resp.result.data_array) else []

        cols: List[str] = []
        if resp.manifest and resp.manifest.schema and resp.manifest.schema.columns:
            cols = [c.name for c in resp.manifest.schema.columns]

        if not cols:
            if rows:
                cols = [f"col_{i+1}" for i in range(len(rows[0]))]
            else:
                return pd.DataFrame()

        return pd.DataFrame(rows, columns=cols)

    except Exception as e:
        st.error(f"Query failed. {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_table_columns(catalog: str, schema: str, table: str) -> List[str]:
    """Return list of column names for a table using information_schema."""
    info_schema = f"{q_ident(catalog)}.information_schema.columns"
    q = f"""
    SELECT column_name
    FROM {info_schema}
    WHERE table_schema = '{schema}'
      AND table_name = '{table}'
    ORDER BY ordinal_position
    """
    df = run_query(q)
    if df.empty or "column_name" not in df.columns:
        return []
    return df["column_name"].dropna().astype(str).tolist()


def pick_first_existing(existing: List[str], candidates: List[str]) -> Optional[str]:
    """
    Return the first candidate that exists in existing columns (case-insensitive),
    but return the actual stored column name.
    """
    lower_map = {c.lower(): c for c in existing}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


# =========================
# Column mappings (based on your Create_schema.ipynb cleaning)
# =========================
PERF_COLS: Dict[str, List[str]] = {
    # Date/time (support both cleaned snake_case and raw CSV headers)
    "datetime": ["datetime", "date_time", "timestamp", "time", "dt", "Datetime", "DateTime"],
    # Server (support cleaned and raw)
    "server_name": ["server_name", "servername", "server", "machine", "machine_name", "server name", "Server Name", "MachineName"],
    "max_cpu_utilization": ["max_cpu_utilization", "max_cpu", "cpu_max", "max_cpu_util", "max CPU utilization", "Max CPU utilization"],
    "max_memory_utilization": ["max_memory_utilization", "max_memory", "memory_max", "max_mem_util", "max Memory Utilization", "Max Memory Utilization"],
    # Waits: support cleaned variants AND raw CSV headers
    "wait_type": ["wait_type", "waittype", "Wait type", "WAIT TYPE", "wait type"],
    # Depending on cleaning rules, may become wait_timems or wait_time_ms or remain raw
    "wait_time_ms": [
        "wait_timems",
        "wait_time_ms",
        "wait_time",
        "wait_times",
        "waittime_ms",
        "wait_time_milliseconds",
        "Wait time(ms)",
        "WAIT TIME(MS)",
        "wait time(ms)",
        "Wait time(ms) ",
        "Wait time (ms)",
        "wait time (ms)",
    ],
    "cache_hit_ratio": ["cache_hit_ratio", "cach_hit_ratio", "cache_hit", "cach_hit", "Cach Hit Ratio", "Cache Hit Ratio"],
    "most_expensive_query_by_cpu": ["most_expensive_query_by_cpu", "Most expensive query by cpu"],
    "most_expensive_query_by_cpu_url": ["most_expensive_query_by_cpu_url", "Most expensive query by cpu  URL", "Most expensive query by cpu URL"],
    "most_expensive_query_by_physical_reads": ["most_expensive_query_by_physical_reads", "Most expensive query by physical reads"],
    "most_expensive_query_by_physical_reads_url": ["most_expensive_query_by_physical_reads_url", "Most expensive query by physical reads URL"],
    "jobsurl": ["jobsurl", "jobs_url", "JobsURL", "Jobs Url", "Jobs URL"],
}

EVENTS_COLS: Dict[str, List[str]] = {
    "server_name": ["server_name", "machinename", "machine_name", "server"],
    "time_created": ["time_created", "timecreated", "time_created_utc"],
    "level": ["LevelDisplayName", "leveldisplayname", "level", "severity"],
    "provider": ["ProviderName", "providername", "provider"],
    "id": ["ID", "id", "event_id"],
    "message": ["Message", "message", "msg"],
}


def must_pick(existing: List[str], key: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    return pick_first_existing(existing, mapping.get(key, []))


# =========================
# Load Servers
# =========================
servers_df = run_query(f"""
SELECT DISTINCT server_name
FROM {fqtn(CATALOG, SCHEMA, SERVERS_VIEW)}
ORDER BY server_name
""")

if servers_df.empty or "server_name" not in servers_df.columns:
    st.warning("No servers found (or the query failed).")
    st.stop()

server_list = servers_df["server_name"].dropna().astype(str).tolist()

# Search + Select on the same row (keeps the selector compact)
col_search, col_select, col_spacer = st.columns([1.2, 1.8, 3])

with col_search:
    search_text = st.text_input("Search Server", placeholder="Type to filter...")

filtered_servers = (
    [s for s in server_list if search_text.lower() in s.lower()]
    if search_text
    else server_list
)

with col_select:
    selected_server = st.selectbox("Select Server", filtered_servers, label_visibility="visible")


# =========================
# PERF: Read columns once
# =========================
perf_cols = get_table_columns(CATALOG, SCHEMA, PERF_TABLE)
if not perf_cols:
    st.warning(f"Could not read columns for {CATALOG}.{SCHEMA}.{PERF_TABLE}.")
    st.stop()

perf_dt = must_pick(perf_cols, "datetime", PERF_COLS)
perf_srv = must_pick(perf_cols, "server_name", PERF_COLS)
cpu_col = must_pick(perf_cols, "max_cpu_utilization", PERF_COLS)
mem_col = must_pick(perf_cols, "max_memory_utilization", PERF_COLS)
wait_type_col = must_pick(perf_cols, "wait_type", PERF_COLS)
wait_time_col = must_pick(perf_cols, "wait_time_ms", PERF_COLS)
cache_hit_col = must_pick(perf_cols, "cache_hit_ratio", PERF_COLS)

if not perf_srv or not perf_dt:
    st.error("Perf table is missing required keys (server_name and/or datetime).")
    st.stop()


# =========================
# Data fetch helpers (single-value snapshots)
# =========================
def latest_single_value(col_name: Optional[str], alias: str) -> Optional[float]:
    if not col_name:
        return None
    df = run_query(f"""
    SELECT {q_ident(col_name)} AS {alias}
    FROM {fqtn(CATALOG, SCHEMA, PERF_TABLE)}
    WHERE {q_ident(perf_srv)} = '{selected_server}'
    ORDER BY {q_ident(perf_dt)} DESC
    LIMIT 1
    """)
    if df.empty or alias not in df.columns:
        return None
    return pd.to_numeric(df[alias], errors="coerce").iloc[0]


cpu_val = latest_single_value(cpu_col, "cpu")
mem_val = latest_single_value(mem_col, "mem")
cache_val = latest_single_value(cache_hit_col, "cache_hit") if cache_hit_col else None


# =========================
# Main navigation (keeps Windows Events from being "hidden")
# =========================
tabs = st.tabs(["Overview", "Windows Events", "Most Expensive Queries"])

# ---------- Overview ----------
with tabs[0]:
    # KPI row
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])

    with k1:
        st.metric("Max CPU Utilization", value=f"{cpu_val:.0f}%" if pd.notna(cpu_val) else "N/A")
    with k2:
        st.metric("Max Memory Utilization", value=f"{mem_val:.0f}%" if pd.notna(mem_val) else "N/A")
    with k3:
        st.metric("Cache Hit Ratio", value=f"{cache_val:.0f}%" if (cache_hit_col and pd.notna(cache_val)) else "N/A")
    with k4:
        st.metric("Selected Server", value=selected_server)

    st.divider()

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.subheader("I/O Stats")
        st.info("I/O fields are not present in the provided perfmon CSV. Add disk metrics columns and I’ll wire this block.")

    with right:
        st.subheader("Waits breakdown")

        missing = []
        if not wait_type_col:
            missing.append("wait_type")
        if not wait_time_col:
            missing.append("wait_time (expected one of: wait_timems / wait_time_ms / wait_time / wait_times ...)")

        if missing:
            st.warning("Missing wait columns: " + ", ".join(missing))
        else:
            waits_df = run_query(f"""
            SELECT
              {q_ident(wait_type_col)} AS wait_type,
              SUM(CAST({q_ident(wait_time_col)} AS DOUBLE)) AS total_wait_ms,
              COUNT(*) AS samples
            FROM {fqtn(CATALOG, SCHEMA, PERF_TABLE)}
            WHERE {q_ident(perf_srv)} = '{selected_server}'
            GROUP BY {q_ident(wait_type_col)}
            ORDER BY total_wait_ms DESC
            LIMIT 10
            """)

            n_rows = int(waits_df.shape[0]) if isinstance(waits_df, pd.DataFrame) else 0
            table_height = min(10, max(1, n_rows)) * 35 + 38
            st.dataframe(waits_df, use_container_width=True, height=table_height)

# ---------- Windows Events ----------
with tabs[1]:
    st.subheader("Windows Events")

    ev_cols = get_table_columns(CATALOG, SCHEMA, EVENTS_TABLE)
    if not ev_cols:
        st.warning(f"Could not read columns for {CATALOG}.{SCHEMA}.{EVENTS_TABLE}.")
    else:
        ev_srv = must_pick(ev_cols, "server_name", EVENTS_COLS)
        ev_time = must_pick(ev_cols, "time_created", EVENTS_COLS)
        ev_lvl = must_pick(ev_cols, "level", EVENTS_COLS)
        ev_prov = must_pick(ev_cols, "provider", EVENTS_COLS)
        ev_id = must_pick(ev_cols, "id", EVENTS_COLS)
        ev_msg = must_pick(ev_cols, "message", EVENTS_COLS)

        if not (ev_srv and ev_time):
            st.warning("Events table missing required keys (server_name / time_created).")
        else:
            # Small controls
            c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
            with c1:
                level_filter = st.multiselect(
                    "Level",
                    options=["Critical", "Error", "Warning", "Information"],
                    default=[],
                    help="Optional filter. Leave empty to show all levels.",
                )
            with c2:
                max_rows = st.selectbox("Rows", options=[25, 50, 100, 200], index=0)

            # Build WHERE filters safely (string literals)
            where_bits = [f"{q_ident(ev_srv)} = '{selected_server}'"]
            if level_filter and ev_lvl:
                levels_sql = ", ".join([f"'{x}'" for x in level_filter])
                where_bits.append(f"{q_ident(ev_lvl)} IN ({levels_sql})")

            where_sql = " AND ".join(where_bits)

            select_bits = [f"{q_ident(ev_time)} AS time_created"]
            if ev_lvl:
                select_bits.append(f"{q_ident(ev_lvl)} AS level")
            if ev_prov:
                select_bits.append(f"{q_ident(ev_prov)} AS provider")
            if ev_id:
                select_bits.append(f"{q_ident(ev_id)} AS id")
            if ev_msg:
                select_bits.append(f"{q_ident(ev_msg)} AS message")

            events_df = run_query(f"""
            SELECT {", ".join(select_bits)}
            FROM {fqtn(CATALOG, SCHEMA, EVENTS_TABLE)}
            WHERE {where_sql}
            ORDER BY {q_ident(ev_time)} DESC
            LIMIT {int(max_rows)}
            """)

            # Dynamic height, but cap so it doesn't explode
            n_rows = int(events_df.shape[0]) if isinstance(events_df, pd.DataFrame) else 0
            table_height = min(18, max(6, n_rows)) * 35 + 38
            st.dataframe(events_df, use_container_width=True, height=table_height)

# ---------- Most Expensive Queries ----------
with tabs[2]:
    st.subheader("Most Expensive Queries")

    cpu_q_col = must_pick(perf_cols, "most_expensive_query_by_cpu", PERF_COLS)
    cpu_u_col = must_pick(perf_cols, "most_expensive_query_by_cpu_url", PERF_COLS)
    phy_q_col = must_pick(perf_cols, "most_expensive_query_by_physical_reads", PERF_COLS)
    phy_u_col = must_pick(perf_cols, "most_expensive_query_by_physical_reads_url", PERF_COLS)
    jobs_col = must_pick(perf_cols, "jobsurl", PERF_COLS)

    select_q = []
    if cpu_q_col:
        select_q.append(f"{q_ident(cpu_q_col)} AS cpu_query")
    if cpu_u_col:
        select_q.append(f"{q_ident(cpu_u_col)} AS cpu_query_url")
    if phy_q_col:
        select_q.append(f"{q_ident(phy_q_col)} AS physical_reads_query")
    if phy_u_col:
        select_q.append(f"{q_ident(phy_u_col)} AS physical_reads_query_url")
    if jobs_col:
        select_q.append(f"{q_ident(jobs_col)} AS jobs_url")
    select_q.append(f"{q_ident(perf_dt)} AS datetime")

    if select_q:
        meq_df = run_query(f"""
        SELECT {", ".join(select_q)}
        FROM {fqtn(CATALOG, SCHEMA, PERF_TABLE)}
        WHERE {q_ident(perf_srv)} = '{selected_server}'
        ORDER BY {q_ident(perf_dt)} DESC
        LIMIT 10
        """)
        # Dynamic height so the table fits returned rows (avoids empty filler rows)
        n_rows = int(meq_df.shape[0]) if isinstance(meq_df, pd.DataFrame) else 0
        table_height = min(12, max(1, n_rows)) * 35 + 38
        st.dataframe(meq_df, use_container_width=True, height=table_height)
    else:
        st.info("No expensive-query columns found in perf table.")

    st.divider()
    
    st.subheader("Recommendations")

    # Build top waits (for context)
    top_waits_df: Optional[pd.DataFrame] = None
    if wait_type_col and wait_time_col:
        top_waits_df = run_query(f"""
        SELECT
          {q_ident(wait_type_col)} AS wait_type,
          SUM(CAST({q_ident(wait_time_col)} AS DOUBLE)) AS total_wait_ms,
          COUNT(*) AS samples
        FROM {fqtn(CATALOG, SCHEMA, PERF_TABLE)}
        WHERE {q_ident(perf_srv)} = '{selected_server}'
        GROUP BY {q_ident(wait_type_col)}
        ORDER BY total_wait_ms DESC
        LIMIT 5
        """)

    # Generate with LLM endpoint (preferred)
    llm_text: Optional[str] = None
    llm_error: Optional[str] = None
    try:
        llm_text = generate_llm_recommendations(
            ENDPOINT,
            server=selected_server,
            cpu_val=cpu_val,
            mem_val=mem_val,
            cache_hit_val=cache_val,
            top_waits_df=top_waits_df,
            meq_df=meq_df if "meq_df" in locals() else None,
        )
    except Exception as e:
        llm_error = str(e)

    # Fallback: deterministic rules (kept as safety net)
    if not llm_text:
        recs: List[str] = []
        if pd.notna(cpu_val) and cpu_val >= 85:
            recs.append("CPU is consistently high → Check top CPU queries, missing indexes, and parallelism settings.")
        if pd.notna(mem_val) and mem_val >= 90:
            recs.append("Memory utilization is high → Review buffer pool pressure, plan cache bloat, and workload spikes.")
        if isinstance(top_waits_df, pd.DataFrame) and (not top_waits_df.empty) and ("wait_type" in top_waits_df.columns):
            top_wait = str(top_waits_df.iloc[0]["wait_type"])
            recs.append(f"Top wait type is {top_wait} → Investigate this wait category (workload vs IO vs CPU scheduling).")
        if cache_hit_col and pd.notna(cache_val) and cache_val < 90:
            recs.append("Cache hit ratio is low → Review memory allocation, IO subsystem, and query patterns causing churn.")
        if not recs:
            recs = ["No critical thresholds triggered → Review trends and waits for early signals."]
        llm_text = "
".join(f"- {r}" for r in recs)

        if llm_error:
            st.caption(f"LLM fallback used (reason: {llm_error})")

    st.text_area("Generated recommendations", value=llm_text, height=240)