# Server Observability Dashboard (Databricks + Streamlit + LLM)

A production-ready **Server Observability Dashboard** built with **Streamlit on Databricks**, enhanced with **LLM-powered SQL performance analysis**.

The application allows infrastructure and database teams to:

- Monitor server performance metrics
- Analyze most expensive SQL queries
- Generate AI-powered performance recommendations
- Ask contextual, custom questions about expensive queries
- Interact dynamically with query-level diagnostics

---

## Architecture Overview

```
Databricks Compute
    │
    ├── Streamlit App (app.py)
    │
    ├── Performance Metrics (Delta / SQL / Excel)
    │
    └── Databricks Model Serving Endpoint
            └── meta-llama-3-70b-instruct
```

---

## Key Features

### 1) Server Observability
- CPU utilization
- Memory utilization
- Wait types
- Cache hit ratio
- Expensive query tracking

### 2) Most Expensive Queries Tab

Displays:
- CPU-heavy queries
- Physical read-heavy queries
- Related URLs (Power BI / Jobs)
- Datetime reference

Table rows are indexed (0,1,2,...) so users can reference them in AI prompts.

### 3) LLM-Powered Recommendations

Two user-controlled workflows:

#### A) Generate Recommendations
A button that:
- Sends the expensive query context to the LLM
- Generates structured recommendations:
  - Executive diagnosis
  - Query-level analysis
  - Optimization guidance

Generation happens **only when the user clicks**.

#### B) Ask Custom Questions
A second button that:
- Reveals a prompt input field
- Allows users to ask conceptual or targeted questions

Example prompts:
```
Why is query 0 expensive?
Compare query 0 and query 2.
Is this replication-related workload?
```

The LLM uses the table data as context and responds accordingly.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Compute | Databricks |
| Model | meta-llama-3-70b-instruct |
| Data | Delta Tables / SQL / Excel |
| Integration | Databricks Model Serving |

---

## Project Structure

Recommended repository layout:

```
.
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── sample_inputs/          # optional sample files (do not include sensitive data)
    └── schema/                 # optional schemas / docs
```

---

## Setup Instructions

### 1) Clone Repository

```bash
git clone https://github.com/your-org/server-observability-dashboard.git
cd server-observability-dashboard
```

### 2) Install Dependencies (Local Testing)

```bash
pip install -r requirements.txt
```

### 3) Run Locally (Optional)

```bash
streamlit run app.py
```

> Note: Local execution requires your environment to have access to the same data sources and (optionally) the same Databricks model endpoint.  
> If using Databricks Model Serving from local, you’ll need a PAT token + workspace URL and to adapt the endpoint call accordingly.

---

## Databricks Deployment

### Prerequisites
- A Databricks workspace with **Databricks Apps** enabled
- A running compute/cluster that supports Streamlit apps
- A deployed **Databricks Model Serving** endpoint (see below)
- Permissions to read the underlying data sources (Delta/SQL/Excel)

### Steps
1. Create a Databricks App.
2. Upload `app.py` (and any required supporting files).
3. Set the app’s compute settings (cluster/warehouse as needed).
4. Confirm the LLM endpoint name and permissions.
5. Launch the app and validate the UI (tabs, table population, LLM actions).

---

## Model Endpoint Configuration

Ensure:
- The model serving endpoint is deployed
- Your app identity/user has permissions to query the endpoint
- The endpoint name matches what is configured in `app.py`

Example configuration (inside `app.py`):
```python
LLM_ENDPOINT = "databricks-meta-llama-3-70b-instruct"
```

---

## Prompt Engineering Strategy

The app sends a compact, structured context to the LLM including:
- Server-level performance signals (CPU/memory/waits/cache)
- Expensive query details (CPU-heavy, read-heavy)
- Row indices (0..n) so users can refer to queries reliably
- Safety & formatting instructions

The LLM is instructed to:
- Avoid hallucination (use only provided context)
- Provide actionable SQL/DB performance advice
- Separate executive diagnosis from query-level findings
- Keep responses readable (headings, bullets, short sections)

---

## Security Considerations

- No external API calls beyond Databricks Model Serving
- No secrets hardcoded in the repository
- No automatic LLM invocation (LLM calls are user-triggered only)
- Query text is analyzed, but **never executed**
- Avoid committing sensitive data (queries, credentials, server names) to GitHub

Recommended:
- Use Databricks secrets / environment variables for tokens and workspace URLs
- Use least-privilege permissions for data sources and serving endpoints

---

## Performance Considerations

- LLM generation is user-triggered to control cost
- Stateless UI interactions (safe reruns)
- Cached data loading is recommended for large datasets
- Use pagination or filtering if expensive query tables grow large

---

## Example Use Cases

- DBA performance triage & incident response
- Memory vs CPU bottleneck analysis (correlate with waits)
- Replication workload identification (e.g., `sp_replcmds` patterns)
- Query optimization planning (indexes, sargability, batching, concurrency)
- Executive performance reporting with AI-generated summaries

---

## Future Improvements

- Vector search / retrieval augmentation for query patterns & KB articles
- Query plan ingestion (actual plans) for more accurate recommendations
- Wait-stat correlation modeling & anomaly detection
- Automated index suggestion engine (with guardrails)
- Historical time-series forecasting & alerting

---

## License

Internal / Private Project (update as needed)

---

## Screenshots

Add screenshots to document the UI:

```
docs/
  screenshots/
    server_observability_dashboard.png
    most_expensive_queries.png
```

Then reference them here:

- `docs/screenshots/server_observability_dashboard.png`
- `docs/screenshots/most_expensive_queries.png`
