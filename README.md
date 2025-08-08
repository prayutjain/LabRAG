# RAG System Design

**Core tech choices (with rationale):**

* **PostgreSQL + pgvector**: single durable store for metadata + embeddings; transactional, scalable enough for ≤30 labs; enables **RLS** for multi-tenant isolation and **hybrid SQL**.
* **Object Storage (S3/GCS/Azure Blob)**: source PDFs & extracted images (cheap, durable).
* **Parser**: pluggable. Default **LlamaParse** for robust layout-aware PDF → JSON; fallback **Unstructured**/**PDFium**; table extraction into Markdown/CSV; images extracted with page/bbox.
* **Embeddings**: OpenAI text-embedding or equivalent; cosine distance in pgvector; **1536–3072 dims** acceptable. Images optionally embedded for image search.
* **Hybrid retrieval**: pgvector (ANN) + **BM25** (pg\_trgm or pgroonga) with **structured filters** (lab\_id, instrument\_id, section\_type). This keeps latency low and boosts keyword recall (your users want both NL + keyword).
* **AuthZ**: **Row-Level Security** in Postgres scoped by `lab_id`; optional per-instrument roles.
* **Citations**: store **(doc\_id, page\_num, bbox, section\_path)** per chunk for exact callouts.
* **Latency**: p95 retrieval target **< 800 ms** end-to-end for ≤30 labs via ANN indexes, caching, and candidate pruning.

---

# Global Conventions (All Modules)

* **Tenancy**: Every row carries `lab_id` (UUID). Postgres **RLS** ON by default:

  * Policy: user can only `SELECT` rows where `lab_id ∈ user.allowed_labs`.
  * Manufacturer docs may be **shared**; enforce via a special `lab_id = NULL` with a **read-only policy** and a **mapping** table that grants visibility per lab.
* **IDs**: UUIDv7 for monotonicity.
* **Time**: `TIMESTAMPTZ` in UTC.
* **Storage layout**: `s3://{env}/labs/{lab_id}/{doc_id}/source.pdf` and `.../images/{image_id}.png`.
* **Chunking**: **200–400 tokens** target; overlap **40–60 tokens**; tables preserved as Markdown; keep page anchors.
* **PII/Secrets**: redact via rule-based patterns + allowlist (SOP authors, emails) prior to indexing (configurable).
* **Quality Gates**: ingestion refuses files > 200 MB, pages > 5,000, or encrypted PDFs unless `allow_encrypted=true`.

---

# Module I — Manufacturer Data Ingestion

## Inputs

* **Files**: PDF, DOCX, HTML (normalized to PDF first), images (PNG/JPG).
* **Metadata (required if manual labeling)**:

  * `manufacturer_name`, `instrument_model`, `document_type` ∈ {manual, troubleshooting, specification, software\_guide, safety, maintenance, install, warranty}, `title`, `version` (optional), `language` (default `en`).

## Pipeline (deterministic steps)

1. **Labeling**

* If `auto_label=true`, infer `document_type`, `title`, `instrument_model` via filename + cover page classifier.
* Else enforce presence of required metadata; reject otherwise.

2. **Parsing**

* Default: LlamaParse to **JSON** structure containing:

  * pages (text with spans + bbox), tables (as matrix + bbox), images (extracted + bbox), hyperlinks, headings.
* Fallback: Unstructured + Camelot/Tabula for tables.

3. **Cleaning**

* Strip **headers/footers**, page numbers, repeated watermarks.
* Deduplicate near-identical pages (cosine similarity on minhash shingles).
* Normalize unicode, units (mm/in, °C/°F).

4. **Hierarchy & Chunking**

* Build **section tree** from headings / TOC.
* Emit **chunks** with:

  * `content`, `chunk_type` ∈ {text, table, procedure, specification}, `page_start`, `page_end`, `bbox_list`, `section_path`, `document_id`.
  * Target 200–400 tokens; keep tables as Markdown fenced blocks.

5. **Hydration**

* **Caption** tables/images; **summary(≤2 sentences)** per chunk; **tags** (top-k keywords); optional **context\_label** (e.g., “Error codes”, “Parameters”).
* Image captioning via a VLM; add `alt_text` and `detected_entities`.

6. **Embedding**

* Text: `text-embedding-*`; Tables: embed **(summary + table header)**; Images: optional CLIP/VLM embedding.

7. **Persist**

* **Object storage**: original file + images.
* **DB**: documents, sections, chunks, embeddings, and search facets.

### Success/Fail Criteria

* Success: ≥95% of pages produce at least one chunk; ≤1% parser errors; TOC alignment ≥90%.
* Fail fast: encrypted/unreadable PDFs, empty text, or no headings if `require_headings=true`.

---

# Module II — Laboratory Data Ingestion

**Same pipeline** with two differences:

1. **Protocol Context**

* Add `protocol_id` and `experiment_context` (JSON) at **document** and **chunk** levels.
* Enforce mapping to **instrument(s)** because labs may **combine instruments**.

  * `instrument_linkage`: array of `{instrument_id, role}` (e.g., “primary”, “pre-processing”).

2. **Access Control & Lifecycle**

* States: `draft → in_review → approved → retired`.
* Only `approved` content is indexed by default; allow `include_draft=true` at ingestion time for sandbox indices.

---

# Module III — Response Engine (MCP-ready)

## Inputs (per request)

* `lab_id` (required), `user_id` (required)
* **Query mode** (one of):

  * `natural_language`: free text
  * `keyword`: boolean search string (supports `AND/OR/NOT`, quotes)
  * `image_query`: image + optional text
  * `query_with_context`: includes `instrument_ids[]`, `protocol_id`, or `experiment_config`
* `mode_hint` (optional): {troubleshoot, protocol, parameters, general}
* `max_context_chunks` (default 12), `max_tokens_answer` (default 800)

## Orchestration Steps

1. **Intent Classifier**

* Route to one of {troubleshoot, protocol, parameters, general}. If provided, trust `mode_hint`.

2. **Target Scoping**

* Determine **candidate instruments/protocols** from:

  * explicit inputs, user’s session defaults, or **recent activity** cache.
* Build SQL **filters**: `lab_id`, `instrument_id IN (...)`, `document_type IN (...)`, states (`approved`).

3. **Hybrid Retrieval**

* **Phase A: Keyword pre-filter (BM25/pg\_trgm)** → top 200 ids
* **Phase B: Vector ANN** on the shortlist (or full if no keywords)
* **Phase C: Weighted fusion**

  * `score = 0.50*vector + 0.35*bm25 + 0.10*source_weight + 0.05*recency`
  * `source_weight`: lab SOPs > lab protocols > manufacturer troubleshooting > manuals

4. **Rerank (optional)**

* LLM rerank over top 40 by **query-chunk Q\&A fit** (latency guard: 100 ms budget per 20 items; disable on tight SLO).

5. **Citations & Snippet Stitching**

* For each selected chunk: include `doc_id, title, section_path, page_num(s), bbox_list`.
* Build **answer context** as structured JSON; **no hallucinated facts**—only copyable spans + normalized units.

6. **Answer Synthesis**

* Produce **pydantic-validated** JSON with:

  * `answer_markdown`, `citations[]`, `supporting_images[]`, `safety_notes[]`, `assumptions[]`
* Strictly enforce **grounding** by injecting context + requiring **inline citation IDs** at sentence level.

7. **Caching**

* Memoize `(query, scope)` → `(topK ids, final answer)` for **120s** TTL (evict on new ingestion in same scope).

---

# Data Model (tables you can hand to the agents)

I’ll keep your schemas and add multi-tenancy + enforcement fields. (Only deltas shown—your originals are good.)

```sql
-- Add lab_id to shared tables where applicable & enforce RLS
ALTER TABLE manufacturer_documents ADD COLUMN lab_visibility JSONB DEFAULT '[]'::jsonb; -- list of lab_ids allowed (empty = global)
ALTER TABLE manufacturer_chunks ADD COLUMN section_path TEXT[];
ALTER TABLE lab_documents ADD COLUMN state TEXT CHECK (state IN ('draft','in_review','approved','retired')) DEFAULT 'draft';
ALTER TABLE lab_chunks ADD COLUMN section_path TEXT[];
ALTER TABLE images ADD COLUMN page_num INTEGER, ADD COLUMN bbox JSONB;

-- Hybrid search helpers
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Example index strategy
CREATE INDEX ON manufacturer_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON manufacturer_chunks USING gin (to_tsvector('english', content));
CREATE INDEX ON lab_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON lab_chunks USING gin (to_tsvector('english', content));

-- RLS sketch
ALTER TABLE lab_documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY lab_docs_policy ON lab_documents
  USING (lab_id = current_setting('app.current_lab')::uuid);

-- Set app.current_lab at session start after OIDC auth.
```

---

# Retrieval SQL (reference queries)

**Cross-source hybrid (scoped):**

```sql
WITH scope AS (
  SELECT unnest($1::uuid[]) AS instrument_id
),
kw AS (
  SELECT mc.id, md.id AS doc_id, md.title, mc.content, mc.embedding,  -- manufacturer
         0.35 * ts_rank_cd(to_tsvector('english', mc.content), plainto_tsquery($2)) AS bm25,
         0.10 * CASE md.document_type
             WHEN 'troubleshooting' THEN 1.0
             WHEN 'manual' THEN 0.6 ELSE 0.3 END AS source_w
  FROM manufacturer_chunks mc
  JOIN manufacturer_documents md ON md.id = mc.document_id
  JOIN instrument_document_mappings map ON map.document_id = md.id AND map.document_source='manufacturer'
  JOIN scope s ON s.instrument_id = map.instrument_id
  WHERE to_tsvector('english', mc.content) @@ plainto_tsquery($2)
  UNION ALL
  SELECT lc.id, ld.id, ld.title, lc.content, lc.embedding,
         0.35 * ts_rank_cd(to_tsvector('english', lc.content), plainto_tsquery($2)) AS bm25,
         0.10 * CASE ld.document_type
             WHEN 'sop' THEN 1.0 WHEN 'protocol' THEN 0.9 ELSE 0.5 END AS source_w
  FROM lab_chunks lc
  JOIN lab_documents ld ON ld.id = lc.document_id AND ld.state='approved'
  JOIN scope s ON s.instrument_id = lc.instrument_id
  WHERE to_tsvector('english', lc.content) @@ plainto_tsquery($2)
),
vec AS (
  SELECT id, doc_id, title, content, embedding,
         0.50 * (1 - (embedding <=> $3)) AS vec_score, bm25, source_w
  FROM kw
)
SELECT id, doc_id, title,
       (vec_score + bm25 + source_w) AS final_score
FROM vec
ORDER BY final_score DESC
LIMIT $4;
```

---

# API Contracts (for AI agents)

## Ingestion API

**POST** `/v1/ingest`

* **Headers**: `Authorization: Bearer <token>`, `X-Lab-ID: <uuid>`
* **Body (multipart)**: `file`, `metadata.json`

```json
{
  "source": "manufacturer|lab",
  "label_mode": "auto|manual",
  "document_type": "manual|troubleshooting|specification|safety|... (required if manual)",
  "manufacturer_name": "string",
  "instrument_model": "string",
  "title": "string",
  "version": "string",
  "language": "en",
  "protocol_id": "uuid (lab only, optional)",
  "instrument_links": [{"instrument_id":"uuid","role":"primary|aux"}],
  "allow_encrypted": false,
  "index_drafts": false
}
```

**Response**

```json
{ "job_id":"uuid", "status":"queued" }
```

**GET** `/v1/ingest/{job_id}`

```json
{
  "status":"succeeded|failed|running",
  "doc_id":"uuid",
  "error": null
}
```

## Retrieval API

**POST** `/v1/retrieve`

```json
{
  "lab_id":"uuid",
  "user_id":"uuid",
  "query_mode":"natural_language|keyword|image_query|query_with_context",
  "query":"string",
  "instrument_ids":["uuid"],
  "protocol_id":"uuid",
  "mode_hint":"troubleshoot|protocol|parameters|general",
  "max_context_chunks":12,
  "max_tokens_answer":800
}
```

**Response (validated)**

```json
{
  "answer_markdown":"string",
  "citations":[
    {
      "doc_id":"uuid",
      "doc_title":"string",
      "source":"manufacturer|lab",
      "page":12,
      "section_path":["3. Maintenance","Filters"],
      "bbox":[[x1,y1,x2,y2]],
      "snippet":"exact text span..."
    }
  ],
  "supporting_images":[{"image_id":"uuid","page":5,"alt_text":"string","url":"signed"}],
  "assumptions":["string"],
  "safety_notes":["string"],
  "latency_ms": 420
}
```

---

# Observability & Guardrails

* **Metrics**: parse success rate, chunk density, avg tokens/chunk, p95 retrieval latency, top-k overlap\@k on eval set, answer-level grounding rate.
* **Tracing**: link **answer → chunk\_ids → (doc\_id,page,bbox)**.
* **Eval set**: 50–100 canonical lab Qs (troubleshooting/protocol/parameters) with **gold citations**.
* **Hallucination guard**: synthesis step requires **citation per sentence**; if none available, return **“insufficient evidence”** with best candidate links.

---

# Security & Isolation

* **RLS** everywhere; every request sets `app.current_lab`.
* **Signed URLs** for object storage downloads (time-boxed).
* **PII scrubbing** before index; logs exclude content by default; opt-in debug sampling.

---

# Performance Targets (≤30 labs)

* Ingestion throughput: **5–10 docs/min** per worker (parallel parser pool).
* Retrieval p95: **<800 ms** (pre-filter 200 via BM25, ANN lists=100).
* Cold start embedding batcher: 256–512 tokens per call; concurrency limited by rate-limits.

---

# YAML Playbook — Ingestion (Module I & II)

```yaml
version: "1.0"
name: "ingestion_playbook"
defaults:
  chunk:
    target_tokens: 300
    overlap_tokens: 50
  acceptance:
    min_page_coverage: 0.95
    max_parser_error_ratio: 0.01
  storage:
    bucket: "rag-labs"
steps:
  - id: label
    run: labeler.apply
    inputs:
      mode: "{{ label_mode | default('auto') }}"
      required_if_manual:
        - document_type
        - title
        - instrument_model
  - id: parse
    run: parser.parse_pdf
    inputs:
      engine: "llamaparse"
      fallback: ["unstructured", "pdfium"]
      extract:
        - text
        - tables
        - images
        - headings
        - links
  - id: clean
    run: cleaner.run
    inputs:
      remove_headers_footers: true
      normalize_units: true
      dedupe_pages: true
  - id: hierarchy_chunk
    run: chunker.build
    inputs:
      from_headings: true
      chunk_size: "{{ defaults.chunk.target_tokens }}"
      chunk_overlap: "{{ defaults.chunk.overlap_tokens }}"
      keep_tables_as_markdown: true
  - id: hydrate
    run: hydrator.apply
    inputs:
      summarizer: "gpt"
      tagger: "keybert"
      image_captioner: "vlm"
  - id: embed
    run: embedder.compute
    inputs:
      model: "text-embedding"
      pooling: "default"
  - id: persist
    run: writer.commit
    inputs:
      db: "postgres"
      vector: "pgvector"
      object_store: "s3"
      path_template: "labs/{{ lab_id }}/{{ doc_id }}/"
  - id: verify
    run: qa.verify
    inputs:
      min_page_coverage: "{{ defaults.acceptance.min_page_coverage }}"
      max_parser_error_ratio: "{{ defaults.acceptance.max_parser_error_ratio }}"
on_failure:
  - run: notify.ops
```

---

# YAML Playbook — Retrieval Orchestration (Module III)

```yaml
version: "1.0"
name: "retrieval_playbook"
sla:
  p95_ms: 800
params:
  max_context_chunks: 12
stages:
  - id: intent
    run: intent.router
    inputs:
      mode_hint: "{{ request.mode_hint }}"
  - id: scope
    run: scope.builder
    inputs:
      lab_id: "{{ request.lab_id }}"
      instrument_ids: "{{ request.instrument_ids }}"
      protocol_id: "{{ request.protocol_id }}"
  - id: prefilter
    run: search.keyword
    inputs:
      query: "{{ request.query }}"
      max_docs: 200
  - id: vector
    run: search.vector
    inputs:
      query_embedder: "text-embedding"
      shortlist_from: "prefilter"
  - id: fuse
    run: rank.fuse
    inputs:
      weights:
        vector: 0.50
        bm25: 0.35
        source_weight: 0.10
        recency: 0.05
      top_k: 40
  - id: rerank
    when: "{{ request.latency_budget_ms | default(800) > 650 }}"
    run: rank.llm
    inputs:
      top_k_in: 40
      top_k_out: "{{ params.max_context_chunks }}"
  - id: assemble
    run: context.assemble
    inputs:
      require_citation_per_sentence: true
  - id: synthesize
    run: llm.answer
    inputs:
      output_schema: "AnswerV1"
      max_tokens: "{{ request.max_tokens_answer | default(800) }}"
  - id: cache
    run: cache.store
    inputs:
      ttl_seconds: 120
```

---

# Pydantic Schemas (key ones)

```python
class Citation(BaseModel):
    doc_id: UUID
    doc_title: str
    source: Literal["manufacturer","lab"]
    page: int
    section_path: List[str]
    bbox: List[Tuple[float,float,float,float]]
    snippet: str

class AnswerV1(BaseModel):
    answer_markdown: str
    citations: List[Citation]
    supporting_images: List[Dict[str, Any]] = []
    assumptions: List[str] = []
    safety_notes: List[str] = []
    latency_ms: int
```


