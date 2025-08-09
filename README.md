# RAG System for Laboratory Instruments - Complete Implementation Guide

## System Overview

This document provides comprehensive specifications for implementing a production-ready Retrieval-Augmented Generation (RAG) system designed specifically for laboratory instrument manuals and protocols. The system manages **30 laboratories**, up to **100 instruments per lab**, and up to **50 manuals per instrument** (2-20MB each), providing intelligent query responses with accurate citations and safety awareness.

## 🔧 Core Technology Stack

**Database & Search:**
- **PostgreSQL + pgvector**: Single durable store for metadata + embeddings with transactional support
- **Row-Level Security (RLS)**: Multi-tenant isolation scoped by lab_id
- **Hybrid Retrieval**: pgvector (ANN) with structured filters

**Document Processing:**
- **Primary Parser**: LlamaParse for robust layout-aware PDF → JSON conversion
- **Fallback Parsers**: Unstructured + PDFium for reliability
- **Image Processing**: Extraction with page/bbox metadata, VLM captioning

**AI & Embeddings:**
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Response Generation**: GPT-4 with structured prompts and safety awareness
- **Vector Search**: Cosine distance similarity with HNSW indexing

**Infrastructure:**
- **Object Storage**: S3/GCS/Azure Blob for source documents and images
- **Caching**: Redis for query results and embeddings
- **API Framework**: FastAPI with async support
- **Protocol Support**: Model Context Protocol (MCP) v1.0 for agent integration

## System Components

### 1. Manufacturer Ingestion Pipeline
**Purpose**: Process official instrument manuals and technical documentation from manufacturers.

**Key Features:**
- Auto-classification of document types (operational, troubleshooting, safety, etc.)
- Hierarchical content analysis with section mapping
- Technical metadata extraction and enrichment
- Multi-modal processing (text, tables, images, diagrams)
- Source citation tracking with page-level references

**Input Documents:**
- Operational manuals and user guides
- Troubleshooting and diagnostic guides
- Technical specifications and datasheets
- Safety documentation and compliance guides
- Software manuals and installation guides

### 2. Laboratory Ingestion Pipeline
**Purpose**: Process lab-specific SOPs, protocols, and custom workflows with approval tracking.

**Key Features:**
- Multi-instrument protocol mapping and dependency tracking
- Approval workflow integration with audit trails
- Safety level classification and compliance monitoring
- Lab data isolation with strict access controls
- Protocol step extraction with quality checkpoints

**Input Documents:**
- Standard Operating Procedures (SOPs)
- Experimental protocols and methods
- Training materials and certification docs
- Maintenance logs and quality control records
- Custom workflows and lab-specific procedures

### 3. Response Engine
**Purpose**: Intelligent query processing with context-aware responses and accurate citations.

**Key Features:**
- Intent classification (troubleshooting, operation, safety, etc.)
- Hybrid retrieval combining vector similarity and keyword matching
- Multi-modal query support (text + images)
- Real-time streaming responses via WebSocket
- Confidence scoring and hallucination prevention
- Safety warning detection and highlighting

## Data Flow & Processing Pipeline

### Document Ingestion Flow
```
PDF/DOCX Input → Classification → Parsing → Content Filtering → 
Hierarchical Analysis → Metadata Enrichment → Embedding Generation → 
Database Storage → Index Creation
```

### Query Processing Flow
```
User Query → Preprocessing → Intent Classification → Target Identification → 
Hybrid Retrieval → Context Ranking → Response Generation → 
Validation → Citation Assembly → Delivery
```

## Database Schema

### Core Tables Structure
```sql
-- Manufacturer documents and content
manufacturer_documents (id, manufacturer_id, instrument_model, document_type, ...)
manufacturer_chunks (id, document_id, content, embedding, metadata, ...)

-- Laboratory documents with approval tracking
lab_documents (id, lab_id, instrument_ids, approval_status, version, ...)
lab_chunks (id, document_id, lab_id, content, embedding, safety_level, ...)

-- Protocol and workflow management
protocol_steps (id, protocol_id, step_number, instruments_required, ...)

-- Instrument and lab management
laboratories (id, name, location, compliance_level, ...)
instruments (id, lab_id, manufacturer_id, model, serial_number, ...)
```

### Multi-Tenant Isolation
- **Row-Level Security**: All queries automatically filtered by lab_id
- **Access Control**: User permissions validated at API level
- **Audit Logging**: Complete trail of document access and modifications
- **Data Separation**: Lab documents only accessible to authorized users

## API Specifications

### Manufacturer Ingestion API
```http
POST /ingest/document
Content-Type: application/json

{
  "file_path": "/uploads/manual.pdf",
  "filename": "instrument_manual_v2.pdf",
  "instrument_name": "Model XYZ-100",
  "doc_category": "operational_guide",
  "manufacturer_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Laboratory Ingestion API
```http
POST /lab/ingest/document
Content-Type: application/json

{
  "file_path": "/uploads/protocol.pdf",
  "lab_id": "550e8400-e29b-41d4-a716-446655440000",
  "instrument_ids": ["abc12345-1234-1234-1234-123456789012"],
  "document_type": "protocol",
  "approval_status": "approved",
  "safety_level": "medium"
}
```

### Response Engine API
```http
POST /api/v1/process-query
Content-Type: application/json

{
  "text": "How do I fix error E001 on my HPLC?",
  "lab_id": "550e8400-e29b-41d4-a716-446655440000",
  "instrument_ids": ["abc12345-1234-1234-1234-123456789012"],
  "search_type": "hybrid",
  "include_images": true
}
```

## Performance & Scalability

### Performance Targets
- **Query Response Time**: p95 < 800ms end-to-end
- **Document Processing**: 2-20MB PDFs in < 60 seconds
- **Concurrent Users**: Support 100+ simultaneous queries
- **Storage Efficiency**: Optimized for 150,000+ documents

### Optimization Strategies
- **Vector Indexing**: HNSW indexes for fast ANN search
- **Query Caching**: Redis-based caching with 60-minute TTL
- **Parallel Processing**: Concurrent retrieval from multiple sources
- **Connection Pooling**: Efficient database connection management
- **Content Pruning**: Intelligent candidate filtering before ranking

## Safety & Compliance Features

### Safety-First Design
- **Safety Warning Detection**: Automatic identification and highlighting
- **Risk Level Classification**: Document and procedure safety scoring
- **Critical Step Identification**: Highlighted procedural checkpoints
- **Emergency Information**: Priority access to safety-critical content

### Quality Assurance
- **Response Validation**: Confidence scoring and accuracy checks
- **Citation Verification**: Source attribution with page references
- **Hallucination Prevention**: Content grounding and fact-checking
- **Review Workflows**: Approval processes for sensitive content

## Deployment & Configuration

### Docker Deployment
```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: rag_lab_system
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: secure_password

  redis:
    image: redis:7-alpine

  manufacturer_ingestion:
    build: ./manufacturer_ingestion
    environment:
      - DATABASE_URL=postgresql://rag_user:secure_password@postgres:5432/rag_lab_system
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLAMAPARSE_API_KEY=${LLAMAPARSE_API_KEY}

  lab_ingestion:
    build: ./lab_ingestion
    environment:
      - DATABASE_URL=postgresql://rag_user:secure_password@postgres:5432/rag_lab_system
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  response_engine:
    build: ./response_engine
    environment:
      - DATABASE_URL=postgresql://rag_user:secure_password@postgres:5432/rag_lab_system
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8000:8000"
```

### Environment Configuration
```bash
# Core API keys
OPENAI_API_KEY=your_openai_api_key
LLAMAPARSE_API_KEY=your_llamaparse_api_key

# Database configuration
DATABASE_URL=postgresql://user:pass@host:port/database
REDIS_URL=redis://localhost:6379

# Object storage (choose one)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=rag-documents

# System configuration
MAX_FILE_SIZE_MB=50
EMBEDDING_MODEL=text-embedding-3-large
RESPONSE_TIMEOUT_SECONDS=30
```

## Development Best Practices

### Code Organization
- **Modular Design**: Clear separation of concerns with domain-driven structure
- **Interface Contracts**: Well-defined APIs between components
- **Error Handling**: Comprehensive error catching with proper logging
- **Testing Strategy**: 70% unit tests, 20% integration, 10% end-to-end

### Security Implementation
- **Input Validation**: Comprehensive parameter checking and sanitization
- **Authentication**: OAuth2/JWT with role-based access control
- **Data Encryption**: At-rest and in-transit encryption for sensitive data
- **Audit Logging**: Complete activity tracking for compliance

### Quality Assurance
- **Code Reviews**: Mandatory peer review for all changes
- **Automated Testing**: CI/CD pipeline with quality gates
- **Documentation**: API documentation and architectural decision records
- **Version Control**: Semantic versioning with backward compatibility

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- OpenAI API access
- LlamaParse API access

### Quick Setup
1. **Clone and Configure**:
   ```bash
   git clone LabRAG
   cd LabRAG
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```

2. **Start Services**:
   ```bash
   docker-compose up -d
   ```

3. **Initialize Database**:
   ```bash
   python scripts/initialize_database.py
   ```

4. **Test System**:
   ```bash
   python scripts/health_check.py
   ```

### Example Usage
```python
# Process manufacturer manual
from manufacturer_ingestion import ManufacturerIngestionPipeline

pipeline = ManufacturerIngestionPipeline('config/processing_config.yaml')
result = await pipeline.process_document({
    "file_path": "manuals/hplc_manual.pdf",
    "instrument_name": "HPLC-2000",
    "doc_category": "operational_guide"
})

# Query the system
from response_engine import ResponseEngine

engine = ResponseEngine('config/response_config.yaml')
response = await engine.process_query({
    "text": "How do I calibrate the HPLC detector?",
    "lab_id": "lab-001"
})
```
