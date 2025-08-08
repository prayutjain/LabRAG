# Laboratory Data Ingestion Pipeline

## Project Structure
```
lab_ingestion/
├── src/
│   ├── segmentation/
│   │   ├── lab_document_classifier.py
│   │   ├── protocol_parser.py
│   │   └── sop_processor.py
│   ├── indexing/
│   │   ├── lab_hierarchy_analyzer.py
│   │   └── protocol_mapper.py
│   ├── hydration/
│   │   ├── lab_metadata_enricher.py
│   │   └── approval_tracker.py
│   └── storage/
│       └── lab_database_writer.py
├── config/
│   └── lab_config.yaml
├── requirements.txt
└── lab_main.py
```

## User Inputs

### LabDocumentInput Schema
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import date

class LabDocumentInput(BaseModel):
    file_path: str = Field(..., description="Path to document file")
    filename: str = Field(..., description="Original filename")
    lab_id: str = Field(..., description="Laboratory identifier")
    instrument_ids: List[str] = Field(..., description="Associated instrument UUIDs")
    document_type: Literal[
        "sop",
        "protocol", 
        "training_material",
        "maintenance_log",
        "quality_control",
        "custom_workflow",
        "safety_procedure",
        "method_validation"
    ] = Field(..., description="Lab document type")
    title: str = Field(..., description="Document title")
    version: str = Field(..., description="Document version")
    author: str = Field(..., description="Document author")
    approval_status: Literal["draft", "review", "approved", "archived"] = Field(..., description="Approval status")
    approved_by: Optional[str] = Field(None, description="Approver name")
    approved_date: Optional[date] = Field(None, description="Approval date")
    review_date: Optional[date] = Field(None, description="Next review date")
    protocol_id: Optional[str] = Field(None, description="Protocol identifier for multi-instrument procedures")
    experiment_types: List[str] = Field(default=[], description="Types of experiments this applies to")
    safety_level: Literal["low", "medium", "high", "critical"] = Field(default="medium", description="Safety classification")
```

**Validation Requirements:**
- `file_path`: Must exist and be readable PDF/DOCX file
- `filename`: Non-empty string, max 255 characters  
- `lab_id`: Valid UUID format
- `instrument_ids`: Non-empty list of valid UUIDs
- `title`: Non-empty string, max 200 characters
- `version`: Non-empty string, max 50 characters
- `author`: Non-empty string, max 100 characters
- File size: Maximum 50MB

If validation fails, respond with HTTP 400 and error details.

## API Specifications

### Framework: FastAPI
### Services Used:
- **OpenAI API** (https://api.openai.com/v1/)
- **LlamaParse API** (https://cloud.llamaindex.ai/)
- **PostgreSQL with pgvector extension**

### Configuration
Read from `config/lab_config.yaml`:
```yaml
apis:
  openai_api_key: ${OPENAI_API_KEY}
  llamaparse_api_key: ${LLAMAPARSE_API_KEY}

database:
  connection_string: ${DATABASE_URL}

lab_processing:
  multi_tenant: true
  data_isolation: strict
  approval_workflow: true
  max_file_size_mb: 50
  supported_formats: ["pdf", "docx"]

lab_classification:
  auto_classify: true
  confidence_threshold: 0.9
  require_manual_review: true

protocol_processing:
  detect_multi_instrument: true
  cross_reference_instruments: true
  track_dependencies: true

metadata_enrichment:
  track_approvals: true
  monitor_review_dates: true
  extract_safety_info: true
  identify_critical_steps: true
```

## API Endpoints

### POST /lab/ingest/document
Process and ingest a laboratory document into the system with protocol mapping and approval tracking.

**Request Body:**
```json
{
  "file_path": "/uploads/protocol.pdf",
  "filename": "hplc_analysis_protocol_v3.pdf",
  "lab_id": "550e8400-e29b-41d4-a716-446655440000",
  "instrument_ids": ["abc12345-1234-1234-1234-123456789012", "def67890-5678-5678-5678-567890123456"],
  "document_type": "protocol",
  "title": "HPLC Analysis Protocol for Drug Compounds",
  "version": "3.1",
  "author": "Dr. Jane Smith",
  "approval_status": "approved",
  "approved_by": "Dr. John Doe",
  "approved_date": "2025-01-15",
  "review_date": "2026-01-15",
  "protocol_id": "PROT-2025-001",
  "experiment_types": ["drug_analysis", "quality_control"],
  "safety_level": "medium"
}
```

### Success Response (HTTP 200)
```json
{
  "success": true,
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "sections_count": 12,
  "chunks_count": 45,
  "protocol_steps_count": 8,
  "processing_time_seconds": 32.1,
  "metadata": {
    "total_pages": 24,
    "tables_extracted": 5,
    "safety_warnings_found": 3,
    "multi_instrument_protocol": true,
    "approval_tracking": {
      "status": "approved",
      "approved_by": "Dr. John Doe",
      "approval_date": "2025-01-15"
    },
    "classification": {
      "confidence": 0.94,
      "detected_type": "protocol",
      "safety_level": "medium"
    }
  }
}
```

### Error Response (HTTP 400/500)
```json
{
  "error": "Invalid lab_id format",
  "code": 400,
  "details": {
    "field": "lab_id",
    "received_value": "invalid-uuid",
    "expected_format": "Valid UUID string"
  }
}
```

## Processing Pipeline Steps

### Step 1: Lab Document Classification
- Auto-classify document type using lab-specific AI classifier
- Validate against laboratory document categories
- Assess safety level and compliance requirements
- Flag for manual review if confidence < 0.9

### Step 2: Protocol Structure Analysis  
- Detect multi-instrument protocols
- Extract protocol steps and dependencies
- Map instrument requirements per step
- Identify critical control points

### Step 3: Content Processing
- Use LlamaParse for structured content extraction
- Preserve laboratory table formats (results, parameters)
- Extract safety warnings and compliance notes
- Process procedural diagrams and flow charts

### Step 4: Lab-Specific Metadata Enrichment
- Generate procedure summaries and safety tags
- Extract critical steps and quality checkpoints
- Identify troubleshooting indicators
- Tag compliance requirements (FDA, ISO, GLP)

### Step 5: Approval Workflow Integration
- Track document approval status and history
- Monitor review dates and renewal requirements
- Enforce lab data isolation policies
- Create audit trail for compliance

### Step 6: Multi-Instrument Protocol Mapping
- Link protocol steps to specific instruments
- Track instrument dependencies and setup requirements
- Generate cross-references between related procedures
- Estimate timing and resource requirements

### Step 7: Database Storage with Lab Isolation
- Store in PostgreSQL with strict lab data isolation
- Maintain protocol step relationships
- Create lab-specific search indexes
- Enable approval workflow queries

## Database Schema

### lab_documents
```sql
CREATE TABLE lab_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lab_id UUID NOT NULL,
    instrument_ids UUID[] NOT NULL,
    protocol_id TEXT,
    document_type TEXT NOT NULL,
    title TEXT NOT NULL,
    version TEXT NOT NULL,
    author TEXT NOT NULL,
    approval_status TEXT NOT NULL CHECK (approval_status IN ('draft', 'review', 'approved', 'archived')),
    approved_by TEXT,
    approved_date DATE,
    review_date DATE,
    file_path TEXT,
    safety_level TEXT CHECK (safety_level IN ('low', 'medium', 'high', 'critical')),
    compliance_requirements TEXT[],
    experiment_types TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### lab_chunks  
```sql
CREATE TABLE lab_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES lab_documents(id),
    section_id UUID REFERENCES lab_sections(id),
    lab_id UUID NOT NULL,
    instrument_ids UUID[] NOT NULL,
    protocol_id TEXT,
    content TEXT NOT NULL,
    markdown_table TEXT,
    chunk_index INTEGER,
    embedding vector(3072),
    chunk_type TEXT NOT NULL,
    safety_level TEXT,
    requires_training BOOLEAN DEFAULT false,
    approval_required BOOLEAN DEFAULT false,
    compliance_tags TEXT[],
    critical_steps TEXT[],
    quality_checkpoints TEXT[],
    metadata JSONB,
    image_refs TEXT[],
    page_references INTEGER[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### protocol_steps
```sql
CREATE TABLE protocol_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    protocol_id TEXT NOT NULL,
    document_id UUID REFERENCES lab_documents(id),
    step_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    instruments_required UUID[] NOT NULL,
    estimated_duration TEXT,
    safety_notes TEXT[],
    critical_parameters JSONB,
    dependencies INTEGER[],
    quality_checks TEXT[],
    troubleshooting_hints TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Error Handling

### Common Error Codes:

**400 Bad Request:**
- Invalid file format or size exceeds 50MB limit
- Invalid lab_id or instrument_ids format
- Missing required approval fields for approved documents
- Invalid document_type or safety_level

**401 Unauthorized:**
- Invalid API keys for OpenAI or LlamaParse
- Insufficient lab access permissions

**403 Forbidden:**
- Lab data isolation policy violation
- Unauthorized access to lab documents

**422 Unprocessable Entity:**
- Document parsing failed
- Protocol structure detection failed
- Approval workflow validation errors

**500 Internal Server Error:**
- Database connection failed
- API service unavailable
- Lab isolation enforcement failed

### Error Response Format:
```json
{
  "error": "Lab data isolation policy violation",
  "code": 403,
  "details": {
    "field": "lab_id",
    "reason": "User does not have access to specified laboratory",
    "suggestion": "Verify lab_id and user permissions"
  },
  "request_id": "req_lab_123456789"
}
```

## Usage Example

```python
import asyncio
from lab_ingestion import LabIngestionPipeline, LabDocumentInput
from datetime import date

async def main():
    # Initialize lab ingestion pipeline
    pipeline = LabIngestionPipeline('config/lab_config.yaml')
    await pipeline.db_writer.initialize()
    
    # Prepare lab document input
    input_data = LabDocumentInput(
        file_path="/uploads/protocol.pdf",
        filename="hplc_analysis_protocol_v3.pdf",
        lab_id="550e8400-e29b-41d4-a716-446655440000",
        instrument_ids=["abc12345-1234-1234-1234-123456789012"],
        document_type="protocol",
        title="HPLC Analysis Protocol for Drug Compounds",
        version="3.1",
        author="Dr. Jane Smith",
        approval_status="approved",
        approved_by="Dr. John Doe",
        approved_date=date(2025, 1, 15),
        review_date=date(2026, 1, 15),
        protocol_id="PROT-2025-001",
        experiment_types=["drug_analysis", "quality_control"],
        safety_level="medium"
    )
    
    # Process lab document
    result = await pipeline.process_lab_document(input_data)
    
    if result["success"]:
        print(f"Lab document processed: {result['document_id']}")
        print(f"Protocol steps created: {result['protocol_steps_count']}")
        print(f"Chunks indexed: {result['chunks_count']}")
    else:
        print(f"Processing failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Files

### requirements.txt
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
asyncpg==0.29.0
openai==1.3.7
llama-cloud==0.0.6
llama-index==0.9.15
unstructured==0.11.6
camelot-py==0.10.1
PyMuPDF==1.23.8
Pillow==10.1.0
pyyaml==6.0.1
tenacity==8.2.3
nest-asyncio==1.5.8
uuid==1.30
```

### config/lab_config.yaml
```yaml
apis:
  openai_api_key: ${OPENAI_API_KEY}
  llamaparse_api_key: ${LLAMAPARSE_API_KEY}

database:
  connection_string: ${DATABASE_URL}
  enable_audit_log: true
  track_document_versions: true
  lab_isolation_enforcement: true

lab_processing:
  multi_tenant: true
  data_isolation: strict
  approval_workflow: true
  max_file_size_mb: 50
  supported_formats: ["pdf", "docx"]

lab_classification:
  auto_classify: true
  confidence_threshold: 0.9
  require_manual_review: true

protocol_processing:
  detect_multi_instrument: true
  cross_reference_instruments: true
  track_dependencies: true

metadata_enrichment:
  track_approvals: true
  monitor_review_dates: true
  extract_safety_info: true
  identify_critical_steps: true

content_filtering:
  remove_patterns:
    - "^Page \\d+ of \\d+$"
    - "^©.+All rights reserved\\.?$"
    - "^Confidential.+"
  min_content_length: 50
  max_repetition_ratio: 0.3

embedding:
  model: "text-embedding-3-large"
  dimension: 3072
  batch_size: 100
```