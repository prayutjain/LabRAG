# Manufacturer Data Ingestion Pipeline

## Project Structure
```
manufacturer_ingestion/
├── src/
│   ├── segmentation/
│   │   ├── document_classifier.py
│   │   ├── llamaparse_processor.py
│   │   └── content_filter.py
│   ├── indexing/
│   │   ├── hierarchy_analyzer.py
│   │   └── structure_mapper.py
│   ├── hydration/
│   │   ├── text_enricher.py
│   │   ├── image_processor.py
│   │   └── embedding_generator.py
│   ├── storage/
│   │   └── database_writer.py
│   └── main.py
├── config/
│   ├── processing_config.yaml
│   └── model_config.yaml
├── requirements.txt
└── main.py
```

## User Inputs

### DocumentInput Schema
```python
class DocumentInput(BaseModel):
    file_path: str = Field(..., description="Path to PDF file")
    filename: str = Field(..., description="Original filename")
    instrument_name: str = Field(..., description="Target instrument model")
    doc_category: Literal[
        "operational_guide", 
        "troubleshooting_guide", 
        "technical_specification", 
        "safety_document",
        "maintenance_manual",
        "software_guide",
        "installation_guide"
    ] = Field(..., description="Document category")
    manufacturer_id: str = Field(..., description="UUID of manufacturer")
    is_auto_classified: bool = Field(default=True, description="Whether to use auto-classification")
    language: str = Field(default="en", description="Document language")
    version: Optional[str] = Field(None, description="Document version")
```

**Validation Requirements:**
- `file_path`: Must exist and be readable PDF/DOCX file
- `filename`: Non-empty string, max 255 characters
- `instrument_name`: Non-empty string, max 100 characters
- `doc_category`: Must be one of the allowed enum values
- `manufacturer_id`: Valid UUID format
- `language`: ISO 639-1 language code
- File size: Maximum 50MB

If validation fails, respond with HTTP 400 and error details.

## API Specifications

### Framework: FastAPI
### Services Used:
- **LlamaParse API** (https://cloud.llamaindex.ai/)
- **OpenAI API** (https://api.openai.com/v1/)
- **PostgreSQL with pgvector extension**

### Configuration
Read from `config/processing_config.yaml`:
```yaml
apis:
  llamaparse_api_key: ${LLAMAPARSE_API_KEY}
  openai_api_key: ${OPENAI_API_KEY}
  
database:
  connection_string: ${DATABASE_URL}
  
document_processing:
  max_file_size_mb: 50
  supported_formats: ["pdf", "docx"]
  auto_classification:
    enabled: true
    confidence_threshold: 0.85
```

## API Endpoints

### POST /ingest/document
Process and ingest a manufacturer document into the system.

**Request Body:**
```json
{
  "file_path": "/uploads/manual.pdf",
  "filename": "xyz_instrument_manual_v2.pdf",
  "instrument_name": "Model XYZ-100",
  "doc_category": "operational_guide",
  "manufacturer_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_auto_classified": true,
  "language": "en",
  "version": "2.1"
}
```

### Success Response (HTTP 200)
```json
{
  "success": true,
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "sections_count": 15,
  "chunks_count": 87,
  "processing_time_seconds": 45.2,
  "metadata": {
    "total_pages": 120,
    "tables_extracted": 23,
    "images_processed": 12,
    "auto_classification": {
      "confidence": 0.92,
      "detected_type": "operational_guide"
    }
  }
}
```

### Error Response (HTTP 400/500)
```json
{
  "error": "File size exceeds maximum limit of 50MB",
  "code": 400,
  "details": {
    "field": "file_path",
    "received_size_mb": 75.3,
    "max_allowed_mb": 50
  }
}
```

## Processing Pipeline Steps

### Step 1: Document Classification
- Auto-classify document type using filename and content preview
- Validate against supported categories
- Flag for manual review if confidence < 0.85

### Step 2: LlamaParse Processing
- Extract structured content using LlamaParse API
- Parse tables, images, headings, and text with bounding boxes
- Fallback to Unstructured + Camelot for table extraction if needed

### Step 3: Content Filtering
- Remove headers, footers, page numbers, watermarks
- Deduplicate near-identical content
- Normalize unicode and units

### Step 4: Hierarchical Analysis
- Build section tree from headings and table of contents
- Create chunks with metadata (1000-1200 tokens per chunk)
- Preserve table structure as Markdown fenced blocks

### Step 5: Content Enrichment
- Generate summaries (≤2 sentences per chunk)
- Extract keywords and tags
- Add context labels (troubleshooting, operation, etc.)
- Process images with VLM for captions and metadata

### Step 6: Embedding Generation
- Generate vector embeddings using OpenAI text-embedding-3-large
- Enhance text with metadata for better semantic understanding
- Store 3072-dimensional vectors

### Step 7: Database Storage
- Store in PostgreSQL with pgvector extension
- Maintain relationships between documents, sections, and chunks
- Create semantic search indexes

## Database Schema

### manufacturer_documents
```sql
CREATE TABLE manufacturer_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manufacturer_id UUID REFERENCES manufacturers(id),
    instrument_model TEXT NOT NULL,
    document_type TEXT NOT NULL,
    title TEXT NOT NULL,
    version TEXT,
    language TEXT DEFAULT 'en',
    file_path TEXT,
    total_pages INTEGER,
    processing_status TEXT DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### manufacturer_chunks
```sql
CREATE TABLE manufacturer_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES manufacturer_documents(id),
    section_id UUID REFERENCES manufacturer_sections(id),
    content TEXT NOT NULL,
    embedding vector(3072),
    chunk_type TEXT NOT NULL,
    technical_level TEXT,
    metadata JSONB,
    page_references INTEGER[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Error Handling

### Common Error Codes:

**400 Bad Request:**
- Invalid file format
- File size exceeds limit
- Missing required fields
- Invalid manufacturer_id format

**401 Unauthorized:**
- Invalid API keys for LlamaParse or OpenAI

**413 Payload Too Large:**
- File exceeds 50MB limit

**422 Unprocessable Entity:**
- Document parsing failed
- Unsupported document structure

**500 Internal Server Error:**
- Database connection failed
- API service unavailable
- Unexpected processing error

### Error Response Format:
```json
{
  "error": "Descriptive error message",
  "code": 400,
  "details": {
    "field": "field_name",
    "reason": "Specific validation failure",
    "suggestion": "How to fix the issue"
  },
  "request_id": "req_123456789"
}
```

## Usage Example

```python
import asyncio
from manufacturer_ingestion import ManufacturerIngestionPipeline, DocumentInput

async def main():
    # Initialize pipeline
    pipeline = ManufacturerIngestionPipeline('config/processing_config.yaml')
    await pipeline.db_writer.initialize()
    
    # Prepare document input
    input_data = DocumentInput(
        file_path="/uploads/manual.pdf",
        filename="instrument_manual_v2.pdf",
        instrument_name="Model XYZ-100",
        doc_category="operational_guide",
        manufacturer_id="550e8400-e29b-41d4-a716-446655440000"
    )
    
    # Process document
    result = await pipeline.process_document(input_data)
    
    if result["success"]:
        print(f"Document processed successfully: {result['document_id']}")
        print(f"Created {result['chunks_count']} searchable chunks")
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
```

### config/processing_config.yaml
```yaml
apis:
  llamaparse_api_key: ${LLAMAPARSE_API_KEY}
  openai_api_key: ${OPENAI_API_KEY}

database:
  connection_string: ${DATABASE_URL}

document_processing:
  max_file_size_mb: 50
  supported_formats: ["pdf", "docx"]
  auto_classification:
    enabled: true
    confidence_threshold: 0.85
    fallback_to_manual: true

llamaparse:
  result_type: "json"
  premium_mode: true
  auto_mode: true
  parsing_instruction: |
    Extract technical manual content preserving:
    - Section headings and hierarchy
    - Table structures and captions
    - Figure references and captions
    - Procedure steps and numbering
    - Technical specifications
    - Safety warnings and notes

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
