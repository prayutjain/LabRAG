# Response Engine Implementation Guide

## Project Structure
```
response_engine/
├── src/
│   ├── agents/
│   │   ├── intent_classifier.py
│   │   ├── target_identifier.py
│   │   ├── query_orchestrator.py
│   │   └── response_generator.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   ├── relevance_scorer.py
│   │   └── context_ranker.py
│   ├── processing/
│   │   ├── query_preprocessor.py
│   │   ├── image_analyzer.py
│   │   └── multimodal_processor.py
│   ├── validation/
│   │   ├── response_validator.py
│   │   └── citation_manager.py
│   └── api/
│       ├── mcp_interface.py
│       ├── rest_api.py
│       └── websocket_handler.py
├── config/
│   └── response_config.yaml
├── requirements.txt
└── response_main.py
```

## User Inputs
```python
class QueryInput(BaseModel):
    text: str = Field(..., description="User's text query")
    lab_id: Optional[str] = Field(None, description="Lab context")
    instrument_ids: Optional[List[str]] = Field(None, description="Specific instruments")
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    search_type: str = Field(default="hybrid", description="search|keyword|hybrid")
    max_results: int = Field(default=10, description="Maximum results to return")
    include_images: bool = Field(default=True, description="Include images in response")
```

## Input Validation
- **text**: Required string, minimum 3 characters, maximum 2000 characters
- **lab_id**: Optional UUID format string
- **instrument_ids**: Optional list of UUID strings
- **image_data**: Optional base64 encoded string
- **search_type**: Must be one of ["search", "keyword", "hybrid"]
- **max_results**: Integer between 1 and 50

If validation fails, respond with HTTP 400 and error details.

## Create a Laboratory Response Engine with the following specifications:

### Framework & Libraries
- **API Framework**: FastAPI
- **AI Service**: OpenAI GPT-4 API
- **Database**: PostgreSQL with asyncpg
- **Vector Search**: pgvector extension
- **Protocol**: Model Context Protocol (MCP) v1.0

### Configuration
Read from `config/response_config.yaml`:
```yaml
response_engine:
  max_retrieval_results: 50
  response_timeout_seconds: 30
  enable_streaming: true

openai:
  api_key: "your-openai-api-key"

database:
  connection_string: "postgresql://user:pass@host:port/db"

intent_classification:
  confidence_threshold: 0.8
  supported_intents:
    - troubleshooting
    - operation
    - maintenance
    - safety
    - parameter_lookup
    - protocol_guidance
    - general_info
```

### Core Processing Pipeline

#### 1. Query Preprocessing
- **Endpoint**: `POST /api/v1/process-query`
- **Process**: Normalize text, extract entities, analyze images
- **Image Analysis**: Use GPT-4 Vision for equipment/error recognition

#### 2. Intent Classification
- **Method**: LLM-based classification with confidence scoring
- **Output**: Primary intent + confidence score + urgency level

#### 3. Target Identification
- **Process**: Match entities to instruments and documents in database
- **Query**: Find relevant lab instruments and document types
- **Prioritization**: Lab-specific documents over manufacturer manuals

#### 4. Hybrid Retrieval
- **Vector Search**: Use embeddings with pgvector similarity
- **Parallel Retrieval**: Manufacturer docs + lab-specific docs
- **Reranking**: Boost lab-specific and intent-matched content

#### 5. Response Generation
- **LLM**: GPT-4 with structured prompts
- **Components**: Main answer + citations + safety warnings + images
- **Format**: JSON with confidence scores and metadata

## Output Format

### Success Response (HTTP 200)
```json
{
  "success": true,
  "response": "Your HPLC error E001 indicates a pressure sensor malfunction...",
  "confidence": 0.87,
  "citations": [
    {
      "id": 1,
      "document_title": "HPLC Troubleshooting Guide",
      "section_title": "Pressure System Errors",
      "source_type": "lab",
      "page_references": [23, 24],
      "relevance_score": 0.94
    }
  ],
  "images": [
    {
      "image_path": "/images/hplc_pressure_diagram.png",
      "context": "From HPLC Troubleshooting Guide - Pressure System Errors",
      "source_type": "lab"
    }
  ],
  "safety_warnings": [
    "Ensure system is depressurized before inspection"
  ],
  "follow_up_suggestions": [
    "Check pressure sensor connections",
    "Inspect pump seals for leaks"
  ],
  "procedure_steps": [
    "Turn off the HPLC system",
    "Depressurize the system completely",
    "Check sensor cable connections"
  ],
  "metadata": {
    "intent": "troubleshooting",
    "intent_confidence": 0.92,
    "sources_used": 5,
    "processing_time": 2.3
  }
}
```

### Error Response (HTTP 400/500)
```json
{
  "success": false,
  "error": "Invalid query parameters",
  "message": "Query text must be between 3 and 2000 characters",
  "code": 400
}
```

## MCP Protocol Integration

### Capabilities Response
```json
{
  "capabilities": {
    "query_processing": true,
    "streaming_responses": true,
    "multimodal_input": true,
    "source_citation": true,
    "image_analysis": true,
    "lab_data_isolation": true
  },
  "supported_document_types": [
    "operational_guide",
    "troubleshooting_guide", 
    "technical_specification",
    "safety_document",
    "sop",
    "protocol"
  ],
  "version": "1.0"
}
```

### Streaming Support
- **WebSocket Endpoint**: `/ws/query-stream`
- **Real-time Updates**: Progress notifications during processing
- **Partial Results**: Incremental response building

## Database Schema Requirements

### Core Tables
- `manufacturer_documents` - Official instrument manuals
- `manufacturer_chunks` - Processed text chunks with embeddings
- `lab_documents` - Lab-specific SOPs and procedures
- `lab_chunks` - Lab document chunks with embeddings
- `instruments` - Lab instrument inventory

### Vector Search
- Use pgvector extension for similarity search
- Store embeddings as `vector(1536)` for OpenAI ada-002
- Index with HNSW for performance

## Security & Isolation

### Lab Data Isolation
- All queries scoped to specific lab_id
- Lab documents only accessible to associated labs
- User permissions validated at API level

### Content Validation
- Response accuracy scoring
- Source citation verification
- Safety warning detection
- Hallucination prevention

## Performance Optimizations

### Caching
- Query result caching (60 minutes TTL)
- Embedding caching for repeated queries
- Document chunk preloading

### Parallel Processing
- Concurrent manufacturer + lab document retrieval
- Async LLM calls for classification and generation
- Background validation processing

## Error Handling

### Common Error Scenarios
- **No relevant documents found**: Suggest query refinement
- **Low confidence responses**: Request clarification
- **API rate limits**: Implement exponential backoff
- **Database timeouts**: Graceful degradation with cached results

### Monitoring
- Query processing times
- Confidence score distributions
- Citation accuracy metrics
- User satisfaction feedback