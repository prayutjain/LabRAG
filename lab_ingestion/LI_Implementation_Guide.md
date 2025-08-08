## Module II: Laboratory Data Ingestion Pipeline
Same pipeline as Module I Manufacturer Ingestion pipeline with two differences:

- Protocol Context
  - Add protocol_id and experiment_context (JSON) at document and chunk levels.
  - Enforce mapping to instrument(s) because labs may combine instruments.
    - instrument_linkage: array of {instrument_id, role} (e.g., “primary”, “pre-processing”).
- Access Control & Lifecycle
  - States: draft → in_review → approved → retired.
  - Only approved content is indexed by default; allow include_draft=true at ingestion time for sandbox indices.
  
### Project Structure
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

### Enhanced Input Validation for Lab Documents

```python
# Input validation for lab-specific documents
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

### Lab-Specific Configuration

```yaml
# config/lab_config.yaml
lab_processing:
  multi_tenant: true
  data_isolation: strict
  approval_workflow: true
  
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
  
database:
  enable_audit_log: true
  track_document_versions: true
  lab_isolation_enforcement: true
```

### Lab Document Classifier

```python
# src/segmentation/lab_document_classifier.py
from typing import Dict, List, Any
import openai

class LabDocumentClassifier:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def classify_lab_document(self, content: str, filename: str, lab_context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify lab-specific documents with enhanced context"""
        
        classification_prompt = f"""
        Classify this laboratory document with the following context:
        
        Lab Context: {lab_context}
        Filename: {filename}
        Content Preview: {content[:2000]}
        
        Analyze and classify this document, returning JSON in this format:
        {{
            "document_type": "sop|protocol|training_material|maintenance_log|quality_control|custom_workflow|safety_procedure|method_validation",
            "confidence": 0.0-1.0,
            "is_multi_instrument": true|false,
            "applicable_instruments": ["instrument1", "instrument2"],
            "safety_level": "low|medium|high|critical",
            "requires_approval": true|false,
            "approval_authority": "lab_manager|safety_officer|quality_manager|department_head",
            "estimated_complexity": "basic|intermediate|advanced|expert",
            "primary_functions": ["function1", "function2"],
            "dependencies": ["doc1", "doc2"],
            "compliance_requirements": ["FDA", "ISO", "GLP"],
            "review_frequency": "monthly|quarterly|annually|as_needed"
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": classification_prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

### Protocol-Specific Processing

```python
# src/indexing/protocol_mapper.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ProtocolStep(BaseModel):
    step_number: int
    title: str
    description: str
    instruments_required: List[str]
    estimated_duration: Optional[str] = None
    safety_notes: List[str] = []
    critical_parameters: Dict[str, Any] = {}
    dependencies: List[int] = []  # Other step numbers this depends on
    
class MultiInstrumentProtocol(BaseModel):
    protocol_id: str
    title: str
    instruments_involved: List[str]
    steps: List[ProtocolStep]
    setup_requirements: List[str]
    safety_requirements: List[str]
    estimated_total_time: Optional[str] = None

class ProtocolMapper:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def analyze_protocol_structure(self, content: str, instrument_ids: List[str]) -> MultiInstrumentProtocol:
        """Extract detailed protocol structure for multi-instrument procedures"""
        
        analysis_prompt = f"""
        Analyze this laboratory protocol and extract structured information.
        
        Available Instruments: {instrument_ids}
        Protocol Content: {content}
        
        Extract protocol structure in this JSON format:
        {{
            "title": "Protocol title",
            "instruments_involved": ["instrument_id1", "instrument_id2"],
            "estimated_total_time": "2 hours",
            "setup_requirements": ["requirement1", "requirement2"],
            "safety_requirements": ["safety1", "safety2"],
            "steps": [
                {{
                    "step_number": 1,
                    "title": "Step title",
                    "description": "Detailed description",
                    "instruments_required": ["instrument_id1"],
                    "estimated_duration": "15 minutes",
                    "safety_notes": ["note1"],
                    "critical_parameters": {{"temperature": "25°C", "pressure": "1 atm"}},
                    "dependencies": []
                }}
            ]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"}
        )
        
        protocol_data = json.loads(response.choices[0].message.content)
        return MultiInstrumentProtocol(**protocol_data)
```

### Lab-Specific Metadata Enrichment

```python
# src/hydration/lab_metadata_enricher.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import openai

class LabChunkMetadata(BaseModel):
    summary: str
    tags: List[str]
    chunk_type: str
    safety_level: str
    requires_training: bool
    approval_required: bool
    compliance_tags: List[str] = []
    instrument_specific: List[str] = []
    procedure_type: Optional[str] = None
    critical_steps: List[str] = []
    quality_checkpoints: List[str] = []
    troubleshooting_indicators: List[str] = []

class LabMetadataEnricher:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def enrich_lab_chunk(self, content: str, document_context: Dict[str, Any]) -> LabChunkMetadata:
        """Enrich lab document chunks with specialized metadata"""
        
        enrichment_prompt = f"""
        Analyze this laboratory procedure content and generate specialized metadata.
        
        Document Context: {document_context}
        Content: {content[:1500]}
        
        Generate metadata in this JSON format:
        {{
            "summary": "Brief procedural summary",
            "tags": ["tag1", "tag2"],
            "chunk_type": "procedure_step|safety_warning|quality_check|troubleshooting|setup|cleanup",
            "safety_level": "low|medium|high|critical",
            "requires_training": true|false,
            "approval_required": true|false,
            "compliance_tags": ["FDA", "ISO", "GLP"],
            "instrument_specific": ["instrument1", "instrument2"],
            "procedure_type": "calibration|maintenance|operation|troubleshooting|safety|quality_control|null",
            "critical_steps": ["step1", "step2"],
            "quality_checkpoints": ["checkpoint1", "checkpoint2"],
            "troubleshooting_indicators": ["indicator1", "indicator2"]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": enrichment_prompt}],
            response_format={"type": "json_object"}
        )
        
        metadata_dict = json.loads(response.choices[0].message.content)
        return LabChunkMetadata(**metadata_dict)
```

### Lab Database Schema Extensions

```python
# src/storage/lab_database_writer.py
import asyncpg
from typing import Dict, Any, List
import json

class LabDatabaseWriter:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def setup_lab_tables(self):
        """Create lab-specific database tables"""
        async with self.pool.acquire() as conn:
            # Lab documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lab_documents (
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
            """)
            
            # Lab sections table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lab_sections (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES lab_documents(id),
                    title TEXT NOT NULL,
                    level INTEGER,
                    parent_section_id UUID REFERENCES lab_sections(id),
                    order_index INTEGER,
                    section_type TEXT, -- 'procedure', 'safety', 'setup', 'troubleshooting'
                    estimated_duration TEXT,
                    required_instruments UUID[],
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Lab chunks table with enhanced metadata
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lab_chunks (
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
            """)
            
            # Protocol steps table for multi-instrument procedures
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS protocol_steps (
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
                    dependencies INTEGER[], -- step numbers this depends on
                    quality_checks TEXT[],
                    troubleshooting_hints TEXT[],
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Audit log table for compliance
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lab_document_audit (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES lab_documents(id),
                    action TEXT NOT NULL, -- 'created', 'updated', 'approved', 'archived'
                    user_id TEXT NOT NULL,
                    old_values JSONB,
                    new_values JSONB,
                    notes TEXT,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Lab data isolation indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lab_chunks_lab_isolation 
                ON lab_chunks(lab_id, document_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lab_chunks_embedding 
                ON lab_chunks USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lab_chunks_instruments 
                ON lab_chunks USING GIN(instrument_ids);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_steps_protocol 
                ON protocol_steps(protocol_id, step_number);
            """)

    async def store_lab_document_batch(self, document_data: Dict[str, Any], user_id: str) -> str:
        """Store lab document with audit trail"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Store document
                document_id = await self._store_lab_document(conn, document_data)
                
                # Store protocol steps if applicable
                if document_data.get('protocol_steps'):
                    await self._store_protocol_steps(conn, document_id, document_data['protocol_steps'])
                
                # Store sections and chunks
                section_mapping = {}
                for section_data in document_data.get('sections', []):
                    section_id = await self._store_lab_section(conn, document_id, section_data, section_mapping)
                    section_mapping[section_data['temp_id']] = section_id
                
                for chunk_data in document_data.get('chunks', []):
                    await self._store_lab_chunk(conn, document_id, chunk_data, section_mapping, document_data['lab_id'])
                
                # Create audit log entry
                await self._create_audit_entry(conn, document_id, 'created', user_id, {}, document_data)
                
                return document_id

    async def _store_lab_document(self, conn, document_data: Dict[str, Any]) -> str:
        """Store lab document record"""
        document_id = str(uuid4())
        
        await conn.execute("""
            INSERT INTO lab_documents 
            (id, lab_id, instrument_ids, protocol_id, document_type, title, version, 
             author, approval_status, approved_by, approved_date, review_date, 
             file_path, safety_level, compliance_requirements, experiment_types, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        """,
            document_id,
            document_data['lab_id'],
            document_data['instrument_ids'],
            document_data.get('protocol_id'),
            document_data['document_type'],
            document_data['title'],
            document_data['version'],
            document_data['author'],
            document_data['approval_status'],
            document_data.get('approved_by'),
            document_data.get('approved_date'),
            document_data.get('review_date'),
            document_data['file_path'],
            document_data.get('safety_level', 'medium'),
            document_data.get('compliance_requirements', []),
            document_data.get('experiment_types', []),
            json.dumps(document_data.get('metadata', {}))
        )
        
        return document_id

    async def _store_protocol_steps(self, conn, document_id: str, protocol_steps: List[Dict[str, Any]]):
        """Store protocol steps for multi-instrument procedures"""
        for step_data in protocol_steps:
            await conn.execute("""
                INSERT INTO protocol_steps 
                (protocol_id, document_id, step_number, title, description, 
                 instruments_required, estimated_duration, safety_notes, 
                 critical_parameters, dependencies, quality_checks, troubleshooting_hints)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                step_data['protocol_id'],
                document_id,
                step_data['step_number'],
                step_data['title'],
                step_data['description'],
                step_data['instruments_required'],
                step_data.get('estimated_duration'),
                step_data.get('safety_notes', []),
                json.dumps(step_data.get('critical_parameters', {})),
                step_data.get('dependencies', []),
                step_data.get('quality_checks', []),
                step_data.get('troubleshooting_hints', [])
            )

    async def _create_audit_entry(self, conn, document_id: str, action: str, user_id: str, old_values: Dict, new_values: Dict):
        """Create audit trail entry"""
        await conn.execute("""
            INSERT INTO lab_document_audit 
            (document_id, action, user_id, old_values, new_values)
            VALUES ($1, $2, $3, $4, $5)
        """,
            document_id,
            action,
            user_id,
            json.dumps(old_values),
            json.dumps(new_values)
        )
```
