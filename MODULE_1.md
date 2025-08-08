## Module I: Manufacturer Data Ingestion Pipeline

### Project Structure
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
│   └── storage/
│       └── database_writer.py
├── config/
│   ├── processing_config.yaml
│   └── model_config.yaml
├── requirements.txt
└── main.py
```

### Step 1: Document Labeling and Classification

**Input Validation:**
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from pathlib import Path

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

**Processing Configuration:**
```yaml
# config/processing_config.yaml
document_processing:
  max_file_size_mb: 50
  supported_formats: ["pdf", "docx"]
  auto_classification:
    enabled: true
    confidence_threshold: 0.85
    fallback_to_manual: true
  
llamaparse:
    api_key_env: "LLAMAPARSE_API_KEY"
    parsing_instruction: |
      You are parsing a technical manual for laboratory instruments. 
      Preserve all structural information including:
      - Section headings and their hierarchy
      - Table structures and captions
      - Figure references and captions
      - Procedure steps and numbering
      - Technical specifications in tables
      - Safety warnings and notes
      Extract tables as markdown format.
    result_type: "markdown"
    premium_mode: true
    continuous_mode: true
    
content_filtering:
  remove_patterns:
    - "^Page \\d+ of \\d+$"
    - "^©.+All rights reserved\\.?$"
    - "^Confidential.+"
    - "^\\s*\\d{1,3}\\s*$"  # Page numbers
  min_content_length: 50
  max_repetition_ratio: 0.3
```

**Implementation:**
```python
# src/segmentation/document_classifier.py
import openai
from pydantic import BaseModel

class DocumentClassifier:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def classify_document(self, content_preview: str, filename: str) -> dict:
        """Auto-classify document type based on content preview"""
        
        classification_prompt = f"""
        Classify this laboratory instrument document based on the preview content and filename.
        
        Content preview: {content_preview[:2000]}
        Filename: {filename}
        
        Return classification in this exact JSON format:
        {{
            "document_type": "operational_guide|troubleshooting_guide|technical_specification|safety_document|maintenance_manual|software_guide|installation_guide",
            "confidence": 0.0-1.0,
            "primary_topics": ["topic1", "topic2"],
            "instrument_mentions": ["instrument1", "instrument2"]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": classification_prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

### Step 2: LlamaParse Processing

**Configuration:**
```python
# src/segmentation/llamaparse_processor.py
from llama_parse import LlamaParse
import asyncio

class LlamaParseProcessor:
    def __init__(self, api_key: str, config: dict):
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            premium_mode=config['premium_mode'],
            continuous_mode=config['continuous_mode'],
            parsing_instruction=config['parsing_instruction']
        )
    
    async def process_document(self, file_path: str) -> dict:
        """Process PDF through LlamaParse"""
        try:
            documents = await self.parser.aload_data(file_path)
            
            return {
                "success": True,
                "content": documents[0].text if documents else "",
                "metadata": documents[0].metadata if documents else {},
                "total_pages": len(documents),
                "processing_time": None  # Add timing
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": "",
                "metadata": {}
            }
```

### Step 3: Content Filtering

**Implementation:**
```python
# src/segmentation/content_filter.py
import re
from typing import List, Dict

class ContentFilter:
    def __init__(self, config: dict):
        self.remove_patterns = [re.compile(pattern) for pattern in config['remove_patterns']]
        self.min_content_length = config['min_content_length']
        self.max_repetition_ratio = config['max_repetition_ratio']
    
    def clean_content(self, content: str) -> str:
        """Remove headers, footers, and irrelevant content"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines matching removal patterns
            if any(pattern.match(line) for pattern in self.remove_patterns):
                continue
                
            # Skip lines that are too short
            if len(line) < self.min_content_length and not self._is_important_short_line(line):
                continue
                
            cleaned_lines.append(line)
        
        # Remove excessive repetition
        return self._remove_repetitive_content('\n'.join(cleaned_lines))
    
    def _is_important_short_line(self, line: str) -> bool:
        """Identify important short lines (headings, labels, etc.)"""
        important_patterns = [
            r'^#+ ',  # Markdown headings
            r'^\d+\.',  # Numbered items
            r'^[A-Z][A-Z\s]+:$',  # Labels like "WARNING:"
            r'^Figure \d+',  # Figure references
            r'^Table \d+'   # Table references
        ]
        return any(re.match(pattern, line) for pattern in important_patterns)
    
    def _remove_repetitive_content(self, content: str) -> str:
        """Remove sections with excessive repetition"""
        # Implementation for detecting and removing repetitive content
        # This is a simplified version - real implementation would be more sophisticated
        return content
```

### Step 4: Hierarchical Structure Analysis

**Implementation:**
```python
# src/indexing/hierarchy_analyzer.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re

class Section(BaseModel):
    id: str
    title: str
    level: int
    content: str
    parent_id: Optional[str] = None
    children: List[str] = []
    metadata: Dict[str, Any] = {}
    order_index: int

class HierarchyAnalyzer:
    def __init__(self):
        self.heading_patterns = [
            r'^# (.+)$',      # Level 1
            r'^## (.+)$',     # Level 2  
            r'^### (.+)$',    # Level 3
            r'^#### (.+)$',   # Level 4
            r'^##### (.+)$',  # Level 5
        ]
    
    def analyze_document_structure(self, content: str, document_id: str) -> List[Section]:
        """Extract hierarchical structure from markdown content"""
        lines = content.split('\n')
        sections = []
        current_sections_by_level = {}
        section_counter = 0
        
        current_content = []
        current_section = None
        
        for line_idx, line in enumerate(lines):
            heading_match = self._match_heading(line)
            
            if heading_match:
                # Save previous section content
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Create new section
                level, title = heading_match
                section_id = f"{document_id}_section_{section_counter}"
                section_counter += 1
                
                # Determine parent
                parent_id = None
                for parent_level in range(level - 1, 0, -1):
                    if parent_level in current_sections_by_level:
                        parent_id = current_sections_by_level[parent_level].id
                        break
                
                current_section = Section(
                    id=section_id,
                    title=title.strip(),
                    level=level,
                    content="",
                    parent_id=parent_id,
                    order_index=len(sections),
                    metadata={
                        "line_number": line_idx,
                        "has_subsections": False
                    }
                )
                
                # Update tracking
                current_sections_by_level[level] = current_section
                # Clear deeper levels
                for deeper_level in list(current_sections_by_level.keys()):
                    if deeper_level > level:
                        del current_sections_by_level[deeper_level]
                
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def _match_heading(self, line: str) -> Optional[tuple]:
        """Match line against heading patterns"""
        for level, pattern in enumerate(self.heading_patterns, 1):
            match = re.match(pattern, line)
            if match:
                return (level, match.group(1))
        return None
```

### Step 5: Data Hydration and Metadata Enrichment

**Text and Table Hydration:**
```python
# src/hydration/text_enricher.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openai

class ChunkMetadata(BaseModel):
    caption: Optional[str] = None
    summary: str
    tags: List[str] = []
    context_label: Optional[str] = None
    chunk_type: str  # 'text', 'table', 'procedure', 'specification'
    technical_level: str  # 'basic', 'intermediate', 'advanced'
    keywords: List[str] = []
    references: List[str] = []

class TextEnricher:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def enrich_chunk(self, content: str, section_title: str, document_type: str) -> ChunkMetadata:
        """Generate metadata for a content chunk"""
        
        enrichment_prompt = f"""
        Analyze this technical content from a laboratory instrument manual and generate metadata.
        
        Document Type: {document_type}
        Section: {section_title}
        Content: {content[:1500]}
        
        Generate metadata in this exact JSON format:
        {{
            "summary": "Brief 1-2 sentence summary",
            "tags": ["tag1", "tag2", "tag3"],
            "chunk_type": "text|table|procedure|specification",
            "technical_level": "basic|intermediate|advanced",
            "keywords": ["keyword1", "keyword2"],
            "context_label": "troubleshooting|operation|maintenance|safety|calibration|null",
            "references": ["Figure 1", "Table 2"],
            "caption": "Caption if this is a table or figure, else null"
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": enrichment_prompt}],
            response_format={"type": "json_object"}
        )
        
        metadata_dict = json.loads(response.choices[0].message.content)
        return ChunkMetadata(**metadata_dict)
```

**Image Processing and Captioning:**
```python
# src/hydration/image_processor.py
import openai
import base64
from pathlib import Path
from typing import Dict, List, Optional

class ImageMetadata(BaseModel):
    alt_text: str
    detailed_caption: str
    image_type: str  # 'diagram', 'screenshot', 'photo', 'chart', 'flowchart'
    contains_text: bool
    extracted_text: Optional[str] = None
    technical_elements: List[str] = []
    referenced_in_sections: List[str] = []

class ImageProcessor:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def process_image(self, image_path: str, context: str) -> ImageMetadata:
        """Generate comprehensive metadata for images"""
        
        # Encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        analysis_prompt = f"""
        Analyze this image from a laboratory instrument manual in the context: {context}
        
        Provide detailed analysis in this JSON format:
        {{
            "alt_text": "Brief accessible description",
            "detailed_caption": "Comprehensive description of what's shown",
            "image_type": "diagram|screenshot|photo|chart|flowchart|schematic",
            "contains_text": true|false,
            "extracted_text": "Any text visible in the image or null",
            "technical_elements": ["element1", "element2"],
            "key_components": ["component1", "component2"]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        metadata_dict = json.loads(response.choices[0].message.content)
        return ImageMetadata(**metadata_dict)
```

### Step 6: Embedding Generation

**Configuration:**
```python
# src/hydration/embedding_generator.py
import openai
import numpy as np
from typing import List, Dict, Any
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingGenerator:
    def __init__(self, openai_api_key: str, model: str = "text-embedding-3-large"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.dimension = 3072 if "3-large" in model else 1536
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_embedding(self, text: str, metadata: Dict[str, Any] = None) -> List[float]:
        """Generate embedding for text content with metadata enhancement"""
        
        # Enhance text with metadata for better embeddings
        enhanced_text = self._enhance_text_with_metadata(text, metadata)
        
        response = await self.client.embeddings.acreate(
            model=self.model,
            input=enhanced_text
        )
        
        return response.data[0].embedding
    
    def _enhance_text_with_metadata(self, text: str, metadata: Dict[str, Any]) -> str:
        """Enhance text with metadata for better semantic understanding"""
        if not metadata:
            return text
        
        enhancements = []
        
        if metadata.get('chunk_type'):
            enhancements.append(f"Content type: {metadata['chunk_type']}")
        
        if metadata.get('context_label'):
            enhancements.append(f"Context: {metadata['context_label']}")
        
        if metadata.get('tags'):
            enhancements.append(f"Topics: {', '.join(metadata['tags'])}")
        
        if metadata.get('keywords'):
            enhancements.append(f"Keywords: {', '.join(metadata['keywords'])}")
        
        enhanced_prefix = " | ".join(enhancements)
        return f"{enhanced_prefix} | {text}" if enhancements else text

    async def batch_generate_embeddings(self, texts: List[str], metadatas: List[Dict] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        tasks = [
            self.generate_embedding(text, metadata) 
            for text, metadata in zip(texts, metadatas)
        ]
        
        return await asyncio.gather(*tasks)
```

### Step 7: Database Storage

**Database Schema Implementation:**
```python
# src/storage/database_writer.py
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from uuid import uuid4
import json

class DatabaseWriter:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        await self._setup_database()
    
    async def _setup_database(self):
        """Create tables and extensions if they don't exist"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create manufacturer tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS manufacturers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name TEXT NOT NULL,
                    website TEXT,
                    contact_info JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS manufacturer_documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    manufacturer_id UUID REFERENCES manufacturers(id),
                    instrument_model TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    version TEXT,
                    language TEXT DEFAULT 'en',
                    file_path TEXT,
                    total_pages INTEGER,
                    file_size_mb FLOAT,
                    processing_status TEXT DEFAULT 'pending',
                    is_public BOOLEAN DEFAULT true,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS manufacturer_sections (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES manufacturer_documents(id),
                    title TEXT NOT NULL,
                    level INTEGER,
                    parent_section_id UUID REFERENCES manufacturer_sections(id),
                    order_index INTEGER,
                    content_preview TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS manufacturer_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES manufacturer_documents(id),
                    section_id UUID REFERENCES manufacturer_sections(id),
                    content TEXT NOT NULL,
                    markdown_table TEXT,
                    chunk_index INTEGER,
                    embedding vector(3072),
                    chunk_type TEXT NOT NULL,
                    technical_level TEXT,
                    metadata JSONB,
                    image_refs TEXT[],
                    page_references INTEGER[],
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_manufacturer_chunks_embedding 
                ON manufacturer_chunks USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_manufacturer_chunks_document 
                ON manufacturer_chunks(document_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_manufacturer_chunks_type 
                ON manufacturer_chunks(chunk_type);
            """)

    async def store_document_batch(self, document_data: Dict[str, Any]) -> str:
        """Store complete document with all sections and chunks"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Store document
                document_id = await self._store_document(conn, document_data)
                
                # Store sections
                section_mapping = {}
                for section_data in document_data.get('sections', []):
                    section_id = await self._store_section(conn, document_id, section_data, section_mapping)
                    section_mapping[section_data['temp_id']] = section_id
                
                # Store chunks
                for chunk_data in document_data.get('chunks', []):
                    await self._store_chunk(conn, document_id, chunk_data, section_mapping)
                
                return document_id
    
    async def _store_document(self, conn, document_data: Dict[str, Any]) -> str:
        """Store document record"""
        document_id = str(uuid4())
        
        await conn.execute("""
            INSERT INTO manufacturer_documents 
            (id, manufacturer_id, instrument_model, document_type, title, version, 
             language, file_path, total_pages, file_size_mb, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """, 
            document_id,
            document_data['manufacturer_id'],
            document_data['instrument_model'],
            document_data['document_type'],
            document_data['title'],
            document_data.get('version'),
            document_data.get('language', 'en'),
            document_data['file_path'],
            document_data.get('total_pages'),
            document_data.get('file_size_mb'),
            json.dumps(document_data.get('metadata', {}))
        )
        
        return document_id
    
    async def _store_section(self, conn, document_id: str, section_data: Dict[str, Any], section_mapping: Dict[str, str]) -> str:
        """Store section record"""
        section_id = str(uuid4())
        parent_id = section_mapping.get(section_data.get('parent_temp_id'))
        
        await conn.execute("""
            INSERT INTO manufacturer_sections 
            (id, document_id, title, level, parent_section_id, order_index, content_preview, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            section_id,
            document_id,
            section_data['title'],
            section_data['level'],
            parent_id,
            section_data['order_index'],
            section_data.get('content', '')[:500],  # Preview
            json.dumps(section_data.get('metadata', {}))
        )
        
        return section_id
    
    async def _store_chunk(self, conn, document_id: str, chunk_data: Dict[str, Any], section_mapping: Dict[str, str]):
        """Store chunk record with embedding"""
        section_id = section_mapping.get(chunk_data.get('section_temp_id'))
        
        await conn.execute("""
            INSERT INTO manufacturer_chunks 
            (document_id, section_id, content, markdown_table, chunk_index, embedding, 
             chunk_type, technical_level, metadata, image_refs, page_references)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
            document_id,
            section_id,
            chunk_data['content'],
            chunk_data.get('markdown_table'),
            chunk_data['chunk_index'],
            chunk_data['embedding'],
            chunk_data['chunk_type'],
            chunk_data.get('technical_level'),
            json.dumps(chunk_data.get('metadata', {})),
            chunk_data.get('image_refs', []),
            chunk_data.get('page_references', [])
        )
```

### Main Processing Pipeline

**Orchestration:**
```python
# main.py
import asyncio
from pathlib import Path
import yaml
from src.segmentation.document_classifier import DocumentClassifier
from src.segmentation.llamaparse_processor import LlamaParseProcessor
from src.segmentation.content_filter import ContentFilter
from src.indexing.hierarchy_analyzer import HierarchyAnalyzer
from src.hydration.text_enricher import TextEnricher
from src.hydration.image_processor import ImageProcessor
from src.hydration.embedding_generator import EmbeddingGenerator
from src.storage.database_writer import DatabaseWriter

class ManufacturerIngestionPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classifier = DocumentClassifier(self.config['openai']['api_key'])
        self.parser = LlamaParseProcessor(
            self.config['llamaparse']['api_key'],
            self.config['llamaparse']
        )
        self.filter = ContentFilter(self.config['content_filtering'])
        self.hierarchy_analyzer = HierarchyAnalyzer()
        self.text_enricher = TextEnricher(self.config['openai']['api_key'])
        self.image_processor = ImageProcessor(self.config['openai']['api_key'])
        self.embedding_generator = EmbeddingGenerator(self.config['openai']['api_key'])
        self.db_writer = DatabaseWriter(self.config['database']['connection_string'])
    
    async def process_document(self, input_data: DocumentInput) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # Step 1: Parse document
            parse_result = await self.parser.process_document(input_data.file_path)
            if not parse_result['success']:
                return {"success": False, "error": parse_result['error']}
            
            # Step 2: Clean content
            cleaned_content = self.filter.clean_content(parse_result['content'])
            
            # Step 3: Auto-classify if needed
            if input_data.is_auto_classified:
                classification = self.classifier.classify_document(
                    cleaned_content[:2000], 
                    input_data.filename
                )
                if classification['confidence'] < 0.85:
                    # Flag for manual review
                    pass
            
            # Step 4: Analyze hierarchy
            sections = self.hierarchy_analyzer.analyze_document_structure(
                cleaned_content, 
                f"doc_{uuid4()}"
            )
            
            # Step 5: Chunk content
            chunks = await self._create_chunks(sections, cleaned_content)
            
            # Step 6: Enrich metadata
            enriched_chunks = []
            for chunk in chunks:
                metadata = self.text_enricher.enrich_chunk(
                    chunk['content'],
                    chunk.get('section_title', ''),
                    input_data.doc_category
                )
                chunk['metadata'] = metadata.dict()
                enriched_chunks.append(chunk)
            
            # Step 7: Generate embeddings
            texts = [chunk['content'] for chunk in enriched_chunks]
            metadatas = [chunk['metadata'] for chunk in enriched_chunks]
            embeddings = await self.embedding_generator.batch_generate_embeddings(texts, metadatas)
            
            for chunk, embedding in zip(enriched_chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Step 8: Store in database
            document_data = {
                'manufacturer_id': input_data.manufacturer_id,
                'instrument_model': input_data.instrument_name,
                'document_type': input_data.doc_category,
                'title': input_data.filename,
                'version': input_data.version,
                'language': input_data.language,
                'file_path': input_data.file_path,
                'total_pages': parse_result.get('total_pages'),
                'sections': [section.dict() for section in sections],
                'chunks': enriched_chunks,
                'metadata': {
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    'classification': classification if input_data.is_auto_classified else None
                }
            }
            
            document_id = await self.db_writer.store_document_batch(document_data)
            
            return {
                "success": True,
                "document_id": document_id,
                "sections_count": len(sections),
                "chunks_count": len(enriched_chunks)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _create_chunks(self, sections: List[Section], content: str) -> List[Dict[str, Any]]:
        """Create chunks from sections with smart splitting"""
        chunks = []
        chunk_size = 1000  # tokens
        overlap = 200  # tokens
        
        for section in sections:
            if len(section.content) <= chunk_size:
                # Small section - single chunk
                chunks.append({
                    'content': section.content,
                    'section_title': section.title,
                    'section_temp_id': section.id,
                    'chunk_index': len(chunks),
                    'chunk_type': 'text'
                })
            else:
                # Large section - split into multiple chunks
                section_chunks = self._split_content_smart(section.content, chunk_size, overlap)
                for i, chunk_content in enumerate(section_chunks):
                    chunks.append({
                        'content': chunk_content,
                        'section_title': section.title,
                        'section_temp_id': section.id,
                        'chunk_index': len(chunks),
                        'chunk_type': 'text'
                    })
        
        return chunks
    
    def _split_content_smart(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Smart content splitting that preserves context"""
        # Implementation for smart chunking that respects:
        # - Sentence boundaries
        # - Paragraph boundaries  
        # - Table boundaries
        # - List boundaries
        # This is a simplified version
        
        sentences = content.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                
                # Add overlap
                overlap_sentences = current_chunk[-overlap//20:] if len(current_chunk) > overlap//20 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

# Usage
async def main():
    pipeline = ManufacturerIngestionPipeline('config/processing_config.yaml')
    await pipeline.db_writer.initialize()
    
    # Process document
    input_data = DocumentInput(
        file_path="/path/to/manual.pdf",
        filename="instrument_manual_v2.pdf",
        instrument_name="Model XYZ-100",
        doc_category="operational_guide",
        manufacturer_id="manufacturer-uuid-here"
    )
    
    result = await pipeline.process_document(input_data)
    print(f"Processing result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```
