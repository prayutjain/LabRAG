## Module III: Response Engine (MCP-Ready)

### Project Structure
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

### Response Engine Configuration

```yaml
# config/response_config.yaml
response_engine:
  max_retrieval_results: 50
  response_timeout_seconds: 30
  enable_streaming: true
  
intent_classification:
  confidence_threshold: 0.8
  fallback_to_general: true
  supported_intents:
    - troubleshooting
    - operation
    - maintenance
    - safety
    - parameter_lookup
    - protocol_guidance
    - general_info

target_identification:
  max_instruments_per_query: 5
  max_documents_per_instrument: 10
  prefer_lab_specific: true
  
retrieval:
  vector_similarity_threshold: 0.7
  max_chunks_per_source: 15
  rerank_top_k: 20
  enable_hybrid_search: true
  
response_generation:
  max_response_length: 4000
  include_confidence_scores: true
  require_source_citations: true
  include_images: true
  format_tables: true
  
mcp_protocol:
  version: "1.0"
  max_concurrent_requests: 10
  enable_streaming_responses: true
  
caching:
  enable_query_cache: true
  cache_ttl_minutes: 60
  max_cache_size: 1000
```

### Query Preprocessing and Intent Classification

```python
# src/processing/query_preprocessor.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import openai
import base64
from PIL import Image
import io

class QueryInput(BaseModel):
    text: str = Field(..., description="User's text query")
    lab_id: Optional[str] = Field(None, description="Lab context")
    instrument_ids: Optional[List[str]] = Field(None, description="Specific instruments")
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    search_type: str = Field(default="hybrid", description="search|keyword|hybrid")
    max_results: int = Field(default=10, description="Maximum results to return")
    include_images: bool = Field(default=True, description="Include images in response")

class ProcessedQuery(BaseModel):
    original_text: str
    normalized_text: str
    extracted_keywords: List[str]
    image_description: Optional[str] = None
    detected_entities: Dict[str, List[str]] = {}
    context_hints: List[str] = []

class QueryPreprocessor:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    async def preprocess_query(self, query_input: QueryInput) -> ProcessedQuery:
        """Preprocess and enhance query for better retrieval"""
        
        # Process image if provided
        image_description = None
        if query_input.image_data:
            image_description = await self._analyze_image(query_input.image_data, query_input.text)
        
        # Extract entities and keywords
        enhanced_text = query_input.text
        if image_description:
            enhanced_text = f"{query_input.text} [Image shows: {image_description}]"
        
        analysis_prompt = f"""
        Analyze this laboratory query and extract structured information.
        
        Query: {enhanced_text}
        Lab Context: {query_input.lab_id}
        Instruments: {query_input.instrument_ids}
        
        Extract information in this JSON format:
        {{
            "normalized_text": "Cleaned and normalized query text",
            "extracted_keywords": ["keyword1", "keyword2"],
            "detected_entities": {{
                "instruments": ["instrument1", "instrument2"],
                "procedures": ["procedure1"],
                "parameters": ["temp", "pressure"],
                "errors": ["error_code_1"],
                "chemicals": ["reagent1"]
            }},
            "context_hints": ["hint1", "hint2"]
        }}
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"}
        )
        
        analysis_result = json.loads(response.choices[0].message.content)
        
        return ProcessedQuery(
            original_text=query_input.text,
            normalized_text=analysis_result['normalized_text'],
            extracted_keywords=analysis_result['extracted_keywords'],
            image_description=image_description,
            detected_entities=analysis_result['detected_entities'],
            context_hints=analysis_result['context_hints']
        )
    
    async def _analyze_image(self, base64_image: str, query_context: str) -> str:
        """Analyze uploaded image and generate description"""
        
        image_prompt = f"""
        Analyze this image in the context of a laboratory query: "{query_context}"
        
        Describe what you see focusing on:
        - Laboratory equipment or instruments
        - Error messages or displays
        - Procedures being performed
        - Any text or readings visible
        - Safety concerns or issues
        
        Provide a concise but detailed description that would help with document retrieval.
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content
```

### Intent Classification and Target Identification

- Route to one of {troubleshoot, operation, maintenance, safety, parameter_lookup, protocol_guidance, general_info}.

```python
# src/agents/intent_classifier.py
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import openai

class IntentResult(BaseModel):
    primary_intent: str
    confidence: float
    secondary_intents: List[str] = []
    urgency_level: str  # 'low', 'medium', 'high', 'critical'
    required_expertise: str  # 'basic', 'intermediate', 'advanced', 'expert'

class IntentClassifier:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.intent_definitions = {
            "troubleshooting": "User is experiencing problems or errors and needs help resolving them",
            "operation": "User needs guidance on normal instrument operation or procedures",
            "maintenance": "User needs information about instrument maintenance, cleaning, or upkeep",
            "safety": "User has safety-related questions or concerns",
            "parameter_lookup": "User needs specific parameter values, settings, or specifications",
            "protocol_guidance": "User needs help with experimental protocols or procedures",
            "general_info": "User wants general information about instruments or capabilities"
        }
    
    async def classify_intent(self, processed_query: ProcessedQuery, context: Dict[str, Any]) -> IntentResult:
        """Classify user intent for targeted retrieval"""
        
        classification_prompt = f"""
        Classify the intent of this laboratory query.
        
        Intent Definitions:
        {json.dumps(self.intent_definitions, indent=2)}
        
        Query: {processed_query.normalized_text}
        Keywords: {processed_query.extracted_keywords}
        Detected Entities: {processed_query.detected_entities}
        Context: {context}
        Image Description: {processed_query.image_description or "None"}
        
        Classify in this JSON format:
        {{
            "primary_intent": "intent_name",
            "confidence": 0.0-1.0,
            "secondary_intents": ["intent2", "intent3"],
            "urgency_level": "low|medium|high|critical",
            "required_expertise": "basic|intermediate|advanced|expert",
            "reasoning": "Why this classification was chosen"
        }}
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": classification_prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return IntentResult(**{k: v for k, v in result.items() if k != 'reasoning'})

# src/agents/target_identifier.py
class TargetIdentifier:
    ```
    Implement using LLM by finding the closest match to instruments in database
    ```

    def __init__(self, database_pool):
        self.db_pool = database_pool
    
    async def identify_targets(self, processed_query: ProcessedQuery, intent: IntentResult, lab_id: str) -> Dict[str, Any]:
        """Identify specific instruments, documents, and contexts relevant to the query"""
        
        async with self.db_pool.acquire() as conn:
            # Get lab instruments
            lab_instruments = await conn.fetch("""
                SELECT i.id, i.model_name, i.serial_number, m.name as manufacturer
                FROM instruments i
                JOIN manufacturers m ON i.manufacturer_id = m.id
                WHERE i.lab_asset_id = ANY(
                    SELECT unnest(string_to_array($1, ','))
                ) OR $2 IS NULL
            """, lab_id, lab_id)
            
            # Match entities to instruments
            relevant_instruments = []
            detected_instruments = processed_query.detected_entities.get('instruments', [])
            
            for instrument in lab_instruments:
                if any(entity.lower() in instrument['model_name'].lower() 
                      for entity in detected_instruments):
                    relevant_instruments.append(instrument)
            
            # If no specific instruments identified, include all lab instruments
            if not relevant_instruments and lab_id:
                relevant_instruments = list(lab_instruments)
            
            # Get relevant document types based on intent
            document_priorities = self._get_document_priorities(intent.primary_intent)
            
            # Get recent lab documents for these instruments
            lab_documents = await conn.fetch("""
                SELECT ld.*, array_agg(i.model_name) as instrument_models
                FROM lab_documents ld
                JOIN unnest(ld.instrument_ids) AS instrument_id ON true
                JOIN instruments i ON i.id = instrument_id::uuid
                WHERE ld.lab_id = $1 
                AND ld.approval_status = 'approved'
                AND ld.document_type = ANY($2)
                GROUP BY ld.id
                ORDER BY ld.updated_at DESC
                LIMIT 20
            """, lab_id, document_priorities[:3])
            
            return {
                "relevant_instruments": [dict(r) for r in relevant_instruments],
                "relevant_lab_documents": [dict(r) for r in lab_documents],
                "document_priorities": document_priorities,
                "search_scope": {
                    "lab_specific": bool(lab_id),
                    "manufacturer_docs": True,
                    "max_instruments": 5
                }
            }
    
    def _get_document_priorities(self, intent: str) -> List[str]:
        """Get prioritized document types based on intent"""
        priority_map = {
            "troubleshooting": ["troubleshooting_guide", "sop", "maintenance_manual", "operational_guide"],
            "operation": ["operational_guide", "sop", "protocol", "software_guide"],
            "maintenance": ["maintenance_manual", "sop", "safety_document", "operational_guide"],
            "safety": ["safety_document", "sop", "troubleshooting_guide", "operational_guide"],
            "parameter_lookup": ["technical_specification", "operational_guide", "software_guide"],
            "protocol_guidance": ["protocol", "sop", "method_validation", "operational_guide"],
            "general_info": ["operational_guide", "technical_specification", "installation_guide"]
        }
        return priority_map.get(intent, ["operational_guide", "sop", "troubleshooting_guide"])
```

### Hybrid Retrieval System

```python
# src/retrieval/hybrid_retriever.py
from typing import List, Dict, Any, Tuple
import asyncpg
import asyncio
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    source_type: str  # 'manufacturer' or 'lab'
    document_title: str
    section_title: str
    relevance_score: float
    metadata: Dict[str, Any]
    page_references: List[int]
    image_refs: List[str]

class HybridRetriever:
    def __init__(self, database_pool, embedding_generator):
        self.db_pool = database_pool
        self.embedding_generator = embedding_generator
    
    async def retrieve(self, processed_query: ProcessedQuery, intent: IntentResult, 
                      targets: Dict[str, Any], max_results: int = 20) -> List[RetrievalResult]:
        """Perform hybrid retrieval across manufacturer and lab documents"""
        
        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_embedding(
            processed_query.normalized_text,
            {"intent": intent.primary_intent, "keywords": processed_query.extracted_keywords}
        )
        
        # Parallel retrieval from both sources
        manufacturer_task = self._retrieve_manufacturer_docs(
            query_embedding, processed_query, targets, max_results // 2
        )
        lab_task = self._retrieve_lab_docs(
            query_embedding, processed_query, targets, max_results // 2
        )
        
        manufacturer_results, lab_results = await asyncio.gather(
            manufacturer_task, lab_task
        )
        
        # Combine and rerank results
        all_results = manufacturer_results + lab_results
        reranked_results = await self._rerank_results(all_results, processed_query, intent)
        
        return reranked_results[:max_results]
    
    async def _retrieve_manufacturer_docs(self, query_embedding: List[float], 
                                        processed_query: ProcessedQuery, 
                                        targets: Dict[str, Any], 
                                        max_results: int) -> List[RetrievalResult]:
        """Retrieve from manufacturer documents"""
        
        async with self.db_pool.acquire() as conn:
            # Build instrument filter
            instrument_models = [inst['model_name'] for inst in targets['relevant_instruments']]
            
            results = await conn.fetch("""
                WITH ranked_chunks AS (
                    SELECT 
                        mc.id,
                        mc.content,
                        mc.metadata,
                        mc.page_references,
                        mc.image_refs,
                        md.title as document_title,
                        ms.title as section_title,
                        mc.embedding <-> $1 as distance,
                        ROW_NUMBER() OVER (
                            PARTITION BY md.id 
                            ORDER BY mc.embedding <-> $1
                        ) as rank_in_doc
                    FROM manufacturer_chunks mc
                    JOIN manufacturer_documents md ON mc.document_id = md.id
                    JOIN manufacturer_sections ms ON mc.section_id = ms.id
                    WHERE md.instrument_model = ANY($2)
                    AND md.document_type = ANY($3)
                    AND mc.embedding <-> $1 < 0.8
                )
                SELECT *
                FROM ranked_chunks
                WHERE rank_in_doc <= 3
                ORDER BY distance
                LIMIT $4
            """, 
                query_embedding,
                instrument_models,
                targets['document_priorities'],
                max_results
            )
            
            return [
                RetrievalResult(
                    chunk_id=str(r['id']),
                    content=r['content'],
                    source_type='manufacturer',
                    document_title=r['document_title'],
                    section_title=r['section_title'],
                    relevance_score=1.0 - r['distance'],
                    metadata=r['metadata'] or {},
                    page_references=r['page_references'] or [],
                    image_refs=r['image_refs'] or []
                )
                for r in results
            ]
    
    async def _retrieve_lab_docs(self, query_embedding: List[float], 
                                processed_query: ProcessedQuery, 
                                targets: Dict[str, Any], 
                                max_results: int) -> List[RetrievalResult]:
        """Retrieve from lab-specific documents"""
        
        async with self.db_pool.acquire() as conn:
            # Get lab document IDs
            lab_doc_ids = [doc['id'] for doc in targets['relevant_lab_documents']]
            
            if not lab_doc_ids:
                return []
            
            results = await conn.fetch("""
                WITH ranked_chunks AS (
                    SELECT 
                        lc.id,
                        lc.content,
                        lc.metadata,
                        lc.page_references,
                        lc.image_refs,
                        ld.title as document_title,
                        ls.title as section_title,
                        lc.embedding <-> $1 as distance,
                        lc.safety_level,
                        lc.requires_training,
                        ROW_NUMBER() OVER (
                            PARTITION BY ld.id 
                            ORDER BY lc.embedding <-> $1
                        ) as rank_in_doc
                    FROM lab_chunks lc
                    JOIN lab_documents ld ON lc.document_id = ld.id
                    JOIN lab_sections ls ON lc.section_id = ls.id
                    WHERE ld.id = ANY($2)
                    AND lc.embedding <-> $1 < 0.8
                )
                SELECT *
                FROM ranked_chunks
                WHERE rank_in_doc <= 3
                ORDER BY distance
                LIMIT $3
            """,
                query_embedding,
                lab_doc_ids,
                max_results
            )
            
            return [
                RetrievalResult(
                    chunk_id=str(r['id']),
                    content=r['content'],
                    source_type='lab',
                    document_title=r['document_title'],
                    section_title=r['section_title'],
                    relevance_score=1.0 - r['distance'],
                    metadata=r['metadata'] or {},
                    page_references=r['page_references'] or [],
                    image_refs=r['image_refs'] or []
                )
                for r in results
            ]
    
    async def _rerank_results(self, results: List[RetrievalResult], 
                            processed_query: ProcessedQuery, 
                            intent: IntentResult) -> List[RetrievalResult]:
        """Rerank results based on intent and query context"""
        
        for result in results:
            # Base score from vector similarity
            score = result.relevance_score
            
            # Boost lab-specific documents
            if result.source_type == 'lab':
                score *= 1.2
            
            # Boost based on intent matching
            if intent.primary_intent == 'troubleshooting' and 'troubleshoot' in result.content.lower():
                score *= 1.3
            elif intent.primary_intent == 'safety' and any(word in result.content.lower() 
                                                          for word in ['warning', 'caution', 'danger', 'safety']):
                score *= 1.4
            
            # Boost if contains detected entities
            for entity_list in processed_query.detected_entities.values():
                for entity in entity_list:
                    if entity.lower() in result.content.lower():
                        score *= 1.1
            
            # Boost recent lab documents
            if result.source_type == 'lab':
                score *= 1.15
            
            result.relevance_score = min(score, 1.0)
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
```

### Response Generation and Validation

```python
# src/agents/response_generator.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai

class ResponseComponents(BaseModel):
    main_answer: str
    confidence_score: float
    source_citations: List[Dict[str, Any]]
    related_images: List[Dict[str, str]] = []
    safety_warnings: List[str] = []
    follow_up_suggestions: List[str] = []
    procedure_steps: List[str] = []
    additional_context: Optional[str] = None

class ResponseGenerator:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    async def generate_response(self, processed_query: ProcessedQuery, 
                              intent: IntentResult, 
                              retrieval_results: List[RetrievalResult],
                              context: Dict[str, Any]) -> ResponseComponents:
        """Generate comprehensive response with citations and multimedia"""
        
        # Prepare context for response generation
        source_context = self._prepare_source_context(retrieval_results)
        
        response_prompt = f"""
        You are a laboratory assistant helping with instrument-related queries. Generate a comprehensive, accurate response.
        
        User Query: {processed_query.original_text}
        Intent: {intent.primary_intent} (confidence: {intent.confidence})
        Image Context: {processed_query.image_description or "None"}
        
        Available Information:
        {source_context}
        
        Instructions:
        1. Provide a direct, helpful answer to the user's question
        2. Prioritize lab-specific information over manufacturer generic info
        3. Include safety warnings if relevant
        4. Cite specific sources with page numbers where possible
        5. Suggest follow-up actions if appropriate
        6. If this is a procedure, break it into clear steps
        7. Be concise but thorough
        
        Generate response in this JSON format:
        {{
            "main_answer": "Direct answer to the query",
            "confidence_score": 0.0-1.0,
            "safety_warnings": ["warning1", "warning2"] or [],
            "procedure_steps": ["step1", "step2"] or [],
            "follow_up_suggestions": ["suggestion1", "suggestion2"],
            "additional_context": "Any relevant background information or null"
        }}
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": response_prompt}],
            response_format={"type": "json_object"}
        )
        
        response_data = json.loads(response.choices[0].message.content)
        
        # Generate citations
        citations = self._generate_citations(retrieval_results)
        
        # Find related images
        related_images = self._extract_related_images(retrieval_results)
        
        return ResponseComponents(
            main_answer=response_data['main_answer'],
            confidence_score=response_data['confidence_score'],
            source_citations=citations,
            related_images=related_images,
            safety_warnings=response_data.get('safety_warnings', []),
            follow_up_suggestions=response_data.get('follow_up_suggestions', []),
            procedure_steps=response_data.get('procedure_steps', []),
            additional_context=response_data.get('additional_context')
        )
    
    def _prepare_source_context(self, results: List[RetrievalResult]) -> str:
        """Prepare context from retrieval results"""
        context_parts = []
        
        for i, result in enumerate(results[:10]):  # Limit context size
            source_info = f"[Source {i+1}: {result.document_title} - {result.section_title}]"
            if result.page_references:
                source_info += f" (Pages: {', '.join(map(str, result.page_references))})"
            
            context_parts.append(f"{source_info}\n{result.content[:800]}...")
        
        return "\n\n".join(context_parts)
    
    def _generate_citations(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Generate properly formatted citations"""
        citations = []
        
        for i, result in enumerate(results):
            citation = {
                "id": i + 1,
                "document_title": result.document_title,
                "section_title": result.section_title,
                "source_type": result.source_type,
                "page_references": result.page_references,
                "relevance_score": round(result.relevance_score, 3),
                "chunk_id": result.chunk_id
            }
            citations.append(citation)
        
        return citations
    
    def _extract_related_images(self, results: List[RetrievalResult]) -> List[Dict[str, str]]:
        """Extract and organize related images"""
        images = []
        seen_images = set()
        
        for result in results:
            for image_ref in result.image_refs:
                if image_ref not in seen_images:
                    images.append({
                        "image_path": image_ref,
                        "context": f"From {result.document_title} - {result.section_title}",
                        "source_type": result.source_type
                    })
                    seen_images.add(image_ref)
        
        return images[:5]  # Limit to 5 most relevant images

# src/validation/response_validator.py
class ResponseValidator:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    async def validate_response(self, query: ProcessedQuery, response: ResponseComponents, 
                              sources: List[RetrievalResult]) -> Dict[str, Any]:
        """Validate response accuracy and completeness"""
        
        validation_prompt = f"""
        Validate this laboratory assistant response for accuracy and completeness.
        
        Original Query: {query.original_text}
        Generated Response: {response.main_answer}
        
        Available Sources: {len(sources)} documents
        
        Check for:
        1. Factual accuracy based on provided sources
        2. Completeness of answer
        3. Appropriate safety considerations
        4. Proper citation usage
        5. Potential hallucinations or unsupported claims
        
        Return validation in JSON format:
        {{
            "accuracy_score": 0.0-1.0,
            "completeness_score": 0.0-1.0,
            "safety_score": 0.0-1.0,
            "citation_score": 0.0-1.0,
            "overall_quality": 0.0-1.0,
            "issues_found": ["issue1", "issue2"] or [],
            "recommendations": ["rec1", "rec2"] or []
        }}
        """
        
        validation_response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": validation_prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(validation_response.choices[0].message.content)

# src/api/mcp_interface.py
from typing import Dict, Any, AsyncGenerator
import json
import asyncio

class MCPInterface:
    """Model Context Protocol interface for agent integration"""
    
    def __init__(self, response_engine):
        self.response_engine = response_engine
        self.protocol_version = "1.0"
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        
        if request.get("method") == "query":
            return await self._handle_query_request(request)
        elif request.get("method") == "stream_query":
            return await self._handle_streaming_query(request)
        elif request.get("method") == "get_capabilities":
            return self._get_capabilities()
        else:
            return {
                "error": "unsupported_method",
                "message": f"Method {request.get('method')} not supported"
            }
    
    async def _handle_query_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard query request"""
        
        try:
            query_params = request.get("params", {})
            result = await self.response_engine.process_query(query_params)
            
            return {
                "id": request.get("id"),
                "result": {
                    "response": result.main_answer,
                    "confidence": result.confidence_score,
                    "citations": result.source_citations,
                    "images": result.related_images,
                    "safety_warnings": result.safety_warnings,
                    "follow_up": result.follow_up_suggestions,
                    "procedure_steps": result.procedure_steps
                }
            }
        except Exception as e:
            return {
                "id": request.get("id"),
                "error": "processing_error",
                "message": str(e)
            }
    
    async def _handle_streaming_query(self, request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming query with real-time updates"""
        
        query_params = request.get("params", {})
        
        # Yield initial response
        yield {
            "id": request.get("id"),
            "type": "start",
            "message": "Processing query..."
        }
        
        try:
            # Stream processing steps
            async for update in self.response_engine.process_query_streaming(query_params):
                yield {
                    "id": request.get("id"),
                    "type": "progress",
                    "data": update
                }
            
            # Final result
            result = await self.response_engine.get_final_result()
            yield {
                "id": request.get("id"),
                "type": "complete",
                "result": {
                    "response": result.main_answer,
                    "confidence": result.confidence_score,
                    "citations": result.source_citations,
                    "images": result.related_images
                }
            }
            
        except Exception as e:
            yield {
                "id": request.get("id"),
                "type": "error",
                "error": "processing_error",
                "message": str(e)
            }
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Return system capabilities"""
        
        return {
            "capabilities": {
                "query_processing": True,
                "streaming_responses": True,
                "multimodal_input": True,
                "source_citation": True,
                "image_analysis": True,
                "lab_data_isolation": True,
                "real_time_validation": True
            },
            "supported_document_types": [
                "operational_guide",
                "troubleshooting_guide", 
                "technical_specification",
                "safety_document",
                "sop",
                "protocol",
                "maintenance_manual"
            ],
            "max_query_length": 2000,
            "max_response_length": 2000,
            "supported_file_types": ["pdf", "docx", "images"],
            "version": self.protocol_version
        }
```

### Main Response Engine Orchestrator

```python
# response_main.py
import asyncio
from typing import Dict, Any, AsyncGenerator
from src.processing.query_preprocessor import QueryPreprocessor, QueryInput
from src.agents.intent_classifier import IntentClassifier
from src.agents.target_identifier import TargetIdentifier
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.response_generator import ResponseGenerator
from src.validation.response_validator import ResponseValidator
from src.hydration.embedding_generator import EmbeddingGenerator
import asyncpg

class ResponseEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.query_preprocessor = QueryPreprocessor(config['openai']['api_key'])
        self.intent_classifier = IntentClassifier(config['openai']['api_key'])
        self.embedding_generator = EmbeddingGenerator(config['openai']['api_key'])
        self.response_generator = ResponseGenerator(config['openai']['api_key'])
        self.response_validator = ResponseValidator(config['openai']['api_key'])
        
        # Database connection
        self.db_pool = None
        
        # Will be initialized with database
        self.target_identifier = None
        self.hybrid_retriever = None
    
    async def initialize(self):
        """Initialize database connections and components"""
        self.db_pool = await asyncpg.create_pool(self.config['database']['connection_string'])
        
        self.target_identifier = TargetIdentifier(self.db_pool)
        self.hybrid_retriever = HybridRetriever(self.db_pool, self.embedding_generator)
    
    async def process_query(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        """Main query processing pipeline"""
        
        try:
            # Step 1: Preprocess query
            query_obj = QueryInput(**query_input)
            processed_query = await self.query_preprocessor.preprocess_query(query_obj)
            
            # Step 2a: Classify intent
            intent = await self.intent_classifier.classify_intent(
                processed_query, 
                {"lab_id": query_obj.lab_id, "instruments": query_obj.instrument_ids}
            )
            
            # Step 2b: Identify targets
            targets = await self.target_identifier.identify_targets(
                processed_query, 
                intent, 
                query_obj.lab_id or ""
            )
            
            # Step 3: Retrieve relevant information
            retrieval_results = await self.hybrid_retriever.retrieve(
                processed_query,
                intent,
                targets,
                query_obj.max_results
            )
            
            # Step 4: Validate retrieval relevance
            if not retrieval_results or retrieval_results[0].relevance_score < 0.5:
                return {
                    "success": False,
                    "message": "No relevant information found for your query",
                    "suggestions": ["Try rephrasing your question", "Check if the instrument model is correct"]
                }
            
            # Step 5: Generate response
            response = await self.response_generator.generate_response(
                processed_query,
                intent,
                retrieval_results,
                {"lab_id": query_obj.lab_id}
            )
            
            # Step 6: Validate response
            validation = await self.response_validator.validate_response(
                processed_query,
                response,
                retrieval_results
            )
            
            # Step 7: Format final response
            return {
                "success": True,
                "response": response.main_answer,
                "confidence": response.confidence_score,
                "citations": response.source_citations,
                "images": response.related_images if query_obj.include_images else [],
                "safety_warnings": response.safety_warnings,
                "follow_up_suggestions": response.follow_up_suggestions,
                "procedure_steps": response.procedure_steps,
                "additional_context": response.additional_context,
                "validation": validation,
                "metadata": {
                    "intent": intent.primary_intent,
                    "intent_confidence": intent.confidence,
                    "sources_used": len(retrieval_results),
                    "processing_time": None  # Add timing
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred while processing your query"
            }
    
    async def process_query_streaming(self, query_input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of query processing for real-time updates"""
        
        yield {"step": "preprocessing", "message": "Analyzing your query..."}
        
        query_obj = QueryInput(**query_input)
        processed_query = await self.query_preprocessor.preprocess_query(query_obj)
        
        yield {"step": "intent_classification", "message": "Understanding your intent..."}
        
        intent = await self.intent_classifier.classify_intent(
            processed_query, 
            {"lab_id": query_obj.lab_id}
        )
        
        yield {
            "step": "intent_classified", 
            "data": {"intent": intent.primary_intent, "confidence": intent.confidence}
        }
        
        yield {"step": "target_identification", "message": "Finding relevant instruments and documents..."}
        
        targets = await self.target_identifier.identify_targets(processed_query, intent, query_obj.lab_id or "")
        
        yield {
            "step": "targets_identified",
            "data": {
                "instruments": len(targets['relevant_instruments']),
                "documents": len(targets['relevant_lab_documents'])
            }
        }
        
        yield {"step": "retrieval", "message": "Searching for relevant information..."}
        
        retrieval_results = await self.hybrid_retriever.retrieve(
            processed_query, intent, targets, query_obj.max_results
        )
        
        yield {
            "step": "retrieval_complete",
            "data": {"sources_found": len(retrieval_results)}
        }
        
        yield {"step": "generating_response", "message": "Generating your answer..."}
        
        response = await self.response_generator.generate_response(
            processed_query, intent, retrieval_results, {"lab_id": query_obj.lab_id}
        )
        
        yield {"step": "response_complete", "data": {"confidence": response.confidence_score}}

# Usage Example
async def main():
    config = {
        "openai": {"api_key": "your-openai-key"},
        "database": {"connection_string": "postgresql://..."},
        "response": {"max_results": 20}
    }
    
    engine = ResponseEngine(config)
    await engine.initialize()
    
    # Example query
    query = {
        "text": "My HPLC is showing error code E001 and the pressure is fluctuating. What should I do?",
        "lab_id": "lab-123",
        "instrument_ids": ["instrument-456"],
        "include_images": True
    }
    
    result = await engine.process_query(query)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```
