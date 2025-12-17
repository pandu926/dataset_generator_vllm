"""
Phase 1 Redesign: Semantic Chunking with Entity Extraction

Principles:
- "Chunk is a unit of retrieval, not text splitting"
- Self-contained, Atomic, Retrievable, Linkable

Components:
A. Document Profiling
B. Entity & Concept Extraction
C. Semantic Chunking (by content type, not tokens)
D. Chunk Enrichment (3-layer metadata)
E. Cross-Chunk Linking
F. Table Processing
G. Quality Gates
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Entity:
    """Extracted entity from documents"""
    id: str
    name: str
    type: str  # named, numerical, procedural, relational
    value: Any
    source_sections: List[str]
    mentions: int = 1


@dataclass
class SemanticChunk:
    """Semantic chunk with rich metadata (3 layers)"""
    id: str
    content: str
    content_type: str  # faq, procedure, table_row, description, policy, narrative
    
    # Layer 1: Structural Metadata
    source_file: str
    section_path: List[str]
    position: str  # start, middle, end
    sibling_chunks: List[str]
    parent_chunk: Optional[str]
    
    # Layer 2: Semantic Metadata
    primary_topic: str
    secondary_topics: List[str]
    entities_mentioned: List[str]
    question_types_answerable: List[str]
    requires_other_chunks: bool
    
    # Layer 3: Retrieval Metadata
    summary: str
    hypothetical_questions: List[str]
    search_keywords: List[str]
    
    # Quality metrics
    token_count: int
    char_count: int
    quality_score: float = 0.0
    
    # Additional metadata (for overlap context, table info, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # E5-Multilingual embedding (optional, stored as list for JSON serialization)
    embedding: Optional[List[float]] = None


@dataclass
class ChunkRelation:
    """Relation between chunks"""
    source_id: str
    target_id: str
    relation_type: str  # relates_to, compares_to, requires, details, generalizes
    strength: float = 1.0


@dataclass
class DocumentProfile:
    """Profile of a document before chunking"""
    filename: str
    doc_type: str  # faq, procedure, table, narrative, mixed
    hierarchy_levels: int
    section_count: int
    table_count: int
    list_count: int
    dense_sections: List[str]
    sparse_sections: List[str]


# =============================================================================
# ENTITY EXTRACTION (Step B)
# =============================================================================

class EntityExtractor:
    """Extract entities from PMB documents"""
    
    # Patterns for different entity types
    PATTERNS = {
        # Named entities
        "prodi": [
            r"(?:Program Studi|Prodi)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(Teknik Informatika|Keperawatan|Kebidanan|Manajemen|Akuntansi|"
            r"Pendidikan Agama Islam|Pendidikan Bahasa Arab|Arsitektur|"
            r"Teknik Sipil|Teknik Mesin|Ilmu Hukum|Perbankan Syariah)"
        ],
        "fakultas": [
            r"(Fakultas\s+[A-Z][a-z]+(?:\s+(?:dan|&)\s+[A-Z][a-z]+)*(?:\s+\([A-Z]+\))?)",
            r"(FASTIKOM|FIKES|FSH|FITK|FKSP)"
        ],
        "beasiswa": [
            r"[Bb]easiswa\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(Beasiswa Tahfidz|Beasiswa Prestasi|KIP-K|BIDIKMISI|PPA)"
        ],
        
        # Numerical entities
        "biaya": [
            r"Rp\.?\s*([\d.,]+)",
            r"([\d.,]+)\s*(?:rupiah|juta)"
        ],
        "tanggal": [
            r"(\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ],
        "durasi": [
            r"(\d+)\s*(?:semester|tahun|bulan|minggu|hari)"
        ],
        "kuota": [
            r"[Kk]uota[:\s]+(\d+)",
            r"(\d+)\s*mahasiswa"
        ],
        
        # Procedural entities
        "langkah": [
            r"(?:Langkah|Tahap|Step)\s+(\d+)[:\s]",
            r"(\d+)\.\s+[A-Z]"
        ],
        "syarat": [
            r"[Ss]yarat[:\s]+(.+?)(?:\n|$)",
        ],
        "dokumen": [
            r"(?:dokumen|berkas)[:\s]+(.+?)(?:\n|$)",
            r"(Ijazah|SKHUN|KTP|KK|Akta|Foto|Surat)"
        ]
    }
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.entity_counter = 0
    
    def extract_all(self, content: str, source_section: str = "") -> List[Entity]:
        """Extract all entities from content"""
        found_entities = []
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    match = match.strip()
                    if len(match) > 2:  # Skip very short matches
                        entity = self._add_entity(match, entity_type, source_section)
                        found_entities.append(entity)
        
        return found_entities
    
    def _add_entity(self, name: str, entity_type: str, source_section: str) -> Entity:
        """Add or update entity"""
        # Create key for dedup
        key = f"{entity_type}:{name.lower()}"
        
        if key in self.entities:
            self.entities[key].mentions += 1
            if source_section not in self.entities[key].source_sections:
                self.entities[key].source_sections.append(source_section)
        else:
            self.entity_counter += 1
            entity_id = f"ent_{self.entity_counter:04d}"
            
            self.entities[key] = Entity(
                id=entity_id,
                name=name,
                type=entity_type,
                value=self._extract_value(name, entity_type),
                source_sections=[source_section] if source_section else []
            )
        
        return self.entities[key]
    
    def _extract_value(self, name: str, entity_type: str) -> Any:
        """Extract structured value from entity"""
        if entity_type == "biaya":
            # Parse currency
            nums = re.findall(r'[\d.,]+', name)
            if nums:
                return int(nums[0].replace('.', '').replace(',', ''))
        elif entity_type in ["durasi", "kuota"]:
            nums = re.findall(r'\d+', name)
            if nums:
                return int(nums[0])
        return name
    
    def get_all_entities(self) -> Dict[str, Entity]:
        return self.entities
    
    def save(self, path: str):
        """Save entities to JSON"""
        data = {k: asdict(v) for k, v in self.entities.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# =============================================================================
# DOCUMENT PROFILER (Step A)
# =============================================================================

class DocumentProfiler:
    """Analyze document structure before chunking"""
    
    def profile(self, content: str, filename: str) -> DocumentProfile:
        """Create document profile"""
        lines = content.split('\n')
        
        # Count structural elements
        h1_count = len(re.findall(r'^#\s', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s', content, re.MULTILINE))
        h4_count = len(re.findall(r'^####\s', content, re.MULTILINE))
        
        table_count = content.count('|---|')
        list_count = len(re.findall(r'^\s*[-*]\s', content, re.MULTILINE))
        
        # Detect document type
        if '?' in content and 'A:' in content:
            doc_type = "faq"
        elif table_count > 3:
            doc_type = "table"
        elif list_count > 20:
            doc_type = "procedure"
        elif h3_count > 10:
            doc_type = "mixed"
        else:
            doc_type = "narrative"
        
        # Find sections and density
        sections = self._extract_sections(content)
        dense_sections = [s for s, c in sections.items() if len(c) > 500]
        sparse_sections = [s for s, c in sections.items() if len(c) < 100]
        
        return DocumentProfile(
            filename=filename,
            doc_type=doc_type,
            hierarchy_levels=sum(1 for c in [h1_count, h2_count, h3_count, h4_count] if c > 0),
            section_count=h2_count + h3_count,
            table_count=table_count,
            list_count=list_count,
            dense_sections=dense_sections[:10],
            sparse_sections=sparse_sections[:10]
        )
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections and their content"""
        sections = {}
        current_section = "Intro"
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections


# =============================================================================
# SEMANTIC CHUNKER (Step C)
# =============================================================================

class SemanticChunker:
    """Chunk by semantic boundaries, not token count"""
    
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
        self.chunk_counter = 0
        self.chunks: List[SemanticChunk] = []
    
    def _generate_id(self, content: str, category: str = "chunk") -> str:
        """Generate unique chunk ID with category prefix"""
        self.chunk_counter += 1
        # Clean category: lowercase, remove special chars
        cat = category.lower().replace(" ", "_").replace("-", "_")
        return f"{cat}_{self.chunk_counter:04d}"
    
    def _extract_category_from_source(self, source_file: str) -> str:
        """Extract category from source filename for chunk ID prefix"""
        name = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file
        
        # Known category mappings - each file type gets its own category
        category_map = {
            'profil': 'profil', 
            'alur': 'alur', 
            'beasiswa': 'beasiswa',
            'fasilitas': 'fasilitas', 
            'syarat': 'syarat', 
            'singkatan': 'singkatan',
            # CSV files - each gets own category
            'biaya': 'biaya',
            'jadwal': 'jadwal',
            'kontak': 'kontak',
            'program': 'prodi',  # pmb_program_studi.csv -> prodi
        }
        
        name_lower = name.lower()
        for key, cat in category_map.items():
            if key in name_lower:
                return f"{cat}_faq" if 'faq' in name_lower else cat
        
        # Fallback: first word
        first_word = name.split('_')[0].lower() if '_' in name else name.lower()
        return first_word if first_word else "unknown"

    
    def chunk_document(self, content: str, source_file: str, 
                       profile: DocumentProfile) -> List[SemanticChunk]:
        """Chunk document using HYBRID approach (FAQ + Section + Table)"""
        
        # Always use hybrid chunking for comprehensive coverage
        return self._chunk_hybrid(content, source_file, profile)
    
    def _chunk_hybrid(self, content: str, source_file: str, 
                      profile: DocumentProfile) -> List[SemanticChunk]:
        """
        HYBRID CHUNKING: Combine multiple strategies for comprehensive coverage.
        
        1. Extract FAQ pairs (Q&A format)
        2. Extract table rows 
        3. Extract sections (## and ###)
        4. Deduplicate overlapping content
        """
        all_chunks = []
        used_content_hashes = set()  # For deduplication
        
        # 1. Extract FAQs first
        faq_chunks = self._chunk_faq(content, source_file)
        for chunk in faq_chunks:
            content_hash = hash(chunk.content[:100])
            if content_hash not in used_content_hashes:
                all_chunks.append(chunk)
                used_content_hashes.add(content_hash)
        
        # 2. Extract table rows
        table_chunks = self._chunk_tables(content, source_file)
        for chunk in table_chunks:
            content_hash = hash(chunk.content[:100])
            if content_hash not in used_content_hashes:
                all_chunks.append(chunk)
                used_content_hashes.add(content_hash)
        
        # 3. Extract sections (## and ### headers)
        section_chunks = self._chunk_by_section(content, source_file)
        for chunk in section_chunks:
            content_hash = hash(chunk.content[:100])
            if content_hash not in used_content_hashes:
                all_chunks.append(chunk)
                used_content_hashes.add(content_hash)
        
        # 4. If still no chunks, use fallback paragraph chunking
        if len(all_chunks) == 0:
            fallback_chunks = self._chunk_paragraphs(content, source_file)
            all_chunks.extend(fallback_chunks)
        
        return all_chunks
    
    def _chunk_paragraphs(self, content: str, source_file: str) -> List[SemanticChunk]:
        """Fallback: Chunk by paragraphs when other methods fail"""
        chunks = []
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para or para.startswith('#'):
                continue
            
            if len(current_chunk) + len(para) < 1500:
                current_chunk += para + "\n\n"
            else:
                if len(current_chunk) > 100:
                    chunk = self._create_chunk(
                        content=current_chunk.strip(),
                        content_type="description",
                        source_file=source_file,
                        section_path=["Content"],
                        primary_topic=self._detect_topic(current_chunk)
                    )
                    chunks.append(chunk)
                current_chunk = para + "\n\n"
        
        # Last chunk
        if len(current_chunk) > 100:
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                content_type="description",
                source_file=source_file,
                section_path=["Content"],
                primary_topic=self._detect_topic(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks

    
    def _chunk_faq(self, content: str, source_file: str) -> List[SemanticChunk]:
        """Chunk FAQ: 1 Q-A pair = 1 chunk (FIX #4 - Multiple patterns)"""
        chunks = []
        all_matches = []
        
        # Multiple FAQ patterns (FIX #4)
        faq_patterns = [
            # Q: / A: format
            r'(?:Q|###\s*Q)[:\s]*(.+?)\n+(?:A|###\s*A)[:\s]*(.+?)(?=(?:Q|###\s*Q)[:\s]|$)',
            # **Pertanyaan** / **Jawaban** format
            r'\*\*(?:Pertanyaan|Q)[:\s]*\*\*(.+?)\n+\*\*(?:Jawaban|A)[:\s]*\*\*(.+?)(?=\*\*(?:Pertanyaan|Q)|$)',
            # Numbered: 1. Question? Answer...
            r'(?:\d+)\.\s*([^?]+\?)\n+([^0-9]+?)(?=\d+\.|$)',
        ]
        
        for pattern in faq_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            all_matches.extend(matches)
        
        for q, a in all_matches:
            q = q.strip()
            a = a.strip()
            if len(q) < 10 or len(a) < 20:  # Skip invalid
                continue
            full_content = f"Pertanyaan: {q}\n\nJawaban: {a}"
            
            chunk = self._create_chunk(
                content=full_content,
                content_type="faq",
                source_file=source_file,
                section_path=["FAQ"],
                primary_topic=self._detect_topic(full_content),
                hypothetical_questions=[q]  # The actual question!
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_tables(self, content: str, source_file: str) -> List[SemanticChunk]:
        """Chunk tables: Each row + header = 1 chunk"""
        chunks = []
        
        # Find tables
        table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)'
        tables = re.findall(table_pattern, content)
        
        for table_idx, table in enumerate(tables):
            lines = table.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Parse header
            headers = [h.strip() for h in lines[0].split('|') if h.strip()]
            
            # Parse each row
            for row_idx, line in enumerate(lines[2:]):  # Skip header and separator
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) != len(headers):
                    continue
                
                # Create row content
                row_data = {h: c for h, c in zip(headers, cells)}
                row_content = self._format_table_row(headers, cells)
                
                # Detect topic from row content
                topic = self._detect_topic(row_content)
                
                # Generate hypothetical questions for this row
                hypo_questions = self._generate_table_questions(headers, cells)
                
                chunk = self._create_chunk(
                    content=row_content,
                    content_type="table_row",
                    source_file=source_file,
                    section_path=["Tabel", f"Row {row_idx+1}"],
                    primary_topic=topic,
                    hypothetical_questions=hypo_questions,
                    metadata={"table_index": table_idx, "row_index": row_idx, "headers": headers}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_procedures(self, content: str, source_file: str) -> List[SemanticChunk]:
        """Chunk procedures: Keep steps together"""
        chunks = []
        
        # Find procedural sections (numbered lists or step markers)
        sections = self._split_by_headers(content)
        
        for section_title, section_content in sections.items():
            # Check if this section has steps
            has_steps = bool(re.search(r'^\s*\d+\.', section_content, re.MULTILINE))
            has_bullets = bool(re.search(r'^\s*[-*]', section_content, re.MULTILINE))
            
            if has_steps or has_bullets:
                # Clean section title (remove numbering like "1. " or "1.1 ")
                clean_title = re.sub(r'^[\d.]+\s*', '', section_title).strip()
                
                # Keep entire procedure together
                chunk = self._create_chunk(
                    content=f"## {section_title}\n\n{section_content}",
                    content_type="procedure",
                    source_file=source_file,
                    section_path=[section_title],
                    primary_topic=self._detect_topic(section_content),
                    question_types=["procedural"],
                    hypothetical_questions=[
                        f"Bagaimana {clean_title.lower()}?",
                        f"Apa langkah-langkah {clean_title.lower()}?"
                    ]
                )
                chunks.append(chunk)
            else:
                # Split narrative by paragraphs with overlap
                para_chunks = self._chunk_narrative_with_overlap(
                    section_content, source_file, section_title
                )
                chunks.extend(para_chunks)
        
        return chunks
    
    def _chunk_by_section(self, content: str, source_file: str) -> List[SemanticChunk]:
        """Chunk by semantic sections (Level 3 headers)"""
        chunks = []
        sections = self._split_by_headers(content, level=3)
        
        for section_title, section_content in sections.items():
            if len(section_content.strip()) < 50:
                continue  # Skip empty sections
            
            # Decide chunking strategy based on content length
            tokens = len(section_content.split())
            
            if tokens <= 400:
                # Section fits in one chunk
                chunk = self._create_chunk(
                    content=f"### {section_title}\n\n{section_content}",
                    content_type="description",
                    source_file=source_file,
                    section_path=self._extract_section_path(section_title),
                    primary_topic=self._detect_topic(section_content),
                    hypothetical_questions=self._generate_section_questions(section_title, section_content)
                )
                chunks.append(chunk)
            else:
                # Split with semantic overlap
                para_chunks = self._chunk_narrative_with_overlap(
                    section_content, source_file, section_title
                )
                chunks.extend(para_chunks)
        
        return chunks
    
    def _chunk_narrative_with_overlap(self, content: str, source_file: str, 
                                       section_title: str) -> List[SemanticChunk]:
        """Split narrative with overlap context (FIX #2 - no content duplication)"""
        chunks = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return chunks
        
        current_chunk = []
        current_tokens = 0
        previous_context = ""  # Store summary of previous chunk, not full content
        
        for para in paragraphs:
            para_tokens = len(para.split())
            
            if current_tokens + para_tokens <= 350:
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                # Save current chunk
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    
                    chunk = self._create_chunk(
                        content=chunk_content,
                        content_type="narrative",
                        source_file=source_file,
                        section_path=[section_title],
                        primary_topic=self._detect_topic(chunk_content),
                        # Store overlap context as metadata, not in content (FIX #2)
                        metadata={"previous_context": previous_context} if previous_context else None
                    )
                    chunks.append(chunk)
                    
                    # Store last sentence as context (not full paragraphs)
                    last_para = current_chunk[-1]
                    sentences = re.split(r'(?<=[.!?])\s+', last_para)
                    previous_context = sentences[-1] if sentences else last_para[:100]
                
                # Start new chunk
                current_chunk = [para]
                current_tokens = para_tokens
        
        # Save remaining
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            
            chunk = self._create_chunk(
                content=chunk_content,
                content_type="narrative",
                source_file=source_file,
                section_path=[section_title],
                primary_topic=self._detect_topic(chunk_content),
                metadata={"previous_context": previous_context} if previous_context else None
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, content_type: str, source_file: str,
                      section_path: List[str], primary_topic: str,
                      hypothetical_questions: List[str] = None,
                      question_types: List[str] = None,
                      metadata: Dict = None) -> SemanticChunk:
        """Create a semantic chunk with all metadata layers"""
        
        # Extract category from source file name
        category = self._extract_category_from_source(source_file)
        
        # Extract entities from this chunk
        entities = self.entity_extractor.extract_all(content, section_path[0] if section_path else "")
        entity_names = [e.name for e in entities]
        
        # Detect question types this chunk can answer
        if question_types is None:
            question_types = self._detect_answerable_question_types(content)
        
        # Generate hypothetical questions if not provided
        if hypothetical_questions is None:
            hypothetical_questions = self._generate_hypothetical_questions(content, primary_topic)
        
        # GUARANTEE non-empty questions (FIX #7)
        if not hypothetical_questions:
            hypothetical_questions = [f"Informasi tentang {primary_topic}?"]
        
        # Generate summary
        summary = self._generate_summary(content)
        
        # Extract search keywords
        keywords = self._extract_keywords(content, entity_names)
        
        # Calculate quality score (FIX #1)
        quality_score = self._calculate_quality_score(content, entity_names, question_types)
        
        chunk = SemanticChunk(
            id=self._generate_id(content, category),
            content=content,
            content_type=content_type,
            
            # Layer 1: Structural
            source_file=source_file,
            section_path=section_path,
            position="middle",  # Updated later in linking
            sibling_chunks=[],  # Updated later in linking
            parent_chunk=None,  # Updated later in linking
            
            # Layer 2: Semantic
            primary_topic=primary_topic,
            secondary_topics=self._detect_secondary_topics(content, primary_topic),
            entities_mentioned=entity_names,
            question_types_answerable=question_types,
            requires_other_chunks=self._check_requires_context(content),
            
            # Layer 3: Retrieval
            summary=summary,
            hypothetical_questions=hypothetical_questions[:5],
            search_keywords=keywords[:20],
            
            # Quality - NOW CALCULATED (FIX #1)
            token_count=len(content.split()),
            char_count=len(content),
            quality_score=quality_score,
            
            # Additional metadata (FIX: now stored)
            metadata=metadata or {}
        )
        
        self.chunks.append(chunk)
        return chunk
    
    def _calculate_quality_score(self, content: str, entities: List[str], 
                                  question_types: List[str]) -> float:
        """Calculate chunk quality score (FIX #1 - was dead code)"""
        score = 0.0
        
        # 1. Entity density (0-0.3)
        entity_ratio = len(entities) / max(1, len(content.split()) / 50)
        score += min(0.3, entity_ratio * 0.1)
        
        # 2. Question type diversity (0-0.3)  
        score += len(question_types) * 0.1
        
        # 3. Content length appropriateness (0-0.2)
        tokens = len(content.split())
        if 50 <= tokens <= 300:
            score += 0.2
        elif 30 <= tokens <= 400:
            score += 0.1
        
        # 4. Has specific data (0-0.2)
        if re.search(r'Rp\.?|[0-9]+', content):
            score += 0.1
        if entities:
            score += 0.1
        
        return min(1.0, score)
    
    # --- Helper methods ---
    
    def _split_by_headers(self, content: str, level: int = 2) -> Dict[str, str]:
        """Split content by markdown headers"""
        pattern = r'^' + '#' * level + r'\s+(.+)$'
        sections = {}
        current_title = "Intro"
        current_content = []
        
        for line in content.split('\n'):
            match = re.match(pattern, line)
            if match:
                if current_content:
                    sections[current_title] = '\n'.join(current_content)
                current_title = match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_title] = '\n'.join(current_content)
        
        return sections
    
    def _detect_topic(self, content: str) -> str:
        """Detect primary topic of content"""
        content_lower = content.lower()
        
        topic_scores = {
            "biaya": len(re.findall(r'biaya|spp|sks|rp\.?|pembayaran', content_lower)),
            "beasiswa": len(re.findall(r'beasiswa|tahfidz|prestasi|kip', content_lower)),
            "program_studi": len(re.findall(r'prodi|program studi|fakultas|jurusan|akreditasi', content_lower)),
            "pendaftaran": len(re.findall(r'daftar|registrasi|pendaftaran|pmb', content_lower)),
            "persyaratan": len(re.findall(r'syarat|dokumen|ijazah|berkas', content_lower)),
            "jadwal": len(re.findall(r'jadwal|gelombang|tanggal|periode', content_lower)),
            "kontak": len(re.findall(r'kontak|telepon|whatsapp|email', content_lower)),
            "orientasi": len(re.findall(r'orientasi|matrikulasi|ospek|pkkmb', content_lower)),
            "institusi": len(re.findall(r'unsiq|universitas|visi|misi|wonosobo', content_lower)),
        }
        
        if max(topic_scores.values()) == 0:
            return "umum"
        
        return max(topic_scores, key=topic_scores.get)
    
    def _detect_secondary_topics(self, content: str, primary: str) -> List[str]:
        """Detect secondary topics"""
        content_lower = content.lower()
        topics = []
        
        topic_keywords = {
            "biaya": ["rp", "bayar", "spp"],
            "beasiswa": ["beasiswa", "gratis"],
            "jadwal": ["tanggal", "bulan"],
            "persyaratan": ["syarat", "dokumen"],
        }
        
        for topic, keywords in topic_keywords.items():
            if topic != primary and any(kw in content_lower for kw in keywords):
                topics.append(topic)
        
        return topics[:3]
    
    def _detect_answerable_question_types(self, content: str) -> List[str]:
        """Detect what types of questions this chunk can answer"""
        types = []
        
        # Factual: has specific data
        if re.search(r'Rp\.?|[0-9]+|adalah|merupakan', content):
            types.append("factual")
        
        # Procedural: has steps
        if re.search(r'langkah|tahap|cara|[0-9]+\.\s', content):
            types.append("procedural")
        
        # Comparative: has multiple items
        if content.count('|') > 5 or re.search(r'dibanding|versus|vs|lebih', content.lower()):
            types.append("comparative")
        
        # Conditional: has conditions
        if re.search(r'jika|apabila|dalam hal|kecuali', content.lower()):
            types.append("conditional")
        
        return types if types else ["factual"]
    
    def _check_requires_context(self, content: str) -> bool:
        """Check if chunk needs other chunks for full understanding"""
        indicators = [
            r'^(?:Selain itu|Sebagai tambahan|Selanjutnya)',
            r'(?:lihat|merujuk pada|sesuai dengan)\s+(?:bagian|section)',
            r'(?:seperti disebutkan|hal ini)',
        ]
        return any(re.search(p, content, re.IGNORECASE) for p in indicators)
    
    def _generate_summary(self, content: str) -> str:
        """Generate extractive summary using key sentences (FIX #3)"""
        # Extractive summarization: find sentences with most entities/keywords
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        if not sentences:
            return content[:100]
        
        # Score each sentence by keyword density
        scored = []
        key_patterns = [
            r'Rp\.?\s*[\d.,]+',  # Money
            r'\d+',  # Numbers
            r'adalah|merupakan|yaitu',  # Definitions
            r'UNSIQ|universitas|fakultas|prodi',  # Key entities
        ]
        
        for sent in sentences:
            if len(sent) < 20:  # Skip very short
                continue
            score = sum(len(re.findall(p, sent, re.I)) for p in key_patterns)
            # Boost first sentences (often topic sentences)
            if sentences.index(sent) < 2:
                score += 1
            scored.append((score, sent))
        
        if not scored:
            return sentences[0][:200] if sentences else content[:100]
        
        # Return top 1-2 sentences
        scored.sort(reverse=True)
        top_sentences = [s[1] for s in scored[:2]]
        return ' '.join(top_sentences)[:300]
    
    def _generate_hypothetical_questions(self, content: str, topic: str) -> List[str]:
        """Generate content-based questions (FIX #2 - not just templates)"""
        questions = []
        
        # 1. Extract question patterns from interrogative keywords in content
        q_patterns = [
            (r'(apa\s+(?:itu|saja)?\s+[^.?]+)', 'Apa'),
            (r'(berapa\s+[^.?]+)', 'Berapa'),
            (r'(bagaimana\s+[^.?]+)', 'Bagaimana'),
            (r'(kapan\s+[^.?]+)', 'Kapan'),
        ]
        
        content_lower = content.lower()
        
        # 2. Generate questions based on ACTUAL content patterns
        # Find subjects/entities mentioned
        entities = self.entity_extractor.extract_all(content, "")
        entity_names = [e.name for e in entities[:3]]
        
        # Find key phrases that could be answers
        if re.search(r'Rp\.?\s*[\d.,]+', content):
            for entity in entity_names:
                questions.append(f"Berapa biaya {entity}?")
        
        if re.search(r'\d+\.\s+', content):  # Numbered list = procedure
            questions.append(f"Bagaimana langkah-langkah dalam {topic}?")
        
        if re.search(r'syarat|dokumen|persyaratan', content_lower):
            questions.append("Apa saja persyaratan yang diperlukan?")
        
        if re.search(r'tanggal|bulan|periode', content_lower):
            questions.append("Kapan jadwal atau periodenya?")
        
        # 3. Fallback: Generate from entities
        if not questions and entity_names:
            questions.append(f"Informasi tentang {entity_names[0]}?")
            questions.append(f"Apa yang dimaksud dengan {entity_names[0]}?")
        
        # 4. Topic-based fallback
        if not questions:
            topic_qs = {
                "biaya": "Berapa biaya kuliah?",
                "beasiswa": "Beasiswa apa yang tersedia?",
                "pendaftaran": "Bagaimana cara mendaftar?",
            }
            questions.append(topic_qs.get(topic, f"Informasi tentang {topic}?"))
        
        return questions[:5]
    
    def _generate_section_questions(self, title: str, content: str) -> List[str]:
        """Generate questions based on section"""
        title_lower = title.lower()
        questions = []
        
        if "biaya" in title_lower:
            questions.append(f"Berapa {title}?")
        elif "syarat" in title_lower or "persyaratan" in title_lower:
            questions.append(f"Apa saja {title}?")
        elif "jadwal" in title_lower:
            questions.append(f"Kapan {title}?")
        else:
            questions.append(f"Apa itu {title}?")
            questions.append(f"Jelaskan tentang {title}")
        
        return questions
    
    def _generate_table_questions(self, headers: List[str], cells: List[str]) -> List[str]:
        """Generate questions for table row"""
        questions = []
        
        # Use first cell as subject
        subject = cells[0] if cells else "ini"
        
        for header in headers[1:]:  # Skip first (usually name)
            questions.append(f"Berapa {header.lower()} untuk {subject}?")
        
        return questions[:3]
    
    def _format_table_row(self, headers: List[str], cells: List[str]) -> str:
        """Format table row as readable text"""
        lines = []
        for h, c in zip(headers, cells):
            lines.append(f"- {h}: {c}")
        return '\n'.join(lines)
    
    def _extract_section_path(self, section_title: str) -> List[str]:
        """Extract section path from title"""
        # Remove numbering
        clean = re.sub(r'^[\d.]+\s*', '', section_title)
        return [clean]
    
    def _get_overlap_text(self, paragraph: str) -> str:
        """Get last 2-3 sentences for overlap"""
        sentences = re.split(r'[.!?]\s+', paragraph)
        return '. '.join(sentences[-3:]) if len(sentences) >= 3 else paragraph[-200:]
    
    def _extract_keywords(self, content: str, entities: List[str]) -> List[str]:
        """Extract search keywords"""
        keywords = set(entities)
        
        # Add important words
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        keywords.update(words[:10])
        
        # Add numbers with context
        nums = re.findall(r'Rp\.?\s*[\d.,]+', content)
        keywords.update(nums[:5])
        
        return list(keywords)


# =============================================================================
# CHUNK GRAPH (Step E)
# =============================================================================

class ChunkGraph:
    """Build and manage chunk relations"""
    
    def __init__(self):
        self.relations: List[ChunkRelation] = []
    
    def build_graph(self, chunks: List[SemanticChunk]) -> List[ChunkRelation]:
        """Build relations between chunks"""
        
        # Create indices
        by_topic = {}
        by_entity = {}
        by_section = {}
        
        for chunk in chunks:
            # Index by topic
            topic = chunk.primary_topic
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(chunk)
            
            # Index by entity
            for entity in chunk.entities_mentioned:
                if entity not in by_entity:
                    by_entity[entity] = []
                by_entity[entity].append(chunk)
            
            # Index by section
            if chunk.section_path:
                section = chunk.section_path[0]
                if section not in by_section:
                    by_section[section] = []
                by_section[section].append(chunk)
        
        # Build relations
        self._link_same_topic(by_topic)
        self._link_same_entity(by_entity)
        self._link_siblings(by_section)
        self._link_table_rows(chunks)
        
        return self.relations
    
    def _link_same_topic(self, by_topic: Dict[str, List[SemanticChunk]]):
        """Link chunks with same topic - WITH THRESHOLD (FIX #6)"""
        MAX_RELATIONS_PER_TOPIC = 10  # Limit to reduce noise
        
        for topic, chunks in by_topic.items():
            if len(chunks) <= 1:
                continue
            
            # Only link most related pairs, not all combinations
            relations_added = 0
            for i, chunk1 in enumerate(chunks):
                if relations_added >= MAX_RELATIONS_PER_TOPIC:
                    break
                # Link to next 2 chunks only (locality)
                for chunk2 in chunks[i+1:i+3]:
                    self.relations.append(ChunkRelation(
                        source_id=chunk1.id,
                        target_id=chunk2.id,
                        relation_type="relates_to",
                        strength=0.7
                    ))
                    relations_added += 1
    
    def _link_same_entity(self, by_entity: Dict[str, List[SemanticChunk]]):
        """Link chunks mentioning same entity - WITH DEDUP (FIX #9)"""
        seen_pairs = set()
        
        for entity, chunks in by_entity.items():
            if len(chunks) <= 1 or len(chunks) > 5:  # Skip very common entities
                continue
            
            for i, chunk1 in enumerate(chunks):
                for chunk2 in chunks[i+1:]:
                    pair_key = tuple(sorted([chunk1.id, chunk2.id]))
                    if pair_key not in seen_pairs:
                        self.relations.append(ChunkRelation(
                            source_id=chunk1.id,
                            target_id=chunk2.id,
                            relation_type="relates_to",
                            strength=0.9
                        ))
                        seen_pairs.add(pair_key)
    
    def _link_siblings(self, by_section: Dict[str, List[SemanticChunk]]):
        """Link sibling chunks in same section"""
        for section, chunks in by_section.items():
            for i, chunk in enumerate(chunks):
                # Update position
                if i == 0:
                    chunk.position = "start"
                elif i == len(chunks) - 1:
                    chunk.position = "end"
                else:
                    chunk.position = "middle"
                
                # Link siblings
                if i > 0:
                    chunk.sibling_chunks.append(chunks[i-1].id)
                if i < len(chunks) - 1:
                    chunk.sibling_chunks.append(chunks[i+1].id)
    
    def _link_table_rows(self, chunks: List[SemanticChunk]):
        """Link table rows for comparison (FIX #3 - Limited to 10 max)"""
        table_chunks = [c for c in chunks if c.content_type == "table_row"]
        MAX_TABLE_RELATIONS = 10
        relations_added = 0
        
        for i, chunk1 in enumerate(table_chunks):
            if relations_added >= MAX_TABLE_RELATIONS:
                break
            # Only link to next 2 rows (locality)
            for chunk2 in table_chunks[i+1:i+3]:
                self.relations.append(ChunkRelation(
                    source_id=chunk1.id,
                    target_id=chunk2.id,
                    relation_type="compares_to",
                    strength=0.8
                ))
                relations_added += 1
    
    def save(self, path: str):
        """Save graph to JSON"""
        data = [asdict(r) for r in self.relations]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def build_semantic_relations(self, chunks: List[SemanticChunk],
                                  embedding_service,
                                  similarity_threshold: float = 0.75,
                                  max_per_item: int = 5) -> List[ChunkRelation]:
        """
        Build semantic relations based on E5 embeddings.
        
        Args:
            chunks: List of chunks with embeddings
            embedding_service: E5EmbeddingService instance
            similarity_threshold: Minimum similarity for relation
            max_per_item: Maximum relations per chunk
            
        Returns:
            List of new semantic relations
        """
        import numpy as np
        
        # Get embeddings from chunks
        embeddings = []
        valid_chunks = []
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if len(embeddings) < 2:
            return []
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Find related pairs
        related_pairs = embedding_service.find_related_pairs(
            embeddings_array,
            threshold=similarity_threshold,
            max_per_item=max_per_item
        )
        
        # Create relations
        semantic_relations = []
        for idx1, idx2, similarity in related_pairs:
            chunk1 = valid_chunks[idx1]
            chunk2 = valid_chunks[idx2]
            
            relation = ChunkRelation(
                source_id=chunk1.id,
                target_id=chunk2.id,
                relation_type="semantic_similarity",
                strength=similarity
            )
            self.relations.append(relation)
            semantic_relations.append(relation)
        
        return semantic_relations


# =============================================================================
# SEMANTIC DEDUPLICATOR (E5-based)
# =============================================================================

class SemanticDeduplicator:
    """
    Deduplicate chunks using E5-multilingual embeddings.
    
    Detects semantically similar chunks across different source files
    and keeps the one with higher quality score.
    """
    
    def __init__(self, embedding_service, similarity_threshold: float = 0.92):
        """
        Initialize deduplicator.
        
        Args:
            embedding_service: E5EmbeddingService instance
            similarity_threshold: Similarity above which chunks are considered duplicates
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, chunks: List[SemanticChunk]) -> Tuple[List[SemanticChunk], List[Dict]]:
        """
        Remove semantically duplicate chunks.
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            Tuple of (unique_chunks, duplicate_info)
        """
        import numpy as np
        
        if len(chunks) < 2:
            return chunks, []
        
        # Collect embeddings
        embeddings = []
        valid_indices = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_indices.append(i)
        
        if len(embeddings) < 2:
            return chunks, []
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Find duplicates
        duplicate_pairs = self.embedding_service.find_duplicates(
            embeddings_array,
            threshold=self.similarity_threshold
        )
        
        # Track which chunks to remove
        to_remove = set()
        duplicate_info = []
        
        for emb_idx1, emb_idx2, similarity in duplicate_pairs:
            chunk_idx1 = valid_indices[emb_idx1]
            chunk_idx2 = valid_indices[emb_idx2]
            
            chunk1 = chunks[chunk_idx1]
            chunk2 = chunks[chunk_idx2]
            
            # SKIP dedup for CSV/tabular data - they have similar structure but different content
            if chunk1.content_type == "table_row" or chunk2.content_type == "table_row":
                continue
            if chunk1.id.startswith("csv_") or chunk2.id.startswith("csv_"):
                continue
            
            # Keep the one with higher quality score
            if chunk1.quality_score >= chunk2.quality_score:
                kept = chunk1
                removed = chunk2
                to_remove.add(chunk_idx2)
            else:
                kept = chunk2
                removed = chunk1
                to_remove.add(chunk_idx1)
            
            duplicate_info.append({
                "kept_id": kept.id,
                "removed_id": removed.id,
                "kept_source": kept.source_file,
                "removed_source": removed.source_file,
                "similarity": similarity,
                "kept_quality": kept.quality_score,
                "removed_quality": removed.quality_score,
            })
        
        # Create list of unique chunks
        unique_chunks = [c for i, c in enumerate(chunks) if i not in to_remove]
        
        return unique_chunks, duplicate_info
    
    def compute_embeddings(self, chunks: List[SemanticChunk], 
                          batch_size: int = 32) -> List[SemanticChunk]:
        """
        Compute and store embeddings for all chunks.
        
        Args:
            chunks: List of chunks
            batch_size: Batch size for encoding
            
        Returns:
            Same chunks with embeddings populated
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.encode_passages(texts, batch_size=batch_size)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        
        return chunks


# =============================================================================
# QUALITY GATES (Step G)
# =============================================================================


class QualityGates:
    """Validate chunks before proceeding"""
    
    MIN_TOKENS = 20  # Lowered for table rows and short content
    MAX_TOKENS = 600  # Increased for longer procedures
    MIN_QUALITY = 0.3  # More lenient for diverse content
    
    def validate_chunks(self, chunks: List[SemanticChunk]) -> Tuple[List[SemanticChunk], List[Dict]]:
        """Run all quality checks"""
        valid = []
        issues = []
        
        seen_content = {}
        
        for chunk in chunks:
            chunk_issues = []
            
            # Check 1: Too short (but lenient for table_row and faq types)
            min_tokens = 10 if chunk.content_type in ["table_row", "faq"] else self.MIN_TOKENS
            if chunk.token_count < min_tokens:
                chunk_issues.append({
                    "check": "too_short",
                    "value": chunk.token_count,
                    "action": "merge_with_adjacent"
                })
            
            # Check 2: Too long
            if chunk.token_count > self.MAX_TOKENS:
                chunk_issues.append({
                    "check": "too_long",
                    "value": chunk.token_count,
                    "action": "resplit"
                })
            
            # Check 3: No clear topic
            if chunk.primary_topic == "umum" and not chunk.entities_mentioned:
                chunk_issues.append({
                    "check": "no_clear_topic",
                    "value": chunk.primary_topic,
                    "action": "manual_review"
                })
            
            # Check 4: Orphan chunk
            if not chunk.sibling_chunks and chunk.content_type == "narrative":
                chunk_issues.append({
                    "check": "orphan",
                    "value": "no_siblings",
                    "action": "check_standalone"
                })
            
            # Check 5: Duplicate
            content_hash = hashlib.md5(chunk.content[:200].encode()).hexdigest()
            if content_hash in seen_content:
                chunk_issues.append({
                    "check": "duplicate",
                    "value": seen_content[content_hash],
                    "action": "deduplicate"
                })
            else:
                seen_content[content_hash] = chunk.id
            
            # Decide fate
            if not chunk_issues or all(i["check"] in ["orphan"] for i in chunk_issues):
                valid.append(chunk)
            else:
                issues.append({
                    "chunk_id": chunk.id,
                    "issues": chunk_issues
                })
        
        return valid, issues


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_phase1(
    config_path: str = "config/config.yaml",
    base_dir: str = ".",
    output_dir: str = "data/chunks",
    offline_mode: bool = True,  # For compatibility with main_pipeline.py
    use_embeddings: bool = True  # Use E5-multilingual embeddings
) -> Dict[str, Any]:
    """
    Run Phase 1: Semantic Chunking from multiple markdown documents.
    
    Reads documents from new_dokument_rag/ folder and creates chunks.
    
    Args:
        config_path: Path to config.yaml
        base_dir: Base directory for the project
        output_dir: Output directory for chunks
        offline_mode: Ignored, kept for compatibility
        
    Returns:
        Results dict with all outputs
    """
    import yaml
    
    start_time = datetime.now()
    
    print("=" * 60)
    print("PHASE 1: Semantic Chunking (Multiple Documents)")
    print("=" * 60)
    
    # Load config
    config_full_path = os.path.join(base_dir, config_path)
    with open(config_full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    prep_config = config.get('preparation', {})
    docs_dir = prep_config.get('docs_dir', './new_dokument_rag')
    markdown_files = prep_config.get('markdown_files', [])
    csv_files = prep_config.get('csv_files', [])
    
    # Setup paths
    full_docs_dir = os.path.join(base_dir, docs_dir)
    full_output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    os.makedirs(os.path.join(full_output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(full_output_dir, "indices"), exist_ok=True)
    
    # Initialize components
    extractor = EntityExtractor()
    chunker = SemanticChunker(extractor)
    profiler = DocumentProfiler()
    
    all_chunks = []
    all_profiles = []
    
    # Process each markdown file
    print(f"\n1. Processing {len(markdown_files)} markdown documents from {docs_dir}:")
    
    for i, md_file in enumerate(markdown_files, 1):
        file_path = os.path.join(full_docs_dir, md_file)
        
        if not os.path.exists(file_path):
            print(f"   ⚠️ [{i}] {md_file} - NOT FOUND, skipping")
            continue
            
        print(f"   📄 [{i}] {md_file}")
        
        # Load document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"       Loaded {len(content)} characters")
        
        # Profile document
        profile = profiler.profile(content, md_file)
        all_profiles.append(profile)
        print(f"       Type: {profile.doc_type}, Tables: {profile.table_count}")
        
        # Create chunks
        chunks = chunker.chunk_document(content, md_file, profile)
        all_chunks.extend(chunks)
        print(f"       Created {len(chunks)} chunks")
    
    print(f"\n   Total chunks from markdown: {len(all_chunks)}")
    
    # Process CSV files
    print(f"\n2. Processing {len(csv_files)} CSV files:")
    
    for csv_file in csv_files:
        csv_path = os.path.join(full_docs_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"   ⚠️ {csv_file} - NOT FOUND, skipping")
            continue
            
        print(f"   📊 {csv_file}")
        
        # Read CSV and create table chunks
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            # Create a markdown table from CSV for chunking
            lines = csv_content.strip().split('\n')
            if len(lines) > 1:
                # Parse CSV
                headers = [h.strip() for h in lines[0].split(',')]
                
                for row_idx, line in enumerate(lines[1:], 1):
                    cells = [c.strip() for c in line.split(',')]
                    if len(cells) >= len(headers):
                        # Format as readable content
                        row_content = f"Data dari {csv_file}:\n"
                        for h, c in zip(headers, cells):
                            row_content += f"- {h}: {c}\n"
                        
                        # Create chunk
                        chunk = chunker._create_chunk(
                            content=row_content,
                            content_type="table_row",
                            source_file=csv_file,
                            section_path=["CSV", csv_file],
                            primary_topic=chunker._detect_topic(row_content),
                            hypothetical_questions=[
                                f"Informasi tentang {cells[0]}?" if cells else f"Data dari {csv_file}?"
                            ]
                        )
                        all_chunks.append(chunk)
                
                print(f"       Created {len(lines) - 1} chunks from rows")
        except Exception as e:
            print(f"       Error processing CSV: {e}")
    
    print(f"\n   Total chunks including CSV: {len(all_chunks)}")
    
    # Build chunk graph (basic relations first)
    print("\n3. Building chunk graph...")
    graph = ChunkGraph()
    relations = graph.build_graph(all_chunks)
    print(f"   Created {len(relations)} basic relations")
    
    # E5 Embedding processing (optional but recommended)
    semantic_dedup_count = 0
    semantic_relations_count = 0
    
    if use_embeddings:
        try:
            import sys
            import os as os_module
            # Add src directory to path for import
            src_dir = os_module.path.dirname(os_module.path.abspath(__file__))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from e5_embedding import E5EmbeddingService
            
            print("\n3b. Computing E5-multilingual embeddings...")
            embedding_service = E5EmbeddingService()
            
            # Compute embeddings for all chunks
            texts = [chunk.content for chunk in all_chunks]
            embeddings = embedding_service.encode_passages(texts, batch_size=32)
            
            for chunk, emb in zip(all_chunks, embeddings):
                chunk.embedding = emb.tolist()
            print(f"    Computed {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
            
            # Semantic deduplication
            print("\n3c. Semantic deduplication (threshold=0.92)...")
            deduplicator = SemanticDeduplicator(embedding_service, similarity_threshold=0.92)
            unique_chunks, duplicates = deduplicator.deduplicate(all_chunks)
            semantic_dedup_count = len(duplicates)
            print(f"    Found {len(duplicates)} semantic duplicates")
            print(f"    Remaining chunks: {len(unique_chunks)}")
            
            # Save dedup report
            if duplicates:
                dedup_path = os.path.join(full_output_dir, "dedup_report.json")
                with open(dedup_path, 'w', encoding='utf-8') as f:
                    json.dump(duplicates, f, ensure_ascii=False, indent=2)
                print(f"    Saved dedup report to {dedup_path}")
            
            # Update all_chunks with deduplicated list
            all_chunks = unique_chunks
            
            # Build semantic relations
            print("\n3d. Building semantic relations (threshold=0.75)...")
            semantic_relations = graph.build_semantic_relations(
                all_chunks, 
                embedding_service,
                similarity_threshold=0.75,
                max_per_item=5
            )
            semantic_relations_count = len(semantic_relations)
            print(f"    Created {len(semantic_relations)} semantic relations")
            print(f"    Total relations: {len(graph.relations)}")
            
        except ImportError as e:
            print(f"\n   ⚠️ E5 embeddings disabled: {e}")
            print("      Install with: pip install sentence-transformers")
        except Exception as e:
            print(f"\n   ⚠️ E5 embedding error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n   (E5 embeddings disabled)")
    
    # Quality gates
    print("\n4. Running quality gates...")
    gates = QualityGates()
    valid_chunks, issues = gates.validate_chunks(all_chunks)
    print(f"   Valid: {len(valid_chunks)}")
    print(f"   Issues: {len(issues)}")
    
    # Save outputs
    print("\n5. Saving outputs...")
    
    # Chunks
    chunks_path = os.path.join(full_output_dir, "chunks.jsonl")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in valid_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')
    print(f"   - {chunks_path}")
    
    # Entities
    entities_path = os.path.join(full_output_dir, "entities.json")
    extractor.save(entities_path)
    print(f"   - {entities_path}")
    
    # Graph
    graph_path = os.path.join(full_output_dir, "chunk_graph.json")
    graph.save(graph_path)
    print(f"   - {graph_path}")
    
    # Indices
    by_topic = {}
    by_entity = {}
    hypo_qa = {}
    
    for chunk in valid_chunks:
        # By topic
        if chunk.primary_topic not in by_topic:
            by_topic[chunk.primary_topic] = []
        by_topic[chunk.primary_topic].append(chunk.id)
        
        # By entity
        for entity in chunk.entities_mentioned:
            if entity not in by_entity:
                by_entity[entity] = []
            by_entity[entity].append(chunk.id)
        
        # Hypothetical Q-A
        for q in chunk.hypothetical_questions:
            hypo_qa[q] = chunk.id
    
    with open(os.path.join(full_output_dir, "indices", "by_topic.json"), 'w', encoding='utf-8') as f:
        json.dump(by_topic, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(full_output_dir, "indices", "by_entity.json"), 'w', encoding='utf-8') as f:
        json.dump(by_entity, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(full_output_dir, "indices", "hypothetical_qa.json"), 'w', encoding='utf-8') as f:
        json.dump(hypo_qa, f, ensure_ascii=False, indent=2)
    
    print(f"   - indices/by_topic.json ({len(by_topic)} topics)")
    print(f"   - indices/by_entity.json ({len(by_entity)} entities)")
    print(f"   - indices/hypothetical_qa.json ({len(hypo_qa)} questions)")
    
    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "docs_processed": markdown_files,
        "csv_processed": csv_files,
        "total_chunks": len(valid_chunks),
        "total_entities": len(extractor.entities),
        "total_relations": len(relations),
        "quality_issues": len(issues)
    }
    
    with open(os.path.join(full_output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    results = {
        "phase": 1,
        "name": "Semantic Chunking (Multiple Documents)",
        "status": "completed",
        "duration_seconds": duration,
        "chunks_created": len(valid_chunks),
        "total_chunks": len(valid_chunks),  # For compatibility
        "entities_extracted": len(extractor.entities),
        "relations_created": len(graph.relations),
        "basic_relations": len(relations),
        "semantic_relations": semantic_relations_count,
        "semantic_duplicates_removed": semantic_dedup_count,
        "quality_issues": len(issues),
        "documents_processed": len(markdown_files),
        "csv_processed": len(csv_files),
        "embeddings_enabled": use_embeddings,
    }
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"   Documents: {len(markdown_files)} markdown + {len(csv_files)} CSV")
    print(f"   Chunks: {len(valid_chunks)}")
    print(f"   Entities: {len(extractor.entities)}")
    print(f"   Relations: {len(graph.relations)} ({len(relations)} basic + {semantic_relations_count} semantic)")
    if use_embeddings:
        print(f"   Semantic duplicates removed: {semantic_dedup_count}")
    print(f"   Duration: {duration:.1f}s")
    
    return results


# Alias for backward compatibility
run_phase1_redesign = run_phase1


if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    results = run_phase1(base_dir=base_dir)

