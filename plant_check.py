# Fix PyTorch/Streamlit compatibility issues - Enhanced
import os
import sys

# Set environment variables BEFORE any imports
os.environ['TORCH_EXTENSION_VERBOSITY'] = '0'
os.environ['TORCH_LOGS'] = '0'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Suppress all warnings before importing anything
import warnings
warnings.filterwarnings('ignore')

# Basic imports first
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
import numpy as np
import re
import json
import io
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import hashlib
import sqlite3

# Page configuration - Move to very beginning
st.set_page_config(
    page_title="Plant Check - Smart Analyzer",
    page_icon="🏭",
    layout="wide"
)

# Enhanced logging setup to suppress torch warnings
class StreamlitLogFilter(logging.Filter):
    def filter(self, record):
        # Filter out torch-related log messages
        msg = record.getMessage().lower()
        return not any(word in msg for word in ['torch', '_classes', 'examining the path'])

# Setup logging with custom filter
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[logging.StreamHandler()]
)

# Add filter to suppress torch warnings
for handler in logging.root.handlers:
    handler.addFilter(StreamlitLogFilter())

logger = logging.getLogger(__name__)

# PDF processing
try:
    import pdfplumber
    PDF_AVAILABLE = True
    logger.info("✅ PDF processing available")
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("❌ PDF processing not available. Install pdfplumber: pip install pdfplumber")

# Machine learning - Lazy loading to avoid Streamlit conflicts
TORCH_AVAILABLE = False
device = 'cpu'

# Global variables for ML components - will be loaded only when needed
_torch_modules = {}

def get_torch_module(module_name):
    """Lazy load torch modules only when actually needed"""
    global _torch_modules, TORCH_AVAILABLE, device
    
    if module_name in _torch_modules:
        return _torch_modules[module_name]
    
    if not TORCH_AVAILABLE and not _torch_modules:
        try:
            import contextlib
            import io as io_module
            
            with contextlib.redirect_stdout(io_module.StringIO()), \
                 contextlib.redirect_stderr(io_module.StringIO()), \
                 warnings.catch_warnings():
                
                warnings.simplefilter("ignore")
                
                try:
                    import torch as _torch
                    import torch.nn as _nn
                    import torch.nn.functional as _F
                    from sentence_transformers import SentenceTransformer as _SentenceTransformer
                    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
                    from sklearn.cluster import KMeans as _KMeans
                    
                    _torch_modules.update({
                        'torch': _torch,
                        'nn': _nn,
                        'F': _F,
                        'SentenceTransformer': _SentenceTransformer,
                        'cosine_similarity': _cosine_similarity,
                        'KMeans': _KMeans
                    })
                    
                    TORCH_AVAILABLE = True
                    device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
                    logger.info("✅ Machine learning modules loaded")
                    
                except Exception as e:
                    TORCH_AVAILABLE = False
                    device = 'cpu'
                    logger.info(f"⚠️ Machine learning features disabled: {e}")
                    
        except Exception as e:
            TORCH_AVAILABLE = False
            device = 'cpu'
            logger.info("⚠️ Machine learning features disabled (torch not available)")
            return None
    
    return _torch_modules.get(module_name)

# Ontology support
try:
    import rdflib
    from rdflib import Graph, Namespace, RDF, RDFS, OWL
    RDF_AVAILABLE = True
    logger.info("✅ RDF ontology features available")
except ImportError:
    RDF_AVAILABLE = False
    logger.info("⚠️ RDF ontology features disabled (rdflib not available)")

class SmartConfig:
    """Dynamic configuration with learning capabilities"""
    
    # File paths
    DB_PATH = "plant_knowledge.db"
    ONTOLOGY_PATH = "ontology.ttl"
    FIELD_MAPPING_CONFIG = "field_mappings.json"
    USER_FEEDBACK_FILE = "user_feedback.json"
    EXTRACTION_RULES_FILE = "extraction_rules.json"
    
    # Quality thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MIN_FIELD_LENGTH = 2
    MAX_FIELD_LENGTH = 150
    MAX_VALUE_LENGTH = 1000
    
    # Table processing thresholds
    MIN_TABLE_ROWS = 2
    MAX_TABLE_COLS = 20
    MIN_TABLE_CELL_LENGTH = 1
    MAX_TABLE_CELL_LENGTH = 200
    
    @staticmethod
    def get_exclude_patterns():
        """Get exclusion patterns from config file"""
        default_patterns = [
            r'page\s*[:]\s*\d+',
            r'handling\s+guide',
            r'rev\s*[:]\s*\d+',
            r'date\s*[:]\s*\d{2,4}',
            r'^\d+\s*of\s*\d+$',
            r'figure\s*\d+',
            r'table\s*\d+',
            r'section\s*\d+',
            r'appendix\s*[a-z]',
            r'note\s*[:]\s*',
            r'reference\s*[:]\s*',
            r'copyright\s*',
            r'proprietary\s*',
            r'confidential\s*'
        ]
        
        try:
            if os.path.exists(SmartConfig.EXTRACTION_RULES_FILE):
                with open(SmartConfig.EXTRACTION_RULES_FILE, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                    return rules.get('exclude_patterns', default_patterns)
        except Exception:
            pass
        
        return default_patterns
    
    @staticmethod
    def get_field_consolidation():
        """Get field consolidation rules"""
        default_consolidation = {
            'capacity_group': {
                'keywords': ['capacity', 'rated capacity', 'normal capacity', 'flow rate', 'flow', 'throughput', 'volume'],
                'standard_name': 'Capacity',
                'units': ['m3/hr', 'l/s', 'gpm', 'm³/h', 'lpm', 'cfm']
            },
            'power_group': {
                'keywords': ['power', 'motor power', 'driver power', 'rated power', 'electrical power', 'consumption'],
                'standard_name': 'Power',
                'units': ['kw', 'hp', 'w', 'mw', 'bhp']
            },
            'temperature_group': {
                'keywords': ['temperature', 'temp', 'operating temp', 'design temp', 'service temp', 'maximum temp'],
                'standard_name': 'Temperature',
                'units': ['°c', '°f', 'celsius', 'fahrenheit', 'k', 'kelvin']
            },
            'pressure_group': {
                'keywords': ['pressure', 'suction pressure', 'discharge pressure', 'operating pressure', 'design pressure', 'working pressure'],
                'standard_name': 'Pressure',
                'units': ['bar', 'barg', 'psi', 'kpa', 'mpa', 'psig', 'kg/cm²']
            },
            'material_group': {
                'keywords': ['material', 'construction material', 'body material', 'casing material', 'wetted parts'],
                'standard_name': 'Material',
                'units': []
            },
            'size_group': {
                'keywords': ['size', 'diameter', 'length', 'width', 'height', 'dimension', 'bore'],
                'standard_name': 'Size',
                'units': ['mm', 'cm', 'm', 'inch', 'ft', '"']
            }
        }
        
        try:
            if os.path.exists(SmartConfig.EXTRACTION_RULES_FILE):
                with open(SmartConfig.EXTRACTION_RULES_FILE, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                    return rules.get('field_consolidation', default_consolidation)
        except Exception:
            pass
        
        return default_consolidation

class SmartDatabaseManager:
    """Enhanced database manager with learning capabilities"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or SmartConfig.DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize enhanced database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Documents table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        quality_score REAL DEFAULT 0.0,
                        extraction_method TEXT DEFAULT 'auto',
                        page_count INTEGER DEFAULT 0,
                        table_count INTEGER DEFAULT 0,
                        field_count INTEGER DEFAULT 0
                    )
                """)
                
                # Fields table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fields (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id INTEGER NOT NULL,
                        original_field_name TEXT NOT NULL DEFAULT '',
                        normalized_field_name TEXT NOT NULL DEFAULT '',
                        standard_field_name TEXT NOT NULL DEFAULT '',
                        field_value TEXT NOT NULL DEFAULT '',
                        field_type TEXT DEFAULT 'text',
                        confidence REAL DEFAULT 0.0,
                        extraction_method TEXT DEFAULT 'pattern',
                        validation_status TEXT DEFAULT 'unknown',
                        context_info TEXT DEFAULT '',
                        source_type TEXT DEFAULT 'text',
                        page_number INTEGER DEFAULT 0,
                        table_index INTEGER DEFAULT -1,
                        user_validated BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (doc_id) REFERENCES documents (id)
                    )
                """)
                
                # User feedback table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_field TEXT NOT NULL,
                        suggested_standard TEXT NOT NULL,
                        user_approved BOOLEAN NOT NULL,
                        confidence_score REAL,
                        feedback_date DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Mapping cache
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS mapping_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        field_pattern TEXT NOT NULL UNIQUE,
                        standard_mapping TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        usage_count INTEGER DEFAULT 1,
                        last_used DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Table extractions
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS table_extractions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id INTEGER NOT NULL,
                        page_number INTEGER DEFAULT 0,
                        table_index INTEGER DEFAULT 0,
                        headers TEXT DEFAULT '',
                        row_count INTEGER DEFAULT 0,
                        col_count INTEGER DEFAULT 0,
                        extraction_confidence REAL DEFAULT 0.0,
                        field_pairs_extracted INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (doc_id) REFERENCES documents (id)
                    )
                """)
                
                self._update_existing_tables(conn)
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def _update_existing_tables(self, conn):
        """기존 테이블 업데이트"""
        try:
            # documents 테이블 컬럼 확인
            cursor = conn.execute("PRAGMA table_info(documents)")
            doc_columns = [row[1] for row in cursor.fetchall()]
            
            new_doc_columns = {
                'quality_score': 'REAL DEFAULT 0.0',
                'page_count': 'INTEGER DEFAULT 0',
                'table_count': 'INTEGER DEFAULT 0',
                'field_count': 'INTEGER DEFAULT 0'
            }
            
            for column_name, column_def in new_doc_columns.items():
                if column_name not in doc_columns:
                    try:
                        conn.execute(f"ALTER TABLE documents ADD COLUMN {column_name} {column_def}")
                    except sqlite3.OperationalError:
                        pass
            
            # fields 테이블 컬럼 확인
            cursor = conn.execute("PRAGMA table_info(fields)")
            field_columns = [row[1] for row in cursor.fetchall()]
            
            new_field_columns = {
                'original_field_name': 'TEXT DEFAULT ""',
                'normalized_field_name': 'TEXT DEFAULT ""',
                'field_type': 'TEXT DEFAULT "text"',
                'confidence': 'REAL DEFAULT 0.0',
                'extraction_method': 'TEXT DEFAULT "pattern"',
                'validation_status': 'TEXT DEFAULT "unknown"',
                'context_info': 'TEXT DEFAULT ""',
                'source_type': 'TEXT DEFAULT "text"',
                'page_number': 'INTEGER DEFAULT 0',
                'table_index': 'INTEGER DEFAULT -1',
                'user_validated': 'BOOLEAN DEFAULT FALSE',
                'created_at': 'DATETIME DEFAULT CURRENT_TIMESTAMP'
            }
            
            for column_name, column_def in new_field_columns.items():
                if column_name not in field_columns:
                    try:
                        conn.execute(f"ALTER TABLE fields ADD COLUMN {column_name} {column_def}")
                    except sqlite3.OperationalError:
                        pass
            
        except Exception as e:
            logger.error(f"❌ Table update failed: {e}")
    
    def save_document(self, filename: str, content_hash: str, quality_score: float = 0.0, 
                     page_count: int = 0, table_count: int = 0, field_count: int = 0) -> int:
        """Save document with enhanced metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO documents 
                       (filename, content_hash, quality_score, page_count, table_count, field_count) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (filename, content_hash, quality_score, page_count, table_count, field_count)
                )
                doc_id = cursor.lastrowid
                return doc_id
        except Exception as e:
            logger.error(f"❌ Save document failed: {e}")
            return 1
    
    def save_field(self, doc_id: int, field_data: Dict):
        """Save field with enhanced metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                safe_field_data = {
                    'doc_id': doc_id,
                    'original_field_name': str(field_data.get('original_field_name', '')),
                    'normalized_field_name': str(field_data.get('normalized_field_name', '')),
                    'standard_field_name': str(field_data.get('standard_field_name', '')),
                    'field_value': str(field_data.get('field_value', '')),
                    'field_type': str(field_data.get('field_type', 'text')),
                    'confidence': float(field_data.get('confidence', 0.0)),
                    'extraction_method': str(field_data.get('extraction_method', 'pattern')),
                    'validation_status': str(field_data.get('validation_status', 'unknown')),
                    'context_info': str(field_data.get('context_info', '')),
                    'source_type': str(field_data.get('source_type', 'text')),
                    'page_number': int(field_data.get('page_number', 0)),
                    'table_index': int(field_data.get('table_index', -1))
                }
                
                conn.execute("""
                    INSERT INTO fields 
                    (doc_id, original_field_name, normalized_field_name, standard_field_name, 
                     field_value, field_type, confidence, extraction_method, validation_status, 
                     context_info, source_type, page_number, table_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    safe_field_data['doc_id'],
                    safe_field_data['original_field_name'],
                    safe_field_data['normalized_field_name'],
                    safe_field_data['standard_field_name'],
                    safe_field_data['field_value'],
                    safe_field_data['field_type'],
                    safe_field_data['confidence'],
                    safe_field_data['extraction_method'],
                    safe_field_data['validation_status'],
                    safe_field_data['context_info'],
                    safe_field_data['source_type'],
                    safe_field_data['page_number'],
                    safe_field_data['table_index']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Save field failed: {e}")
    
    def save_table_extraction(self, doc_id: int, page_number: int, table_index: int, 
                            headers: List[str], row_count: int, col_count: int, 
                            confidence: float, field_pairs: int):
        """Save table extraction results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                headers_str = json.dumps(headers) if headers else ''
                conn.execute("""
                    INSERT INTO table_extractions 
                    (doc_id, page_number, table_index, headers, row_count, col_count, 
                     extraction_confidence, field_pairs_extracted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, page_number, table_index, headers_str, row_count, col_count, confidence, field_pairs))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Save table extraction failed: {e}")
    
    def update_mapping_cache(self, field_pattern: str, standard_mapping: str, confidence: float):
        """Update mapping cache for learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO mapping_cache 
                    (field_pattern, standard_mapping, confidence, usage_count, last_used)
                    VALUES (?, ?, ?, 
                           COALESCE((SELECT usage_count + 1 FROM mapping_cache WHERE field_pattern = ?), 1),
                           CURRENT_TIMESTAMP)
                """, (field_pattern, standard_mapping, confidence, field_pattern))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Update mapping cache failed: {e}")
    
    def get_learned_mappings(self) -> Dict[str, str]:
        """Get learned mappings from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT field_pattern, standard_mapping, confidence 
                    FROM mapping_cache 
                    WHERE confidence >= ? 
                    ORDER BY usage_count DESC, confidence DESC
                """, (SmartConfig.MIN_CONFIDENCE_THRESHOLD,))
                
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"❌ Get learned mappings failed: {e}")
            return {}

class SmartOntologyManager:
    """Intelligent ontology manager with learning capabilities"""
    
    def __init__(self, ontology_path: str = None):
        self.ontology_path = ontology_path or SmartConfig.ONTOLOGY_PATH
        self.field_mappings = {}
        self.learned_mappings = {}
        self.consolidation_rules = SmartConfig.get_field_consolidation()
        self.exclude_patterns = SmartConfig.get_exclude_patterns()
        
        if RDF_AVAILABLE:
            self.graph = Graph()
            self.ex_namespace = Namespace("http://example.org/plant#")
            self.graph.bind("ex", self.ex_namespace)
            self.graph.bind("rdfs", RDFS)
            self.graph.bind("owl", OWL)
        
        self.load_ontology()
        self.load_learned_mappings()
    
    def load_ontology(self):
        """Load ontology with fallback to dynamic generation"""
        try:
            if RDF_AVAILABLE and os.path.exists(self.ontology_path):
                self.graph.parse(self.ontology_path, format="ttl")
                self._extract_rdf_mappings()
                logger.info(f"✅ Loaded RDF ontology from {self.ontology_path}")
            else:
                self._generate_dynamic_ontology()
                logger.info("✅ Generated dynamic ontology")
                
        except Exception as e:
            logger.warning(f"⚠️ Ontology loading failed: {e}")
            self._generate_dynamic_ontology()
    
    def load_learned_mappings(self):
        """Load learned mappings from database and files"""
        db_manager = SmartDatabaseManager()
        self.learned_mappings.update(db_manager.get_learned_mappings())
        
        try:
            if os.path.exists(SmartConfig.USER_FEEDBACK_FILE):
                with open(SmartConfig.USER_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    for item in feedback_data.get('approved_mappings', []):
                        self.learned_mappings[item['original']] = item['standard']
        except Exception as e:
            logger.warning(f"⚠️ Could not load user feedback: {e}")
        
        logger.info(f"✅ Loaded {len(self.learned_mappings)} learned mappings")
    
    def _extract_rdf_mappings(self):
        """Extract mappings from RDF graph"""
        if not RDF_AVAILABLE:
            return
        
        for subj, pred, obj in self.graph.triples((None, RDFS.label, None)):
            label = str(obj).lower()
            concept = str(subj).split('#')[-1]
            self.field_mappings[label] = concept
        
        logger.info(f"✅ Extracted {len(self.field_mappings)} RDF mappings")
    
    def _generate_dynamic_ontology(self):
        """Generate ontology based on consolidation rules"""
        for group_name, group_config in self.consolidation_rules.items():
            standard_name = group_config['standard_name']
            keywords = group_config['keywords']
            
            for keyword in keywords:
                normalized_keyword = self._normalize_field_name(keyword)
                self.field_mappings[normalized_keyword.lower()] = standard_name
                
                variations = self._generate_field_variations(keyword)
                for variation in variations:
                    self.field_mappings[variation.lower()] = standard_name
    
    def _generate_field_variations(self, base_keyword: str) -> List[str]:
        """Generate common variations of a field name"""
        variations = []
        base = base_keyword.lower()
        
        prefixes = ['', 'rated ', 'normal ', 'design ', 'operating ', 'maximum ', 'minimum ']
        suffixes = ['', ' rate', ' value', ' setting', ' condition', ' requirement']
        
        for prefix in prefixes:
            for suffix in suffixes:
                variation = f"{prefix}{base}{suffix}".strip()
                if variation != base:
                    variations.append(variation)
        
        return variations
    
    def find_standard_field(self, field_name: str, context: str = "") -> Dict[str, Any]:
        """Enhanced field matching with context awareness"""
        if not field_name or not field_name.strip():
            return self._empty_field_result(field_name)
        
        if self._should_exclude_field(field_name):
            return self._empty_field_result(field_name, exclude_reason="excluded_pattern")
        
        normalized = self._normalize_field_name(field_name)
        normalized_lower = normalized.lower()
        
        result = {
            'original_field_name': field_name,
            'normalized_field_name': normalized,
            'standard_field_name': normalized,
            'confidence': 0.0,
            'match_type': 'none',
            'validation_status': 'unknown',
            'context_info': context[:200]
        }
        
        # Check learned mappings first
        if normalized_lower in self.learned_mappings:
            standard_name = self.learned_mappings[normalized_lower]
            result.update({
                'standard_field_name': standard_name,
                'confidence': 0.95,
                'match_type': 'learned'
            })
            return result
        
        # Check consolidation groups
        consolidation_match = self._find_consolidation_match(normalized_lower)
        if consolidation_match:
            result.update(consolidation_match)
            return result
        
        # Direct ontology matching
        if normalized_lower in self.field_mappings:
            standard_name = self.field_mappings[normalized_lower]
            result.update({
                'standard_field_name': standard_name,
                'confidence': 0.9,
                'match_type': 'direct_ontology'
            })
            return result
        
        # Fuzzy matching
        fuzzy_match = self._find_fuzzy_match(normalized_lower, context)
        if fuzzy_match:
            result.update(fuzzy_match)
            return result
        
        # Smart fallback
        smart_fallback = self._create_smart_fallback(normalized)
        result.update(smart_fallback)
        
        return result
    
    def _should_exclude_field(self, field_name: str) -> bool:
        """Check if field should be excluded based on patterns"""
        field_lower = field_name.lower()
        
        for pattern in self.exclude_patterns:
            if re.search(pattern, field_lower, re.IGNORECASE):
                return True
        
        if len(field_name) > SmartConfig.MAX_FIELD_LENGTH:
            return True
        
        if len(field_name) < SmartConfig.MIN_FIELD_LENGTH:
            return True
        
        non_field_indicators = ['page', 'figure', 'table', 'section', 'appendix', 'note', 'reference']
        if any(indicator in field_lower for indicator in non_field_indicators):
            return True
        
        return False
    
    def _find_consolidation_match(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Find match in consolidation groups"""
        for group_name, group_config in self.consolidation_rules.items():
            keywords = group_config['keywords']
            standard_name = group_config['standard_name']
            
            for keyword in keywords:
                if keyword.lower() in field_name:
                    confidence = min(len(keyword), len(field_name)) / max(len(keyword), len(field_name))
                    if confidence >= 0.7:
                        return {
                            'standard_field_name': standard_name,
                            'confidence': confidence,
                            'match_type': 'consolidation'
                        }
        
        return None
    
    def _find_fuzzy_match(self, field_name: str, context: str) -> Optional[Dict[str, Any]]:
        """Find fuzzy match with context consideration"""
        best_score = 0.0
        best_match = None
        
        for learned_field, standard_name in self.learned_mappings.items():
            score = self._calculate_field_similarity(field_name, learned_field)
            if score > best_score and score >= 0.7:
                best_score = score
                best_match = {
                    'standard_field_name': standard_name,
                    'confidence': score * 0.9,
                    'match_type': 'fuzzy_learned'
                }
        
        for ontology_field, standard_name in self.field_mappings.items():
            score = self._calculate_field_similarity(field_name, ontology_field)
            if score > best_score and score >= 0.7:
                best_score = score
                best_match = {
                    'standard_field_name': standard_name,
                    'confidence': score * 0.8,
                    'match_type': 'fuzzy_ontology'
                }
        
        return best_match
    
    def _create_smart_fallback(self, field_name: str) -> Dict[str, Any]:
        """Create intelligent fallback mapping"""
        clean_name = self._intelligent_field_cleaning(field_name)
        confidence = 0.3 if clean_name != field_name else 0.1
        
        return {
            'standard_field_name': clean_name,
            'confidence': confidence,
            'match_type': 'smart_fallback'
        }
    
    def _intelligent_field_cleaning(self, field_name: str) -> str:
        """Intelligent field name cleaning"""
        clean = re.sub(r'^\d+\.?\s*', '', field_name)
        
        abbreviations = {
            'temp': 'Temperature',
            'press': 'Pressure',
            'cap': 'Capacity',
            'eff': 'Efficiency',
            'dia': 'Diameter',
            'len': 'Length',
            'wt': 'Weight',
            'max': 'Maximum',
            'min': 'Minimum',
            'nom': 'Nominal'
        }
        
        words = clean.lower().split()
        expanded_words = []
        
        for word in words:
            if word in abbreviations:
                expanded_words.append(abbreviations[word])
            else:
                expanded_words.append(word.capitalize())
        
        result = '_'.join(expanded_words)
        return result if result else 'Unknown_Field'
    
    def validate_field_value(self, field_name: str, field_value: str) -> bool:
        """Validate if field-value pair makes sense"""
        if not field_value or len(field_value) > SmartConfig.MAX_VALUE_LENGTH:
            return False
        
        return True
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Advanced field name normalization"""
        if not field_name:
            return ""
        
        normalized = re.sub(r'^\d+\.?\s*', '', field_name.strip())
        
        prefixes_to_remove = ['item', 'no', 'ref', 'spec']
        words = normalized.lower().split()
        if words and words[0] in prefixes_to_remove:
            normalized = ' '.join(words[1:])
        
        normalized = re.sub(r'[^\w\s°℃°F]', ' ', normalized)
        normalized = re.sub(r'[°℃°F]+', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_field_similarity(self, field1: str, field2: str) -> float:
        """Calculate advanced field similarity"""
        words1 = set(field1.lower().split())
        words2 = set(field2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union)
        
        char_sim = 1.0 - (abs(len(field1) - len(field2)) / max(len(field1), len(field2)))
        
        return (jaccard * 0.7) + (char_sim * 0.3)
    
    def _empty_field_result(self, field_name: str, exclude_reason: str = "") -> Dict[str, Any]:
        """Return empty field result"""
        return {
            'original_field_name': field_name or 'Empty_Field',
            'normalized_field_name': field_name or 'Empty_Field',
            'standard_field_name': 'Excluded_Field' if exclude_reason else 'Unknown_Field',
            'confidence': 0.0,
            'match_type': exclude_reason or 'empty',
            'validation_status': 'excluded' if exclude_reason else 'unknown',
            'context_info': f"Excluded: {exclude_reason}" if exclude_reason else ""
        }

class SmartPDFProcessor:
    """Enhanced PDF processor with intelligent extraction"""
    
    def __init__(self):
        self.exclude_patterns = SmartConfig.get_exclude_patterns()
        self.consolidation_rules = SmartConfig.get_field_consolidation()
        self.checkbox_patterns = [
            r'[■▣☑✓✔]\s*([A-Za-z][A-Za-z\s]+)',
            r'[□☐]\s*([A-Za-z][A-Za-z\s]+)',
        ]
        self.revision_patterns = [
            r'REV\.?\s*:?\s*(\d+[A-Z]*)',
            r'REVISION\s*:?\s*(\d+[A-Z]*)',
            r'Rev\s*\.?\s*(\d+[A-Z]*)',
        ]
        self.table_field_patterns = [
            r'^([A-Za-z][A-Za-z\s]{2,50})\s*[:=]\s*(.+)$',
            r'^(\d+\.?\s*[A-Za-z][A-Za-z\s]{2,50})\s*[:=]?\s*(.+)$',
        ]
    
    def extract_text_and_tables(self, pdf_file) -> Dict[str, Any]:
        """Enhanced PDF extraction with quality assessment"""
        if not PDF_AVAILABLE:
            st.error("PDF processing not available. Please install pdfplumber")
            return self._empty_result()
        
        results = {
            'text': '',
            'tables': [],
            'fields': {},
            'checkboxes': {},
            'revisions': [],
            'quality_score': 0.0,
            'metadata': {},
            'extraction_quality': {}
        }
        
        try:
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                all_tables = []
                all_checkboxes = {}
                all_revisions = []
                page_qualities = []
                
                total_pages = len(pdf.pages)
                logger.info(f"📄 Processing {total_pages} pages")
                
                progress_placeholder = st.empty()
                
                for page_num, page in enumerate(pdf.pages):
                    progress_placeholder.text(f"📄 Processing page {page_num + 1}/{total_pages}...")
                    
                    page_result = self._process_page_enhanced(page, page_num)
                    
                    if page_result['text']:
                        full_text += f"\n--- PAGE {page_num + 1} ---\n" + page_result['text'] + "\n"
                    
                    for table_info in page_result['tables']:
                        table_info['page_number'] = page_num + 1
                        all_tables.append(table_info)
                    
                    all_checkboxes.update(page_result['checkboxes'])
                    all_revisions.extend(page_result['revisions'])
                    page_qualities.append(page_result['quality'])
                
                progress_placeholder.empty()
                
                extracted_fields = self._extract_fields_comprehensively(full_text, all_tables, all_checkboxes)
                
                results.update({
                    'text': full_text,
                    'tables': all_tables,
                    'checkboxes': all_checkboxes,
                    'revisions': all_revisions,
                    'final_revision': self._get_latest_revision(all_revisions),
                    'fields': extracted_fields,
                    'quality_score': sum(page_qualities) / len(page_qualities) if page_qualities else 0.0,
                    'metadata': {
                        'total_pages': total_pages,
                        'total_tables': len(all_tables),
                        'checkboxes_found': len(all_checkboxes),
                        'revisions_found': len(all_revisions),
                        'fields_extracted': len(extracted_fields),
                        'successful_tables': sum(1 for t in all_tables if t.get('confidence', 0) > 0.5)
                    },
                    'extraction_quality': {
                        'page_qualities': page_qualities,
                        'avg_quality': sum(page_qualities) / len(page_qualities) if page_qualities else 0.0,
                        'table_success_rate': sum(1 for t in all_tables if t.get('confidence', 0) > 0.5) / len(all_tables) if all_tables else 0
                    }
                })
                
                logger.info(f"✅ PDF extraction completed: {len(extracted_fields)} fields extracted from {total_pages} pages")
                
        except Exception as e:
            logger.error(f"❌ PDF processing error: {e}")
            st.error(f"PDF processing error: {str(e)}")
            return self._empty_result()
            
        return results
    
    def _process_page_enhanced(self, page, page_num: int) -> Dict[str, Any]:
        """Enhanced page processing"""
        page_result = {
            'text': '',
            'tables': [],
            'checkboxes': {},
            'revisions': [],
            'quality': 0.0
        }
        
        try:
            page_text = page.extract_text()
            if page_text:
                filtered_text = self._filter_page_noise_enhanced(page_text)
                page_result['text'] = filtered_text
                page_result['checkboxes'] = self._extract_checkboxes_enhanced(filtered_text)
                page_result['revisions'] = self._extract_revisions(filtered_text)
                page_result['quality'] = self._assess_page_quality_enhanced(filtered_text)
            
            try:
                tables = page.extract_tables()
                logger.info(f"📊 Page {page_num + 1}: Found {len(tables)} raw tables")
                
                for table_idx, table in enumerate(tables):
                    if table and len(table) >= SmartConfig.MIN_TABLE_ROWS:
                        processed_table = self._process_table_enhanced(table, page_num, table_idx)
                        if processed_table and processed_table.get('confidence', 0) > 0.2:
                            page_result['tables'].append(processed_table)
                            logger.info(f"✅ Table {table_idx} processed successfully (confidence: {processed_table['confidence']:.2f})")
                        else:
                            logger.warning(f"⚠️ Table {table_idx} skipped (low confidence)")
                            
            except Exception as e:
                logger.error(f"❌ Table extraction error on page {page_num + 1}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Page processing error for page {page_num + 1}: {e}")
        
        return page_result
    
    def _filter_page_noise_enhanced(self, text: str) -> str:
        """Enhanced page noise filtering"""
        if not text:
            return ""
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            
            if len(line) < 2:
                continue
            
            should_exclude = False
            for pattern in self.exclude_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    should_exclude = True
                    break
            
            noise_patterns = [
                r'^\d+$',
                r'^[^\w]*$',
                r'^\s*$'
            ]
            
            for pattern in noise_patterns:
                if re.match(pattern, line):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _process_table_enhanced(self, table: List[List[str]], page_num: int, table_idx: int) -> Dict[str, Any]:
        """Enhanced table processing"""
        if not table or len(table) < SmartConfig.MIN_TABLE_ROWS:
            return None
        
        # Clean table data
        cleaned_table = []
        for row in table:
            if row:
                cleaned_row = []
                for cell in row:
                    if cell is not None:
                        cell_str = str(cell).strip()
                        if len(cell_str) > SmartConfig.MAX_TABLE_CELL_LENGTH:
                            cell_str = cell_str[:SmartConfig.MAX_TABLE_CELL_LENGTH] + "..."
                        cleaned_row.append(cell_str)
                    else:
                        cleaned_row.append("")
                
                if any(cell for cell in cleaned_row):
                    cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < SmartConfig.MIN_TABLE_ROWS:
            return None
        
        # Identify headers
        headers = self._identify_table_headers(cleaned_table)
        data_rows = cleaned_table[1:] if headers else cleaned_table
        
        # Extract field-value pairs
        field_value_pairs = {}
        checkbox_data = {}
        confidence_scores = []
        
        # Method 1: Two-column table extraction
        if len(headers) == 2 and len(data_rows) > 0:
            field_pairs_2col = self._extract_from_two_column_table(data_rows, page_num, table_idx)
            field_value_pairs.update(field_pairs_2col['fields'])
            checkbox_data.update(field_pairs_2col['checkboxes'])
            confidence_scores.extend(field_pairs_2col['confidences'])
        
        # Method 2: Multi-column table extraction
        elif len(headers) > 2:
            multi_col_pairs = self._extract_from_multi_column_table(headers, data_rows, page_num, table_idx)
            field_value_pairs.update(multi_col_pairs['fields'])
            checkbox_data.update(multi_col_pairs['checkboxes'])
            confidence_scores.extend(multi_col_pairs['confidences'])
        
        # Method 3: Row pattern extraction
        row_pattern_pairs = self._extract_from_row_patterns(data_rows, page_num, table_idx)
        field_value_pairs.update(row_pattern_pairs['fields'])
        checkbox_data.update(row_pattern_pairs['checkboxes'])
        confidence_scores.extend(row_pattern_pairs['confidences'])
        
        # Calculate confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        structure_quality = self._assess_table_structure_quality(cleaned_table, headers)
        final_confidence = (avg_confidence + structure_quality) / 2
        
        result = {
            'headers': headers,
            'data': data_rows,
            'field_value_pairs': field_value_pairs,
            'checkbox_data': checkbox_data,
            'confidence': final_confidence,
            'page_number': page_num + 1,
            'table_index': table_idx,
            'structure_quality': structure_quality,
            'extraction_stats': {
                'total_rows': len(data_rows),
                'total_cols': len(headers) if headers else len(cleaned_table[0]) if cleaned_table else 0,
                'field_pairs_found': len(field_value_pairs),
                'checkbox_pairs_found': len(checkbox_data),
                'confidence_scores': confidence_scores
            }
        }
        
        return result
    
    def _identify_table_headers(self, table: List[List[str]]) -> List[str]:
        """Identify table headers"""
        if not table or len(table) < 1:
            return []
        
        first_row = table[0]
        headers = []
        
        for cell in first_row:
            if cell and isinstance(cell, str):
                cell = cell.strip()
                if len(cell) > 0:
                    headers.append(cell)
                else:
                    headers.append(f"Col_{len(headers)}")
            else:
                headers.append(f"Col_{len(headers)}")
        
        return headers
    
    def _extract_from_two_column_table(self, data_rows: List[List[str]], page_num: int, table_idx: int) -> Dict[str, Any]:
        """Extract from two-column table"""
        result = {
            'fields': {},
            'checkboxes': {},
            'confidences': []
        }
        
        for row_idx, row in enumerate(data_rows):
            if len(row) >= 2 and row[0] and row[1]:
                field_name = str(row[0]).strip()
                field_value = str(row[1]).strip()
                
                if not field_name or not field_value:
                    continue
                
                cell_confidence = self._assess_cell_pair_quality(field_name, field_value)
                result['confidences'].append(cell_confidence)
                
                if cell_confidence > 0.3:
                    if self._is_checkbox_pattern(field_value):
                        checkbox_result = self._parse_checkbox_value(field_name, field_value)
                        if checkbox_result:
                            result['checkboxes'].update(checkbox_result)
                        continue
                    
                    clean_field_name = self._clean_field_name_enhanced(field_name)
                    clean_field_value = self._clean_field_value(field_value)
                    
                    if clean_field_name and clean_field_value:
                        result['fields'][clean_field_name] = clean_field_value
        
        return result
    
    def _extract_from_multi_column_table(self, headers: List[str], data_rows: List[List[str]], 
                                       page_num: int, table_idx: int) -> Dict[str, Any]:
        """Extract from multi-column table"""
        result = {
            'fields': {},
            'checkboxes': {},
            'confidences': []
        }
        
        key_col_idx = self._find_key_column(headers)
        value_col_indices = self._find_value_columns(headers, key_col_idx)
        
        for row in data_rows:
            if len(row) > max(key_col_idx, max(value_col_indices) if value_col_indices else 0):
                field_name = str(row[key_col_idx]).strip() if key_col_idx < len(row) else ""
                
                if not field_name:
                    continue
                
                for val_idx in value_col_indices:
                    if val_idx < len(row) and row[val_idx]:
                        field_value = str(row[val_idx]).strip()
                        
                        if field_value:
                            combined_field_name = f"{field_name}"
                            if val_idx < len(headers) and headers[val_idx]:
                                combined_field_name = f"{field_name}_{headers[val_idx]}"
                            
                            confidence = self._assess_cell_pair_quality(combined_field_name, field_value)
                            result['confidences'].append(confidence)
                            
                            if confidence > 0.3:
                                if self._is_checkbox_pattern(field_value):
                                    checkbox_result = self._parse_checkbox_value(combined_field_name, field_value)
                                    if checkbox_result:
                                        result['checkboxes'].update(checkbox_result)
                                else:
                                    clean_field_name = self._clean_field_name_enhanced(combined_field_name)
                                    clean_field_value = self._clean_field_value(field_value)
                                    
                                    if clean_field_name and clean_field_value:
                                        result['fields'][clean_field_name] = clean_field_value
        
        return result
    
    def _extract_from_row_patterns(self, data_rows: List[List[str]], page_num: int, table_idx: int) -> Dict[str, Any]:
        """Extract from row patterns"""
        result = {
            'fields': {},
            'checkboxes': {},
            'confidences': []
        }
        
        for row in data_rows:
            for cell in row:
                if not cell:
                    continue
                
                cell_str = str(cell).strip()
                
                for pattern in self.table_field_patterns:
                    matches = re.findall(pattern, cell_str, re.MULTILINE)
                    
                    for match in matches:
                        if len(match) == 2:
                            field_name, field_value = match
                            field_name = field_name.strip()
                            field_value = field_value.strip()
                            
                            if field_name and field_value:
                                confidence = self._assess_cell_pair_quality(field_name, field_value)
                                result['confidences'].append(confidence)
                                
                                if confidence > 0.4:
                                    if self._is_checkbox_pattern(field_value):
                                        checkbox_result = self._parse_checkbox_value(field_name, field_value)
                                        if checkbox_result:
                                            result['checkboxes'].update(checkbox_result)
                                    else:
                                        clean_field_name = self._clean_field_name_enhanced(field_name)
                                        clean_field_value = self._clean_field_value(field_value)
                                        
                                        if clean_field_name and clean_field_value:
                                            result['fields'][clean_field_name] = clean_field_value
        
        return result
    
    def _find_key_column(self, headers: List[str]) -> int:
        """Find key column"""
        key_indicators = ['item', 'field', 'parameter', 'description', 'name', 'property']
        
        for idx, header in enumerate(headers):
            header_lower = header.lower()
            if any(indicator in header_lower for indicator in key_indicators):
                return idx
        
        return 0
    
    def _find_value_columns(self, headers: List[str], key_col_idx: int) -> List[int]:
        """Find value columns"""
        value_indices = []
        
        for idx, header in enumerate(headers):
            if idx != key_col_idx:
                value_indices.append(idx)
        
        return value_indices if value_indices else [i for i in range(len(headers)) if i != key_col_idx]
    
    def _assess_table_structure_quality(self, table: List[List[str]], headers: List[str]) -> float:
        """Assess table structure quality"""
        if not table:
            return 0.0
        
        score = 0.5
        
        if len(table) >= 3:
            score += 0.2
        elif len(table) >= 2:
            score += 0.1
        
        col_count = len(headers) if headers else (len(table[0]) if table else 0)
        if 2 <= col_count <= 5:
            score += 0.2
        elif col_count > 1:
            score += 0.1
        
        if table:
            col_counts = [len(row) for row in table]
            if col_counts:
                consistency = 1.0 - (max(col_counts) - min(col_counts)) / max(col_counts, 1)
                score += consistency * 0.1
        
        return min(score, 1.0)
    
    def _assess_cell_pair_quality(self, field_name: str, field_value: str) -> float:
        """Assess cell pair quality"""
        score = 0.3
        
        if len(field_name) >= 3:
            score += 0.2
        
        if re.match(r'^[A-Za-z]', field_name):
            score += 0.1
        
        if not re.search(r'\d{3,}', field_name):
            score += 0.1
        
        if len(field_value) >= 1:
            score += 0.1
        
        if not field_value.lower() in ['', 'n/a', 'tbd', 'see drawing', 'as required', 'null']:
            score += 0.1
        
        if self._is_meaningful_content(field_name, field_value):
            score += 0.1
        
        return min(score, 1.0)
    
    def _is_meaningful_content(self, field_name: str, field_value: str) -> bool:
        """Check if content is meaningful"""
        generic_terms = ['item', 'no', 'number', 'id', 'index']
        if field_name.lower().strip() in generic_terms:
            return False
        
        if len(field_value.strip()) < 1:
            return False
        
        special_char_ratio = sum(1 for c in field_name if not c.isalnum() and c != ' ') / max(len(field_name), 1)
        if special_char_ratio > 0.5:
            return False
        
        return True
    
    def _is_checkbox_pattern(self, value: str) -> bool:
        """Check checkbox pattern"""
        checkbox_symbols = ['■', '□', '☑', '☐', '✓', '✔', '▣']
        return any(symbol in value for symbol in checkbox_symbols)
    
    def _parse_checkbox_value(self, field_name: str, field_value: str) -> Dict[str, str]:
        """Parse checkbox value"""
        result = {}
        
        checkbox_mappings = {
            'MANUAL': 'Manual',
            'AUTOMATIC': 'Automatic', 
            'AUTO': 'Automatic',
            'CONTINUOUS': 'Continuous',
            'INTERMITTENT': 'Intermittent',
            'YES': 'Yes',
            'NO': 'No'
        }
        
        if '■' in field_value or '☑' in field_value or '✓' in field_value or '✔' in field_value:
            for keyword, standard_value in checkbox_mappings.items():
                if keyword in field_value.upper():
                    clean_field_name = self._clean_field_name_enhanced(field_name)
                    if clean_field_name:
                        result[clean_field_name] = standard_value
                    break
        
        return result
    
    def _clean_field_name_enhanced(self, field_name: str) -> str:
        """Enhanced field name cleaning"""
        if not field_name:
            return ""
        
        clean = re.sub(r'^\d+\.?\s*', '', field_name.strip())
        
        prefixes_to_remove = ['item', 'no', 'ref', 'spec', 'field']
        words = clean.lower().split()
        if words and words[0] in prefixes_to_remove and len(words) > 1:
            clean = ' '.join(words[1:])
        
        clean = re.sub(r'[^\w\s°℃°F/-]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        if len(clean) < SmartConfig.MIN_FIELD_LENGTH or len(clean) > SmartConfig.MAX_FIELD_LENGTH:
            return ""
        
        return clean
    
    def _clean_field_value(self, field_value: str) -> str:
        """Clean field value"""
        if not field_value:
            return ""
        
        clean = field_value.strip()
        
        if len(clean) > SmartConfig.MAX_VALUE_LENGTH:
            clean = clean[:SmartConfig.MAX_VALUE_LENGTH] + "..."
        
        return clean
    
    def _extract_fields_comprehensively(self, text: str, tables: List[Dict], checkboxes: Dict) -> Dict[str, str]:
        """Comprehensive field extraction"""
        fields = {}
        
        # Add checkbox data
        fields.update(checkboxes)
        
        # Add table fields
        for table in tables:
            if table.get('confidence', 0) > 0.3:
                table_fields = table.get('field_value_pairs', {})
                table_checkboxes = table.get('checkbox_data', {})
                
                for field_name, field_value in table_fields.items():
                    existing_key = self._find_similar_existing_key(field_name, list(fields.keys()))
                    if existing_key:
                        if len(field_value) > len(fields[existing_key]):
                            fields[existing_key] = field_value
                    else:
                        fields[field_name] = field_value
                
                fields.update(table_checkboxes)
        
        # Add text fields
        text_fields = self._extract_text_patterns_enhanced(text)
        
        for field_name, field_value in text_fields.items():
            if self._is_valid_field_value_pair_enhanced(field_name, field_value):
                existing_key = self._find_similar_existing_key(field_name, list(fields.keys()))
                if existing_key:
                    if len(field_value) > len(fields[existing_key]):
                        fields[existing_key] = field_value
                else:
                    fields[field_name] = field_value
        
        logger.info(f"✅ Comprehensive extraction completed: {len(fields)} fields total")
        return fields
    
    def _extract_text_patterns_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced text pattern extraction"""
        fields = {}
        
        patterns = [
            r'([A-Za-z][A-Za-z\s]{2,50})\s*:\s*([^\n\r]{2,200})',
            r'([A-Za-z][A-Za-z\s]{2,50})\s*=\s*([^\n\r]{2,200})',
            r'([A-Za-z][A-Za-z\s]{2,50})\s{3,}([^\n\r]{2,200})',
            r'([A-Za-z][A-Za-z\s]{2,50})\t+([^\n\r\t]{2,200})',
            r'(\d{1,2}\.?\s+[A-Za-z][A-Za-z\s]{2,50})\s*[:=]?\s*([^\n\r]{2,200})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for field_name, field_value in matches:
                field_name = field_name.strip()
                field_value = field_value.strip()
                
                clean_field_name = self._clean_field_name_enhanced(field_name)
                clean_field_value = self._clean_field_value(field_value)
                
                if clean_field_name and clean_field_value:
                    if self._is_meaningful_content(clean_field_name, clean_field_value):
                        fields[clean_field_name] = clean_field_value
        
        return fields
    
    def _is_valid_field_value_pair_enhanced(self, field_name: str, field_value: str) -> bool:
        """Enhanced field-value pair validation"""
        if len(field_name) < SmartConfig.MIN_FIELD_LENGTH or len(field_name) > SmartConfig.MAX_FIELD_LENGTH:
            return False
        
        if len(field_value) < 1 or len(field_value) > SmartConfig.MAX_VALUE_LENGTH:
            return False
        
        for pattern in self.exclude_patterns:
            if re.search(pattern, field_name, re.IGNORECASE):
                return False
        
        meaningless_values = ['', 'n/a', 'tbd', 'see drawing', 'as required', 'null', 'none', '-']
        if field_value.lower().strip() in meaningless_values:
            return False
        
        if re.match(r'^\d+$', field_name.strip()):
            return False
        
        return True
    
    def _find_similar_existing_key(self, new_key: str, existing_keys: List[str]) -> Optional[str]:
        """Find similar existing key"""
        new_key_lower = new_key.lower().strip()
        
        for existing_key in existing_keys:
            existing_lower = existing_key.lower().strip()
            
            if new_key_lower == existing_lower:
                return existing_key
            
            new_words = set(new_key_lower.split())
            existing_words = set(existing_lower.split())
            
            if new_words and existing_words:
                overlap = len(new_words.intersection(existing_words))
                union = len(new_words.union(existing_words))
                
                if union > 0 and overlap / union > 0.8:
                    return existing_key
        
        return None
    
    def _extract_checkboxes_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced checkbox extraction"""
        checkboxes = {}
        
        checkbox_mappings = {
            'MANUAL': 'Start_Method',
            'AUTOMATIC': 'Start_Method', 
            'AUTO': 'Start_Method',
            'CONTINUOUS': 'Operation_Mode',
            'INTERMITTENT': 'Operation_Mode',
            'BATCH': 'Operation_Mode'
        }
        
        checked_patterns = [
            r'[■▣☑✓✔]\s*([A-Z][A-Z\s]+)',
            r'([A-Z][A-Z\s]+)\s*[■▣☑✓✔]'
        ]
        
        for pattern in checked_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                
                field_name = str(match).strip()
                
                for keyword, standard_field in checkbox_mappings.items():
                    if keyword in field_name.upper():
                        checkboxes[standard_field] = keyword
                        break
        
        return checkboxes
    
    def _extract_revisions(self, text: str) -> List[str]:
        """Extract revisions from text"""
        revisions = []
        
        for pattern in self.revision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            revisions.extend(matches)
        
        return list(set(revisions))
    
    def _get_latest_revision(self, revisions: List[str]) -> str:
        """Get latest revision"""
        if not revisions:
            return "0"
        
        def revision_key(rev):
            try:
                num_match = re.findall(r'\d+', rev)
                if num_match:
                    num = int(num_match[0])
                    letter_bonus = 0
                    letter_match = re.findall(r'[A-Z]', rev)
                    if letter_match:
                        letter_bonus = ord(letter_match[0]) - ord('A') + 1
                    return num * 100 + letter_bonus
                return 0
            except:
                return 0
        
        sorted_revisions = sorted(revisions, key=revision_key, reverse=True)
        return sorted_revisions[0]
    
    def _assess_page_quality_enhanced(self, text: str) -> float:
        """Enhanced page quality assessment"""
        if not text:
            return 0.0
        
        score = 0.1
        
        if len(text) > 500:
            score += 0.3
        elif len(text) > 100:
            score += 0.2
        
        structure_indicators = [':' in text, '=' in text, '\t' in text, re.search(r'\s{3,}', text)]
        score += sum(structure_indicators) * 0.1
        
        technical_indicators = ['pressure', 'temperature', 'capacity', 'pump', 'motor', 'flow', 'power']
        tech_score = sum(1 for indicator in technical_indicators if indicator in text.lower())
        score += min(tech_score * 0.05, 0.2)
        
        field_patterns = len(re.findall(r'[A-Za-z][A-Za-z\s]{2,30}\s*[:=]\s*[^\n\r]{2,}', text))
        score += min(field_patterns * 0.02, 0.2)
        
        return min(score, 1.0)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'text': '',
            'tables': [],
            'fields': {},
            'checkboxes': {},
            'revisions': [],
            'quality_score': 0.0,
            'metadata': {
                'total_pages': 0,
                'total_tables': 0,
                'checkboxes_found': 0,
                'revisions_found': 0,
                'fields_extracted': 0
            },
            'extraction_quality': {
                'page_qualities': [],
                'avg_quality': 0.0,
                'table_success_rate': 0.0
            }
        }

class SmartSemanticMatcher:
    """Enhanced semantic matcher with learning capabilities"""
    
    def __init__(self, ontology_manager):
        self.ontology = ontology_manager
        self.model = None
    
    def _load_model(self):
        """Load sentence transformer model"""
        SentenceTransformer = get_torch_module('SentenceTransformer')
        
        if not TORCH_AVAILABLE or SentenceTransformer is None:
            logger.warning("⚠️ SentenceTransformer not available")
            return None
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                import contextlib
                import io as io_module
                
                with contextlib.redirect_stdout(io_module.StringIO()), \
                     contextlib.redirect_stderr(io_module.StringIO()):
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                
                return model
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed: {e}")
            return None
    
    def match_fields(self, field1: str, field2: str) -> Dict[str, float]:
        """Enhanced field matching"""
        results = {
            'ontology_score': 0.0,
            'semantic_score': 0.0,
            'string_score': 0.0,
            'consolidation_score': 0.0,
            'learned_score': 0.0,
            'final_score': 0.0,
            'confidence': 0.0,
            'match_details': {}
        }
        
        field1_info = self.ontology.find_standard_field(field1)
        field2_info = self.ontology.find_standard_field(field2)
        
        std_field1 = field1_info['standard_field_name']
        std_field2 = field2_info['standard_field_name']
        
        if std_field1 == std_field2:
            results['ontology_score'] = 1.0
        
        results['consolidation_score'] = self._calculate_consolidation_similarity(std_field1, std_field2)
        results['learned_score'] = self._calculate_learned_similarity(field1, field2)
        results['string_score'] = self._calculate_string_similarity(field1, field2)
        
        cosine_similarity = get_torch_module('cosine_similarity')
        
        if cosine_similarity is not None:
            try:
                if self.model is None:
                    self.model = self._load_model()
                
                if self.model is not None:
                    emb1 = self.model.encode([field1])
                    emb2 = self.model.encode([field2])
                    results['semantic_score'] = float(cosine_similarity(emb1, emb2)[0][0])
                else:
                    results['semantic_score'] = results['string_score']
            except Exception as e:
                logger.warning(f"Semantic matching failed: {e}")
                results['semantic_score'] = results['string_score']
        else:
            results['semantic_score'] = results['string_score']
        
        confidence1 = field1_info['confidence']
        confidence2 = field2_info['confidence']
        avg_confidence = (confidence1 + confidence2) / 2
        
        if avg_confidence > SmartConfig.HIGH_CONFIDENCE_THRESHOLD:
            weights = {
                'ontology': 0.4,
                'consolidation': 0.25,
                'learned': 0.15,
                'semantic': 0.15,
                'string': 0.05
            }
            results['confidence'] = 0.9
        elif avg_confidence > SmartConfig.MIN_CONFIDENCE_THRESHOLD:
            weights = {
                'ontology': 0.3,
                'consolidation': 0.2,
                'learned': 0.2,
                'semantic': 0.2,
                'string': 0.1
            }
            results['confidence'] = 0.7
        else:
            weights = {
                'ontology': 0.2,
                'consolidation': 0.15,
                'learned': 0.15,
                'semantic': 0.3,
                'string': 0.2
            }
            results['confidence'] = 0.5
        
        results['final_score'] = (
            weights['ontology'] * results['ontology_score'] +
            weights['consolidation'] * results['consolidation_score'] +
            weights['learned'] * results['learned_score'] +
            weights['semantic'] * results['semantic_score'] +
            weights['string'] * results['string_score']
        )
        
        results['match_details'] = {
            'field1_info': field1_info,
            'field2_info': field2_info,
            'weights': weights
        }
        
        return results
    
    def _calculate_consolidation_similarity(self, field1: str, field2: str) -> float:
        """Calculate similarity based on consolidation groups"""
        consolidation_rules = self.ontology.consolidation_rules
        
        field1_groups = []
        field2_groups = []
        
        for group_name, group_config in consolidation_rules.items():
            if field1.lower() in [kw.lower() for kw in group_config['keywords']]:
                field1_groups.append(group_name)
            if field2.lower() in [kw.lower() for kw in group_config['keywords']]:
                field2_groups.append(group_name)
        
        if field1_groups and field2_groups:
            common_groups = set(field1_groups).intersection(set(field2_groups))
            if common_groups:
                return 0.9
        
        return 0.0
    
    def _calculate_learned_similarity(self, field1: str, field2: str) -> float:
        """Calculate similarity based on learned patterns"""
        learned_mappings = self.ontology.learned_mappings
        
        field1_lower = field1.lower()
        field2_lower = field2.lower()
        
        if field1_lower in learned_mappings and field2_lower in learned_mappings:
            if learned_mappings[field1_lower] == learned_mappings[field2_lower]:
                return 0.95
        
        return 0.0
    
    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """Enhanced string similarity calculation"""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union)
        
        char_sim = 1.0 - (abs(len(s1) - len(s2)) / max(len(s1), len(s2)))
        
        return (jaccard * 0.8) + (char_sim * 0.2)

class SmartPlantAnalyzer:
    """Enhanced plant analyzer with learning and validation"""
    
    def __init__(self):
        self.db_manager = SmartDatabaseManager()
        self.ontology_manager = SmartOntologyManager()
        self.pdf_processor = SmartPDFProcessor()
        self.semantic_matcher = SmartSemanticMatcher(self.ontology_manager)
        self.documents = {}
        self.knowledge_graph = nx.DiGraph()
        
        logger.info("🚀 Smart Plant Analyzer initialized successfully")
    
    def process_document(self, pdf_file, filename: str) -> Dict[str, Any]:
        """Enhanced document processing with validation"""
        try:
            st.info(f"🔍 Analyzing {filename}...")
            extraction_result = self.pdf_processor.extract_text_and_tables(pdf_file)
            
            if not extraction_result or extraction_result.get('quality_score', 0) == 0:
                st.warning(f"⚠️ Low quality extraction for {filename}")
            
            content_for_hash = {
                'fields': extraction_result.get('fields', {}),
                'tables': len(extraction_result.get('tables', [])),
                'text_length': len(extraction_result.get('text', ''))
            }
            content_hash = hashlib.sha256(str(content_for_hash).encode()).hexdigest()
            
            metadata = extraction_result.get('metadata', {})
            doc_id = self.db_manager.save_document(
                filename=filename,
                content_hash=content_hash,
                quality_score=extraction_result.get('quality_score', 0.0),
                page_count=metadata.get('total_pages', 0),
                table_count=metadata.get('total_tables', 0),
                field_count=len(extraction_result.get('fields', {}))
            )
            
            for table in extraction_result.get('tables', []):
                self.db_manager.save_table_extraction(
                    doc_id=doc_id,
                    page_number=table.get('page_number', 0),
                    table_index=table.get('table_index', 0),
                    headers=table.get('headers', []),
                    row_count=table.get('extraction_stats', {}).get('total_rows', 0),
                    col_count=table.get('extraction_stats', {}).get('total_cols', 0),
                    confidence=table.get('confidence', 0.0),
                    field_pairs=len(table.get('field_value_pairs', {}))
                )
            
            self.documents[str(doc_id)] = {
                'doc_id': doc_id,
                'filename': filename,
                'fields': extraction_result.get('fields', {}),
                'checkboxes': extraction_result.get('checkboxes', {}),
                'revisions': extraction_result.get('revisions', []),
                'final_revision': extraction_result.get('final_revision', '0'),
                'tables': extraction_result.get('tables', []),
                'text': extraction_result.get('text', ''),
                'quality_score': extraction_result.get('quality_score', 0.0),
                'metadata': metadata,
                'extraction_quality': extraction_result.get('extraction_quality', {}),
                'processed_at': datetime.now().isoformat(),
                'ontology_enhanced_fields': {}
            }
            
            st.info(f"🧠 Applying AI ontology to {len(extraction_result.get('fields', {}))} fields...")
            
            ontology_enhanced_fields = {}
            high_confidence_count = 0
            validated_count = 0
            
            all_fields = {**extraction_result.get('fields', {}), **extraction_result.get('checkboxes', {})}
            
            for field_name, field_value in all_fields.items():
                field_info = self.ontology_manager.find_standard_field(
                    field_name, 
                    context=extraction_result.get('text', '')[:500]
                )
                
                is_valid = self.ontology_manager.validate_field_value(
                    field_info['standard_field_name'], 
                    field_value
                )
                
                if is_valid:
                    validated_count += 1
                    field_info['validation_status'] = 'valid'
                else:
                    field_info['validation_status'] = 'invalid'
                
                if field_info['confidence'] > SmartConfig.HIGH_CONFIDENCE_THRESHOLD:
                    high_confidence_count += 1
                
                source_type = 'checkbox' if field_name in extraction_result.get('checkboxes', {}) else 'text'
                
                for table in extraction_result.get('tables', []):
                    if field_name in table.get('field_value_pairs', {}) or field_name in table.get('checkbox_data', {}):
                        source_type = 'table'
                        break
                
                enhanced_field_data = {
                    'original_field_name': field_name,
                    'normalized_field_name': field_info['normalized_field_name'],
                    'standard_field_name': field_info['standard_field_name'],
                    'field_value': field_value,
                    'field_type': self._detect_field_type_enhanced(field_value),
                    'confidence': field_info['confidence'],
                    'extraction_method': field_info['match_type'],
                    'validation_status': field_info['validation_status'],
                    'context_info': field_info['context_info'],
                    'source_type': source_type,
                    'page_number': 0,
                    'table_index': -1
                }
                
                self.db_manager.save_field(doc_id, enhanced_field_data)
                ontology_enhanced_fields[field_name] = field_info
                
                if field_info['confidence'] > SmartConfig.MIN_CONFIDENCE_THRESHOLD:
                    self.db_manager.update_mapping_cache(
                        field_info['normalized_field_name'], 
                        field_info['standard_field_name'], 
                        field_info['confidence']
                    )
            
            self.documents[str(doc_id)]['ontology_enhanced_fields'] = ontology_enhanced_fields
            
            self._update_knowledge_graph_enhanced(doc_id, extraction_result, ontology_enhanced_fields)
            
            processing_result = {
                'doc_id': doc_id,
                'field_count': len(all_fields),
                'checkbox_count': len(extraction_result.get('checkboxes', {})),
                'quality_score': extraction_result.get('quality_score', 0.0),
                'tables_count': len(extraction_result.get('tables', [])),
                'successful_tables': sum(1 for t in extraction_result.get('tables', []) if t.get('confidence', 0) > 0.5),
                'final_revision': extraction_result.get('final_revision', '0'),
                'ontology_enhanced_count': len(ontology_enhanced_fields),
                'high_confidence_count': high_confidence_count,
                'validated_count': validated_count,
                'validation_rate': validated_count / len(all_fields) if all_fields else 0,
                'extraction_quality': extraction_result.get('extraction_quality', {}),
                'pages_processed': metadata.get('total_pages', 0)
            }
            
            logger.info(f"✅ Document {filename} processed successfully: "
                       f"{processing_result['field_count']} fields, "
                       f"{processing_result['high_confidence_count']} high confidence, "
                       f"{processing_result['validated_count']} validated")
            
            return processing_result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed for {filename}: {e}")
            st.error(f"Error processing {filename}: {str(e)}")
            return {
                'doc_id': 0,
                'field_count': 0,
                'checkbox_count': 0,
                'quality_score': 0.0,
                'tables_count': 0,
                'successful_tables': 0,
                'final_revision': '0',
                'ontology_enhanced_count': 0,
                'high_confidence_count': 0,
                'validated_count': 0,
                'validation_rate': 0.0,
                'extraction_quality': {},
                'pages_processed': 0,
                'error': str(e)
            }
    
    def _detect_field_type_enhanced(self, value: str) -> str:
        """Enhanced field type detection"""
        value = str(value).strip()
        
        if re.match(r'^-?\d+\.?\d*\s*[a-zA-Z/°℃°F]+', value):
            return "measurement"
        elif re.match(r'^-?\d+\.?\d*ailable. Install: `pip install pdfplumber`")
                return
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF documents for analysis",
                type=['pdf'],
                accept_multiple_files=True,
                help="AI will learn from your documents"
            )
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files ready")
                
                if st.button("🚀 Process Documents", type="primary"):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    for i, file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {file.name}...")
                        
                        try:
                            result = analyzer.process_document(file, file.name)
                            results.append(result)
                            
                            quality_emoji = "🌟" if result['quality_score'] > 0.8 else "✅" if result['quality_score'] > 0.6 else "⚠️"
                            st.success(f"{quality_emoji} {file.name} processed!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Fields", result['field_count'])
                            with col2:
                                st.metric("Quality", f"{result['quality_score']:.2f}")
                            with col3:
                                st.metric("High Confidence", result['high_confidence_count'])
                            with col4:
                                st.metric("Validated", result['validated_count'])
                        
                        except Exception as e:
                            st.error(f"❌ Failed: {file.name} - {str(e)}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    if results:
                        st.balloons()
                        st.rerun()
            
            # Document summary
            if analyzer.documents:
                st.markdown("### 📋 Processed Documents")
                
                summary_data = []
                for doc_data in analyzer.documents.values():
                    ontology_fields = doc_data.get('ontology_enhanced_fields', {})
                    high_conf = sum(1 for f in ontology_fields.values() if f.get('confidence', 0) > SmartConfig.HIGH_CONFIDENCE_THRESHOLD)
                    validated = sum(1 for f in ontology_fields.values() if f.get('validation_status') == 'valid')
                    
                    summary_data.append({
                        "📄 Document": doc_data['filename'],
                        "📊 Fields": len(doc_data.get('fields', {})),
                        "🎯 High Conf": high_conf,
                        "✅ Validated": validated,
                        "📈 Quality": f"{doc_data['quality_score']:.2f}",
                        "📝 Rev": doc_data.get('final_revision', '0'),
                        "📑 Pages": doc_data.get('metadata', {}).get('total_pages', 0)
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Document Comparison")
            if len(analyzer.documents) < 2:
                st.warning("⚠️ Upload at least 2 documents for comparison")
            else:
                comparison_data = analyzer.get_comparison_data()
                if comparison_data['comparison_ready']:
                    st.success(f"✅ Ready to compare {comparison_data['total_documents']} documents")
                    st.info("🚧 Comparison interface coming soon...")
        
        with tab3:
            st.markdown("## 📊 Quality Analytics")
            if not analyzer.documents:
                st.info("📈 Upload documents to see analytics")
            else:
                qualities = [doc['quality_score'] for doc in analyzer.documents.values()]
                if qualities:
                    quality_df = pd.DataFrame({'Quality': qualities})
                    
                    fig = px.histogram(
                        quality_df, 
                        x='Quality',
                        nbins=10,
                        title="Document Quality Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show details"):
            st.exception(e)

# Run the application
if __name__ == "__main__":
    main()df, 
                        x='Quality',
                        nbins=10,
                        title="Document Quality Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show details"):
            st.exception(e)

# Run the application
if __name__ == "__main__":
    main()total_validated / total_fields * 100) if total_fields > 0 else 0
                            st.metric("Validation Rate", f"{validation_rate:.1f}%")
                        
                        st.balloons()
                        st.rerun()
            
            if analyzer.documents:
                st.markdown("### 📋 Processed Documents")
                
                summary_data = []
                for doc_key, doc_data in analyzer.documents.items():
                    ontology_fields = doc_data.get('ontology_enhanced_fields', {})
                    high_conf_count = sum(1 for f in ontology_fields.values() if f.get('confidence', 0) > SmartConfig.HIGH_CONFIDENCE_THRESHOLD)
                    validated_count = sum(1 for f in ontology_fields.values() if f.get('validation_status') == 'valid')
                    
                    summary_data.append({
                        "📄 Document": doc_data['filename'],
                        "📊 Fields": len(doc_data.get('fields', {})),
                        "🎯 High Confidence": high_conf_count,
                        "✅ Validated": validated_count,
                        "📈 Quality": f"{doc_data['quality_score']:.2f}",
                        "📝 Revision": doc_data.get('final_revision', '0'),
                        "⏰ Processed": doc_data['processed_at'][:16],
                        "📑 Pages": doc_data.get('metadata', {}).get('total_pages', 0),
                        "📊 Tables": len(doc_data.get('tables', []))
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Intelligent Document Comparison")
            if len(analyzer.documents) < 2:
                st.warning("⚠️ Please upload at least 2 documents for comparison.")
            else:
                comparison_data = analyzer.get_comparison_data()
                if comparison_data['comparison_ready']:
                    st.success(f"✅ Ready to compare {comparison_data['total_documents']} documents with {comparison_data['total_fields']} unique fields")
                else:
                    st.info("🔄 Loading comparison interface...")
        
        with tab3:
            st.markdown("## 📊 Advanced Quality Analytics")
            if not analyzer.documents:
                st.info("📈 Upload documents to see quality analytics.")
            else:
                doc_qualities = [doc_data['quality_score'] for doc_data in analyzer.documents.values()]
                quality_df = pd.DataFrame({'Quality': doc_qualities})
                
                fig_quality_dist = px.histogram(
                    quality_df, 
                    x='Quality',
                    nbins=10,
                    title="Document Quality Score Distribution",
                    labels={'count': 'Number of Documents'}
                )
                st.plotly_chart(fig_quality_dist, use_container_width=True)
        
        with tab4:
            st.markdown("## 🕸️ Enhanced Knowledge Graph")
            if len(analyzer.knowledge_graph.nodes) == 0:
                st.warning("⚠️ No knowledge graph data available. Process documents first.")
            else:
                st.info("🔄 Loading knowledge graph visualization...")
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show error details"):
            st.exception(e)

# System initialization and execution
if __name__ == "__main__":
    try:
        if 'analyzer' not in st.session_state:
            with st.spinner("🚀 Initializing Smart Plant Analyzer..."):
                analyzer = initialize_smart_analyzer()
                if analyzer:
                    st.session_state.analyzer = analyzer
                    logger.info("✅ System initialized successfully")
                else:
                    st.error("❌ Failed to initialize Smart Plant Analyzer")
                    st.stop()
        
        main()
        
    except Exception as e:
        st.error(f"❌ Critical system error: {e}")
        if st.button("🔄 Restart System"):
            st.session_state.clear()
            st.rerun(), value):
            return "numeric"
        elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value):
            return "date"
        elif value.upper() in ['YES', 'NO', 'TRUE', 'FALSE', 'MANUAL', 'AUTOMATIC', 'CONTINUOUS', 'INTERMITTENT']:
            return "boolean"
        elif re.match(r'^[A-Z0-9]+-[A-Z0-9]+', value) or re.match(r'^[A-Z]{2,}\d+', value):
            return "identifier"
        elif any(material in value.lower() for material in ['steel', 'carbon', 'stainless', 'alloy', 'brass', 'bronze']):
            return "material"
        elif any(unit in value.lower() for unit in ['bar', 'psi', 'kpa', 'mpa']):
            return "pressure"
        elif any(unit in value.lower() for unit in ['°c', '°f', 'celsius', 'fahrenheit']):
            return "temperature"
        elif any(unit in value.lower() for unit in ['m3/hr', 'l/s', 'gpm', 'lpm', 'cfm']):
            return "capacity"
        elif len(value) < 50 and len(value) > 0 and value[0].isupper():
            return "category"
        elif len(value) > 100:
            return "description"
        else:
            return "text"
    
    def _update_knowledge_graph_enhanced(self, doc_id: int, extraction_result: Dict, ontology_enhanced_fields: Dict):
        """Enhanced knowledge graph construction"""
        doc_node = f"doc_{doc_id}"
        
        metadata = extraction_result.get('metadata', {})
        self.knowledge_graph.add_node(doc_node, 
                                     type="document", 
                                     filename=metadata.get('filename', ''),
                                     revision=extraction_result.get('final_revision', '0'),
                                     quality=extraction_result.get('quality_score', 0.0),
                                     pages=metadata.get('total_pages', 0),
                                     tables=metadata.get('total_tables', 0),
                                     extraction_method="enhanced_pdf")
        
        all_fields = {**extraction_result.get('fields', {}), **extraction_result.get('checkboxes', {})}
        
        for field_name, field_value in all_fields.items():
            field_info = ontology_enhanced_fields.get(field_name, {})
            std_field = field_info.get('standard_field_name', field_name)
            
            field_node = f"field_{std_field}_{doc_id}"
            value_node = f"value_{hashlib.md5(str(field_value).encode()).hexdigest()[:8]}"
            
            self.knowledge_graph.add_node(field_node, 
                                         type="field", 
                                         name=std_field, 
                                         original_name=field_name,
                                         field_type=self._detect_field_type_enhanced(field_value),
                                         doc_id=doc_id,
                                         confidence=field_info.get('confidence', 0.0),
                                         match_type=field_info.get('match_type', 'unknown'),
                                         validation_status=field_info.get('validation_status', 'unknown'),
                                         source_type=field_info.get('source_type', 'text'))
            
            self.knowledge_graph.add_node(value_node, 
                                         type="value", 
                                         value=field_value,
                                         data_type=self._detect_field_type_enhanced(field_value),
                                         is_valid=field_info.get('validation_status') == 'valid',
                                         length=len(str(field_value)))
            
            self.knowledge_graph.add_edge(doc_node, field_node, 
                                        relation="has_field",
                                        confidence=field_info.get('confidence', 0.0),
                                        extraction_method=field_info.get('match_type', 'unknown'))
            
            self.knowledge_graph.add_edge(field_node, value_node, 
                                        relation="has_value",
                                        validation=field_info.get('validation_status', 'unknown'),
                                        data_type=self._detect_field_type_enhanced(field_value))
        
        for table_idx, table in enumerate(extraction_result.get('tables', [])):
            table_node = f"table_{doc_id}_{table_idx}"
            self.knowledge_graph.add_node(table_node,
                                        type="table",
                                        doc_id=doc_id,
                                        table_index=table_idx,
                                        page_number=table.get('page_number', 0),
                                        confidence=table.get('confidence', 0.0),
                                        row_count=table.get('extraction_stats', {}).get('total_rows', 0),
                                        col_count=table.get('extraction_stats', {}).get('total_cols', 0),
                                        field_pairs_count=len(table.get('field_value_pairs', {})))
            
            self.knowledge_graph.add_edge(doc_node, table_node,
                                        relation="contains_table",
                                        confidence=table.get('confidence', 0.0))
    
    def get_comparison_data(self) -> Dict[str, Any]:
        """Get comparison data for documents"""
        if len(self.documents) < 2:
            return {
                'field_data': {},
                'doc_names': [],
                'comparison_ready': False,
                'message': "⚠️ At least 2 documents required."
            }
        
        field_data = {}
        doc_names = []
        
        for doc_key, doc_data in self.documents.items():
            doc_name = doc_data['filename']
            doc_names.append(doc_name)
            ontology_fields = doc_data.get('ontology_enhanced_fields', {})
            
            all_fields = {**doc_data['fields'], **doc_data.get('checkboxes', {})}
            
            for original_field, field_value in all_fields.items():
                field_info = ontology_fields.get(original_field, {})
                std_field = field_info.get('standard_field_name', original_field)
                
                if std_field not in field_data:
                    field_data[std_field] = {
                        'original_names': set(),
                        'documents': {},
                        'field_info': field_info,
                        'confidence_scores': [],
                        'validation_statuses': []
                    }
                
                field_data[std_field]['original_names'].add(original_field)
                field_data[std_field]['documents'][doc_name] = field_value
                field_data[std_field]['confidence_scores'].append(field_info.get('confidence', 0.0))
                field_data[std_field]['validation_statuses'].append(field_info.get('validation_status', 'unknown'))
        
        return {
            'field_data': field_data,
            'doc_names': doc_names,
            'comparison_ready': True,
            'total_fields': len(field_data),
            'total_documents': len(self.documents)
        }

def get_analyzer():
    """Get or create analyzer instance"""
    if 'analyzer' not in st.session_state:
        try:
            with st.spinner("🚀 Initializing Smart Plant Analyzer..."):
                st.session_state.analyzer = SmartPlantAnalyzer()
                logger.info("✅ System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            st.error(f"❌ System initialization failed: {e}")
            st.stop()
    
    return st.session_state.analyzer

def main():
    """Main application function"""
    
    # Header
    st.title("🏭 Plant Check - Smart Analyzer")
    st.markdown("### AI-Powered Document Processing & Field Extraction")
    
    # Initialize analyzer
    try:
        analyzer = get_analyzer()
        st.success("✅ Smart Plant Analyzer Ready")
    except Exception as e:
        st.error(f"❌ Analyzer error: {e}")
        return
    
    # Sidebar status
    with st.sidebar:
        st.markdown("## 🎯 System Status")
        
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(analyzer.documents))
                st.metric("Graph Nodes", len(analyzer.knowledge_graph.nodes))
            with col2:
                st.metric("Graph Edges", len(analyzer.knowledge_graph.edges))
                
                if analyzer.documents:
                    avg_quality = sum(doc['quality_score'] for doc in analyzer.documents.values()) / len(analyzer.documents)
                    st.metric("Avg Quality", f"{avg_quality:.2f}")
            
            st.markdown("### 🔧 Features")
            features = [
                ("📄 PDF Processing", PDF_AVAILABLE),
                ("🧠 AI Ontology", RDF_AVAILABLE), 
                ("🤖 ML Matching", TORCH_AVAILABLE)
            ]
            
            for feature, status in features:
                status_icon = "✅" if status else "❌"
                st.markdown(f"{status_icon} {feature}")
            
            st.markdown("### 🧠 Intelligence")
            learned_mappings = len(analyzer.ontology_manager.learned_mappings)
            field_mappings = len(analyzer.ontology_manager.field_mappings)
            
            st.metric("Learned Mappings", learned_mappings)
            st.metric("Field Mappings", field_mappings)
            
            total_mappings = learned_mappings + field_mappings
            if total_mappings > 100:
                st.success("🌟 Expert Level")
            elif total_mappings > 50:
                st.info("🚀 Advanced Level")
            elif total_mappings > 20:
                st.warning("📚 Learning Level")
            else:
                st.info("🔰 Beginner Level")
                
        except Exception as e:
            st.error(f"Sidebar error: {e}")
    
    # Main tabs
    try:
        tab1, tab2, tab3 = st.tabs(["📄 Document Processing", "🔍 Comparison", "📊 Analytics"])
        
        with tab1:
            st.markdown("## 📄 Document Processing")
            
            if not PDF_AVAILABLE:
                st.error("❌ PDF processing unavailable. Install: `pip install pdfplumber`")
                return
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF documents for analysis",
                type=['pdf'],
                accept_multiple_files=True,
                help="AI will learn from your documents"
            )
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files ready")
                
                if st.button("🚀 Process Documents", type="primary"):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    for i, file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {file.name}...")
                        
                        try:
                            result = analyzer.process_document(file, file.name)
                            results.append(result)
                            
                            quality_emoji = "🌟" if result['quality_score'] > 0.8 else "✅" if result['quality_score'] > 0.6 else "⚠️"
                            st.success(f"{quality_emoji} {file.name} processed!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Fields", result['field_count'])
                            with col2:
                                st.metric("Quality", f"{result['quality_score']:.2f}")
                            with col3:
                                st.metric("High Confidence", result['high_confidence_count'])
                            with col4:
                                st.metric("Validated", result['validated_count'])
                        
                        except Exception as e:
                            st.error(f"❌ Failed: {file.name} - {str(e)}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    if results:
                        st.balloons()
                        st.rerun()
            
            # Document summary
            if analyzer.documents:
                st.markdown("### 📋 Processed Documents")
                
                summary_data = []
                for doc_data in analyzer.documents.values():
                    ontology_fields = doc_data.get('ontology_enhanced_fields', {})
                    high_conf = sum(1 for f in ontology_fields.values() if f.get('confidence', 0) > SmartConfig.HIGH_CONFIDENCE_THRESHOLD)
                    validated = sum(1 for f in ontology_fields.values() if f.get('validation_status') == 'valid')
                    
                    summary_data.append({
                        "📄 Document": doc_data['filename'],
                        "📊 Fields": len(doc_data.get('fields', {})),
                        "🎯 High Conf": high_conf,
                        "✅ Validated": validated,
                        "📈 Quality": f"{doc_data['quality_score']:.2f}",
                        "📝 Rev": doc_data.get('final_revision', '0'),
                        "📑 Pages": doc_data.get('metadata', {}).get('total_pages', 0)
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Document Comparison")
            if len(analyzer.documents) < 2:
                st.warning("⚠️ Upload at least 2 documents for comparison")
            else:
                comparison_data = analyzer.get_comparison_data()
                if comparison_data['comparison_ready']:
                    st.success(f"✅ Ready to compare {comparison_data['total_documents']} documents")
                    st.info("🚧 Comparison interface coming soon...")
        
        with tab3:
            st.markdown("## 📊 Quality Analytics")
            if not analyzer.documents:
                st.info("📈 Upload documents to see analytics")
            else:
                qualities = [doc['quality_score'] for doc in analyzer.documents.values()]
                if qualities:
                    quality_df = pd.DataFrame({'Quality': qualities})
                    
                    fig = px.histogram(
                        quality_df, 
                        x='Quality',
                        nbins=10,
                        title="Document Quality Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show details"):
            st.exception(e)

# Run the application
if __name__ == "__main__":
    main()ailable. Install: `pip install pdfplumber`")
                return
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF documents for analysis",
                type=['pdf'],
                accept_multiple_files=True,
                help="AI will learn from your documents"
            )
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files ready")
                
                if st.button("🚀 Process Documents", type="primary"):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    for i, file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {file.name}...")
                        
                        try:
                            result = analyzer.process_document(file, file.name)
                            results.append(result)
                            
                            quality_emoji = "🌟" if result['quality_score'] > 0.8 else "✅" if result['quality_score'] > 0.6 else "⚠️"
                            st.success(f"{quality_emoji} {file.name} processed!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Fields", result['field_count'])
                            with col2:
                                st.metric("Quality", f"{result['quality_score']:.2f}")
                            with col3:
                                st.metric("High Confidence", result['high_confidence_count'])
                            with col4:
                                st.metric("Validated", result['validated_count'])
                        
                        except Exception as e:
                            st.error(f"❌ Failed: {file.name} - {str(e)}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    if results:
                        st.balloons()
                        st.rerun()
            
            # Document summary
            if analyzer.documents:
                st.markdown("### 📋 Processed Documents")
                
                summary_data = []
                for doc_data in analyzer.documents.values():
                    ontology_fields = doc_data.get('ontology_enhanced_fields', {})
                    high_conf = sum(1 for f in ontology_fields.values() if f.get('confidence', 0) > SmartConfig.HIGH_CONFIDENCE_THRESHOLD)
                    validated = sum(1 for f in ontology_fields.values() if f.get('validation_status') == 'valid')
                    
                    summary_data.append({
                        "📄 Document": doc_data['filename'],
                        "📊 Fields": len(doc_data.get('fields', {})),
                        "🎯 High Conf": high_conf,
                        "✅ Validated": validated,
                        "📈 Quality": f"{doc_data['quality_score']:.2f}",
                        "📝 Rev": doc_data.get('final_revision', '0'),
                        "📑 Pages": doc_data.get('metadata', {}).get('total_pages', 0)
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Document Comparison")
            if len(analyzer.documents) < 2:
                st.warning("⚠️ Upload at least 2 documents for comparison")
            else:
                comparison_data = analyzer.get_comparison_data()
                if comparison_data['comparison_ready']:
                    st.success(f"✅ Ready to compare {comparison_data['total_documents']} documents")
                    st.info("🚧 Comparison interface coming soon...")
        
        with tab3:
            st.markdown("## 📊 Quality Analytics")
            if not analyzer.documents:
                st.info("📈 Upload documents to see analytics")
            else:
                qualities = [doc['quality_score'] for doc in analyzer.documents.values()]
                if qualities:
                    quality_df = pd.DataFrame({'Quality': qualities})
                    
                    fig = px.histogram(
                        quality_df, 
                        x='Quality',
                        nbins=10,
                        title="Document Quality Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show details"):
            st.exception(e)

# Run the application
if __name__ == "__main__":
    main()ailable. Install: `pip install pdfplumber`")
                return
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF documents for analysis",
                type=['pdf'],
                accept_multiple_files=True,
                help="AI will learn from your documents"
            )
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files ready")
                
                if st.button("🚀 Process Documents", type="primary"):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    for i, file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {file.name}...")
                        
                        try:
                            result = analyzer.process_document(file, file.name)
                            results.append(result)
                            
                            quality_emoji = "🌟" if result['quality_score'] > 0.8 else "✅" if result['quality_score'] > 0.6 else "⚠️"
                            st.success(f"{quality_emoji} {file.name} processed!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Fields", result['field_count'])
                            with col2:
                                st.metric("Quality", f"{result['quality_score']:.2f}")
                            with col3:
                                st.metric("High Confidence", result['high_confidence_count'])
                            with col4:
                                st.metric("Validated", result['validated_count'])
                        
                        except Exception as e:
                            st.error(f"❌ Failed: {file.name} - {str(e)}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    if results:
                        st.balloons()
                        st.rerun()
            
            # Document summary
            if analyzer.documents:
                st.markdown("### 📋 Processed Documents")
                
                summary_data = []
                for doc_data in analyzer.documents.values():
                    ontology_fields = doc_data.get('ontology_enhanced_fields', {})
                    high_conf = sum(1 for f in ontology_fields.values() if f.get('confidence', 0) > SmartConfig.HIGH_CONFIDENCE_THRESHOLD)
                    validated = sum(1 for f in ontology_fields.values() if f.get('validation_status') == 'valid')
                    
                    summary_data.append({
                        "📄 Document": doc_data['filename'],
                        "📊 Fields": len(doc_data.get('fields', {})),
                        "🎯 High Conf": high_conf,
                        "✅ Validated": validated,
                        "📈 Quality": f"{doc_data['quality_score']:.2f}",
                        "📝 Rev": doc_data.get('final_revision', '0'),
                        "📑 Pages": doc_data.get('metadata', {}).get('total_pages', 0)
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Document Comparison")
            if len(analyzer.documents) < 2:
                st.warning("⚠️ Upload at least 2 documents for comparison")
            else:
                comparison_data = analyzer.get_comparison_data()
                if comparison_data['comparison_ready']:
                    st.success(f"✅ Ready to compare {comparison_data['total_documents']} documents")
                    st.info("🚧 Comparison interface coming soon...")
        
        with tab3:
            st.markdown("## 📊 Quality Analytics")
            if not analyzer.documents:
                st.info("📈 Upload documents to see analytics")
            else:
                qualities = [doc['quality_score'] for doc in analyzer.documents.values()]
                if qualities:
                    quality_df = pd.DataFrame({'Quality': qualities})
                    
                    fig = px.histogram(
                        quality_df, 
                        x='Quality',
                        nbins=10,
                        title="Document Quality Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show details"):
            st.exception(e)

# Run the application
if __name__ == "__main__":
    main()df, 
                        x='Quality',
                        nbins=10,
                        title="Document Quality Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show details"):
            st.exception(e)

# Run the application
if __name__ == "__main__":
    main()total_validated / total_fields * 100) if total_fields > 0 else 0
                            st.metric("Validation Rate", f"{validation_rate:.1f}%")
                        
                        st.balloons()
                        st.rerun()
            
            if analyzer.documents:
                st.markdown("### 📋 Processed Documents")
                
                summary_data = []
                for doc_key, doc_data in analyzer.documents.items():
                    ontology_fields = doc_data.get('ontology_enhanced_fields', {})
                    high_conf_count = sum(1 for f in ontology_fields.values() if f.get('confidence', 0) > SmartConfig.HIGH_CONFIDENCE_THRESHOLD)
                    validated_count = sum(1 for f in ontology_fields.values() if f.get('validation_status') == 'valid')
                    
                    summary_data.append({
                        "📄 Document": doc_data['filename'],
                        "📊 Fields": len(doc_data.get('fields', {})),
                        "🎯 High Confidence": high_conf_count,
                        "✅ Validated": validated_count,
                        "📈 Quality": f"{doc_data['quality_score']:.2f}",
                        "📝 Revision": doc_data.get('final_revision', '0'),
                        "⏰ Processed": doc_data['processed_at'][:16],
                        "📑 Pages": doc_data.get('metadata', {}).get('total_pages', 0),
                        "📊 Tables": len(doc_data.get('tables', []))
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🔍 Intelligent Document Comparison")
            if len(analyzer.documents) < 2:
                st.warning("⚠️ Please upload at least 2 documents for comparison.")
            else:
                comparison_data = analyzer.get_comparison_data()
                if comparison_data['comparison_ready']:
                    st.success(f"✅ Ready to compare {comparison_data['total_documents']} documents with {comparison_data['total_fields']} unique fields")
                else:
                    st.info("🔄 Loading comparison interface...")
        
        with tab3:
            st.markdown("## 📊 Advanced Quality Analytics")
            if not analyzer.documents:
                st.info("📈 Upload documents to see quality analytics.")
            else:
                doc_qualities = [doc_data['quality_score'] for doc_data in analyzer.documents.values()]
                quality_df = pd.DataFrame({'Quality': doc_qualities})
                
                fig_quality_dist = px.histogram(
                    quality_df, 
                    x='Quality',
                    nbins=10,
                    title="Document Quality Score Distribution",
                    labels={'count': 'Number of Documents'}
                )
                st.plotly_chart(fig_quality_dist, use_container_width=True)
        
        with tab4:
            st.markdown("## 🕸️ Enhanced Knowledge Graph")
            if len(analyzer.knowledge_graph.nodes) == 0:
                st.warning("⚠️ No knowledge graph data available. Process documents first.")
            else:
                st.info("🔄 Loading knowledge graph visualization...")
    
    except Exception as e:
        st.error(f"❌ Interface error: {e}")
        if st.checkbox("Show error details"):
            st.exception(e)

# System initialization and execution
if __name__ == "__main__":
    try:
        if 'analyzer' not in st.session_state:
            with st.spinner("🚀 Initializing Smart Plant Analyzer..."):
                analyzer = initialize_smart_analyzer()
                if analyzer:
                    st.session_state.analyzer = analyzer
                    logger.info("✅ System initialized successfully")
                else:
                    st.error("❌ Failed to initialize Smart Plant Analyzer")
                    st.stop()
        
        main()
        
    except Exception as e:
        st.error(f"❌ Critical system error: {e}")
        if st.button("🔄 Restart System"):
            st.session_state.clear()
            st.rerun()