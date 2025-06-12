# 완전한 필드 추출기 - 모든 필드 비교 시스템
# 핵심: 모든 필드 강제 추출, 필터링 없음, 완전 비교
import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import io
import re
import hashlib
import uuid  # 🆕 사용자별 고유 ID 생성용
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass, asdict
import traceback
import warnings
import logging
from collections import defaultdict, Counter

# 🆕 사용자 세션 관리 함수 추가
def get_user_session():
    """사용자별 고유 세션 ID 생성 및 관리"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]  # 8자리 고유 ID
        st.session_state.session_start_time = datetime.now()
    
    return st.session_state.user_id

# 🆕 사용자별 안전한 세션 상태 접근
def get_user_state(key: str, default=None):
    """사용자별 격리된 세션 상태 가져오기"""
    user_id = get_user_session()
    user_key = f"{key}_{user_id}"
    return st.session_state.get(user_key, default)

def set_user_state(key: str, value):
    """사용자별 격리된 세션 상태 설정"""
    user_id = get_user_session()
    user_key = f"{key}_{user_id}"
    st.session_state[user_key] = value

def clear_user_state(key: str = None):
    """사용자별 세션 상태 초기화"""
    user_id = get_user_session()
    
    if key:
        # 특정 키만 삭제
        user_key = f"{key}_{user_id}"
        if user_key in st.session_state:
            del st.session_state[user_key]
    else:
        # 해당 사용자의 모든 데이터 삭제
        keys_to_delete = [k for k in st.session_state.keys() 
                         if k.endswith(f"_{user_id}") and k != f"user_id_{user_id}"]
        for k in keys_to_delete:
            del st.session_state[k]

# 경고 필터링
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PDF 처리 라이브러리
try:
   import pdfplumber
   import fitz  # PyMuPDF
   PDF_LIBRARIES_AVAILABLE = True
   logger.info("✅ PDF 처리 라이브러리 로드 완료")
except ImportError:
   PDF_LIBRARIES_AVAILABLE = False
   logger.warning("❌ PDF 처리 라이브러리 없음")

@dataclass
class FieldContext:
   """필드의 컨텍스트 정보"""
   field_name: str
   value: Any
   bbox: Tuple[float, float, float, float]
   page: int
   confidence: float
   context_type: str
   extraction_method: str
   parent_field: Optional[str] = None
   conditions: Optional[Dict] = None
   hierarchy_level: int = 0
   font_info: Optional[Dict] = None
   
   def to_dict(self):
       """안전한 딕셔너리 변환"""
       try:
           result = asdict(self)
           if result.get('conditions') is None:
               result['conditions'] = {}
           if result.get('font_info') is None:
               result['font_info'] = {}
           if result.get('parent_field') is None:
               result['parent_field'] = ""
           
           result['confidence'] = float(result['confidence'])
           result['page'] = int(result['page'])
           result['hierarchy_level'] = int(result['hierarchy_level'])
           
           return result
       except Exception as e:
           logger.error(f"FieldContext.to_dict() 오류: {e}")
           return {
               'field_name': str(self.field_name),
               'value': str(self.value),
               'bbox': (0, 0, 0, 0),
               'page': 0,
               'confidence': 0.50,  # 낮은 기본값
               'context_type': 'simple',
               'extraction_method': 'error_recovery',
               'parent_field': "",
               'conditions': {},
               'hierarchy_level': 0,
               'font_info': {}
           }

class ComprehensiveOntologyManager:
  def __init__(self, ttl_file_path: str = "ontology.ttl"):
      self.ttl_file_path = ttl_file_path
      self.field_definitions = {}
      self.feed_types = set()
      self.condition_types = set()
      self.equipment_properties = {}
      self.process_requirements = {}
      
      # TTL 파일 로드
      self._load_ontology_from_ttl()
      
      # 기본 패턴 (TTL 보완용)
      self.basic_patterns = {
          'numbered_field': r'(\d{1,3})\s+([A-Z][A-Z\s\(\)]+)',
          'temperature': r'(\d+\.?\d*)\s*(°?C|DEG\s*C)',
          'pressure': r'(\d+\.?\d*)\s*(kg/cm2[AG]?)',
          'flow_rate': r'(\d+\.?\d*)\s*(m3/h|m3/hr)',
          'percentage': r'(\d+\.?\d*)\s*(%|wt%)',
          'api_class': r'(API\s+CLASS\s+[A-Z0-9\-]+)',
          'feed_type': r'(AM\s+FEED|AH\s+FEED)',
          'sulfur_content': r'Sulfur\s*\(([0-9\.]+)\s*wt%\)'
      }
      
      logger.info(f"TTL 온톨로지 로드 완료: {len(self.field_definitions)} 필드 정의")
  
  def _load_ontology_from_ttl(self):
      """TTL 파일에서 온톨로지 로드"""
      try:
          if not os.path.exists(self.ttl_file_path):
              logger.warning(f"TTL 파일 없음: {self.ttl_file_path}")
              self._create_fallback_definitions()
              return
          
          with open(self.ttl_file_path, 'r', encoding='utf-8') as f:
              ttl_content = f.read()
          
          # TTL 파싱 (간단한 정규식 기반)
          self._parse_ttl_content(ttl_content)
          
      except Exception as e:
          logger.error(f"TTL 로드 실패: {e}")
          self._create_fallback_definitions()
  
  def _parse_ttl_content(self, content: str):
      """TTL 내용 파싱"""
      try:
          # 데이터 속성 추출
          datatype_props = re.findall(
              r'ex:(\w+)\s+a\s+owl:DatatypeProperty.*?rdfs:range\s+(\w+:\w+)',
              content, re.DOTALL
          )
          
          for prop_name, range_type in datatype_props:
              self.field_definitions[prop_name.upper()] = {
                  'property': prop_name,
                  'type': range_type,
                  'patterns': self._get_patterns_for_property(prop_name)
              }
          
          # FeedType 인스턴스 추출
          feed_types = re.findall(r'ex:(\w+)\s+a\s+ex:FeedType', content)
          self.feed_types = set(feed_types)
          
          # ConditionType 인스턴스 추출
          condition_types = re.findall(r'ex:(\w+)\s+a\s+ex:ConditionType', content)
          self.condition_types = set(condition_types)
          
          # ProcessRequirement 인스턴스와 값 추출
          self._parse_process_requirements(content)
          
          # Equipment 속성 추출
          self._parse_equipment_properties(content)
          
          logger.info(f"TTL 파싱 완료 - 필드: {len(self.field_definitions)}, Feed타입: {len(self.feed_types)}")
          
      except Exception as e:
          logger.error(f"TTL 파싱 오류: {e}")
  
  def _parse_process_requirements(self, content: str):
      """ProcessRequirement 인스턴스 파싱"""
      try:
          # ProcessRequirement 블록 찾기
          pr_blocks = re.findall(
              r'ex:(PR_\w+)\s+a\s+ex:ProcessRequirement\s*;(.*?)\.', 
              content, re.DOTALL
          )
          
          for pr_id, pr_content in pr_blocks:
              pr_data = {'id': pr_id}
              
              # ex:value 추출
              value_match = re.search(r'ex:value\s+([^;]+)', pr_content)
              if value_match:
                  value_str = value_match.group(1).strip()
                  if value_str.startswith('"') and value_str.endswith('"'):
                      pr_data['value'] = value_str[1:-1]
                  else:
                      try:
                          pr_data['value'] = float(value_str)
                      except:
                          pr_data['value'] = value_str
              
              # FeedType 추출
              feed_match = re.search(r'ex:hasFeedType\s+ex:(\w+)', pr_content)
              if feed_match:
                  pr_data['feed_type'] = feed_match.group(1)
              
              # ConditionType 추출
              condition_match = re.search(r'ex:hasCondition\s+ex:(\w+)', pr_content)
              if condition_match:
                  pr_data['condition'] = condition_match.group(1)
              
              # Label 추출
              label_match = re.search(r'rdfs:label\s+"([^"]+)"', pr_content)
              if label_match:
                  pr_data['label'] = label_match.group(1)
              
              self.process_requirements[pr_id] = pr_data
          
      except Exception as e:
          logger.error(f"ProcessRequirement 파싱 오류: {e}")
  
  def _parse_equipment_properties(self, content: str):
      """Equipment 속성 파싱"""
      try:
          # Equipment 블록 찾기
          eq_match = re.search(
              r'ex:(Pump_\w+)\s+a\s+ex:Equipment\s*;(.*?)ex:hasRevision', 
              content, re.DOTALL
          )
          
          if eq_match:
              eq_id, eq_content = eq_match.groups()
              
              # 기본 속성들 추출
              properties = {
                  'pumpType': r'ex:pumpType\s+"([^"]+)"',
                  'driverType': r'ex:driverType\s+"([^"]+)"',
                  'numberRequired': r'ex:numberRequired\s+"([^"]+)"',
                  'dutyType': r'ex:dutyType\s+"([^"]+)"',
                  'itemNo': r'ex:itemNo\s+"([^"]+)"'
              }
              
              eq_data = {'id': eq_id}
              for prop, pattern in properties.items():
                  match = re.search(pattern, eq_content)
                  if match:
                      eq_data[prop] = match.group(1)
              
              self.equipment_properties[eq_id] = eq_data
              
      except Exception as e:
          logger.error(f"Equipment 속성 파싱 오류: {e}")
  
  def _get_patterns_for_property(self, prop_name: str) -> List[str]:
      """속성별 매칭 패턴 반환"""
      prop_lower = prop_name.lower()
      
      pattern_map = {
          'pumptype': ['pump type', 'pump', 'centrifugal'],
          'drivertype': ['driver type', 'motor', 'driver'],
          'numberrequired': ['no required', 'number required', 'required'],
          'dutytype': ['duty', 'continuous', 'intermittent'],
          'value': ['value', 'normal', 'maximum', 'minimum', 'rated'],
          'jobno': ['job no', 'job'],
          'projectname': ['project', 'centrifugal pump'],
          'docno': ['doc no', 'document'],
          'itemno': ['item no', 'equipment'],
          'client': ['client', 'yonsei'],
          'service': ['service', 'overflash', 'pumparound'],
          'pagecount': ['page', 'of'],
          'casingclass': ['casing', 'api class'],
          'estimatedshutoff': ['estimated shutoff', 'shutoff']
      }
      
      return pattern_map.get(prop_lower, [prop_lower])
  
  def _create_fallback_definitions(self):
      """TTL 로드 실패시 기본 정의 생성"""
      self.field_definitions = {
          'PUMP_TYPE': {'property': 'pumpType', 'type': 'string', 'patterns': ['pump type', 'pump']},
          'DRIVER_TYPE': {'property': 'driverType', 'type': 'string', 'patterns': ['driver type', 'motor']},
          'CAPACITY': {'property': 'value', 'type': 'decimal', 'patterns': ['capacity', 'm3/hr']},
          'TEMPERATURE': {'property': 'value', 'type': 'decimal', 'patterns': ['temperature', '℃']},
          'PRESSURE': {'property': 'value', 'type': 'decimal', 'patterns': ['pressure', 'kg/cm2']},
          'LIQUID_NAME': {'property': 'value', 'type': 'string', 'patterns': ['liquid name', 'overflash']}
      }
      
      self.feed_types = {'AM_Feed', 'AH_Feed'}
      self.condition_types = {'Normal', 'Maximum', 'Minimum', 'Rated'}
      
      logger.info("기본 정의 생성 완료")
  
  def map_field_to_ontology(self, field_candidate: str) -> str:
      """필드명을 온톨로지 구조에 매핑"""
      if not field_candidate:
          return "UNKNOWN_FIELD"
      
      field_str = str(field_candidate).strip().upper()
      
      # TTL에서 로드한 정의와 매칭
      for ont_field, definition in self.field_definitions.items():
          patterns = definition.get('patterns', [])
          
          for pattern in patterns:
              if pattern.upper() in field_str or field_str in pattern.upper():
                  return ont_field
      
      # 번호 제거 후 매핑 시도
      normalized = re.sub(r'^\d+\s*', '', field_str)
      normalized = re.sub(r'[^\w\s]', '_', normalized)
      normalized = re.sub(r'\s+', '_', normalized.strip())
      
      return normalized if normalized else "UNNAMED_FIELD"
  
  def get_expected_values_for_field(self, field_name: str, feed_type: str = None, condition: str = None) -> List:
      """필드에 대한 예상 값들 반환 (TTL 기반)"""
      expected_values = []
      
      try:
          # ProcessRequirement에서 해당 필드의 예상 값 찾기
          for pr_id, pr_data in self.process_requirements.items():
              if self._matches_field_criteria(pr_data, field_name, feed_type, condition):
                  if 'value' in pr_data:
                      expected_values.append(pr_data['value'])
          
          # Equipment 속성에서 찾기
          for eq_id, eq_data in self.equipment_properties.items():
              if field_name.upper() in ['PUMP_TYPE', 'PUMPTYPE']:
                  if 'pumpType' in eq_data:
                      expected_values.append(eq_data['pumpType'])
              elif field_name.upper() in ['DRIVER_TYPE', 'DRIVERTYPE']:
                  if 'driverType' in eq_data:
                      expected_values.append(eq_data['driverType'])
      
      except Exception as e:
          logger.debug(f"예상 값 검색 오류: {e}")
      
      return expected_values
  
  def _matches_field_criteria(self, pr_data: Dict, field_name: str, feed_type: str, condition: str) -> bool:
      """ProcessRequirement가 필드 조건과 일치하는지 확인"""
      try:
          # 라벨 기반 매칭
          label = pr_data.get('label', '').upper()
          field_upper = field_name.upper()
          
          field_keywords = {
              'TEMPERATURE': ['TEMP', 'TEMPERATURE'],
              'CAPACITY': ['CAPACITY'],
              'PRESSURE': ['PRESSURE'],
              'VAPOR': ['VAPOR'],
              'GRAVITY': ['GRAVITY'],
              'VISCOSITY': ['VISCOSITY'],
              'NPSH': ['NPSH'],
              'CORROSION': ['CORROSION']
          }
          
          # 필드 타입 매칭
          field_matched = False
          for key, keywords in field_keywords.items():
              if key in field_upper:
                  if any(keyword in label for keyword in keywords):
                      field_matched = True
                      break
          
          if not field_matched:
              return False
          
          # Feed 타입 매칭
          if feed_type:
              pr_feed = pr_data.get('feed_type', '')
              if feed_type.upper() not in pr_feed.upper():
                  return False
          
          # Condition 타입 매칭
          if condition:
              pr_condition = pr_data.get('condition', '')
              if condition.upper() not in pr_condition.upper():
                  return False
          
          return True
          
      except Exception:
          return False
  
  def get_field_validation_info(self, field_name: str) -> Dict:
      """필드 검증 정보 반환"""
      field_upper = field_name.upper()
      
      if field_upper in self.field_definitions:
          definition = self.field_definitions[field_upper]
          return {
              'exists_in_ontology': True,
              'property_type': definition.get('type', 'unknown'),
              'expected_patterns': definition.get('patterns', []),
              'validation_rules': self._get_validation_rules(field_upper)
          }
      
      return {
          'exists_in_ontology': False,
          'property_type': 'unknown',
          'expected_patterns': [],
          'validation_rules': {}
      }
  
  def _get_validation_rules(self, field_name: str) -> Dict:
      """필드별 검증 규칙 반환"""
      rules = {
          'PUMP_TYPE': {'expected_values': ['CENTRIFUGAL'], 'type': 'string'},
          'DRIVER_TYPE': {'expected_values': ['MOTOR'], 'type': 'string'},
          'DUTY_TYPE': {'expected_values': ['CONTINUOUS'], 'type': 'string'},
          'TEMPERATURE': {'min': 0, 'max': 1000, 'type': 'numeric', 'unit': '℃'},
          'CAPACITY': {'min': 0, 'max': 10000, 'type': 'numeric', 'unit': 'm3/h'},
          'PRESSURE': {'min': 0, 'max': 100, 'type': 'numeric', 'unit': 'kg/cm2'}
      }
      
      return rules.get(field_name, {})

class MaximumExtractorProcessor:
  def __init__(self, ontology_manager):
      self.ontology = ontology_manager
      self.confidence_threshold = 0.20
      self.min_confidence = 0.25
      
  def extract_all_possible_fields(self, file_content: bytes, filename: str) -> Dict[str, Any]:
      try:
          logger.info(f"구조적 필드 추출 시작: {filename}")
          
          metadata = self._extract_metadata(file_content)
          all_fields = {}
          
          # 문서 구조 분석 (좌표 기반)
          doc_structure = self._analyze_document_structure_spatial(file_content)
          
          # 핵심: 좌표 기반 구조적 추출
          structural_fields = self._extract_structural_fields(file_content)
          all_fields.update(structural_fields)
          
          # TTL 기반 검증 및 보완
          validated_fields = self._validate_and_enhance_with_ttl(all_fields)
          
          doc_id = f"doc_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
          processed_fields = self._process_final_fields(validated_fields, doc_id)
          
          return {
              'doc_id': doc_id,
              'extracted_fields': processed_fields,
              'metadata': metadata,
              'total_confidence': self._calculate_avg_confidence(processed_fields),
              'extraction_summary': {
                  'total_fields': len(processed_fields),
                  'structural_fields': sum(1 for f in processed_fields if f.get('structural', False)),
                  'ttl_validated_fields': sum(1 for f in processed_fields if f.get('ttl_validated', False)),
                  'extraction_methods_used': list(set(f.get('extraction_method', '') for f in processed_fields))
              }
          }
          
      except Exception as e:
          logger.error(f"구조적 추출 실패: {e}")
          return self._create_error_result(filename, str(e))
  
  def _analyze_document_structure_spatial(self, file_content: bytes) -> Dict:
      """좌표 기반 문서 구조 분석"""
      if not PDF_LIBRARIES_AVAILABLE:
          return {'is_structured_table': False}
      
      try:
          with pdfplumber.open(io.BytesIO(file_content)) as pdf:
              page = pdf.pages[0]  # 첫 페이지 분석
              
              # 단어들의 좌표 정보 추출
              words = page.extract_words()
              
              # 번호 매겨진 구조 감지
              numbered_structure = self._detect_numbered_structure(words)
              
              # 테이블 구조 감지
              table_structure = self._detect_table_structure(words)
              
              # 컬럼 구조 분석
              column_analysis = self._analyze_column_structure(words)
              
              return {
                  'is_structured_table': numbered_structure['has_numbered_fields'],
                  'numbered_field_count': numbered_structure['count'],
                  'column_count': column_analysis['estimated_columns'],
                  'table_bounds': table_structure['bounds'],
                  'field_positions': numbered_structure['positions']
              }
      
      except Exception as e:
          logger.error(f"구조 분석 오류: {e}")
          return {'is_structured_table': False}
  
  def _detect_numbered_structure(self, words: List[Dict]) -> Dict:
      """번호 매겨진 구조 감지"""
      numbered_fields = []
      
      for word in words:
          text = word['text'].strip()
          # 01, 02, 03... 형태의 번호 감지
          if re.match(r'^\d{2}$', text):
              numbered_fields.append({
                  'number': text,
                  'x': word['x0'],
                  'y': word['top'],
                  'bbox': (word['x0'], word['top'], word['x1'], word['bottom'])
              })
      
      return {
          'has_numbered_fields': len(numbered_fields) > 5,
          'count': len(numbered_fields),
          'positions': numbered_fields
      }
  
  def _detect_table_structure(self, words: List[Dict]) -> Dict:
      """테이블 구조 감지"""
      if not words:
          return {'bounds': None}
      
      # 좌표 범위 계산
      min_x = min(word['x0'] for word in words)
      max_x = max(word['x1'] for word in words)
      min_y = min(word['top'] for word in words)
      max_y = max(word['bottom'] for word in words)
      
      return {
          'bounds': (min_x, min_y, max_x, max_y),
          'width': max_x - min_x,
          'height': max_y - min_y
      }
  
  def _analyze_column_structure(self, words: List[Dict]) -> Dict:
      """컬럼 구조 분석"""
      if not words:
          return {'estimated_columns': 1}
      
      # X 좌표별 단어 그룹핑
      x_positions = [word['x0'] for word in words]
      
      # 주요 X 좌표 클러스터링 (간단한 방법)
      sorted_x = sorted(set(x_positions))
      column_boundaries = []
      
      if len(sorted_x) > 1:
          threshold = 50  # 50px 이상 차이나면 다른 컬럼
          prev_x = sorted_x[0]
          column_boundaries.append(prev_x)
          
          for x in sorted_x[1:]:
              if x - prev_x > threshold:
                  column_boundaries.append(x)
                  prev_x = x
      
      return {
          'estimated_columns': len(column_boundaries),
          'column_boundaries': column_boundaries
      }
  
  def _extract_structural_fields(self, file_content: bytes) -> Dict[str, FieldContext]:
      """구조적 필드 추출 (내 방식 적용)"""
      results = {}
      
      if not PDF_LIBRARIES_AVAILABLE:
          return results
      
      try:
          with pdfplumber.open(io.BytesIO(file_content)) as pdf:
              page = pdf.pages[0]
              words = page.extract_words()
              tables = page.extract_tables()
              
              # 1. 번호-필드명-값 관계 매핑
              numbered_mappings = self._extract_numbered_field_mappings(words)
              results.update(numbered_mappings)
              
              # 2. 테이블 기반 구조적 추출
              table_mappings = self._extract_table_structure_mappings(tables, words)
              results.update(table_mappings)
              
              # 3. 공간적 관계 기반 추출
              spatial_mappings = self._extract_spatial_relationships(words)
              results.update(spatial_mappings)
              
      except Exception as e:
          logger.error(f"구조적 추출 오류: {e}")
      
      return results
  
  def _extract_numbered_field_mappings(self, words: List[Dict]) -> Dict[str, FieldContext]:
      """번호-필드명-값 관계 매핑"""
      results = {}
      
      # 번호별로 필드명과 값 찾기
      for i, word in enumerate(words):
          text = word['text'].strip()
          
          # 번호 감지 (01, 02, 03...)
          if re.match(r'^\d{2}$', text):
              number = text
              
              # 같은 행에서 필드명 찾기
              field_name = self._find_field_name_in_row(words, i, word['top'])
              
              if field_name:
                  # 같은 행에서 값 찾기
                  values = self._find_values_in_row(words, word['top'], word['x1'])
                  
                  for value_idx, value in enumerate(values):
                      if value and len(value.strip()) > 0:
                          field_id = f"numbered_{number}_{value_idx}"
                          
                          # TTL과 매칭하여 신뢰도 계산
                          confidence = self._calculate_structural_confidence(number, field_name, value)
                          
                          results[field_id] = FieldContext(
                              field_name=f"{number}_{field_name}",
                              value=value.strip(),
                              bbox=word.get('bbox', (0, 0, 0, 0)),
                              page=0,
                              confidence=confidence,
                              context_type='numbered_structure',
                              extraction_method='structural_numbered',
                              conditions={
                                  'number': number,
                                  'structural': True,
                                  'field_name_raw': field_name
                              }
                          )
      
      return results
  
  def _find_field_name_in_row(self, words: List[Dict], start_idx: int, target_y: float) -> str:
      """같은 행에서 필드명 찾기"""
      y_tolerance = 5  # Y 좌표 허용 오차
      field_parts = []
      
      # 번호 다음 단어들을 필드명으로 수집
      for i in range(start_idx + 1, min(start_idx + 10, len(words))):
          word = words[i]
          
          # 같은 행인지 확인
          if abs(word['top'] - target_y) <= y_tolerance:
              text = word['text'].strip()
              
              # 필드명 패턴 (대문자, 특정 키워드)
              if re.match(r'^[A-Z]', text) and len(text) > 1:
                  field_parts.append(text)
              elif text in ['TYPE', 'PRESSURE', 'TEMPERATURE', 'CAPACITY', 'REQUIRED']:
                  field_parts.append(text)
              else:
                  # 값으로 보이면 필드명 수집 중단
                  if re.match(r'^\d+\.?\d*$', text) or text in [':', '=']:
                      break
      
      return ' '.join(field_parts) if field_parts else ""
  
  def _find_values_in_row(self, words: List[Dict], target_y: float, start_x: float) -> List[str]:
      """같은 행에서 값들 찾기"""
      y_tolerance = 5
      values = []
      
      # X 좌표가 start_x보다 오른쪽에 있는 단어들 수집
      row_words = []
      for word in words:
          if (abs(word['top'] - target_y) <= y_tolerance and 
              word['x0'] > start_x):
              row_words.append(word)
      
      # X 좌표 순으로 정렬
      row_words.sort(key=lambda w: w['x0'])
      
      # 값들 추출
      current_value = []
      for word in row_words:
          text = word['text'].strip()
          
          # 구분자나 새로운 값 시작점
          if text in [':', '=', '■', '□'] or (current_value and self._is_new_value_start(text)):
              if current_value:
                  values.append(' '.join(current_value))
                  current_value = []
              
              if text not in [':', '=', '■', '□']:
                  current_value.append(text)
          else:
              current_value.append(text)
      
      # 마지막 값 추가
      if current_value:
          values.append(' '.join(current_value))
      
      return values
  
  def _is_new_value_start(self, text: str) -> bool:
      """새로운 값의 시작점인지 판단"""
      # 숫자로 시작하거나 특정 키워드면 새로운 값
      return (re.match(r'^\d', text) or 
              text.upper() in ['CENTRIFUGAL', 'MOTOR', 'ONE', 'CONTINUOUS', 'HVGO', 'OVERFLASH'])
  
  def _extract_table_structure_mappings(self, tables: List, words: List[Dict]) -> Dict[str, FieldContext]:
      """테이블 구조 기반 매핑"""
      results = {}
      
      if not tables:
          return results
      
      for table_idx, table in enumerate(tables):
          if not table or len(table) < 2:
              continue
          
          # 테이블의 구조 분석
          structure = self._analyze_table_layout(table)
          
          if structure['type'] == 'numbered_field_value':
              # 번호-필드명-값 구조 처리
              for row_idx, row in enumerate(table):
                  if len(row) >= 3:
                      number, field_name, *values = row
                      
                      if number and field_name:
                          for val_idx, value in enumerate(values):
                              if value and str(value).strip():
                                  field_id = f"table_{table_idx}_{row_idx}_{val_idx}"
                                  
                                  confidence = self._calculate_structural_confidence(
                                      str(number), str(field_name), str(value)
                                  )
                                  
                                  results[field_id] = FieldContext(
                                      field_name=f"{number}_{field_name}",
                                      value=str(value).strip(),
                                      bbox=(0, 0, 0, 0),
                                      page=0,
                                      confidence=confidence,
                                      context_type='table_structure',
                                      extraction_method='structural_table',
                                      conditions={
                                          'table_idx': table_idx,
                                          'structural': True,
                                          'column_idx': val_idx + 2
                                      }
                                  )
      
      return results
  
  def _analyze_table_layout(self, table: List[List]) -> Dict:
      """테이블 레이아웃 분석"""
      if not table or len(table) < 2:
          return {'type': 'unknown'}
      
      first_row = table[0]
      if not first_row or len(first_row) < 2:
          return {'type': 'unknown'}
      
      # 첫 번째 컬럼이 번호인지 확인
      first_col_numbers = 0
      for row in table[:10]:  # 처음 10행만 확인
          if row and len(row) > 0 and row[0]:
              if re.match(r'^\d{1,3}\.?$', str(row[0]).strip()):
                  first_col_numbers += 1
      
      if first_col_numbers >= 5:  # 번호가 5개 이상이면
          return {
              'type': 'numbered_field_value',
              'number_column': 0,
              'field_column': 1,
              'value_columns': list(range(2, len(first_row)))
          }
      
      return {'type': 'generic'}
  
  def _extract_spatial_relationships(self, words: List[Dict]) -> Dict[str, FieldContext]:
      """공간적 관계 기반 추출"""
      results = {}
      
      # 특정 키워드 근처의 값들 찾기
      key_terms = {
          'PUMP TYPE': ['CENTRIFUGAL'],
          'CAPACITY': [r'\d+\.?\d*\s*m3/h'],
          'TEMPERATURE': [r'\d+\.?\d*\s*°?C'],
          'LIQUID NAME': ['HVGO', 'OVERFLASH'],
          'AM FEED': [r'\d+\.?\d*'],
          'AH FEED': [r'\d+\.?\d*']
      }
      
      for term, value_patterns in key_terms.items():
          term_positions = self._find_term_positions(words, term)
          
          for pos in term_positions:
              nearby_values = self._find_nearby_values(words, pos, value_patterns)
              
              for val_idx, value in enumerate(nearby_values):
                  field_id = f"spatial_{term}_{val_idx}"
                  
                  # TTL 기반 신뢰도
                  confidence = self._calculate_spatial_confidence(term, value)
                  
                  results[field_id] = FieldContext(
                      field_name=term.replace(' ', '_'),
                      value=value,
                      bbox=pos.get('bbox', (0, 0, 0, 0)),
                      page=0,
                      confidence=confidence,
                      context_type='spatial_relationship',
                      extraction_method='structural_spatial',
                      conditions={'structural': True, 'spatial_term': term}
                  )
      
      return results
  
  def _find_term_positions(self, words: List[Dict], term: str) -> List[Dict]:
      """특정 용어의 위치 찾기"""
      positions = []
      term_words = term.split()
      
      for i in range(len(words) - len(term_words) + 1):
          match = True
          for j, term_word in enumerate(term_words):
              if words[i + j]['text'].upper() != term_word.upper():
                  match = False
                  break
          
          if match:
              positions.append({
                  'start_idx': i,
                  'end_idx': i + len(term_words) - 1,
                  'bbox': (
                      words[i]['x0'],
                      words[i]['top'],
                      words[i + len(term_words) - 1]['x1'],
                      words[i + len(term_words) - 1]['bottom']
                  )
              })
      
      return positions
  
  def _find_nearby_values(self, words: List[Dict], term_pos: Dict, value_patterns: List[str]) -> List[str]:
      """근처 값들 찾기"""
      values = []
      search_radius = 200  # 픽셀
      
      term_x = term_pos['bbox'][0]
      term_y = term_pos['bbox'][1]
      
      for word in words:
          distance = ((word['x0'] - term_x) ** 2 + (word['top'] - term_y) ** 2) ** 0.5
          
          if distance <= search_radius:
              text = word['text']
              
              for pattern in value_patterns:
                  if re.search(pattern, text, re.IGNORECASE):
                      values.append(text)
                      break
      
      return values
  
  def _calculate_structural_confidence(self, number: str, field_name: str, value: str) -> float:
      """구조적 추출 신뢰도 계산"""
      base_confidence = 0.70  # 구조적 추출은 기본적으로 높은 신뢰도
      
      # TTL과 매칭 확인
      expected_values = self.ontology.get_expected_values_for_field(field_name)
      if expected_values and value in [str(v) for v in expected_values]:
          base_confidence += 0.20
      
      # 번호 패턴 확인
      if re.match(r'^\d{2}$', number):
          base_confidence += 0.05
      
      return min(0.95, base_confidence)
  
  def _calculate_spatial_confidence(self, term: str, value: str) -> float:
      """공간적 추출 신뢰도 계산"""
      base_confidence = 0.60
      
      # TTL 기반 검증
      expected_values = self.ontology.get_expected_values_for_field(term)
      if expected_values and value in [str(v) for v in expected_values]:
          base_confidence += 0.25
      
      return min(0.90, base_confidence)
  
  def _validate_and_enhance_with_ttl(self, structural_fields: Dict[str, FieldContext]) -> Dict[str, FieldContext]:
      """TTL로 검증 및 보완"""
      validated = {}
      
      for field_id, context in structural_fields.items():
          # 온톨로지 매핑
          mapped_name = self.ontology.map_field_to_ontology(context.field_name)
          context.field_name = mapped_name
          
          # TTL 검증
          validation_info = self.ontology.get_field_validation_info(mapped_name)
          if validation_info['exists_in_ontology']:
              context.confidence = max(context.confidence, 0.70)
              context.conditions = context.conditions or {}
              context.conditions['ttl_validated'] = True
          
          validated[field_id] = context
      
      return validated
  
  def _process_final_fields(self, fields: Dict[str, FieldContext], doc_id: str) -> List[Dict]:
      processed = []
      for field_id, context in fields.items():
          try:
              field_data = context.to_dict()
              field_data['doc_id'] = doc_id
              processed.append(field_data)
          except Exception as e:
              logger.debug(f"필드 처리 오류 {field_id}: {e}")
      return processed
  
  def _calculate_avg_confidence(self, fields: List[Dict]) -> float:
      if not fields:
          return 0.0
      confidences = [f.get('confidence', self.min_confidence) for f in fields]
      return float(np.mean(confidences))
  
  def _extract_metadata(self, file_content: bytes) -> Dict:
      try:
          if PDF_LIBRARIES_AVAILABLE:
              doc = fitz.open(stream=file_content, filetype="pdf")
              return {'page_count': len(doc)}
          return {'page_count': 1}
      except:
          return {'page_count': 1}
  
  def _create_error_result(self, filename: str, error_msg: str) -> Dict:
      return {
          'doc_id': f"error_{hashlib.md5(filename.encode()).hexdigest()[:8]}",
          'extracted_fields': [],
          'metadata': {'error': error_msg, 'page_count': 1},
          'total_confidence': 0.0,
          'extraction_summary': {
              'total_fields': 0,
              'error': error_msg
          }
      }

# 🔧 완전 비교 시스템
class CompleteComparisonSystem:
  def __init__(self, db_manager):
      self.db = db_manager
      
      self.normalization_patterns = {
          'numeric_with_unit': r'(\d+\.?\d*)\s*([A-Za-z°/%]+)',
          'boolean_yes_no': r'(yes|no|y|n)\b',
          'boolean_manual_auto': r'(manual|automatic|auto|hand)',
          'temperature': r'(\d+\.?\d*)\s*(°?[CF]|deg)',
          'pressure': r'(\d+\.?\d*)\s*(kg/cm2[AG]?|psi|bar)',
          'flow_rate': r'(\d+\.?\d*)\s*(m3/h|m3/hr|gpm)',
          'percentage': r'(\d+\.?\d*)\s*(%|wt%)',
          'api_class': r'(api\s+class\s+[a-z0-9\-]+)'
      }
      
      self.unit_conversions = {
          'temperature': {'°f': 'celsius', 'deg': 'celsius', '°c': 'celsius'},
          'pressure': {'psi': 'kg/cm2', 'bar': 'kg/cm2', 'kg/cm2a': 'kg/cm2', 'kg/cm2g': 'kg/cm2'},
          'flow': {'gpm': 'm3/h', 'm3/hr': 'm3/h', 'l/s': 'm3/h'}
      }
      
      logger.info("비교 시스템 초기화 완료")
  
  def generate_complete_comparison_matrix(self) -> pd.DataFrame:
      try:
          # TTL 매치된 필드들만 우선적으로 비교
          ttl_fields_query = """
              SELECT 
                  d.filename,
                  ef.field_name,
                  ef.field_value,
                  ef.confidence,
                  ef.extraction_method,
                  ef.context_type,
                  JSON_EXTRACT(ef.conditions, '$.ttl_matched') as ttl_matched
              FROM extracted_fields ef
              JOIN documents d ON ef.doc_id = d.doc_id
              WHERE ef.field_name IS NOT NULL AND ef.field_value IS NOT NULL
              ORDER BY 
                  CASE WHEN JSON_EXTRACT(ef.conditions, '$.ttl_matched') = true THEN 0 ELSE 1 END,
                  ef.confidence DESC,
                  ef.field_name, d.filename
          """
          
          all_fields_df = self.db.execute_query(ttl_fields_query)
          
          if all_fields_df.empty:
              logger.warning("추출된 필드가 없습니다")
              return pd.DataFrame()
          
          # TTL 매치 통계
          ttl_matched_count = all_fields_df[all_fields_df.get('ttl_matched', False) == True].shape[0] if 'ttl_matched' in all_fields_df.columns else 0
          logger.info(f"TTL 매치된 필드: {ttl_matched_count}/{len(all_fields_df)}")
          
          field_groups = self._group_similar_fields(all_fields_df)
          comparison_matrix = []
          
          for canonical_field, field_variations in field_groups.items():
              if not canonical_field:
                  continue
                  
              row_data = {'Field_Name': canonical_field}
              
              document_values = {}
              for _, field_data in field_variations.iterrows():
                  doc_name = field_data.get('filename')
                  raw_value = field_data.get('field_value')
                  confidence = field_data.get('confidence', 0.0)
                  ttl_matched = field_data.get('ttl_matched', False)
                  
                  if not doc_name:
                      continue
                  
                  normalized_value = self._normalize_value(raw_value)
                  
                  if doc_name not in document_values:
                      document_values[doc_name] = []
                  
                  document_values[doc_name].append({
                      'raw': raw_value if raw_value is not None else "",
                      'normalized': normalized_value,
                      'confidence': confidence,
                      'ttl_matched': ttl_matched
                  })
              
              all_documents = list(set(all_fields_df['filename'].dropna().unique()))
              
              for doc_name in all_documents:
                  if doc_name in document_values:
                      best_value = self._select_best_value(document_values[doc_name])
                      row_data[f"{doc_name}_Value"] = best_value['raw']
                      
                      normalized_str = self._convert_normalized_to_string(best_value['normalized'])
                      row_data[f"{doc_name}_Normalized"] = normalized_str
                      
                      row_data[f"{doc_name}_Confidence"] = best_value['confidence']
                      row_data[f"{doc_name}_TTL_Matched"] = best_value.get('ttl_matched', False)
                      row_data[f"{doc_name}_Status"] = "✅ 있음"
                  else:
                      row_data[f"{doc_name}_Value"] = "❌ 값 없음"
                      row_data[f"{doc_name}_Normalized"] = ""
                      row_data[f"{doc_name}_Confidence"] = 0.0
                      row_data[f"{doc_name}_TTL_Matched"] = False
                      row_data[f"{doc_name}_Status"] = "❌ 없음"
              
              comparison_result = self._analyze_field_comparison(document_values, all_documents)
              row_data.update(comparison_result)
              
              comparison_matrix.append(row_data)
          
          return pd.DataFrame(comparison_matrix)
          
      except Exception as e:
          logger.error(f"비교 매트릭스 생성 오류: {e}")
          return pd.DataFrame()
  
  def _convert_normalized_to_string(self, normalized_value) -> str:
      try:
          if normalized_value is None:
              return ""
          
          if isinstance(normalized_value, dict):
              value_type = normalized_value.get('type', 'unknown')
              
              if value_type == 'numeric':
                  value = normalized_value.get('value', '')
                  unit = normalized_value.get('unit', '')
                  if unit:
                      return f"{value} {unit}"
                  else:
                      return str(value)
              elif value_type == 'boolean':
                  return str(normalized_value.get('value', ''))
              elif value_type == 'text':
                  return str(normalized_value.get('value', ''))
              elif value_type == 'empty':
                  return ""
              else:
                  return str(normalized_value.get('original', ''))
          else:
              return str(normalized_value)
              
      except Exception as e:
          logger.debug(f"정규화 값 문자열 변환 오류: {e}")
          return ""
  
  def _group_similar_fields(self, fields_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
      field_groups = {}
      processed_fields = set()
      
      unique_fields = fields_df['field_name'].dropna().unique()
      
      for field in unique_fields:
          if not field or field in processed_fields:
              continue
          
          similar_fields = [field]
          field_data = fields_df[fields_df['field_name'] == field]
          
          for other_field in unique_fields:
              if (other_field and other_field != field and 
                  other_field not in processed_fields and 
                  self._are_fields_similar(field, other_field)):
                  similar_fields.append(other_field)
                  processed_fields.add(other_field)
          
          canonical_field = self._select_canonical_field_name(similar_fields)
          if canonical_field:
              group_data = pd.concat([fields_df[fields_df['field_name'] == f] for f in similar_fields])
              field_groups[canonical_field] = group_data
          
          processed_fields.add(field)
      
      return field_groups
  
  def _are_fields_similar(self, field1: str, field2: str) -> bool:
      if not field1 or not field2:
          return False
      
      try:
          norm1 = self._normalize_field_name(field1)
          norm2 = self._normalize_field_name(field2)
          
          if not norm1 or not norm2:
              return False
          
          if norm1 == norm2:
              return True
          
          if norm1 in norm2 or norm2 in norm1:
              return True
          
          words1 = set(norm1.split('_'))
          words2 = set(norm2.split('_'))
          
          if words1 and words2:
              intersection = len(words1 & words2)
              union = len(words1 | words2)
              similarity = intersection / union if union > 0 else 0
              return similarity > 0.6
          
          return False
      except Exception as e:
          logger.debug(f"필드 유사도 비교 오류: {e}")
          return False
  
  def _normalize_field_name(self, field_name) -> str:
      if field_name is None:
          return ""
      
      try:
          normalized = str(field_name).upper()
          normalized = re.sub(r'[^\w\s]', '_', normalized)
          normalized = re.sub(r'\s+', '_', normalized)
          normalized = re.sub(r'_+', '_', normalized)
          return normalized.strip('_')
      except Exception as e:
          logger.debug(f"필드명 정규화 오류: {e}")
          return ""
  
  def _select_canonical_field_name(self, field_names: List[str]) -> str:
      valid_names = [name for name in field_names if name]
      
      if not valid_names:
          return ""
      
      if len(valid_names) == 1:
          return valid_names[0]
      
      try:
          scored_names = []
          for name in valid_names:
              score = len(name) + (10 if '_' in name else 0)
              scored_names.append((score, name))
          
          scored_names.sort(reverse=True)
          return scored_names[0][1]
      except Exception as e:
          logger.debug(f"대표 필드명 선택 오류: {e}")
          return valid_names[0]
  
  def _normalize_value(self, raw_value) -> Dict:
      if raw_value is None:
          return {'type': 'empty', 'value': None, 'unit': None}
      
      try:
          value_str = str(raw_value).strip()
      except Exception:
          return {'type': 'empty', 'value': None, 'unit': None}
      
      if not value_str or value_str == "❌ 값 없음":
          return {'type': 'empty', 'value': None, 'unit': None}
      
      value_lower = value_str.lower()
      
      for pattern_name, pattern in self.normalization_patterns.items():
          try:
              match = re.search(pattern, value_lower, re.IGNORECASE)
              if match:
                  groups = match.groups()
                  if pattern_name == 'numeric_with_unit' and len(groups) >= 2:
                      try:
                          numeric_value = float(groups[0])
                          unit = groups[1].lower()
                          return {
                              'type': 'numeric',
                              'value': numeric_value,
                              'unit': unit,
                              'original': value_str
                          }
                      except (ValueError, TypeError):
                          continue
                  elif pattern_name.startswith('boolean_'):
                      return {
                          'type': 'boolean',
                          'value': groups[0].lower(),
                          'unit': None,
                          'original': value_str
                      }
          except Exception:
              continue
      
      try:
          numeric_value = float(value_lower)
          return {
              'type': 'numeric',
              'value': numeric_value,
              'unit': None,
              'original': value_str
          }
      except (ValueError, TypeError):
          pass
      
      return {
          'type': 'text',
          'value': value_lower,
          'unit': None,
          'original': value_str
      }
  
  def _select_best_value(self, value_candidates: List[Dict]) -> Dict:
      if not value_candidates:
          return {'raw': "", 'normalized': {'type': 'empty', 'value': None, 'unit': None}, 'confidence': 0.0, 'ttl_matched': False}
      
      if len(value_candidates) == 1:
          return value_candidates[0]
      
      try:
          # TTL 매치된 것 우선, 그 다음 신뢰도
          def sort_key(x):
              ttl_matched = x.get('ttl_matched', False)
              confidence = x.get('confidence', 0)
              return (ttl_matched, confidence)
          
          sorted_candidates = sorted(value_candidates, key=sort_key, reverse=True)
          return sorted_candidates[0]
      except Exception:
          return value_candidates[0]
  
  def _analyze_field_comparison(self, document_values: Dict, all_documents: List[str]) -> Dict:
      try:
          non_empty_docs = [doc for doc in all_documents if doc in document_values]
          
          if len(non_empty_docs) == 0:
              return {
                  'Comparison_Result': "⚫ 모든 문서에서 없음",
                  'Documents_With_Field': 0,
                  'Total_Documents': len(all_documents),
                  'Similarity_Score': 0.0,
                  'TTL_Match_Count': 0
              }
          
          if len(non_empty_docs) == 1:
              best_value = self._select_best_value(document_values[non_empty_docs[0]])
              ttl_count = 1 if best_value.get('ttl_matched', False) else 0
              return {
                  'Comparison_Result': "🟡 1개 문서에만 있음",
                  'Documents_With_Field': 1,
                  'Total_Documents': len(all_documents),
                  'Similarity_Score': 0.0,
                  'TTL_Match_Count': ttl_count
              }
          
          normalized_values = []
          ttl_match_count = 0
          
          for doc in non_empty_docs:
              best_value = self._select_best_value(document_values[doc])
              normalized_values.append(best_value['normalized'])
              if best_value.get('ttl_matched', False):
                  ttl_match_count += 1
          
          similarity_score = self._calculate_value_similarity(normalized_values)
          
          if similarity_score >= 0.95:
              result = "🟢 모든 값 동일"
          elif similarity_score >= 0.8:
              result = "🟡 값 유사 (약간 차이)"
          elif similarity_score >= 0.5:
              result = "🟠 값 차이 (중간)"
          else:
              unique_count = len(set(str(v.get('value', '')) for v in normalized_values if v))
              result = f"🔴 값 차이 큼 ({unique_count}개 다른 값)"
          
          # TTL 매치 정보 추가
          if ttl_match_count > 0:
              result += f" [TTL:{ttl_match_count}/{len(non_empty_docs)}]"
          
          return {
              'Comparison_Result': result,
              'Documents_With_Field': len(non_empty_docs),
              'Total_Documents': len(all_documents),
              'Similarity_Score': similarity_score,
              'TTL_Match_Count': ttl_match_count
          }
      except Exception as e:
          logger.error(f"필드 비교 분석 오류: {e}")
          return {
              'Comparison_Result': "❌ 분석 오류",
              'Documents_With_Field': 0,
              'Total_Documents': len(all_documents),
              'Similarity_Score': 0.0,
              'TTL_Match_Count': 0
          }
  
  def _calculate_value_similarity(self, normalized_values: List[Dict]) -> float:
      try:
          if len(normalized_values) <= 1:
              return 1.0
          
          valid_values = [v for v in normalized_values if v and isinstance(v, dict)]
          
          if not valid_values:
              return 0.0
          
          value_types = [v.get('type', 'unknown') for v in valid_values]
          
          if len(set(value_types)) > 1:
              return 0.0
          
          value_type = value_types[0] if value_types else 'unknown'
          
          if value_type == 'numeric':
              return self._calculate_numeric_similarity(valid_values)
          elif value_type == 'boolean':
              return self._calculate_boolean_similarity(valid_values)
          elif value_type == 'text':
              return self._calculate_text_similarity(valid_values)
          else:
              return 0.5
      except Exception as e:
          logger.debug(f"값 유사도 계산 오류: {e}")
          return 0.0
  
  def _calculate_numeric_similarity(self, values: List[Dict]) -> float:
      try:
          numeric_values = []
          units = []
          
          for v in values:
              if v.get('value') is not None:
                  numeric_values.append(v['value'])
                  units.append(v.get('unit', ''))
          
          if not numeric_values:
              return 0.0
          
          unique_units = set(units)
          if len(unique_units) > 1:
              converted_units = set()
              for unit in unique_units:
                  converted = self._convert_unit(unit)
                  converted_units.add(converted)
              
              if len(converted_units) > 1:
                  return 0.3
          
          if len(set(numeric_values)) == 1:
              return 1.0
          
          min_val = min(numeric_values)
          max_val = max(numeric_values)
          
          if min_val == 0:
              return 0.5
          
          relative_diff = abs(max_val - min_val) / max(abs(min_val), abs(max_val))
          
          if relative_diff < 0.01:
              return 0.95
          elif relative_diff < 0.05:
              return 0.85
          elif relative_diff < 0.1:
              return 0.75
          else:
              return max(0.0, 1.0 - relative_diff)
      except Exception as e:
          logger.debug(f"숫자 유사도 계산 오류: {e}")
          return 0.0
  
  def _calculate_boolean_similarity(self, values: List[Dict]) -> float:
      try:
          bool_values = [v.get('value', '') for v in values if v]
          unique_values = set(bool_values)
          return 1.0 if len(unique_values) == 1 else 0.0
      except Exception:
          return 0.0
  
  def _calculate_text_similarity(self, values: List[Dict]) -> float:
      try:
          text_values = [str(v.get('value', '')).lower().strip() for v in values if v]
          unique_values = set(text_values)
          
          if len(unique_values) == 1:
              return 1.0
          
          similarities = []
          for i in range(len(text_values)):
              for j in range(i + 1, len(text_values)):
                  sim = self._text_similarity(text_values[i], text_values[j])
                  similarities.append(sim)
          
          return np.mean(similarities) if similarities else 0.0
      except Exception:
          return 0.0
  
  def _text_similarity(self, text1: str, text2: str) -> float:
      try:
          if not text1 or not text2:
              return 0.0
          
          words1 = set(str(text1).split())
          words2 = set(str(text2).split())
          
          if not words1 and not words2:
              return 1.0
          intersection = len(words1 & words2)
          union = len(words1 | words2)
          
          return intersection / union if union > 0 else 0.0
      except Exception:
          return 0.0
  
  def _convert_unit(self, unit) -> str:
      try:
          if unit is None:
              return ""
          
          unit_lower = str(unit).lower().strip()
          
          for category, conversions in self.unit_conversions.items():
              if unit_lower in conversions:
                  return conversions[unit_lower]
          
          return unit_lower
      except Exception:
          return ""
  
  def save_comparison_results(self, comparison_df: pd.DataFrame):
      try:
          # 스키마 검증 및 수정
          self.db.check_and_fix_schema()
          
          # 기존 비교 결과 삭제
          self.db.conn.execute("DELETE FROM field_comparisons")
          
          for _, row in comparison_df.iterrows():
              comparison_data = row.to_dict()
              
              # 새로운 insert 메서드 사용 (id 자동 생성)
              self.db.insert_comparison_result(
                  field_name=row.get('Field_Name', ''),
                  comparison_result=row.get('Comparison_Result', ''),
                  documents_with_field=row.get('Documents_With_Field', 0),
                  total_documents=row.get('Total_Documents', 0),
                  similarity_score=row.get('Similarity_Score', 0.0),
                  comparison_data=json.dumps(comparison_data)
              )
          
          logger.info(f"비교 결과 저장 완료: {len(comparison_df)} 필드")
          
      except Exception as e:
          logger.error(f"비교 결과 저장 오류: {e}")
  
  def get_ttl_comparison_summary(self) -> Dict:
      """TTL 기반 비교 요약"""
      try:
          ttl_summary = self.db.execute_query("""
              SELECT 
                  COUNT(*) as total_comparisons,
                  SUM(CASE WHEN comparison_result LIKE '%TTL:%' THEN 1 ELSE 0 END) as ttl_matched_comparisons,
                  AVG(similarity_score) as avg_similarity,
                  COUNT(CASE WHEN similarity_score >= 0.8 THEN 1 END) as high_similarity_count
              FROM field_comparisons
          """)
          
          if not ttl_summary.empty:
              return ttl_summary.iloc[0].to_dict()
          
          return {'total_comparisons': 0, 'ttl_matched_comparisons': 0, 'avg_similarity': 0.0, 'high_similarity_count': 0}
          
      except Exception as e:
          logger.error(f"TTL 비교 요약 오류: {e}")
          return {}


# DuckDB 매니저 (확장된 스키마) - 🆕 사용자별 격리 적용
class AdvancedDuckDBManager:
  def __init__(self, db_path: str = "plant_documents.db"):
      # 🆕 사용자별 DB 파일 생성
      user_id = get_user_session()
      self.user_id = user_id
      self.db_path = f"user_{user_id}_{db_path}" if db_path != ":memory:" else ":memory:"
      
      try:
          self.conn = duckdb.connect(self.db_path)
          self.init_database()
          logger.info(f"사용자 {user_id} 데이터베이스 연결 성공: {self.db_path}")
      except Exception as e:
          logger.error(f"데이터베이스 연결 실패: {e}")
          self.conn = duckdb.connect(":memory:")
          self.db_path = ":memory:"
          self.init_database()
  
  def init_database(self):
      try:
          # 문서 테이블
          self.conn.execute("""
              CREATE TABLE IF NOT EXISTS documents (
                  doc_id VARCHAR PRIMARY KEY,
                  filename VARCHAR NOT NULL,
                  upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  file_size INTEGER,
                  page_count INTEGER,
                  total_confidence DECIMAL(4,3),
                  total_fields INTEGER,
                  metadata JSON
              )
          """)
          
          # 필드 테이블
          self.conn.execute("""
              CREATE SEQUENCE IF NOT EXISTS extracted_fields_seq START 1
          """)
          
          self.conn.execute("""
              CREATE TABLE IF NOT EXISTS extracted_fields (
                  id INTEGER PRIMARY KEY DEFAULT nextval('extracted_fields_seq'),
                  doc_id VARCHAR,
                  field_name VARCHAR NOT NULL,
                  field_value TEXT,
                  confidence DECIMAL(5,4),
                  extraction_method VARCHAR,
                  page_number INTEGER,
                  bbox_x1 DECIMAL(10,3),
                  bbox_y1 DECIMAL(10,3),
                  bbox_x2 DECIMAL(10,3),
                  bbox_y2 DECIMAL(10,3),
                  context_type VARCHAR,
                  hierarchy_level INTEGER DEFAULT 0,
                  parent_field VARCHAR,
                  conditions JSON,
                  font_info JSON,
                  extraction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
              )
          """)
          
          # 비교 결과 테이블 (SERIAL로 자동 증가)
          self.conn.execute("DROP TABLE IF EXISTS field_comparisons")
          self.conn.execute("""
              CREATE TABLE field_comparisons (
                  id SERIAL PRIMARY KEY,
                  field_name VARCHAR NOT NULL,
                  comparison_result VARCHAR,
                  documents_with_field INTEGER,
                  total_documents INTEGER,
                  similarity_score DECIMAL(5,4),
                  comparison_data JSON,
                  created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
              )
          """)
          
          # 인덱스 생성
          self.conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted_fields_doc_id ON extracted_fields(doc_id)")
          self.conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted_fields_name ON extracted_fields(field_name)")
          self.conn.execute("CREATE INDEX IF NOT EXISTS idx_field_comparisons_name ON field_comparisons(field_name)")
          
          logger.info(f"사용자 {self.user_id} 데이터베이스 스키마 초기화 완료")
          
      except Exception as e:
          logger.error(f"데이터베이스 초기화 실패: {e}")
  
  def insert_document_with_metadata(self, doc_id: str, filename: str, file_size: int, extraction_result: Dict):
      try:
          metadata_json = json.dumps(extraction_result.get('metadata', {}))
          total_fields = len(extraction_result.get('extracted_fields', []))
          
          self.conn.execute("""
              INSERT OR REPLACE INTO documents (
                  doc_id, filename, file_size, page_count, total_confidence, total_fields, metadata
              ) VALUES (?, ?, ?, ?, ?, ?, ?)
          """, [
              doc_id, filename, file_size,
              extraction_result.get('metadata', {}).get('page_count', 1),
              extraction_result.get('total_confidence', 0.0),
              total_fields,
              metadata_json
          ])
          
      except Exception as e:
          logger.error(f"문서 메타데이터 저장 실패: {e}")
  
  def insert_extracted_fields_batch(self, fields_data: List[Dict]):
      try:
          batch_data = []
          for field in fields_data:
              try:
                  conditions_json = json.dumps(field.get('conditions', {}))
                  font_info_json = json.dumps(field.get('font_info', {}))
                  
                  bbox = field.get('bbox', (0, 0, 0, 0))
                  if not bbox or len(bbox) != 4:
                      bbox = (0, 0, 0, 0)
                  
                  batch_data.append([
                      field.get('doc_id', ''),
                      field.get('field_name', 'Unknown'),
                      field.get('value', ''),
                      float(field.get('confidence', 0.50)),
                      field.get('extraction_method', 'unknown'),
                      int(field.get('page', 0)),
                      float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                      field.get('context_type', 'unknown'),
                      int(field.get('hierarchy_level', 0)),
                      field.get('parent_field', ''),
                      conditions_json,
                      font_info_json
                  ])
              except Exception as e:
                  logger.error(f"필드 데이터 준비 오류: {e}")
                  continue
          
          if batch_data:
              self.conn.executemany("""
                  INSERT INTO extracted_fields (
                      doc_id, field_name, field_value, confidence, extraction_method,
                      page_number, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                      context_type, hierarchy_level, parent_field, conditions, font_info
                  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              """, batch_data)
              
              logger.info(f"사용자 {self.user_id} 필드 일괄 저장 완료: {len(batch_data)}개")
      
      except Exception as e:
          logger.error(f"필드 일괄 저장 실패: {e}")
  
  def execute_query(self, query: str) -> pd.DataFrame:
      try:
          return self.conn.execute(query).df()
      except Exception as e:
          logger.error(f"쿼리 실행 오류: {e}")
          return pd.DataFrame()
  
  def check_and_fix_schema(self):
      """스키마 검증 및 수정"""
      try:
          # field_comparisons 테이블이 올바른 스키마를 가지고 있는지 확인
          result = self.conn.execute("""
              SELECT sql FROM sqlite_master 
              WHERE type='table' AND name='field_comparisons'
          """).fetchall()
          
          if result:
              table_sql = result[0][0]
              if 'SERIAL' not in table_sql and 'AUTO_INCREMENT' not in table_sql:
                  # 테이블 재생성
                  self.conn.execute("DROP TABLE IF EXISTS field_comparisons")
                  self.conn.execute("""
                      CREATE TABLE field_comparisons (
                          id SERIAL PRIMARY KEY,
                          field_name VARCHAR NOT NULL,
                          comparison_result VARCHAR,
                          documents_with_field INTEGER,
                          total_documents INTEGER,
                          similarity_score DECIMAL(5,4),
                          comparison_data JSON,
                          created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                      )
                  """)
                  logger.info("field_comparisons 테이블 재생성 완료")
          
      except Exception as e:
          logger.error(f"스키마 수정 실패: {e}")
  
  def reset_comparison_table(self):
      """비교 테이블 초기화"""
      try:
          self.conn.execute("DROP TABLE IF EXISTS field_comparisons")
          self.conn.execute("""
              CREATE TABLE field_comparisons (
                  id SERIAL PRIMARY KEY,
                  field_name VARCHAR NOT NULL,
                  comparison_result VARCHAR,
                  documents_with_field INTEGER,
                  total_documents INTEGER,
                  similarity_score DECIMAL(5,4),
                  comparison_data JSON,
                  created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
              )
          """)
          logger.info("비교 테이블 초기화 완료")
      except Exception as e:
          logger.error(f"비교 테이블 초기화 실패: {e}")
  
  def insert_comparison_result(self, field_name: str, comparison_result: str, 
                             documents_with_field: int, total_documents: int, 
                             similarity_score: float, comparison_data: str):
      """비교 결과 개별 삽입 (id 자동 생성)"""
      try:
          self.conn.execute("""
              INSERT INTO field_comparisons (
                  field_name, comparison_result, documents_with_field, 
                  total_documents, similarity_score, comparison_data
              ) VALUES (?, ?, ?, ?, ?, ?)
          """, [
              field_name, comparison_result, documents_with_field,
              total_documents, similarity_score, comparison_data
          ])
      except Exception as e:
          logger.error(f"비교 결과 삽입 실패: {e}")
  
  def get_extracted_fields_summary(self):
      """추출된 필드 요약 정보"""
      try:
          summary_query = """
              SELECT 
                  field_name,
                  COUNT(*) as frequency,
                  AVG(confidence) as avg_confidence,
                  COUNT(DISTINCT doc_id) as doc_count,
                  STRING_AGG(DISTINCT field_value, ', ') as sample_values
              FROM extracted_fields 
              GROUP BY field_name 
              ORDER BY frequency DESC
              LIMIT 50
          """
          return self.execute_query(summary_query)
      except Exception as e:
          logger.error(f"필드 요약 조회 실패: {e}")
          return pd.DataFrame()
  
  def get_ttl_matched_fields(self):
      """TTL과 매치된 필드들만 조회"""
      try:
          ttl_query = """
              SELECT 
                  d.filename,
                  ef.field_name,
                  ef.field_value,
                  ef.confidence,
                  ef.extraction_method,
                  JSON_EXTRACT(ef.conditions, '$.ttl_matched') as ttl_matched
              FROM extracted_fields ef
              JOIN documents d ON ef.doc_id = d.doc_id
              WHERE JSON_EXTRACT(ef.conditions, '$.ttl_matched') = true
              ORDER BY ef.field_name, d.filename
          """
          return self.execute_query(ttl_query)
      except Exception as e:
          logger.error(f"TTL 매치 필드 조회 실패: {e}")
          return pd.DataFrame()
  
  def cleanup_database(self):
      """데이터베이스 정리"""
      try:
          # 중복 필드 제거
          self.conn.execute("""
              DELETE FROM extracted_fields 
              WHERE id NOT IN (
                  SELECT MIN(id) 
                  FROM extracted_fields 
                  GROUP BY doc_id, field_name, field_value
              )
          """)
          
          # 빈 값 제거
          self.conn.execute("""
              DELETE FROM extracted_fields 
              WHERE field_value IS NULL OR field_value = '' OR field_name IS NULL OR field_name = ''
          """)
          
          logger.info("데이터베이스 정리 완료")
          
      except Exception as e:
          logger.error(f"데이터베이스 정리 실패: {e}")

def process_files_maximum_extraction(uploaded_files, db, pdf_processor, comparison_system):
  st.header("🚀 최대 필드 추출 및 완전 비교 시스템")
  
  if not uploaded_files:
      st.warning("업로드된 파일이 없습니다.")
      return
  
  progress_container = st.container()
  results_container = st.container()
  
  with progress_container:
      progress_bar = st.progress(0, text="준비 중...")
      status_text = st.empty()
      
      col1, col2, col3, col4 = st.columns(4)
      with col1:
          processed_count = st.metric("처리 완료", 0)
      with col2:
          total_fields_metric = st.metric("총 추출 필드", 0)
      with col3:
          avg_confidence_metric = st.metric("평균 신뢰도", "0%")
      with col4:
          error_count_metric = st.metric("오류 수", 0)
  
  total_files = len(uploaded_files)
  processed_results = []
  total_extracted_fields = 0
  total_confidence_sum = 0.0
  error_count = 0
  detailed_results = []
  
  for i, uploaded_file in enumerate(uploaded_files):
      current_progress = (i + 1) / total_files
      
      with status_text:
          st.info(f"📄 처리 중: {uploaded_file.name} ({i+1}/{total_files})")
      
      try:
          file_size = len(uploaded_file.getvalue())
          
          if file_size > 50 * 1024 * 1024:
              st.warning(f"⚠️ {uploaded_file.name}: 파일이 너무 큽니다 (50MB 초과)")
              error_count += 1
              continue
          
          extraction_start_time = datetime.now()
          
          with st.spinner(f"🔍 '{uploaded_file.name}' 분석 중..."):
              extraction_result = pdf_processor.extract_all_possible_fields(
                  uploaded_file.getvalue(), 
                  uploaded_file.name
              )
          
          extraction_time = (datetime.now() - extraction_start_time).total_seconds()
          
          if not extraction_result or 'extracted_fields' not in extraction_result:
              st.error(f"❌ {uploaded_file.name}: 추출 결과가 비어있습니다")
              error_count += 1
              continue
          
          extracted_fields = extraction_result.get('extracted_fields', [])
          doc_confidence = extraction_result.get('total_confidence', 0.0)
          
          field_analysis = analyze_extraction_result(extracted_fields, uploaded_file.name)
          
          save_success = False
          try:
              if extracted_fields:
                  db.insert_document_with_metadata(
                      extraction_result['doc_id'],
                      uploaded_file.name,
                      file_size,
                      extraction_result
                  )
                  
                  db.insert_extracted_fields_batch(extracted_fields)
                  save_success = True
                  
                  total_extracted_fields += len(extracted_fields)
                  total_confidence_sum += doc_confidence
                  
          except Exception as db_error:
              st.error(f"❌ {uploaded_file.name}: DB 저장 실패 - {str(db_error)}")
              error_count += 1
              continue
          
          if save_success:
              result_data = {
                  'filename': uploaded_file.name,
                  'status': 'success',
                  'total_fields': len(extracted_fields),
                  'confidence': doc_confidence,
                  'extraction_time': extraction_time,
                  'file_size_mb': round(file_size / (1024*1024), 2),
                  'analysis': field_analysis
              }
              
              processed_results.append(result_data)
              detailed_results.append(result_data)
              
              with col1:
                  processed_count.metric("처리 완료", len(processed_results))
              with col2:
                  total_fields_metric.metric("총 추출 필드", total_extracted_fields)
              with col3:
                  avg_conf = (total_confidence_sum / len(processed_results)) if processed_results else 0
                  avg_confidence_metric.metric("평균 신뢰도", f"{avg_conf:.1%}")
              with col4:
                  error_count_metric.metric("오류 수", error_count)
              
              with results_container:
                  with st.expander(f"✅ {uploaded_file.name} - {len(extracted_fields)} 필드", expanded=False):
                      display_file_analysis(result_data, field_analysis)
          
      except Exception as e:
          error_count += 1
          logger.error(f"파일 처리 오류 {uploaded_file.name}: {e}")
          
          with results_container:
              st.error(f"❌ {uploaded_file.name}: {str(e)}")
      
      finally:
          progress_bar.progress(current_progress, text=f"진행률: {current_progress:.1%}")
  
  progress_bar.progress(1.0, text="✅ 모든 파일 처리 완료!")
  
  with results_container:
      st.subheader("📊 처리 결과 요약")
      
      if processed_results:
          success_rate = len(processed_results) / total_files
          
          summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
          
          with summary_col1:
              st.metric("성공률", f"{success_rate:.1%}", f"{len(processed_results)}/{total_files}")
          
          with summary_col2:
              avg_fields_per_doc = total_extracted_fields / len(processed_results)
              st.metric("문서당 평균 필드", f"{avg_fields_per_doc:.0f}")
          
          with summary_col3:
              avg_confidence = total_confidence_sum / len(processed_results)
              st.metric("전체 평균 신뢰도", f"{avg_confidence:.1%}")
          
          with summary_col4:
              total_processing_time = sum(r.get('extraction_time', 0) for r in processed_results)
              st.metric("총 처리 시간", f"{total_processing_time:.1f}초")
          
          if detailed_results:
              st.subheader("📋 파일별 상세 결과")
              
              results_df = pd.DataFrame([
                  {
                      '파일명': r['filename'],
                      '추출 필드 수': r['total_fields'],
                      '신뢰도': f"{r['confidence']:.1%}",
                      '처리 시간(초)': f"{r['extraction_time']:.1f}",
                      '파일 크기(MB)': r['file_size_mb'],
                      '고신뢰도 필드': r['analysis']['high_confidence_fields'],
                      '추출 방법 수': len(r['analysis']['extraction_methods'])
                  }
                  for r in detailed_results
              ])
              
              st.dataframe(results_df, use_container_width=True)
              
              if len(detailed_results) > 1:
                  st.subheader("📈 처리 결과 시각화")
                  
                  fig1 = px.bar(
                      results_df,
                      x='파일명',
                      y='추출 필드 수',
                      title="파일별 추출 필드 수",
                      color='신뢰도'
                  )
                  fig1.update_xaxes(tickangle=45)
                  st.plotly_chart(fig1, use_container_width=True)
          
          if len(processed_results) >= 2:
              st.subheader("🔍 문서 간 비교 분석")
              
              with st.spinner("비교 매트릭스 생성 중..."):
                  try:
                      comparison_matrix = comparison_system.generate_complete_comparison_matrix()
                      
                      if not comparison_matrix.empty:
                          comparison_system.save_comparison_results(comparison_matrix)
                          
                          total_comparable_fields = len(comparison_matrix)
                          identical_fields = len(comparison_matrix[
                              comparison_matrix['Comparison_Result'].str.contains('🟢 모든 값 동일', na=False)
                          ])
                          
                          comp_col1, comp_col2 = st.columns(2)
                          
                          with comp_col1:
                              st.metric("비교 가능 필드", total_comparable_fields)
                          
                          with comp_col2:
                              identical_rate = identical_fields / total_comparable_fields if total_comparable_fields > 0 else 0
                              st.metric("동일 필드 비율", f"{identical_rate:.1%}")
                          
                          st.success("✅ 비교 매트릭스 생성 완료! '🔍 완전 비교' 탭에서 확인하세요.")
                      
                      else:
                          st.warning("⚠️ 비교 매트릭스가 비어있습니다.")
                  
                  except Exception as comp_error:
                      st.error(f"❌ 비교 매트릭스 생성 실패: {str(comp_error)}")
          
          else:
              st.info("💡 문서 간 비교를 위해서는 최소 2개 이상의 문서가 필요합니다.")
      
      else:
          st.error("❌ 성공적으로 처리된 파일이 없습니다.")
          if error_count > 0:
              st.info(f"💡 {error_count}개 파일에서 오류가 발생했습니다. 파일 형식과 크기를 확인해주세요.")

def analyze_extraction_result(extracted_fields: List[Dict], filename: str) -> Dict:
  if not extracted_fields:
      return {
          'high_confidence_fields': 0,
          'medium_confidence_fields': 0,
          'low_confidence_fields': 0,
          'extraction_methods': [],
          'context_types': {},
          'field_name_distribution': {}
      }
  
  high_conf = sum(1 for f in extracted_fields if f.get('confidence', 0) >= 0.7)
  medium_conf = sum(1 for f in extracted_fields if 0.4 <= f.get('confidence', 0) < 0.7)
  low_conf = sum(1 for f in extracted_fields if f.get('confidence', 0) < 0.4)
  
  extraction_methods = list(set(f.get('extraction_method', 'unknown') for f in extracted_fields))
  context_types = Counter(f.get('context_type', 'unknown') for f in extracted_fields)
  field_names = Counter(f.get('field_name', 'unknown') for f in extracted_fields)
  
  return {
      'high_confidence_fields': high_conf,
      'medium_confidence_fields': medium_conf,
      'low_confidence_fields': low_conf,
      'extraction_methods': extraction_methods,
      'context_types': dict(context_types),
      'field_name_distribution': dict(field_names.most_common(10))
  }

def display_file_analysis(result_data: Dict, analysis: Dict):
  col1, col2 = st.columns(2)
  
  with col1:
      st.write(f"**총 필드 수:** {result_data['total_fields']}")
      st.write(f"**전체 신뢰도:** {result_data['confidence']:.1%}")
      st.write(f"**처리 시간:** {result_data['extraction_time']:.1f}초")
  
  with col2:
      st.write(f"**파일 크기:** {result_data['file_size_mb']}MB")
      st.write(f"**추출 방법:** {len(analysis['extraction_methods'])}개")
      st.write(f"**고신뢰도 필드:** {analysis['high_confidence_fields']}개")
  
  if result_data['total_fields'] > 0:
      st.write("**신뢰도 분포:**")
      conf_col1, conf_col2, conf_col3 = st.columns(3)
      
      with conf_col1:
          st.metric("고(≥70%)", analysis['high_confidence_fields'])
      with conf_col2:
          st.metric("중(40-70%)", analysis['medium_confidence_fields'])
      with conf_col3:
          st.metric("저(<40%)", analysis['low_confidence_fields'])
  
  if analysis['field_name_distribution']:
      st.write("**주요 추출 필드:**")
      for field_name, count in list(analysis['field_name_distribution'].items())[:5]:
          st.write(f"- {field_name}: {count}개")

def _display_file_analysis(result_data: Dict, analysis: Dict):
  """개별 파일 분석 결과 표시"""
  
  # 기본 정보
  col1, col2 = st.columns(2)
  
  with col1:
      st.write(f"**총 필드 수:** {result_data['total_fields']}")
      st.write(f"**전체 신뢰도:** {result_data['confidence']:.1%}")
      st.write(f"**처리 시간:** {result_data['extraction_time']:.1f}초")
  
  with col2:
      st.write(f"**파일 크기:** {result_data['file_size_mb']}MB")
      st.write(f"**추출 방법:** {len(analysis['extraction_methods'])}개")
      st.write(f"**고신뢰도 필드:** {analysis['high_confidence_fields']}개")
  
  # 신뢰도 분포
  if result_data['total_fields'] > 0:
      st.write("**신뢰도 분포:**")
      conf_col1, conf_col2, conf_col3 = st.columns(3)
      
      with conf_col1:
          st.metric("고(≥70%)", analysis['high_confidence_fields'])
      with conf_col2:
          st.metric("중(40-70%)", analysis['medium_confidence_fields'])
      with conf_col3:
          st.metric("저(<40%)", analysis['low_confidence_fields'])
  
  # 주요 필드명
  if analysis['field_name_distribution']:
      st.write("**주요 추출 필드:**")
      for field_name, count in list(analysis['field_name_distribution'].items())[:5]:
          st.write(f"- {field_name}: {count}개")

# 메인 애플리케이션 - 🆕 사용자별 격리 적용
def main():
  st.set_page_config(
      page_title="Complete Field Extraction & Comparison System",
      page_icon="🔧",
      layout="wide"
  )
  
  # 🆕 사용자 세션 초기화
  user_id = get_user_session()
  
  st.markdown(f"""
      <div style="padding: 1rem; text-align: center; margin-bottom: 1rem;">                   
          <h1>🔧 Complete Field Extraction & Comparison System</h1>
          <p><strong>모든 필드 추출 → 완전 비교 → DuckDB 저장</strong></p>
          <p style="font-size: 0.8rem; color: #666;">사용자 ID: {user_id}</p>
      </div>
  """, unsafe_allow_html=True)
  
  # 세션 상태 초기화 (에러 복구 포함)
  if not _initialize_session_state():
      st.error("❌ 시스템 초기화 실패. 페이지를 새로고침해주세요.")
      if st.button("🔄 시스템 재초기화"):
          _force_reinitialize_session()
      st.stop()
  
  db = get_user_state('db_manager')
  pdf_processor = get_user_state('pdf_processor')
  comparison_system = get_user_state('comparison_system')
  
  # 사이드바 - 파일 업로드 및 시스템 상태
  with st.sidebar:
      st.header("📁 문서 업로드")
      
      # 🆕 사용자별 초기화 버튼
      if st.button("🔄 내 데이터 초기화"):
          clear_user_state()
          st.success("데이터가 초기화되었습니다!")
          st.rerun()
      
      uploaded_files = st.file_uploader(
          "PDF 파일 선택", 
          type=['pdf'], 
          accept_multiple_files=True,
          key=f"file_uploader_{user_id}"  # 🆕 사용자별 키
      )
      
      st.header("⚙️ 시스템 설정")
      st.success("✅ 신뢰도 임계값: 0.20")
      st.success("✅ 필터링: 비활성화")
      st.success("✅ 모든 패턴: 활성화")
      st.info(f"🔒 사용자별 격리: {user_id}")
      
      # 시스템 상태 모니터링
      _display_system_status(db)
      
      # 메모리 사용량 (간단)
      if st.button("🔧 시스템 상태 새로고침"):
          st.rerun()
  
  # 메인 탭 구성
  tab1, tab2, tab3 = st.tabs(["📄 최대 추출", "🔍 완전 비교", "📊 데이터 분석"])
  
  with tab1:
      _handle_extraction_tab(uploaded_files, db, pdf_processor, comparison_system)
  
  with tab2:
      _handle_comparison_tab(comparison_system, db)
  
  with tab3:
      _handle_analysis_tab(db)

def _initialize_session_state() -> bool:
  """세션 상태 초기화 (에러 복구 포함) - 🆕 사용자별 격리 적용"""
  try:
      # 🆕 사용자별 초기화 확인
      user_id = get_user_session()
      
      # DB 매니저 초기화
      if not get_user_state('db_manager'):
          try:
              db_manager = AdvancedDuckDBManager()
              set_user_state('db_manager', db_manager)
          except Exception as e:
              logger.error(f"DB 매니저 초기화 실패: {e}")
              # 메모리 DB로 폴백
              db_manager = AdvancedDuckDBManager(":memory:")
              set_user_state('db_manager', db_manager)
      
      # 온톨로지 매니저 초기화
      if not get_user_state('ontology_manager'):
          ontology_manager = ComprehensiveOntologyManager()
          set_user_state('ontology_manager', ontology_manager)
      
      # PDF 프로세서 초기화
      if not get_user_state('pdf_processor'):
          pdf_processor = MaximumExtractorProcessor(
              get_user_state('ontology_manager')
          )
          set_user_state('pdf_processor', pdf_processor)
      
      # 비교 시스템 초기화
      if not get_user_state('comparison_system'):
          comparison_system = CompleteComparisonSystem(
              get_user_state('db_manager')
          )
          set_user_state('comparison_system', comparison_system)
      
      # 초기화 상태 기록
      if not get_user_state('system_initialized'):
          set_user_state('system_initialized', True)
          set_user_state('initialization_time', datetime.now())
      
      return True
      
  except Exception as e:
      logger.error(f"세션 상태 초기화 오류: {e}")
      return False

def _force_reinitialize_session():
  """강제 세션 재초기화 - 🆕 사용자별 격리 적용"""
  user_id = get_user_session()
  keys_to_clear = ['db_manager', 'ontology_manager', 'pdf_processor', 'comparison_system', 'system_initialized']
  for key in keys_to_clear:
      clear_user_state(key)

def _display_system_status(db):
  """시스템 상태 표시"""
  st.header("📊 시스템 상태")
  
  if not db:
      st.error("❌ DB 연결 없음")
      return
  
  try:
      # DB 통계
      doc_count = db.execute_query("SELECT COUNT(*) as count FROM documents").iloc[0]['count']
      field_count = db.execute_query("SELECT COUNT(*) as count FROM extracted_fields").iloc[0]['count']
      unique_fields = db.execute_query("SELECT COUNT(DISTINCT field_name) as count FROM extracted_fields").iloc[0]['count']
      
      st.metric("📄 문서 수", doc_count)
      st.metric("🏷️ 총 필드", field_count)
      st.metric("🔖 고유 필드", unique_fields)
      
      # DB 연결 상태
      if db.db_path == ":memory:":
          st.warning("⚠️ 메모리 DB 사용 중")
      else:
          st.success(f"✅ 파일 DB 연결됨")
          st.caption(f"DB: {db.db_path}")
  
  except Exception as e:
      st.error(f"❌ 상태 조회 실패: {str(e)}")
      st.metric("📄 문서 수", "오류")
      st.metric("🏷️ 총 필드", "오류")
      st.metric("🔖 고유 필드", "오류")

def _handle_extraction_tab(uploaded_files, db, pdf_processor, comparison_system):
  """추출 탭 처리"""
  if uploaded_files:
      if st.button("🚀 추출 시작", type="primary"):
          process_files_maximum_extraction(uploaded_files, db, pdf_processor, comparison_system)
  else:
      st.info("📁 PDF 파일을 업로드하여 필드 추출을 시작하세요.")
  
  # 최근 추출 결과 표시
  if db:
      try:
          recent_extractions = db.execute_query("""
              SELECT d.filename, ef.field_name, ef.field_value, ef.confidence, ef.extraction_method
              FROM extracted_fields ef
              JOIN documents d ON ef.doc_id = d.doc_id
              ORDER BY ef.extraction_time DESC
              LIMIT 50
          """)
          
          if not recent_extractions.empty:
              st.subheader("📋 최근 추출 결과 (상위 50개)")
              st.dataframe(recent_extractions, use_container_width=True)
          else:
              st.info("추출된 데이터가 없습니다.")
      
      except Exception as e:
          st.warning(f"데이터 로드 오류: {e}")

def _handle_comparison_tab(comparison_system, db):
  """비교 탭 처리"""
  st.header("🔍 완전 비교 매트릭스")
  
  if not db or not comparison_system:
      st.error("❌ 시스템이 초기화되지 않았습니다.")
      return
  
  try:
      # 문서 수 확인
      doc_count = db.execute_query("SELECT COUNT(*) as count FROM documents").iloc[0]['count']
      
      if doc_count < 2:
          st.warning(f"💡 비교를 위해서는 최소 2개 문서가 필요합니다. (현재: {doc_count}개)")
          return
      
      # 비교 매트릭스 생성
      if st.button("🔄 비교 매트릭스 새로 생성"):
          with st.spinner("비교 매트릭스 생성 중..."):
              comparison_matrix = comparison_system.generate_complete_comparison_matrix()
              if not comparison_matrix.empty:
                  comparison_system.save_comparison_results(comparison_matrix)
                  st.success("✅ 비교 매트릭스 생성 완료!")
                  st.rerun()
      
      # 기존 비교 결과 표시
      comparison_matrix = comparison_system.generate_complete_comparison_matrix()
      
      if not comparison_matrix.empty:
          # 필터링 옵션
          col1, col2, col3 = st.columns(3)
          
          with col1:
              show_differences_only = st.checkbox("차이나는 필드만 표시")
          with col2:
              show_missing_only = st.checkbox("누락 필드만 표시")
          with col3:
              min_documents = st.slider("최소 문서 수", 1, doc_count, 1)
          
          # 필터링 적용
          filtered_matrix = comparison_matrix.copy()
          
          if show_differences_only:
              filtered_matrix = filtered_matrix[
                  filtered_matrix['Comparison_Result'].str.contains('🔴 값 차이', na=False)
              ]
          
          if show_missing_only:
              filtered_matrix = filtered_matrix[
                  filtered_matrix['Comparison_Result'].str.contains('🟡 1개 문서에만', na=False)
              ]
          
          if min_documents > 1:
              filtered_matrix = filtered_matrix[
                  filtered_matrix['Documents_With_Field'] >= min_documents
              ]
          
          # 결과 표시
          if not filtered_matrix.empty:
              st.dataframe(filtered_matrix, use_container_width=True)
              
              # CSV 다운로드
              csv = filtered_matrix.to_csv(index=False)
              st.download_button(
                  label="📥 CSV 다운로드",
                  data=csv,
                  file_name=f"comparison_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                  mime="text/csv"
              )
          else:
              st.warning("필터 조건에 맞는 필드가 없습니다.")
      
      else:
          st.info("비교 매트릭스가 없습니다. '비교 매트릭스 새로 생성' 버튼을 클릭하세요.")
  
  except Exception as e:
      st.error(f"비교 처리 오류: {e}")

def _handle_analysis_tab(db):
  """분석 탭 처리"""
  st.header("📊 추출 데이터 분석")
  
  if not db:
      st.error("❌ DB 연결이 없습니다.")
      return
  
  try:
      # 문서별 통계
      doc_stats = db.execute_query("""
          SELECT 
              d.filename,
              d.total_fields,
              d.total_confidence,
              COUNT(ef.id) as extracted_fields,
              AVG(ef.confidence) as avg_confidence
          FROM documents d
          LEFT JOIN extracted_fields ef ON d.doc_id = ef.doc_id
          GROUP BY d.doc_id, d.filename, d.total_fields, d.total_confidence
          ORDER BY extracted_fields DESC
      """)
      
      if not doc_stats.empty:
          st.subheader("📄 문서별 통계")
          
          # 차트
          fig = px.bar(
              doc_stats,
              x='filename',
              y='extracted_fields',
              title="문서별 추출 필드 수",
              color='avg_confidence'
          )
          fig.update_xaxes(tickangle=45)
          st.plotly_chart(fig, use_container_width=True)
          
          # 상세 테이블
          st.dataframe(doc_stats, use_container_width=True)
      
      # 필드 타입별 통계
      field_stats = db.execute_query("""
          SELECT 
              field_name,
              COUNT(*) as occurrence_count,
              COUNT(DISTINCT doc_id) as document_count,
              AVG(confidence) as avg_confidence
          FROM extracted_fields
          GROUP BY field_name
          HAVING COUNT(*) > 1
          ORDER BY occurrence_count DESC
          LIMIT 20
      """)
      
      if not field_stats.empty:
          st.subheader("🏷️ 필드별 통계 (상위 20개)")
          
          fig2 = px.bar(
              field_stats,
              x='field_name',
              y='occurrence_count',
              title="필드별 출현 빈도",
              color='avg_confidence'
          )
          fig2.update_xaxes(tickangle=45)
          st.plotly_chart(fig2, use_container_width=True)
          
          st.dataframe(field_stats, use_container_width=True)
      
      else:
          st.info("분석할 데이터가 부족합니다.")
  
  except Exception as e:
      st.error(f"분석 오류: {e}")

if __name__ == "__main__":
  main()
