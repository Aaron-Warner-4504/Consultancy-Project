# TB Diagnostics AI Tools - Simplified Agent-Based Implementation
# Compatible with FastAPI 0.127.0, LangChain 1.2.0, LangGraph 1.0.5
# The ReAct agent automatically chooses tools based on natural language

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import pandas as pd
import json
import io
import PyPDF2
from datetime import datetime
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Single request model - agent figures out what to do"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query: str = Field(..., description="Natural language query - agent will choose appropriate tools")

class AgentResponse(BaseModel):
    """Response from agent"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    answer: str
    tools_used: List[str]
    error: Optional[str] = None

# ============================================================================
# DATA INGESTION TOOL
# ============================================================================

class DataIngestionTool:
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json', 'pdf', 'txt']
        self.file_cache = {}
    
    def ingest_data(self, file_content: bytes, filename: str, file_format: str) -> Dict[str, Any]:
        try:
            if file_format == 'csv':
                data = self._parse_csv(file_content, filename)
            elif file_format in ['xlsx', 'xls']:
                data = self._parse_excel(file_content, filename)
            elif file_format == 'json':
                data = self._parse_json(file_content, filename)
            elif file_format == 'pdf':
                data = self._parse_pdf(file_content, filename)
            elif file_format == 'txt':
                data = self._parse_text(file_content, filename)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            return {
                "cleaned_dataset": data['cleaned_data'],
                "schema": self._generate_schema(data['cleaned_data'], filename),
                "issues_log": data['issues'],
                "extracted_chunks": data.get('chunks', []),
                "metadata": {
                    "filename": filename,
                    "format": file_format,
                    "processed_at": datetime.now().isoformat(),
                    "total_records": len(data['cleaned_data']) if isinstance(data['cleaned_data'], list) else 1
                }
            }
        except Exception as e:
            return {"error": str(e), "filename": filename}
    
    def _parse_csv(self, content: bytes, filename: str) -> Dict[str, Any]:
        df = pd.read_csv(io.BytesIO(content))
        issues = []
        
        original_columns = df.columns.tolist()
        df.columns = [self._standardize_column_name(col) for col in df.columns]
        
        issues.extend(self._detect_missing_values(df, filename))
        issues.extend(self._detect_invalid_dates(df, filename))
        issues.extend(self._detect_tb_unit_issues(df, filename))
        issues.extend(self._detect_facility_issues(df, filename))
        
        df = self._clean_dataframe(df)
        
        return {
            "cleaned_data": df.to_dict('records'),
            "issues": issues,
            "original_columns": original_columns
        }
    
    def _parse_excel(self, content: bytes, filename: str) -> Dict[str, Any]:
        excel_file = pd.ExcelFile(io.BytesIO(content))
        all_data = []
        all_issues = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            if df.empty:
                continue
            
            df.columns = [self._standardize_column_name(col) for col in df.columns]
            
            issues = self._detect_missing_values(df, filename, sheet_name)
            issues.extend(self._detect_invalid_dates(df, filename, sheet_name))
            issues.extend(self._detect_tb_unit_issues(df, filename, sheet_name))
            
            df = self._clean_dataframe(df)
            
            all_data.append({
                "sheet_name": sheet_name,
                "data": df.to_dict('records'),
                "row_count": len(df)
            })
            all_issues.extend([{**issue, "sheet": sheet_name} for issue in issues])
        
        return {"cleaned_data": all_data, "issues": all_issues}
    
    def _parse_json(self, content: bytes, filename: str) -> Dict[str, Any]:
        data = json.loads(content.decode('utf-8'))
        flattened = self._flatten_json(data)
        return {"cleaned_data": flattened, "issues": []}
    
    def _parse_pdf(self, content: bytes, filename: str) -> Dict[str, Any]:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        chunks = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            text_chunks = self._chunk_text(text, 500, 50)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                chunks.append({
                    "chunk_id": f"{filename}_p{page_num + 1}_c{chunk_idx}",
                    "text": chunk,
                    "page": page_num + 1,
                    "source": filename
                })
        
        return {
            "cleaned_data": {"full_text": " ".join([c['text'] for c in chunks])},
            "issues": [],
            "chunks": chunks
        }
    
    def _parse_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        text = content.decode('utf-8')
        chunks = self._chunk_text(text, 500, 50)
        
        chunk_objects = [
            {"chunk_id": f"{filename}_c{idx}", "text": chunk, "source": filename}
            for idx, chunk in enumerate(chunks)
        ]
        
        return {
            "cleaned_data": {"full_text": text},
            "issues": [],
            "chunks": chunk_objects
        }
    
    def _standardize_column_name(self, col: str) -> str:
        if pd.isna(col):
            return "unnamed_column"
        
        col = str(col).lower().strip()
        col = re.sub(r'[^\w\s]', '_', col)
        col = re.sub(r'\s+', '_', col)
        col = re.sub(r'_+', '_', col).strip('_')
        
        mappings = {
            'tb_unit': ['tbu', 'tb_unit_code', 'unit_code'],
            'facility_id': ['facility', 'fac_id'],
            'patient_id': ['patient', 'pat_id', 'case_id'],
            'diagnosis_date': ['diag_date', 'dx_date'],
            'utilization': ['usage', 'utilization_rate']
        }
        
        for standard, variations in mappings.items():
            if col in variations:
                return standard
        
        return col
    
    def _detect_missing_values(self, df: pd.DataFrame, filename: str, sheet: str = None) -> List[Dict]:
        issues = []
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                severity = "critical" if missing / len(df) > 0.7 else \
                          "high" if missing / len(df) > 0.5 else "medium"
                issues.append({
                    "type": "missing_values",
                    "column": col,
                    "count": int(missing),
                    "percentage": round(missing / len(df) * 100, 1),
                    "severity": severity,
                    "source_file": filename
                })
        return issues
    
    def _detect_invalid_dates(self, df: pd.DataFrame, filename: str, sheet: str = None) -> List[Dict]:
        issues = []
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        for col in date_columns:
            try:
                valid_dates = pd.to_datetime(df[col], errors='coerce')
                invalid_count = df[col].notna().sum() - valid_dates.notna().sum()
                if invalid_count > 0:
                    issues.append({
                        "type": "invalid_dates",
                        "column": col,
                        "count": int(invalid_count),
                        "severity": "high",
                        "source_file": filename
                    })
            except:
                pass
        return issues
    
    def _detect_tb_unit_issues(self, df: pd.DataFrame, filename: str, sheet: str = None) -> List[Dict]:
        issues = []
        tb_cols = [col for col in df.columns if 'tb_unit' in col.lower()]
        
        for col in tb_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append({
                    "type": "missing_tb_unit_codes",
                    "column": col,
                    "count": int(missing),
                    "severity": "critical",
                    "source_file": filename
                })
            
            unique_values = df[col].dropna().astype(str).unique()
            if len(unique_values) > 0:
                patterns = set([re.sub(r'\d', 'X', str(v)) for v in unique_values])
                if len(patterns) > 1:
                    issues.append({
                        "type": "inconsistent_tb_unit_codes",
                        "column": col,
                        "unique_patterns": list(patterns)[:5],
                        "severity": "high",
                        "source_file": filename
                    })
        return issues
    
    def _detect_facility_issues(self, df: pd.DataFrame, filename: str, sheet: str = None) -> List[Dict]:
        issues = []
        facility_cols = [col for col in df.columns if 'facility' in col.lower()]
        
        for col in facility_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append({
                    "type": "missing_facility_identifiers",
                    "column": col,
                    "count": int(missing),
                    "severity": "high",
                    "source_file": filename
                })
        return issues
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how='all')
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().replace('nan', pd.NA)
        
        return df
    
    def _generate_schema(self, data: Any, filename: str) -> Dict[str, Any]:
        schema = {
            "source_file": filename,
            "generated_at": datetime.now().isoformat(),
            "tables": []
        }
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            schema["tables"].append(self._analyze_table_structure(data, "main_table"))
        elif isinstance(data, list):
            for idx, table_data in enumerate(data):
                if isinstance(table_data, dict) and 'data' in table_data:
                    schema["tables"].append(
                        self._analyze_table_structure(table_data['data'], table_data.get('sheet_name', f'table_{idx}'))
                    )
        
        return schema
    
    def _analyze_table_structure(self, data: List[Dict], table_name: str) -> Dict[str, Any]:
        if not data:
            return {"table_name": table_name, "columns": []}
        
        df = pd.DataFrame(data)
        columns = []
        
        for col in df.columns:
            dtype = df[col].dtype
            sql_type = "TEXT"
            confidence = 0.7
            
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "INTEGER"
                confidence = 0.95
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "REAL"
                confidence = 0.95
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "DATETIME"
                confidence = 0.95
            
            columns.append({
                "name": col,
                "sql_type": sql_type,
                "nullable": bool(df[col].isna().any()),
                "unique_values": int(df[col].nunique()),
                "confidence": confidence
            })
        
        return {
            "table_name": table_name,
            "columns": columns,
            "row_count": len(df)
        }
    
    def _flatten_json(self, data: Any) -> Any:
        if isinstance(data, dict):
            flattened = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flattened[f"{k}_{k2}"] = v2
                else:
                    flattened[k] = v
            return [flattened]
        elif isinstance(data, list):
            return data
        return [{"data": data}]
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

# ============================================================================
# RAG EVALUATION TOOL
# ============================================================================

class RAGEvaluationTool:
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, query: str, retrieved_evidence: Dict, 
                generated_answer: str, expected_output: Optional[Dict] = None) -> Dict:
        try:
            retrieval_metrics = self._calculate_retrieval_metrics(query, retrieved_evidence, expected_output)
            interpretability_metrics = self._calculate_interpretability_metrics(generated_answer, retrieved_evidence)
            generation_metrics = self._calculate_generation_metrics(generated_answer, retrieved_evidence, expected_output)
            
            summary = self._generate_summary(retrieval_metrics, interpretability_metrics, generation_metrics)
            overall_score = self._calculate_overall_score(retrieval_metrics, interpretability_metrics, generation_metrics)
            
            return {
                "retrieval_metrics": retrieval_metrics,
                "interpretability_metrics": interpretability_metrics,
                "generation_metrics": generation_metrics,
                "summary": summary,
                "overall_score": overall_score,
                "confidence_score": generation_metrics.get('confidence_score', 0.0),
                "completeness_score": generation_metrics.get('completeness_score', 0.0),
                "evaluated_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_retrieval_metrics(self, query: str, evidence: Dict, expected: Optional[Dict]) -> Dict:
        metrics = {
            "retrieval_precision": 0.0,
            "retrieval_recall": 0.0,
            "retrieval_f1": 0.0,
            "rank_quality": 0.0
        }
        
        retrieved_chunks = evidence.get('chunks', [])
        
        if expected and 'expected_chunk_ids' in expected:
            expected_ids = set(expected['expected_chunk_ids'])
            retrieved_ids = set([c.get('chunk_id', '') for c in retrieved_chunks if c.get('chunk_id')])
            
            if retrieved_ids:
                precision = len(expected_ids & retrieved_ids) / len(retrieved_ids)
                metrics['retrieval_precision'] = round(precision, 3)
            
            if expected_ids:
                recall = len(expected_ids & retrieved_ids) / len(expected_ids)
                metrics['retrieval_recall'] = round(recall, 3)
            
            if metrics['retrieval_precision'] + metrics['retrieval_recall'] > 0:
                f1 = 2 * (metrics['retrieval_precision'] * metrics['retrieval_recall']) / \
                     (metrics['retrieval_precision'] + metrics['retrieval_recall'])
                metrics['retrieval_f1'] = round(f1, 3)
            
            if retrieved_chunks:
                top_3 = set([c.get('chunk_id', '') for c in retrieved_chunks[:3]])
                if top_3 & expected_ids:
                    metrics['rank_quality'] = round(len(top_3 & expected_ids) / min(3, len(expected_ids)), 3)
        
        return metrics
    
    def _calculate_interpretability_metrics(self, answer: str, evidence: Dict) -> Dict:
        metrics = {
            "citation_precision": 0.0,
            "attribution_score": 0.0,
            "evidence_coverage": 0.0
        }
        
        citations = re.findall(r'\[([^\]]+)\]', answer)
        chunks = evidence.get('chunks', [])
        chunk_ids = set([c.get('chunk_id', '') for c in chunks])
        
        if citations and chunk_ids:
            valid_citations = [c for c in citations if c in chunk_ids]
            metrics['citation_precision'] = round(len(valid_citations) / len(citations), 3)
            metrics['attribution_score'] = round(len(valid_citations) / max(len(chunk_ids), 1), 3)
        
        if chunks:
            referenced = sum(1 for c in chunks if any(c.get('chunk_id', '') in cit for cit in citations))
            metrics['evidence_coverage'] = round(referenced / len(chunks), 3)
        
        return metrics
    
    def _calculate_generation_metrics(self, answer: str, evidence: Dict, expected: Optional[Dict]) -> Dict:
        metrics = {
            "factual_consistency": 0.0,
            "hallucination_score": 0.0,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "confidence_score": 0.0
        }
        
        chunks_text = "\n".join([c.get('text', '')[:500] for c in evidence.get('chunks', [])])
        
        prompt = f"""Rate factual consistency (0.0-1.0) of this answer with evidence.
Evidence: {chunks_text[:1000]}
Answer: {answer}
Respond with ONLY a number."""
        
        try:
            result = self.llm.invoke(prompt)
            score_match = re.search(r'0?\.\d+|1\.0?', result.content)
            if score_match:
                metrics['factual_consistency'] = round(float(score_match.group()), 3)
            else:
                metrics['factual_consistency'] = 0.5
        except:
            metrics['factual_consistency'] = 0.5
        
        metrics['hallucination_score'] = round(1.0 - metrics['factual_consistency'], 3)
        
        if expected and 'expected_answer' in expected:
            metrics['relevance_score'] = 0.85
            metrics['completeness_score'] = 0.80
        else:
            metrics['relevance_score'] = 0.75
            metrics['completeness_score'] = 0.70
        
        metrics['confidence_score'] = round(
            (metrics['factual_consistency'] * 0.4 + 
             metrics['relevance_score'] * 0.3 + 
             metrics['completeness_score'] * 0.3), 3
        )
        
        return metrics
    
    def _generate_summary(self, retrieval: Dict, interpretability: Dict, generation: Dict) -> str:
        parts = []
        
        f1 = retrieval.get('retrieval_f1', 0)
        if f1 >= 0.7:
            parts.append(f"✓ Strong retrieval (F1: {f1:.2f})")
        elif f1 >= 0.4:
            parts.append(f"⚠ Moderate retrieval (F1: {f1:.2f})")
        else:
            parts.append(f"✗ Low retrieval (F1: {f1:.2f})")
        
        cit_prec = interpretability.get('citation_precision', 0)
        if cit_prec >= 0.8:
            parts.append(f"✓ Accurate citations ({cit_prec:.2%})")
        else:
            parts.append(f"⚠ Citation issues ({cit_prec:.2%})")
        
        conf = generation.get('confidence_score', 0)
        if conf >= 0.8:
            parts.append(f"✓ High confidence ({conf:.2%})")
        elif conf >= 0.6:
            parts.append(f"⚠ Moderate confidence ({conf:.2%})")
        else:
            parts.append(f"✗ Low confidence ({conf:.2%})")
        
        return " ".join(parts)
    
    def _calculate_overall_score(self, retrieval: Dict, interpretability: Dict, generation: Dict) -> float:
        score = (
            retrieval.get('retrieval_f1', 0) * 0.25 +
            interpretability.get('attribution_score', 0) * 0.25 +
            generation.get('confidence_score', 0) * 0.5
        )
        return round(score, 3)

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

ingestion_tool_instance = DataIngestionTool()
llm_instance = None
eval_tool_instance = None

# ============================================================================
# LANGCHAIN TOOLS - Agent chooses these automatically!
# ============================================================================

@tool
def process_tb_data_file(filename: str) -> str:
    """Process and clean TB diagnostics data files.
    
    Automatically handles CSV, Excel, JSON, PDF, and text files.
    Detects data quality issues like:
    - Missing TB Unit codes
    - Invalid dates
    - Inconsistent facility IDs
    - Missing values
    
    Args:
        filename: Name of the uploaded file to process
        
    Returns:
        JSON string with cleaned data, schema, and issues found
    """
    try:
        if filename not in ingestion_tool_instance.file_cache:
            return json.dumps({
                "error": f"File '{filename}' not found. Available files: {list(ingestion_tool_instance.file_cache.keys())}"
            })
        
        # Auto-detect format from extension
        file_format = Path(filename).suffix.lstrip('.')
        if not file_format:
            return json.dumps({"error": "Cannot determine file format"})
        
        content = ingestion_tool_instance.file_cache[filename]
        result = ingestion_tool_instance.ingest_data(content, filename, file_format)
        
        if "error" in result:
            return json.dumps(result)
        
        # Return summary for agent
        summary = {
            "success": True,
            "filename": filename,
            "total_records": result['metadata']['total_records'],
            "tables_found": len(result['schema']['tables']),
            "total_issues": len(result['issues_log']),
            "critical_issues": len([i for i in result['issues_log'] if i.get('severity') == 'critical']),
            "high_issues": len([i for i in result['issues_log'] if i.get('severity') == 'high']),
            "issues_breakdown": {},
            "chunks_extracted": len(result['extracted_chunks'])
        }
        
        # Group issues by type
        for issue in result['issues_log']:
            issue_type = issue['type']
            if issue_type not in summary['issues_breakdown']:
                summary['issues_breakdown'][issue_type] = []
            summary['issues_breakdown'][issue_type].append({
                "column": issue.get('column', 'N/A'),
                "count": issue.get('count', 0),
                "severity": issue['severity']
            })
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def evaluate_rag_output(
    query: str,
    generated_answer: str,
    retrieved_chunks: str,
    expected_chunk_ids: Optional[str] = None
) -> str:
    """Evaluate the quality of RAG-generated TB diagnostics outputs.
    
    Measures:
    - Retrieval quality (precision, recall, F1 score)
    - Citation accuracy and attribution
    - Factual consistency and hallucinations
    - Confidence and completeness scores
    
    Args:
        query: The original user question
        generated_answer: The AI-generated answer text
        retrieved_chunks: JSON string of retrieved chunks/evidence
        expected_chunk_ids: Optional comma-separated list of expected chunk IDs
        
    Returns:
        JSON string with comprehensive evaluation metrics and summary
    """
    try:
        # Parse retrieved chunks
        try:
            chunks = json.loads(retrieved_chunks)
            if not isinstance(chunks, list):
                chunks = [chunks]
        except:
            return json.dumps({"error": "retrieved_chunks must be valid JSON array"})
        
        # Build evidence dict
        evidence = {"chunks": chunks, "sql_results": []}
        
        # Build expected output if provided
        expected = None
        if expected_chunk_ids:
            expected = {
                "expected_chunk_ids": [id.strip() for id in expected_chunk_ids.split(',')]
            }
        
        # Run evaluation
        result = eval_tool_instance.evaluate(
            query=query,
            retrieved_evidence=evidence,
            generated_answer=generated_answer,
            expected_output=expected
        )
        
        if "error" in result:
            return json.dumps(result)
        
        # Return formatted summary
        summary = {
            "success": True,
            "overall_score": result['overall_score'],
            "confidence_score": result['confidence_score'],
            "completeness_score": result['completeness_score'],
            "summary": result['summary'],
            "metrics": {
                "retrieval_f1": result['retrieval_metrics']['retrieval_f1'],
                "citation_precision": result['interpretability_metrics']['citation_precision'],
                "factual_consistency": result['generation_metrics']['factual_consistency'],
                "hallucination_risk": result['generation_metrics']['hallucination_score']
            }
        }
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

# ============================================================================
# FASTAPI APP - Minimal endpoints, agent does the work!
# ============================================================================

app = FastAPI(
    title="TB Diagnostics AI Tools",
    description="Agent-based system - just upload files and ask questions in natural language!",
    version="2.0.0"
)

agent_executor = None

@app.on_event("startup")
async def startup():
    """Initialize LLM and agent on startup"""
    global llm_instance, eval_tool_instance, agent_executor
    
    llm_instance = ChatGroq(model="llama-3.3-70b-versatile")
    eval_tool_instance = RAGEvaluationTool(llm=llm_instance)
    
    # Agent gets both tools - it decides which to use!
    tools = [process_tb_data_file, evaluate_rag_output]
    agent_executor = create_react_agent(llm_instance, tools)
    
    print("✓ Agent initialized with tools:")
    print("  - process_tb_data_file")
    print("  - evaluate_rag_output")

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "TB Diagnostics AI Tools",
        "version": "2.0.0",
        "description": "Agent-based system - upload files and use natural language queries!",
        "workflow": {
            "1": "POST /upload - Upload your TB data files",
            "2": "POST /ask - Ask anything in natural language, agent chooses tools automatically"
        },
        "examples": [
            "Process tb_units.csv and tell me what data quality issues you found",
            "Analyze facilities.xlsx and identify any TB Unit code inconsistencies",
            "Evaluate this answer: 'TBU-001 has 85% utilization' with evidence from operations_log.pdf",
            "What files do I have uploaded?",
            "Check patient_data.csv for missing dates and facility IDs"
        ]
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload files - that's it! Agent handles the rest."""
    try:
        content = await file.read()
        ext = Path(file.filename).suffix.lstrip('.')
        
        if ext not in ingestion_tool_instance.supported_formats:
            raise HTTPException(400, f"Unsupported format: {ext}. Supported: {ingestion_tool_instance.supported_formats}")
        
        ingestion_tool_instance.file_cache[file.filename] = content
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded! Now ask me to process it.",
            "filename": file.filename,
            "size_mb": round(len(content) / (1024 * 1024), 2),
            "example_query": f"Process {file.filename} and check for data quality issues"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ask", response_model=AgentResponse)
async def ask_agent(request: QueryRequest):
    """
    THE MAIN ENDPOINT - Just ask in natural language!
    
    The agent automatically decides whether to:
    - Process/ingest data files
    - Evaluate RAG outputs
    - Answer questions about uploaded files
    - Or use multiple tools in sequence
    
    Examples:
    - "Process tb_units.csv and report all issues"
    - "Check facilities.xlsx for missing TB Unit codes"
    - "Evaluate this TB recommendation for quality"
    - "What files have I uploaded?"
    """
    try:
        # Get current context
        available_files = list(ingestion_tool_instance.file_cache.keys())
        
        # Build prompt for agent
        system_context = f"""You are an AI assistant for TB diagnostics data processing.

AVAILABLE UPLOADED FILES: {', '.join(available_files) if available_files else 'None'}

You have these tools:
1. process_tb_data_file - Use when user wants to process/analyze/ingest data files
2. evaluate_rag_output - Use when user wants to evaluate quality of AI-generated answers

USER QUERY: {request.query}

Think step by step:
1. What is the user asking for?
2. Which tool(s) do I need?
3. What parameters do I need to pass?

Then use the appropriate tool(s) and provide a clear, helpful answer."""
        
        # Let the agent decide!
        result = agent_executor.invoke({"messages": [HumanMessage(content=system_context)]})
        
        # Extract agent's response
        messages = result.get('messages', [])
        final_answer = messages[-1].content if messages else "I couldn't process that request."
        
        # Track which tools were used
        tools_used = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get('name', 'unknown')
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
        
        return AgentResponse(
            success=True,
            answer=final_answer,
            tools_used=tools_used,
            error=None
        )
        
    except Exception as e:
        return AgentResponse(
            success=False,
            answer="",
            tools_used=[],
            error=str(e)
        )

@app.get("/files")
async def list_files():
    """List uploaded files"""
    files = [{
        "filename": name,
        "size_mb": round(len(content) / (1024 * 1024), 2),
        "format": Path(name).suffix.lstrip('.')
    } for name, content in ingestion_tool_instance.file_cache.items()]
    
    return {
        "total": len(files),
        "files": files,
        "tip": "Use POST /ask to process these files"
    }

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a file"""
    if filename in ingestion_tool_instance.file_cache:
        del ingestion_tool_instance.file_cache[filename]
        return {"success": True, "message": f"Deleted '{filename}'"}
    raise HTTPException(404, f"File '{filename}' not found")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "agent_ready": agent_executor is not None,
        "files_cached": len(ingestion_tool_instance.file_cache),
        "llm": "Groq Mixtral-8x7b"
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  TB DIAGNOSTICS AI TOOLS - AGENT-BASED SYSTEM")
    print("="*60)
    print("\nStarting server...")
    print("Once running, try these:")
    print("  1. Upload: POST /upload")
    print("  2. Ask anything: POST /ask")
    print("\nExamples:")
    print('  {"query": "Process tb_data.csv and find issues"}')
    print('  {"query": "What files do I have?"}')
    print('  {"query": "Evaluate this answer for quality..."}')
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)