"""
STREAMLIT UI FOR UNIVERSAL DATA INGESTION TOOL
Interactive web interface to demonstrate all features
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import zipfile
import io

# Add app.py to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    UniversalDataIngester, 
    CONFIG,
    SchemaRegistry,
    DataProfiler
)

# Page configuration
st.set_page_config(
    page_title="Universal Data Ingestion Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    h1 {
        color: #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ingestion_result' not in st.session_state:
    st.session_state.ingestion_result = None
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# Helper functions
def create_temp_directory():
    """Create temporary directory for file processing"""
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()
    return st.session_state.temp_dir

def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory"""
    file_path = Path(temp_dir) / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def create_download_zip(result, temp_dir):
    """Create a ZIP file with all outputs"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add cleaned datasets
        for key, df in result.cleaned_datasets.items():
            csv_str = df.to_csv(index=False)
            zip_file.writestr(f"cleaned_data_{key}.csv", csv_str)
        
        # Add schema files
        for key, schema in result.schema_files.items():
            json_str = json.dumps(schema, indent=2)
            zip_file.writestr(f"schema_{key}.json", json_str)
        
        # Add issues log
        issues_data = [
            {
                'severity': i.severity,
                'column': i.column,
                'issue': i.issue,
                'affected_rows': i.affected_rows[:10] if len(i.affected_rows) > 10 else i.affected_rows,
                'recommendation': i.recommendation
            }
            for i in result.issues_log
        ]
        zip_file.writestr("issues_log.json", json.dumps(issues_data, indent=2))
        
        # Add RAG chunks
        chunks_data = [
            {
                'chunk_id': c.chunk_id,
                'content': c.content[:500],  # Truncate for size
                'source_file': c.source_file,
                'chunk_type': c.chunk_type,
                'page': c.page,
                'metadata': c.metadata
            }
            for c in result.rag_chunks
        ]
        zip_file.writestr("rag_chunks.json", json.dumps(chunks_data, indent=2))
        
        # Add metadata
        zip_file.writestr("metadata.json", json.dumps(result.metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer

def render_confidence_chart(schema_files):
    """Render confidence score chart"""
    data = []
    for file_name, schema in schema_files.items():
        data.append({
            'File': file_name,
            'Overall Confidence': schema['confidence'],
            'Quality Score': schema['quality_score']
        })
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['File'],
        y=df['Overall Confidence'],
        name='Schema Match Confidence',
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        x=df['File'],
        y=df['Quality Score'],
        name='Data Quality Score',
        marker_color='#38ef7d'
    ))
    
    fig.update_layout(
        title="Confidence & Quality Scores by File",
        xaxis_title="File",
        yaxis_title="Score (%)",
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig

def render_method_distribution(schema_files):
    """Render matching method distribution"""
    method_counts = {}
    
    for schema in schema_files.values():
        for mapping in schema['field_mappings']:
            method = mapping['method']
            method_counts[method] = method_counts.get(method, 0) + 1
    
    fig = px.pie(
        values=list(method_counts.values()),
        names=list(method_counts.keys()),
        title="Field Matching Methods Used",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def render_issues_chart(issues_log):
    """Render issues severity distribution"""
    severity_counts = {}
    for issue in issues_log:
        severity = issue.severity
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    if not severity_counts:
        return None
    
    colors = {
        'critical': '#dc2626',
        'high': '#ea580c',
        'medium': '#f59e0b',
        'low': '#84cc16',
        'info': '#3b82f6'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=[colors.get(k, '#667eea') for k in severity_counts.keys()]
        )
    ])
    
    fig.update_layout(
        title="Data Quality Issues by Severity",
        xaxis_title="Severity",
        yaxis_title="Count",
        height=400
    )
    
    return fig

# ==================== MAIN UI ====================

# Header
st.title("📊 Universal Data Ingestion & Preparation Tool")
st.markdown("**Hybrid AI-Powered Data Ingestion**: Rule-Based + Semantic Embeddings + LLM Reasoning")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Schema selection
    st.subheader("Schema Files")
    schema_option = st.radio(
        "Schema Source:",
        ["Use Sample Schemas", "Upload Custom Schemas"]
    )
    
    if schema_option == "Use Sample Schemas":
        default_schemas = [
            "/mnt/user-data/uploads/dm_patients.json",
            "/mnt/user-data/uploads/dm_regions.json",
            "/mnt/user-data/uploads/dm_talukas.json"
        ]
        schema_files = default_schemas
        st.success(f"✅ Using {len(schema_files)} default schemas")
        
        # Show schemas
        with st.expander("View Schemas"):
            for schema_path in schema_files:
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                    st.write(f"**{schema['table_name']}**")
                    st.caption(f"{len(schema['schema'])} fields")
    else:
        uploaded_schemas = st.file_uploader(
            "Upload Schema JSON files",
            type=['json'],
            accept_multiple_files=True,
            key='schema_upload'
        )
        
        if uploaded_schemas:
            temp_dir = create_temp_directory()
            schema_files = []
            for schema_file in uploaded_schemas:
                path = save_uploaded_file(schema_file, temp_dir)
                schema_files.append(path)
            st.success(f"✅ Loaded {len(schema_files)} schemas")
        else:
            schema_files = None
            st.warning("⚠️ Please upload schema files")
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=5.0
        )
        
        use_llm = st.checkbox(
            "Enable LLM Fallback (requires Groq API key)",
            value=False
        )
        
        if use_llm:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_..."
            )
            if api_key:
                CONFIG['groq_api_key'] = api_key
        
        chunk_size = st.number_input(
            "RAG Chunk Size (tokens)",
            min_value=100,
            max_value=1000,
            value=500
        )
        
        CONFIG['min_confidence_threshold'] = confidence_threshold
        CONFIG['chunk_size'] = chunk_size
    
    st.divider()
    
    # Info
    st.info("""
    **How it works:**
    1. Upload data files
    2. Tool matches to schemas
    3. Cleans & transforms data
    4. Generates all outputs
    """)

# Main content
tab1, tab2 = st.tabs(["📥 Upload & Process", "📊 Results Dashboard"])

# ==================== TAB 1: UPLOAD & PROCESS ====================
with tab1:
    st.header("Upload Data Files")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files to ingest",
            type=['csv', 'xlsx', 'xls', 'json', 'pdf', 'txt'],
            accept_multiple_files=True,
            help="Supports: CSV, Excel, JSON, PDF, Text files"
        )
    
    with col2:
        st.markdown("### 📋 Supported Formats")
        st.markdown("""
        - 📄 CSV/TSV
        - 📊 Excel (xlsx, xls)
        - 🗂️ JSON
        - 📑 PDF (with tables)
        - 📝 Text files
        """)
    
    if uploaded_files and schema_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready for processing")
        
        # Show file preview
        with st.expander("🔍 Preview Uploaded Files"):
            for file in uploaded_files:
                st.write(f"**{file.name}**")
                st.caption(f"Size: {file.size / 1024:.1f} KB | Type: {file.type}")
        
        # Process button
        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            
            # Save files temporarily
            temp_dir = create_temp_directory()
            file_paths = []
            
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file, temp_dir)
                file_paths.append(file_path)
            
            # Process with progress bar
            with st.spinner("🔄 Processing files..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize ingester
                    status_text.text("Initializing ingester...")
                    progress_bar.progress(10)
                    
                    ingester = UniversalDataIngester(schema_files, CONFIG)
                    
                    # Process files
                    status_text.text("Processing files...")
                    progress_bar.progress(30)
                    
                    result = ingester.ingest_batch(file_paths)
                    
                    progress_bar.progress(90)
                    status_text.text("Finalizing results...")
                    
                    # Store in session state
                    st.session_state.ingestion_result = result
                    st.session_state.uploaded_files_processed = True
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    # Success message
                    st.balloons()
                    st.success("✅ Ingestion completed successfully!")
                    
                    # Show summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⏱️ {result.metadata['processing_time_seconds']:.1f}s</h3>
                            <p>Processing Time</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="success-card">
                            <h3>✅ {result.metadata['files_succeeded']}</h3>
                            <p>Files Processed</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="info-card">
                            <h3>📊 {result.metadata['total_rows']:,}</h3>
                            <p>Total Rows</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        severity = "success-card" if result.metadata['total_issues'] < 5 else "warning-card"
                        st.markdown(f"""
                        <div class="{severity}">
                            <h3>⚠️ {result.metadata['total_issues']}</h3>
                            <p>Issues Found</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("👉 Switch to **Results Dashboard** tab to view detailed results")
                    
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {str(e)}")
                    st.exception(e)
                    
    elif not schema_files:
        st.warning("⚠️ Please configure schema files in the sidebar first")
    else:
        st.info("👆 Upload files above to get started")

# ==================== TAB 2: RESULTS DASHBOARD ====================
with tab2:
    if st.session_state.ingestion_result:
        result = st.session_state.ingestion_result
        
        st.header("📊 Ingestion Results Dashboard")
        
        # Download all button
        col1, col2 = st.columns([3, 1])
        with col2:
            zip_buffer = create_download_zip(result, st.session_state.temp_dir)
            st.download_button(
                label="📦 Download All Outputs",
                data=zip_buffer,
                file_name=f"ingestion_outputs_{result.batch_id}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
        
        st.divider()
        
        # Tabs for each output
        output_tabs = st.tabs([
            "1️⃣ Cleaned Data",
            "2️⃣ Schema Files",
            "3️⃣ Issues Log",
            "4️⃣ RAG Chunks",
            "5️⃣ Metadata"
        ])
        
        # OUTPUT 1: CLEANED DATA
        with output_tabs[0]:
            st.subheader("🧹 Cleaned Structured Datasets")
            st.markdown("*Database-ready, cleaned and standardized data*")
            
            if result.cleaned_datasets:
                for file_key, df in result.cleaned_datasets.items():
                    with st.expander(f"📄 {file_key} ({len(df)} rows × {len(df.columns)} columns)", expanded=True):
                        
                        # Stats
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Rows", f"{len(df):,}")
                        col2.metric("Columns", len(df.columns))
                        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                        
                        null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                        col4.metric("Null %", f"{null_pct:.1f}%")
                        
                        # Data preview
                        st.markdown("**Data Preview:**")
                        st.dataframe(df.head(100), use_container_width=True, height=400)
                        
                        # Column info
                        st.markdown("**Column Information:**")
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Type': df.dtypes.astype(str),
                            'Non-Null': df.count(),
                            'Null %': (df.isnull().sum() / len(df) * 100).round(1),
                            'Unique': df.nunique()
                        })
                        st.dataframe(col_info, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {file_key}.csv",
                            data=csv,
                            file_name=f"cleaned_{file_key}.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("No cleaned datasets available")
        
        # OUTPUT 2: SCHEMA FILES
        with output_tabs[1]:
            st.subheader("📋 Schema Definitions & Mappings")
            st.markdown("*Field mappings with confidence scores and methods*")
            
            if result.schema_files:
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        render_confidence_chart(result.schema_files),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        render_method_distribution(result.schema_files),
                        use_container_width=True
                    )
                
                st.divider()
                
                # Detailed mappings
                for file_key, schema in result.schema_files.items():
                    with st.expander(f"📄 {file_key} → {schema['detected_schema']}", expanded=True):
                        
                        # Overall info
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Detected Schema", schema['detected_schema'])
                        col2.metric("Confidence", f"{schema['confidence']:.1f}%")
                        col3.metric("Quality Score", f"{schema['quality_score']:.1f}%")
                        col4.metric("Matched Fields", len(schema['field_mappings']))
                        
                        # Field mappings table
                        st.markdown("**Field Mappings:**")
                        mapping_df = pd.DataFrame([
                            {
                                'Data Column': m['data_column'],
                                'Schema Field': m['schema_field'],
                                'Confidence': f"{m['confidence']:.1f}%",
                                'Method': m['method'],
                                'Data Type': m['data_type'],
                                'Schema Type': m['schema_type']
                            }
                            for m in schema['field_mappings']
                        ])
                        
                        # Color code by confidence
                        def highlight_confidence(row):
                            conf = float(row['Confidence'].rstrip('%'))
                            if conf >= 90:
                                return ['background-color: #d4edda'] * len(row)
                            elif conf >= 75:
                                return ['background-color: #fff3cd'] * len(row)
                            else:
                                return ['background-color: #f8d7da'] * len(row)
                        
                        st.dataframe(
                            mapping_df.style.apply(highlight_confidence, axis=1),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Unmapped columns
                        if schema['unmapped_columns']:
                            st.warning(f"**⚠️ Unmapped Columns ({len(schema['unmapped_columns'])}):** {', '.join(schema['unmapped_columns'])}")
                        
                        # Missing required fields
                        if schema['missing_required_fields']:
                            st.error(f"**❌ Missing Required Fields:** {', '.join(schema['missing_required_fields'])}")
                        
                        # Download schema
                        schema_json = json.dumps(schema, indent=2)
                        st.download_button(
                            label=f"📥 Download schema_{file_key}.json",
                            data=schema_json,
                            file_name=f"schema_{file_key}.json",
                            mime="application/json"
                        )
            else:
                st.warning("No schema files available")
        
        # OUTPUT 3: ISSUES LOG
        with output_tabs[2]:
            st.subheader("⚠️ Data Quality Issues")
            st.markdown("*Detected issues with severity levels and recommendations*")
            
            if result.issues_log:
                # Chart
                chart = render_issues_chart(result.issues_log)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                st.divider()
                
                # Filter by severity
                severities = list(set(i.severity for i in result.issues_log))
                selected_severity = st.multiselect(
                    "Filter by Severity:",
                    severities,
                    default=severities
                )
                
                # Display issues
                filtered_issues = [i for i in result.issues_log if i.severity in selected_severity]
                
                for i, issue in enumerate(filtered_issues, 1):
                    severity_color = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢',
                        'info': '🔵'
                    }
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{severity_color.get(issue.severity, '⚪')} Issue #{i}:** {issue.issue}")
                            if issue.column:
                                st.caption(f"Column: `{issue.column}`")
                            if issue.affected_rows:
                                st.caption(f"Affected rows: {len(issue.affected_rows)} rows")
                        
                        with col2:
                            st.markdown(f"**Severity:** `{issue.severity.upper()}`")
                        
                        if issue.recommendation:
                            st.info(f"💡 **Recommendation:** {issue.recommendation}")
                        
                        st.divider()
                
                # Download issues
                issues_json = json.dumps([
                    {
                        'severity': i.severity,
                        'column': i.column,
                        'issue': i.issue,
                        'affected_rows': i.affected_rows[:100],
                        'recommendation': i.recommendation
                    }
                    for i in result.issues_log
                ], indent=2)
                
                st.download_button(
                    label="📥 Download issues_log.json",
                    data=issues_json,
                    file_name="issues_log.json",
                    mime="application/json"
                )
            else:
                st.success("✅ No issues found! Data quality is excellent.")
        
        # OUTPUT 4: RAG CHUNKS
        with output_tabs[3]:
            st.subheader("📦 RAG Chunks")
            st.markdown("*Text and table chunks for vector database / RAG systems*")
            
            if result.rag_chunks:
                # Stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Chunks", len(result.rag_chunks))
                
                text_chunks = sum(1 for c in result.rag_chunks if c.chunk_type == 'text')
                col2.metric("Text Chunks", text_chunks)
                
                table_chunks = sum(1 for c in result.rag_chunks if c.chunk_type == 'table')
                col3.metric("Table Chunks", table_chunks)
                
                avg_size = sum(len(c.content) for c in result.rag_chunks) / len(result.rag_chunks)
                col4.metric("Avg Size", f"{avg_size:.0f} chars")
                
                st.divider()
                
                # Filter
                chunk_type_filter = st.selectbox(
                    "Filter by Type:",
                    ["All", "Text", "Table"]
                )
                
                filtered_chunks = result.rag_chunks
                if chunk_type_filter != "All":
                    filtered_chunks = [c for c in result.rag_chunks if c.chunk_type.lower() == chunk_type_filter.lower()]
                
                # Display chunks
                for i, chunk in enumerate(filtered_chunks[:50], 1):  # Limit to 50 for performance
                    with st.expander(f"Chunk {i}: {chunk.chunk_id}"):
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"**Type:** {chunk.chunk_type}")
                        col2.write(f"**Source:** {Path(chunk.source_file).name}")
                        if chunk.page is not None:
                            col3.write(f"**Page:** {chunk.page}")
                        
                        st.markdown("**Content:**")
                        st.text_area(
                            label="Chunk Content",
                            value=chunk.content,
                            height=150,
                            key=f"chunk_{i}",
                            label_visibility="collapsed"
                        )
                        
                        if chunk.metadata:
                            st.json(chunk.metadata)
                
                if len(filtered_chunks) > 50:
                    st.info(f"Showing 50 of {len(filtered_chunks)} chunks")
                
                # Download chunks
                chunks_json = json.dumps([
                    {
                        'chunk_id': c.chunk_id,
                        'content': c.content,
                        'source_file': c.source_file,
                        'chunk_type': c.chunk_type,
                        'page': c.page,
                        'metadata': c.metadata
                    }
                    for c in result.rag_chunks
                ], indent=2)
                
                st.download_button(
                    label="📥 Download rag_chunks.json",
                    data=chunks_json,
                    file_name="rag_chunks.json",
                    mime="application/json"
                )
            else:
                st.info("No RAG chunks generated")
        
        # OUTPUT 5: METADATA
        with output_tabs[4]:
            st.subheader("📊 Processing Metadata")
            st.markdown("*Complete information about the ingestion process*")
            
            # Display as formatted JSON
            st.json(result.metadata)
            
            # Key metrics
            st.divider()
            st.markdown("### Key Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Batch ID", result.metadata['batch_id'][:8] + "...")
                st.metric("Files Processed", f"{result.metadata['files_succeeded']}/{result.metadata['files_processed']}")
            
            with col2:
                st.metric("Total Rows", f"{result.metadata['total_rows']:,}")
                st.metric("Total Issues", result.metadata['total_issues'])
            
            with col3:
                st.metric("Processing Time", f"{result.metadata['processing_time_seconds']:.2f}s")
                st.metric("Total Chunks", result.metadata['total_chunks'])
            
            # Download metadata
            metadata_json = json.dumps(result.metadata, indent=2)
            st.download_button(
                label="📥 Download metadata.json",
                data=metadata_json,
                file_name="metadata.json",
                mime="application/json"
            )
            
    else:
        st.info("👈 Process files in the **Upload & Process** tab first to see results here")



# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Universal Data Ingestion Tool v1.0</strong></p>
    <p>Powered by Rule-Based Matching + Semantic Embeddings + LLM Reasoning</p>
    <p>Built with ❤️ using Streamlit, FastAPI, and Groq AI</p>
</div>
""", unsafe_allow_html=True)
