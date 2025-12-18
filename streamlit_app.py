"""
Streamlit Dashboard: Milvus vs Weaviate Benchmark Results
=========================================================

Interactive visualization of vector database benchmark results.
Parses complete_benchmark_*.txt files for all data.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Milvus vs Weaviate Benchmark",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .winner-badge {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def get_benchmark_files():
    """Get all complete_benchmark_*.txt files from results directory."""
    results_dir = Path('results')
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob('complete_benchmark_*.txt'), reverse=True)


def parse_benchmark_file(filepath):
    """
    Parse a complete_benchmark_*.txt file and extract all data.
    
    Returns a dict with all benchmark data.
    """
    data = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'timestamp': '',
        'date': '',
        'dataset': '',
        'vectors': 0,
        'dimensions': 0,
        'loading': {
            'Milvus': {'load_time': 0, 'peak_memory': 0},
            'Weaviate': {'load_time': 0, 'peak_memory': 0}
        },
        'raw_data_size': 0,
        'query_performance': None,  # DataFrame
        'recall': None  # DataFrame
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract timestamp from filename
        ts_match = re.search(r'(\d{8}_\d{6})', filepath.name)
        if ts_match:
            ts = ts_match.group(1)
            try:
                dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
                data['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                data['timestamp'] = ts
        
        # Extract date from content
        date_match = re.search(r'Date:\s*(.+)', content)
        if date_match:
            data['date'] = date_match.group(1).strip()
        
        # Extract dataset info (handle names with hyphens like glove-200)
        dataset_match = re.search(r'Dataset:\s*([\w-]+)\s*\(([0-9,]+)\s*vectors,\s*(\d+)D\)', content)
        if dataset_match:
            data['dataset'] = dataset_match.group(1)
            data['vectors'] = int(dataset_match.group(2).replace(',', ''))
            data['dimensions'] = int(dataset_match.group(3))
        
        # Extract Milvus loading stats
        milvus_section = re.search(r'Milvus:\s*\n\s*Load Time:\s*([\d.]+)\s*seconds\s*\n\s*Peak Memory:\s*([\d.]+)\s*MB', content)
        if milvus_section:
            data['loading']['Milvus']['load_time'] = float(milvus_section.group(1))
            data['loading']['Milvus']['peak_memory'] = float(milvus_section.group(2))
        
        # Extract Weaviate loading stats
        weaviate_section = re.search(r'Weaviate:\s*\n\s*Load Time:\s*([\d.]+)\s*seconds\s*\n\s*Peak Memory:\s*([\d.]+)\s*MB', content)
        if weaviate_section:
            data['loading']['Weaviate']['load_time'] = float(weaviate_section.group(1))
            data['loading']['Weaviate']['peak_memory'] = float(weaviate_section.group(2))
        
        # Extract raw data size
        size_match = re.search(r'Raw Data Size:\s*([\d.]+)\s*MB', content)
        if size_match:
            data['raw_data_size'] = float(size_match.group(1))
        
        # Extract query performance table
        perf_match = re.search(r'QUERY PERFORMANCE\s*-+\s*\n\s*\n(.+?)(?=\n-{10,}|\n={10,}|$)', content, re.DOTALL)
        if perf_match:
            table_text = perf_match.group(1).strip()
            lines = [l.strip() for l in table_text.split('\n') if l.strip()]
            
            if len(lines) >= 2:
                perf_data = []
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            row = {
                                'Database': parts[1],
                                'Test': parts[2],
                                'P50 (ms)': float(parts[3]),
                                'P95 (ms)': float(parts[4]),
                                'P99 (ms)': float(parts[5]),
                                'Mean (ms)': float(parts[6]),
                                'QPS': float(parts[7]) if len(parts) > 7 else 0
                            }
                            perf_data.append(row)
                        except (ValueError, IndexError):
                            continue
                
                if perf_data:
                    data['query_performance'] = pd.DataFrame(perf_data)
        
        # Extract recall data
        recall_matches = re.findall(r'Recall@(\d+):\s*Milvus=([\d.]+),\s*Weaviate=([\d.]+)', content)
        if recall_matches:
            recall_data = []
            for k, m, w in recall_matches:
                recall_data.append({
                    'K': int(k),
                    'Milvus': float(m),
                    'Weaviate': float(w)
                })
            data['recall'] = pd.DataFrame(recall_data)
    
    except Exception as e:
        st.error(f"Error parsing {filepath}: {e}")
    
    return data


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_loading_time_chart(benchmark):
    """Create a horizontal bar chart comparing load times."""
    load_data = {
        'Database': ['Milvus', 'Weaviate'],
        'Load Time (s)': [
            benchmark['loading']['Milvus']['load_time'],
            benchmark['loading']['Weaviate']['load_time']
        ]
    }
    df = pd.DataFrame(load_data)
    
    fig = go.Figure()
    
    colors = ['#667eea', '#e17055']
    
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Database']],
            x=[row['Load Time (s)']],
            orientation='h',
            name=row['Database'],
            marker_color=colors[i],
            text=[f"{row['Load Time (s)']:.1f}s"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="⏱️ Data Loading Time",
        xaxis_title="Time (seconds)",
        yaxis_title="",
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_memory_chart(benchmark):
    """Create a horizontal bar chart comparing peak memory usage."""
    mem_data = {
        'Database': ['Milvus', 'Weaviate'],
        'Peak Memory (MB)': [
            benchmark['loading']['Milvus']['peak_memory'],
            benchmark['loading']['Weaviate']['peak_memory']
        ]
    }
    df = pd.DataFrame(mem_data)
    
    fig = go.Figure()
    
    colors = ['#667eea', '#e17055']
    
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Database']],
            x=[row['Peak Memory (MB)']],
            orientation='h',
            name=row['Database'],
            marker_color=colors[i],
            text=[f"{row['Peak Memory (MB)']:.1f} MB"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="💾 Peak Memory Usage",
        xaxis_title="Memory (MB)",
        yaxis_title="",
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_latency_comparison_chart(perf_df):
    """Create grouped bar chart comparing P50/P95/P99 latencies."""
    if perf_df is None or perf_df.empty:
        return None
    
    # Filter to no-filter tests for cleaner comparison
    df = perf_df[perf_df['Test'].str.contains('nofilter')].copy()
    
    fig = go.Figure()
    
    milvus_df = df[df['Database'] == 'Milvus']
    weaviate_df = df[df['Database'] == 'Weaviate']
    
    tests = milvus_df['Test'].tolist()
    test_labels = [t.replace('_nofilter', '').upper() for t in tests]
    
    fig.add_trace(go.Bar(
        name='Milvus P50',
        x=test_labels,
        y=milvus_df['P50 (ms)'].tolist(),
        marker_color='#667eea',
        text=[f"{v:.1f}" for v in milvus_df['P50 (ms)']],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        name='Weaviate P50',
        x=test_labels,
        y=weaviate_df['P50 (ms)'].tolist(),
        marker_color='#e17055',
        text=[f"{v:.1f}" for v in weaviate_df['P50 (ms)']],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="⚡ Query Latency (P50) - Lower is Better",
        xaxis_title="Query Type",
        yaxis_title="Latency (ms)",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_qps_chart(perf_df):
    """Create bar chart comparing QPS (throughput)."""
    if perf_df is None or perf_df.empty:
        return None
    
    df = perf_df[perf_df['Test'].str.contains('nofilter')].copy()
    
    fig = go.Figure()
    
    milvus_df = df[df['Database'] == 'Milvus']
    weaviate_df = df[df['Database'] == 'Weaviate']
    
    tests = milvus_df['Test'].tolist()
    test_labels = [t.replace('_nofilter', '').upper() for t in tests]
    
    fig.add_trace(go.Bar(
        name='Milvus',
        x=test_labels,
        y=milvus_df['QPS'].tolist(),
        marker_color='#667eea',
        text=[f"{v:.0f}" for v in milvus_df['QPS']],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        name='Weaviate',
        x=test_labels,
        y=weaviate_df['QPS'].tolist(),
        marker_color='#e17055',
        text=[f"{v:.0f}" for v in weaviate_df['QPS']],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="🚀 Throughput (QPS) - Higher is Better",
        xaxis_title="Query Type",
        yaxis_title="Queries per Second",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_latency_heatmap(perf_df):
    """Create heatmap showing all latency metrics."""
    if perf_df is None or perf_df.empty:
        return None
    
    metrics = ['P50 (ms)', 'P95 (ms)', 'P99 (ms)']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Milvus Latencies', 'Weaviate Latencies'),
        horizontal_spacing=0.15
    )
    
    for idx, db in enumerate(['Milvus', 'Weaviate']):
        df_db = perf_df[perf_df['Database'] == db]
        
        z_data = df_db[metrics].values
        x_labels = ['P50', 'P95', 'P99']
        y_labels = df_db['Test'].tolist()
        
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlGn_r',
                showscale=(idx == 1),
                text=[[f"{v:.1f}" for v in row] for row in z_data],
                texttemplate="%{text}",
                textfont={"size": 10},
            ),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        title="📊 Latency Heatmap (ms) - Cooler Colors = Faster",
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    
    return fig


def create_recall_chart(recall_df):
    """Create line chart comparing recall@K."""
    if recall_df is None or recall_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall_df['K'],
        y=recall_df['Milvus'],
        mode='lines+markers+text',
        name='Milvus',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10),
        text=[f"{v:.2%}" for v in recall_df['Milvus']],
        textposition='top center',
    ))
    
    fig.add_trace(go.Scatter(
        x=recall_df['K'],
        y=recall_df['Weaviate'],
        mode='lines+markers+text',
        name='Weaviate',
        line=dict(color='#e17055', width=3),
        marker=dict(size=10),
        text=[f"{v:.2%}" for v in recall_df['Weaviate']],
        textposition='bottom center',
    ))
    
    fig.update_layout(
        title="🎯 Recall@K (Search Accuracy) - Higher is Better",
        xaxis_title="K (Top-K Results)",
        yaxis_title="Recall",
        yaxis=dict(range=[0, 1.05], tickformat='.0%'),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_radar_chart(benchmark):
    """Create radar chart for overall comparison."""
    perf_df = benchmark['query_performance']
    recall_df = benchmark['recall']
    
    if perf_df is None:
        return None
    
    milvus_scores = []
    weaviate_scores = []
    categories = []
    
    # Load time (lower is better, invert)
    m_load = benchmark['loading']['Milvus']['load_time']
    w_load = benchmark['loading']['Weaviate']['load_time']
    max_load = max(m_load, w_load)
    if max_load > 0:
        milvus_scores.append((1 - m_load/max_load) * 100)
        weaviate_scores.append((1 - w_load/max_load) * 100)
        categories.append('Load Speed')
    
    # Memory (lower is better, invert)
    m_mem = benchmark['loading']['Milvus']['peak_memory']
    w_mem = benchmark['loading']['Weaviate']['peak_memory']
    max_mem = max(m_mem, w_mem)
    if max_mem > 0:
        milvus_scores.append((1 - m_mem/max_mem) * 100)
        weaviate_scores.append((1 - w_mem/max_mem) * 100)
        categories.append('Memory Efficiency')
    
    # P50 Latency for k10 (lower is better, invert)
    m_p50 = perf_df[(perf_df['Database'] == 'Milvus') & (perf_df['Test'] == 'k10_nofilter')]['P50 (ms)'].values
    w_p50 = perf_df[(perf_df['Database'] == 'Weaviate') & (perf_df['Test'] == 'k10_nofilter')]['P50 (ms)'].values
    if len(m_p50) > 0 and len(w_p50) > 0:
        max_p50 = max(m_p50[0], w_p50[0])
        if max_p50 > 0:
            milvus_scores.append((1 - m_p50[0]/max_p50) * 100)
            weaviate_scores.append((1 - w_p50[0]/max_p50) * 100)
            categories.append('Query Speed')
    
    # QPS for k10 (higher is better)
    m_qps = perf_df[(perf_df['Database'] == 'Milvus') & (perf_df['Test'] == 'k10_nofilter')]['QPS'].values
    w_qps = perf_df[(perf_df['Database'] == 'Weaviate') & (perf_df['Test'] == 'k10_nofilter')]['QPS'].values
    if len(m_qps) > 0 and len(w_qps) > 0:
        max_qps = max(m_qps[0], w_qps[0])
        if max_qps > 0:
            milvus_scores.append((m_qps[0]/max_qps) * 100)
            weaviate_scores.append((w_qps[0]/max_qps) * 100)
            categories.append('Throughput')
    
    # Recall@10 (higher is better)
    if recall_df is not None and not recall_df.empty:
        r10 = recall_df[recall_df['K'] == 10]
        if not r10.empty:
            milvus_scores.append(r10['Milvus'].values[0] * 100)
            weaviate_scores.append(r10['Weaviate'].values[0] * 100)
            categories.append('Accuracy')
    
    if not categories:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=milvus_scores + [milvus_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Milvus',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)',
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=weaviate_scores + [weaviate_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Weaviate',
        line_color='#e17055',
        fillcolor='rgba(225, 112, 85, 0.3)',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="🏆 Overall Performance Comparison",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    
    return fig


def create_filter_impact_chart(perf_df):
    """Create chart showing impact of filters on performance."""
    if perf_df is None or perf_df.empty:
        return None
    
    has_filters = perf_df['Test'].str.contains('filter').any()
    if not has_filters:
        return None
    
    fig = go.Figure()
    
    for db, color in [('Milvus', '#667eea'), ('Weaviate', '#e17055')]:
        df_db = perf_df[perf_df['Database'] == db]
        
        tests = ['k10_nofilter', 'k10_filter', 'k100_nofilter', 'k100_filter']
        labels = ['K10', 'K10+Filter', 'K100', 'K100+Filter']
        
        values = []
        for t in tests:
            row = df_db[df_db['Test'] == t]
            if not row.empty:
                values.append(row['P50 (ms)'].values[0])
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=db,
            x=labels,
            y=values,
            marker_color=color,
            text=[f"{v:.1f}" for v in values],
            textposition='outside',
        ))
    
    fig.update_layout(
        title="🔍 Filter Impact on Latency",
        xaxis_title="Query Configuration",
        yaxis_title="P50 Latency (ms)",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


# =============================================================================
# TOTAL ANALYSIS CHARTS (Cross-benchmark)
# =============================================================================

def create_load_time_comparison_all(benchmarks):
    """Bar chart comparing load times across all datasets."""
    data = []
    for b in benchmarks:
        if b['dataset']:
            data.append({'Dataset': b['dataset'], 'Database': 'Milvus', 'Load Time (s)': b['loading']['Milvus']['load_time']})
            data.append({'Dataset': b['dataset'], 'Database': 'Weaviate', 'Load Time (s)': b['loading']['Weaviate']['load_time']})
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x='Dataset', y='Load Time (s)', color='Database',
        barmode='group',
        color_discrete_map={'Milvus': '#667eea', 'Weaviate': '#e17055'},
        text='Load Time (s)'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
    fig.update_layout(
        title="⏱️ Load Time Comparison Across All Datasets",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_memory_comparison_all(benchmarks):
    """Bar chart comparing memory usage across all datasets."""
    data = []
    for b in benchmarks:
        if b['dataset']:
            data.append({'Dataset': b['dataset'], 'Database': 'Milvus', 'Peak Memory (MB)': b['loading']['Milvus']['peak_memory']})
            data.append({'Dataset': b['dataset'], 'Database': 'Weaviate', 'Peak Memory (MB)': b['loading']['Weaviate']['peak_memory']})
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x='Dataset', y='Peak Memory (MB)', color='Database',
        barmode='group',
        color_discrete_map={'Milvus': '#667eea', 'Weaviate': '#e17055'},
        text='Peak Memory (MB)'
    )
    
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(
        title="💾 Peak Memory Usage Across All Datasets",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_scalability_chart(benchmarks):
    """Scatter plot showing how performance scales with dataset size."""
    data = []
    for b in benchmarks:
        if b['dataset'] and b['query_performance'] is not None:
            # Get k10 no filter P50
            perf = b['query_performance']
            m_row = perf[(perf['Database'] == 'Milvus') & (perf['Test'] == 'k10_nofilter')]
            w_row = perf[(perf['Database'] == 'Weaviate') & (perf['Test'] == 'k10_nofilter')]
            
            if not m_row.empty:
                data.append({
                    'Dataset': b['dataset'],
                    'Vectors': b['vectors'],
                    'Database': 'Milvus',
                    'P50 Latency (ms)': m_row['P50 (ms)'].values[0]
                })
            if not w_row.empty:
                data.append({
                    'Dataset': b['dataset'],
                    'Vectors': b['vectors'],
                    'Database': 'Weaviate',
                    'P50 Latency (ms)': w_row['P50 (ms)'].values[0]
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.scatter(
        df, x='Vectors', y='P50 Latency (ms)', color='Database',
        size='P50 Latency (ms)',
        hover_data=['Dataset'],
        color_discrete_map={'Milvus': '#667eea', 'Weaviate': '#e17055'},
    )
    
    # Add trend lines
    for db, color in [('Milvus', '#667eea'), ('Weaviate', '#e17055')]:
        df_db = df[df['Database'] == db].sort_values('Vectors')
        if len(df_db) > 1:
            fig.add_trace(go.Scatter(
                x=df_db['Vectors'],
                y=df_db['P50 Latency (ms)'],
                mode='lines',
                name=f'{db} trend',
                line=dict(color=color, dash='dash', width=2),
                showlegend=False
            ))
    
    fig.update_layout(
        title="📈 Scalability: Latency vs Dataset Size (K10 queries)",
        xaxis_title="Number of Vectors",
        yaxis_title="P50 Latency (ms)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_qps_scalability_chart(benchmarks):
    """Line chart showing QPS across different dataset sizes."""
    data = []
    for b in benchmarks:
        if b['dataset'] and b['query_performance'] is not None:
            perf = b['query_performance']
            m_row = perf[(perf['Database'] == 'Milvus') & (perf['Test'] == 'k10_nofilter')]
            w_row = perf[(perf['Database'] == 'Weaviate') & (perf['Test'] == 'k10_nofilter')]
            
            if not m_row.empty:
                data.append({
                    'Dataset': b['dataset'],
                    'Vectors': b['vectors'],
                    'Database': 'Milvus',
                    'QPS': m_row['QPS'].values[0]
                })
            if not w_row.empty:
                data.append({
                    'Dataset': b['dataset'],
                    'Vectors': b['vectors'],
                    'Database': 'Weaviate',
                    'QPS': w_row['QPS'].values[0]
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    for db, color in [('Milvus', '#667eea'), ('Weaviate', '#e17055')]:
        df_db = df[df['Database'] == db].sort_values('Vectors')
        fig.add_trace(go.Scatter(
            x=df_db['Dataset'],
            y=df_db['QPS'],
            mode='lines+markers+text',
            name=db,
            line=dict(color=color, width=3),
            marker=dict(size=12),
            text=[f"{v:.0f}" for v in df_db['QPS']],
            textposition='top center',
        ))
    
    fig.update_layout(
        title="🚀 Throughput (QPS) Across Datasets",
        xaxis_title="Dataset",
        yaxis_title="Queries per Second",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_recall_comparison_all(benchmarks):
    """Compare Recall@10 across all datasets that have recall data."""
    data = []
    for b in benchmarks:
        if b['dataset'] and b['recall'] is not None:
            r10 = b['recall'][b['recall']['K'] == 10]
            if not r10.empty:
                data.append({
                    'Dataset': b['dataset'],
                    'Database': 'Milvus',
                    'Recall@10': r10['Milvus'].values[0]
                })
                data.append({
                    'Dataset': b['dataset'],
                    'Database': 'Weaviate',
                    'Recall@10': r10['Weaviate'].values[0]
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x='Dataset', y='Recall@10', color='Database',
        barmode='group',
        color_discrete_map={'Milvus': '#667eea', 'Weaviate': '#e17055'},
        text='Recall@10'
    )
    
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        title="🎯 Recall@10 Accuracy Across Datasets",
        yaxis=dict(range=[0, 1.1], tickformat='.0%'),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_win_rate_chart(benchmarks):
    """Pie charts showing win rates across all benchmarks."""
    wins = {'Milvus': {'load_time': 0, 'memory': 0, 'latency': 0, 'qps': 0, 'recall': 0},
            'Weaviate': {'load_time': 0, 'memory': 0, 'latency': 0, 'qps': 0, 'recall': 0}}
    
    for b in benchmarks:
        if not b['dataset']:
            continue
        
        # Load time (lower is better)
        if b['loading']['Milvus']['load_time'] < b['loading']['Weaviate']['load_time']:
            wins['Milvus']['load_time'] += 1
        else:
            wins['Weaviate']['load_time'] += 1
        
        # Memory (lower is better)
        if b['loading']['Milvus']['peak_memory'] < b['loading']['Weaviate']['peak_memory']:
            wins['Milvus']['memory'] += 1
        else:
            wins['Weaviate']['memory'] += 1
        
        # Latency and QPS
        if b['query_performance'] is not None:
            perf = b['query_performance']
            m_row = perf[(perf['Database'] == 'Milvus') & (perf['Test'] == 'k10_nofilter')]
            w_row = perf[(perf['Database'] == 'Weaviate') & (perf['Test'] == 'k10_nofilter')]
            
            if not m_row.empty and not w_row.empty:
                # Latency (lower is better)
                if m_row['P50 (ms)'].values[0] < w_row['P50 (ms)'].values[0]:
                    wins['Milvus']['latency'] += 1
                else:
                    wins['Weaviate']['latency'] += 1
                
                # QPS (higher is better)
                if m_row['QPS'].values[0] > w_row['QPS'].values[0]:
                    wins['Milvus']['qps'] += 1
                else:
                    wins['Weaviate']['qps'] += 1
        
        # Recall
        if b['recall'] is not None:
            r10 = b['recall'][b['recall']['K'] == 10]
            if not r10.empty:
                if r10['Milvus'].values[0] > r10['Weaviate'].values[0]:
                    wins['Milvus']['recall'] += 1
                else:
                    wins['Weaviate']['recall'] += 1
    
    # Create subplot with pie charts
    fig = make_subplots(
        rows=1, cols=5,
        specs=[[{'type': 'pie'}] * 5],
        subplot_titles=['Load Time', 'Memory', 'Latency', 'Throughput', 'Accuracy']
    )
    
    categories = ['load_time', 'memory', 'latency', 'qps', 'recall']
    
    for i, cat in enumerate(categories):
        m_wins = wins['Milvus'][cat]
        w_wins = wins['Weaviate'][cat]
        
        fig.add_trace(
            go.Pie(
                values=[m_wins, w_wins],
                labels=['Milvus', 'Weaviate'],
                marker_colors=['#667eea', '#e17055'],
                textinfo='value',
                showlegend=(i == 0),
                hole=0.4,
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="🏆 Win Rate by Category (across all benchmarks)",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    
    return fig, wins


def create_overall_radar_all(benchmarks):
    """Radar chart showing average performance across all benchmarks."""
    metrics = {
        'Milvus': {'load_speed': [], 'memory_eff': [], 'query_speed': [], 'throughput': [], 'accuracy': []},
        'Weaviate': {'load_speed': [], 'memory_eff': [], 'query_speed': [], 'throughput': [], 'accuracy': []}
    }
    
    for b in benchmarks:
        if not b['dataset']:
            continue
        
        m_load = b['loading']['Milvus']['load_time']
        w_load = b['loading']['Weaviate']['load_time']
        max_load = max(m_load, w_load, 0.001)
        metrics['Milvus']['load_speed'].append((1 - m_load/max_load) * 100)
        metrics['Weaviate']['load_speed'].append((1 - w_load/max_load) * 100)
        
        m_mem = b['loading']['Milvus']['peak_memory']
        w_mem = b['loading']['Weaviate']['peak_memory']
        max_mem = max(m_mem, w_mem, 0.001)
        metrics['Milvus']['memory_eff'].append((1 - m_mem/max_mem) * 100)
        metrics['Weaviate']['memory_eff'].append((1 - w_mem/max_mem) * 100)
        
        if b['query_performance'] is not None:
            perf = b['query_performance']
            m_row = perf[(perf['Database'] == 'Milvus') & (perf['Test'] == 'k10_nofilter')]
            w_row = perf[(perf['Database'] == 'Weaviate') & (perf['Test'] == 'k10_nofilter')]
            
            if not m_row.empty and not w_row.empty:
                max_p50 = max(m_row['P50 (ms)'].values[0], w_row['P50 (ms)'].values[0], 0.001)
                metrics['Milvus']['query_speed'].append((1 - m_row['P50 (ms)'].values[0]/max_p50) * 100)
                metrics['Weaviate']['query_speed'].append((1 - w_row['P50 (ms)'].values[0]/max_p50) * 100)
                
                max_qps = max(m_row['QPS'].values[0], w_row['QPS'].values[0], 0.001)
                metrics['Milvus']['throughput'].append((m_row['QPS'].values[0]/max_qps) * 100)
                metrics['Weaviate']['throughput'].append((w_row['QPS'].values[0]/max_qps) * 100)
        
        if b['recall'] is not None:
            r10 = b['recall'][b['recall']['K'] == 10]
            if not r10.empty:
                metrics['Milvus']['accuracy'].append(r10['Milvus'].values[0] * 100)
                metrics['Weaviate']['accuracy'].append(r10['Weaviate'].values[0] * 100)
    
    # Calculate averages
    categories = ['Load Speed', 'Memory Efficiency', 'Query Speed', 'Throughput', 'Accuracy']
    keys = ['load_speed', 'memory_eff', 'query_speed', 'throughput', 'accuracy']
    
    milvus_avg = []
    weaviate_avg = []
    valid_categories = []
    
    for cat, key in zip(categories, keys):
        if metrics['Milvus'][key] and metrics['Weaviate'][key]:
            milvus_avg.append(sum(metrics['Milvus'][key]) / len(metrics['Milvus'][key]))
            weaviate_avg.append(sum(metrics['Weaviate'][key]) / len(metrics['Weaviate'][key]))
            valid_categories.append(cat)
    
    if not valid_categories:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=milvus_avg + [milvus_avg[0]],
        theta=valid_categories + [valid_categories[0]],
        fill='toself',
        name='Milvus (avg)',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)',
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=weaviate_avg + [weaviate_avg[0]],
        theta=valid_categories + [valid_categories[0]],
        fill='toself',
        name='Weaviate (avg)',
        line_color='#e17055',
        fillcolor='rgba(225, 112, 85, 0.3)',
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="🎯 Average Performance Across All Benchmarks",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    
    return fig


def create_summary_table(benchmarks):
    """Create summary statistics table."""
    data = []
    for b in benchmarks:
        if not b['dataset']:
            continue
        
        row = {
            'Dataset': b['dataset'],
            'Vectors': f"{b['vectors']:,}",
            'Dims': b['dimensions'],
            'M Load (s)': f"{b['loading']['Milvus']['load_time']:.1f}",
            'W Load (s)': f"{b['loading']['Weaviate']['load_time']:.1f}",
            'M Mem (MB)': f"{b['loading']['Milvus']['peak_memory']:.0f}",
            'W Mem (MB)': f"{b['loading']['Weaviate']['peak_memory']:.0f}",
        }
        
        if b['query_performance'] is not None:
            perf = b['query_performance']
            m_row = perf[(perf['Database'] == 'Milvus') & (perf['Test'] == 'k10_nofilter')]
            w_row = perf[(perf['Database'] == 'Weaviate') & (perf['Test'] == 'k10_nofilter')]
            
            if not m_row.empty:
                row['M P50 (ms)'] = f"{m_row['P50 (ms)'].values[0]:.1f}"
                row['M QPS'] = f"{m_row['QPS'].values[0]:.0f}"
            if not w_row.empty:
                row['W P50 (ms)'] = f"{w_row['P50 (ms)'].values[0]:.1f}"
                row['W QPS'] = f"{w_row['QPS'].values[0]:.0f}"
        
        if b['recall'] is not None:
            r10 = b['recall'][b['recall']['K'] == 10]
            if not r10.empty:
                row['M Recall'] = f"{r10['Milvus'].values[0]:.2%}"
                row['W Recall'] = f"{r10['Weaviate'].values[0]:.2%}"
        
        data.append(row)
    
    return pd.DataFrame(data)


def get_ai_recommendations_total(benchmarks, api_key):
    """Generate AI recommendations based on ALL benchmarks."""
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        summary = "## Complete Benchmark Summary (All Datasets)\n\n"
        
        for b in benchmarks:
            if not b['dataset']:
                continue
            
            summary += f"### {b['dataset']} ({b['vectors']:,} vectors, {b['dimensions']}D)\n"
            summary += f"- Load: Milvus {b['loading']['Milvus']['load_time']:.1f}s vs Weaviate {b['loading']['Weaviate']['load_time']:.1f}s\n"
            summary += f"- Memory: Milvus {b['loading']['Milvus']['peak_memory']:.0f}MB vs Weaviate {b['loading']['Weaviate']['peak_memory']:.0f}MB\n"
            
            if b['query_performance'] is not None:
                perf = b['query_performance']
                m_row = perf[(perf['Database'] == 'Milvus') & (perf['Test'] == 'k10_nofilter')]
                w_row = perf[(perf['Database'] == 'Weaviate') & (perf['Test'] == 'k10_nofilter')]
                if not m_row.empty and not w_row.empty:
                    summary += f"- P50 Latency (k10): Milvus {m_row['P50 (ms)'].values[0]:.1f}ms vs Weaviate {w_row['P50 (ms)'].values[0]:.1f}ms\n"
                    summary += f"- QPS: Milvus {m_row['QPS'].values[0]:.0f} vs Weaviate {w_row['QPS'].values[0]:.0f}\n"
            
            if b['recall'] is not None:
                r10 = b['recall'][b['recall']['K'] == 10]
                if not r10.empty:
                    summary += f"- Recall@10: Milvus {r10['Milvus'].values[0]:.2%} vs Weaviate {r10['Weaviate'].values[0]:.2%}\n"
            
            summary += "\n"
        
        prompt = f"""
You are an expert in vector databases. Analyze these comprehensive benchmark results comparing Milvus and Weaviate across multiple datasets:

{summary}

Provide a comprehensive analysis:
1. **Overall Trends** (3-4 bullet points about patterns you see across datasets)
2. **Scalability Insights** (How do they perform as dataset size increases?)
3. **Consistency Analysis** (Which database is more consistent across different scenarios?)
4. **Final Verdict**: Choose Milvus if... (3-4 points) / Choose Weaviate if... (3-4 points)

Be specific with numbers and percentages. Format in clean Markdown.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical expert providing comprehensive database analysis. Be data-driven and specific."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AI RECOMMENDATIONS
# =============================================================================

def get_ai_recommendations(benchmark, api_key):
    """Generate AI-powered recommendations based on benchmark data."""
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        summary = f"""
## Benchmark Results Summary

**Dataset:** {benchmark['dataset']} ({benchmark['vectors']:,} vectors, {benchmark['dimensions']}D)
**Raw Data Size:** {benchmark['raw_data_size']:.2f} MB

### Data Loading Performance:
- Milvus: {benchmark['loading']['Milvus']['load_time']:.1f}s, {benchmark['loading']['Milvus']['peak_memory']:.1f} MB peak memory
- Weaviate: {benchmark['loading']['Weaviate']['load_time']:.1f}s, {benchmark['loading']['Weaviate']['peak_memory']:.1f} MB peak memory

### Query Performance:
"""
        if benchmark['query_performance'] is not None:
            summary += benchmark['query_performance'].to_string(index=False)
        
        if benchmark['recall'] is not None:
            summary += "\n\n### Recall@K Accuracy:\n"
            summary += benchmark['recall'].to_string(index=False)
        
        prompt = f"""
You are an expert in vector databases. Analyze these benchmark results comparing Milvus and Weaviate:

{summary}

Provide:
1. **Key Findings** (3-4 bullet points summarizing the most important results with specific numbers)
2. **Choose Milvus if:** (4-5 specific recommendations based on this data)
3. **Choose Weaviate if:** (4-5 specific recommendations based on this data)

Be specific, reference actual numbers, and format in clean Markdown.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical expert providing data-driven database recommendations. Be concise and specific."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Vector Database Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Milvus vs Weaviate Performance Comparison</p>', unsafe_allow_html=True)
    
    # Load benchmark files
    benchmark_files = get_benchmark_files()
    
    if not benchmark_files:
        st.warning("⚠️ No benchmark files found in `results/` directory.")
        st.info("Run `python run_benchmark.py` to generate benchmark results.")
        return
    
    # Parse ALL benchmarks for total analysis
    all_benchmarks = [parse_benchmark_file(f) for f in benchmark_files]
    
    # Sidebar - File selection
    st.sidebar.header("📁 Benchmark Selection")
    
    # Add "Total Analysis" option
    view_options = ["📊 Total Analysis (All Benchmarks)"] + [f.name for f in benchmark_files]
    selected_view = st.sidebar.selectbox(
        "Select View",
        options=view_options,
        format_func=lambda x: x if x.startswith("📊") else x.replace('complete_benchmark_', '').replace('.txt', '')
    )
    
    # AI Analysis section in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("🤖 AI Analysis")
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=env_api_key,
        type="password",
        help="Auto-loaded from .env file"
    )
    analyze_btn = st.sidebar.button("🔍 Analyze with AI", use_container_width=True)
    
    # Check if Total Analysis view
    if selected_view.startswith("📊"):
        # =====================================================================
        # TOTAL ANALYSIS VIEW
        # =====================================================================
        
        st.sidebar.divider()
        st.sidebar.subheader("📊 Total Analysis")
        st.sidebar.markdown(f"**Benchmarks loaded:** {len(all_benchmarks)}")
        datasets = [b['dataset'] for b in all_benchmarks if b['dataset']]
        st.sidebar.markdown(f"**Datasets:** {', '.join(datasets)}")
        
        # Main content - Total Analysis
        st.header("📊 Total Analysis - All Benchmarks")
        st.info(f"Analyzing {len(all_benchmarks)} benchmark runs across {len(datasets)} datasets")
        
        # Summary Table
        st.subheader("📋 Summary Table")
        summary_df = create_summary_table(all_benchmarks)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Win Rate Charts
        st.subheader("🏆 Win Rate Analysis")
        fig, wins = create_win_rate_chart(all_benchmarks)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            total_milvus = sum(wins['Milvus'].values())
            total_weaviate = sum(wins['Weaviate'].values())
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Milvus Total Wins", total_milvus)
            with col2:
                st.metric("Weaviate Total Wins", total_weaviate)
            with col3:
                winner = "Milvus" if total_milvus > total_weaviate else "Weaviate" if total_weaviate > total_milvus else "Tie"
                st.metric("Overall Leader", f"{winner} 🏆")
        
        st.divider()
        
        # Load Time & Memory Comparison
        st.subheader("⏱️ Loading Performance Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_load_time_comparison_all(all_benchmarks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_memory_comparison_all(all_benchmarks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Scalability Analysis
        st.subheader("📈 Scalability Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_scalability_chart(all_benchmarks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_qps_scalability_chart(all_benchmarks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Recall Comparison
        recall_benchmarks = [b for b in all_benchmarks if b['recall'] is not None]
        if recall_benchmarks:
            st.subheader("🎯 Accuracy Comparison")
            fig = create_recall_comparison_all(all_benchmarks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
        
        # Overall Radar Chart
        st.subheader("🎯 Average Performance Radar")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_overall_radar_all(all_benchmarks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            **How to read this chart:**
            
            Each axis represents average performance across all benchmarks:
            - **Load Speed**: Faster loading = higher score
            - **Memory Efficiency**: Lower memory = higher score
            - **Query Speed**: Lower latency = higher score
            - **Throughput**: Higher QPS = higher score
            - **Accuracy**: Higher recall = higher score
            
            Larger area = better overall performance.
            """)
        
        st.divider()
        
        # AI Recommendations for Total Analysis
        st.header("💡 AI Recommendations (All Benchmarks)")
        
        if analyze_btn and api_key:
            with st.spinner("🤖 AI is analyzing all benchmark data..."):
                ai_result = get_ai_recommendations_total(all_benchmarks, api_key)
                if ai_result:
                    st.session_state['ai_recommendations_total'] = ai_result
        
        if 'ai_recommendations_total' in st.session_state:
            st.markdown(st.session_state['ai_recommendations_total'])
        else:
            st.info("👆 Click 'Analyze with AI' in the sidebar to get comprehensive recommendations based on all benchmark data.")
        
    else:
        # =====================================================================
        # SINGLE BENCHMARK VIEW
        # =====================================================================
        
        # Find the selected benchmark file
        file_options = {f.name: f for f in benchmark_files}
        benchmark = parse_benchmark_file(file_options[selected_view])
        
        # Display dataset info in sidebar
        st.sidebar.divider()
        st.sidebar.subheader("📊 Dataset Info")
        st.sidebar.markdown(f"""
        - **Dataset:** {benchmark['dataset']}
        - **Vectors:** {benchmark['vectors']:,}
        - **Dimensions:** {benchmark['dimensions']}D
        - **Raw Size:** {benchmark['raw_data_size']:.2f} MB
        - **Date:** {benchmark['date']}
        """)
        
        # Row 1: Key Metrics
        st.header("📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        m_load = benchmark['loading']['Milvus']['load_time']
        w_load = benchmark['loading']['Weaviate']['load_time']
        load_winner = "Milvus" if m_load < w_load else "Weaviate"
        
        m_mem = benchmark['loading']['Milvus']['peak_memory']
        w_mem = benchmark['loading']['Weaviate']['peak_memory']
        mem_winner = "Milvus" if m_mem < w_mem else "Weaviate"
        
        with col1:
            st.metric("Milvus Load Time", f"{m_load:.1f}s", 
                      f"{'🏆 Faster' if load_winner == 'Milvus' else ''}")
        
        with col2:
            st.metric("Weaviate Load Time", f"{w_load:.1f}s",
                      f"{'🏆 Faster' if load_winner == 'Weaviate' else ''}")
        
        with col3:
            st.metric("Milvus Peak Memory", f"{m_mem:.0f} MB",
                      f"{'🏆 Lower' if mem_winner == 'Milvus' else ''}")
        
        with col4:
            st.metric("Weaviate Peak Memory", f"{w_mem:.0f} MB",
                      f"{'🏆 Lower' if mem_winner == 'Weaviate' else ''}")
        
        st.divider()
        
        # Row 2: Loading & Memory Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_loading_time_chart(benchmark)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_memory_chart(benchmark)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Row 3: Query Performance
        st.header("⚡ Query Performance")
        
        if benchmark['query_performance'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_latency_comparison_chart(benchmark['query_performance'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_qps_chart(benchmark['query_performance'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            fig = create_latency_heatmap(benchmark['query_performance'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Filter impact
            fig = create_filter_impact_chart(benchmark['query_performance'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No query performance data available.")
        
        st.divider()
        
        # Row 4: Recall (if available)
        if benchmark['recall'] is not None:
            st.header("🎯 Search Accuracy (Recall@K)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_recall_chart(benchmark['recall'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Recall Values")
                styled_df = benchmark['recall'].style.format({
                    'Milvus': '{:.2%}',
                    'Weaviate': '{:.2%}'
                }).background_gradient(
                    subset=['Milvus', 'Weaviate'],
                    cmap='RdYlGn',
                    vmin=0.8,
                    vmax=1.0
                )
                st.dataframe(styled_df, use_container_width=True)
            
            st.divider()
        
        # Row 5: Overall Comparison Radar
        st.header("🏆 Overall Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_radar_chart(benchmark)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Summary")
            
            wins = {'Milvus': 0, 'Weaviate': 0}
            
            if m_load < w_load:
                wins['Milvus'] += 1
            else:
                wins['Weaviate'] += 1
            
            if m_mem < w_mem:
                wins['Milvus'] += 1
            else:
                wins['Weaviate'] += 1
            
            if benchmark['query_performance'] is not None:
                m_p50 = benchmark['query_performance'][(benchmark['query_performance']['Database'] == 'Milvus') & 
                                                         (benchmark['query_performance']['Test'] == 'k10_nofilter')]['P50 (ms)']
                w_p50 = benchmark['query_performance'][(benchmark['query_performance']['Database'] == 'Weaviate') & 
                                                         (benchmark['query_performance']['Test'] == 'k10_nofilter')]['P50 (ms)']
                if len(m_p50) > 0 and len(w_p50) > 0:
                    if m_p50.values[0] < w_p50.values[0]:
                        wins['Milvus'] += 1
                    else:
                        wins['Weaviate'] += 1
            
            if benchmark['recall'] is not None:
                r10 = benchmark['recall'][benchmark['recall']['K'] == 10]
                if not r10.empty:
                    if r10['Milvus'].values[0] > r10['Weaviate'].values[0]:
                        wins['Milvus'] += 1
                    else:
                        wins['Weaviate'] += 1
            
            st.metric("Milvus Wins", f"{wins['Milvus']} categories")
            st.metric("Weaviate Wins", f"{wins['Weaviate']} categories")
            
            overall_winner = "Milvus" if wins['Milvus'] > wins['Weaviate'] else "Weaviate" if wins['Weaviate'] > wins['Milvus'] else "Tie"
            st.success(f"**Overall Winner:** {overall_winner} 🏆")
        
        st.divider()
        
        # Row 6: AI Recommendations
        st.header("💡 Recommendations")
        
        if analyze_btn and api_key:
            with st.spinner("🤖 AI is analyzing your benchmark data..."):
                ai_result = get_ai_recommendations(benchmark, api_key)
                if ai_result:
                    st.session_state['ai_recommendations'] = ai_result
        
        if 'ai_recommendations' in st.session_state:
            st.markdown(st.session_state['ai_recommendations'])
        else:
            st.info("👆 Click 'Analyze with AI' in the sidebar to get personalized recommendations based on your benchmark data.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Choose **Milvus** if:
                - Maximum query performance is critical
                - You need fine-grained index control
                - GPU acceleration is needed
                - High throughput is a priority
                """)
            
            with col2:
                st.markdown("""
                ### Choose **Weaviate** if:
                - Search accuracy is paramount
                - You need GraphQL API support
                - Hybrid search (vector + keyword) is important
                - Easier setup and management is preferred
                """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built by minageus, cobra, mountzouris</p>
        <p>Run <code>python run_benchmark.py</code> to generate new benchmarks</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
