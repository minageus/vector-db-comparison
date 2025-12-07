# Milvus vs Weaviate: Vector Database Comparison

A comprehensive benchmarking framework for comparing **Milvus** and **Weaviate** vector databases across multiple performance dimensions.

## 🎯 Project Overview

This project provides a systematic comparison of two popular open-source vector databases:
- **Milvus** - High-performance vector database optimized for AI/ML workloads
- **Weaviate** - GraphQL-based vector search engine with hybrid search capabilities

### Key Features

✅ **Real-World Datasets**: Support for standard ANN benchmarks (SIFT1M, GIST1M, GloVe)  
✅ **Comprehensive Metrics**: Latency, throughput, recall, resource usage, storage efficiency  
✅ **Multiple Test Scenarios**: Various k-values, filtered search, concurrent load testing  
✅ **Advanced Analysis**: Latency distributions, scalability analysis, recall@k accuracy  
✅ **Resource Monitoring**: CPU, memory, disk I/O, GPU utilization tracking  
✅ **Docker Setup**: Easy deployment with Docker Compose  
✅ **Visualization**: Publication-ready charts and comprehensive reports

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)

---

## 🚀 Quick Start

### 1. Clone and Setup

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Start databases with Docker
docker-compose -f docker/docker-compose-milvus.yml up -d
docker-compose -f docker/docker-compose-weaviate.yml up -d

# Wait for databases to be ready (~30 seconds)
```

### 2. Run Basic Benchmark

```powershell
# Run with synthetic data (100K vectors, 128D)
python main.py
```

### 3. Run with Real Dataset

```powershell
# Download and benchmark with SIFT1M
python run_real_data_benchmark.py --dataset sift1m
```

### 4. View Results

Results are saved in the `results/` directory:
- CSV files with detailed metrics
- PNG visualizations (latency, throughput, distributions)
- Text reports with recommendations

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Docker**: For running Milvus and Weaviate
- **Disk Space**: ~35 GB (for real datasets)
- **RAM**: 16 GB minimum, 32 GB recommended

### Step-by-Step Setup

1. **Install Python Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Start Milvus**
   ```powershell
   docker-compose -f docker/docker-compose-milvus.yml up -d
   
   # Verify Milvus is running
   docker ps | findstr milvus
   ```

3. **Start Weaviate**
   ```powershell
   docker-compose -f docker/docker-compose-weaviate.yml up -d
   
   # Verify Weaviate is running
   docker ps | findstr weaviate
   ```

4. **Verify Connections**
   ```powershell
   python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('Milvus OK')"
   python -c "import weaviate; client = weaviate.Client('http://localhost:8080'); print('Weaviate OK')"
   ```

---

## 💾 Datasets

### Synthetic Data (Default)

Generated on-the-fly with configurable size and dimensionality:
- Small: 100K vectors, 128D
- Medium: 1M vectors, 384D
- Large: 10M vectors, 768D

### Real-World Datasets

Standard ANN benchmark datasets with pre-computed ground truth:

| Dataset | Vectors | Dimension | Size | Description |
|---------|---------|-----------|------|-------------|
| **SIFT10K** | 10K | 128 | ~16 MB | SIFT descriptors - Quick test |
| **SIFT1M** | 1M | 128 | ~500 MB | SIFT image descriptors |
| **GIST1M** | 1M | 960 | ~3.6 GB | GIST image features |
| **GloVe-100D** | 1.2M | 100 | ~5 GB | Word embeddings |
| **MNIST-784** | 60K | 784 | ~217 MB | Handwritten digits |
| **Fashion-MNIST** | 60K | 784 | ~217 MB | Fashion items |

#### Download Datasets

```powershell
# List available datasets
python -m utils.dataset_downloader --list

# Download specific dataset
python -m utils.dataset_downloader --dataset sift1m

# Download and cache in custom directory
python -m utils.dataset_downloader --dataset gist1m --cache-dir D:/datasets
```

#### Multi-Dataset Benchmarking

```powershell
# Run benchmarks across multiple datasets automatically
python run_batch_benchmarks.py

# This will:
# 1. Run SIFT10K (quick test)
# 2. Run SIFT1M with 100K subset
# 3. Run SIFT1M with 500K subset
# 4. Run SIFT1M full (1M vectors)
# 5. Generate comparison visualizations
```

---

## 🔬 Benchmarks

### 1. Basic Performance Benchmark

```powershell
python main.py
```

**Tests:**
- k=10, 100, 1000 (no filters)
- k=10 with filters
- Metrics: P50/P95/P99 latency, QPS

### 2. Advanced Analysis

```powershell
python run_advanced_analysis.py
```

**Options:**
1. Latency Distribution Analysis
2. Recall@K Accuracy Test
3. Scalability Analysis (10K → 500K vectors)
4. Comprehensive Report Generation
5. Run ALL analyses

### 3. Concurrent Load Testing

```powershell
python run_concurrent_test.py
```

**Tests:**
- 1, 5, 10, 25, 50 concurrent clients
- 30-second duration per test
- Metrics: QPS, latency percentiles, success rate

### 4. Real Dataset Benchmark

```powershell
python run_real_data_benchmark.py --dataset sift1m
```

**Features:**
- Uses real-world data distributions
- Ground truth for recall calculation
- Industry-standard comparison

---

## 📊 Results

### Output Files

All results are saved in `results/` with timestamps:

| File Pattern | Description |
|-------------|-------------|
| `comparison_*.csv` | Basic benchmark results |
| `latency_comparison_*.png` | Latency bar charts |
| `qps_comparison_*.png` | Throughput bar charts |
| `latency_distribution_*.png` | Detailed latency analysis |
| `recall_comparison_*.csv` | Recall@K accuracy |
| `concurrent_test_*.csv` | Concurrent load results |
| `scalability_analysis_*.png` | Scaling behavior |
| `comprehensive_report_*.txt` | Summary report |
| `storage_analysis_*.json` | Storage efficiency |

### Sample Results

```
==============================================================================
BENCHMARK RESULTS
==============================================================================

Test: k10_nofilter
  Milvus:   P50=2.5ms, P95=4.2ms, QPS=400
  Weaviate: P50=3.8ms, P95=6.1ms, QPS=263
  Winner: Milvus (37% faster)

Test: k10_filter (10% selectivity)
  Milvus:   P50=3.1ms, P95=5.8ms, QPS=322
  Weaviate: P50=4.5ms, P95=8.2ms, QPS=222
  Winner: Milvus (31% faster)

Storage Efficiency:
  Milvus:   555 MB (1.32x compression)
  Weaviate: 760 MB (1.0x compression)
  Winner: Milvus (27% smaller)
```

---

## 📁 Project Structure

```
vector-db-comparison/
├── data/
│   ├── generators/
│   │   └── vector_generator.py       # Synthetic data generation
│   └── loaders/
│       ├── milvus_loader.py          # Milvus data loader
│       ├── weaviate_loader.py        # Weaviate data loader
│       └── real_dataset_loader.py    # Real dataset loader (NEW)
├── queries/
│   ├── query_generator.py            # Query generation
│   ├── milvus_queries.py             # Milvus query executor
│   └── weaviate_queries.py           # Weaviate query executor
├── benchmarks/
│   └── benchmark_runner.py           # Benchmark orchestration
├── analysis/
│   └── performance_analyzer.py       # Visualization & reporting
├── utils/
│   ├── recall_calculator.py          # Recall@K calculation
│   ├── concurrent_tester.py          # Concurrent load testing
│   ├── dataset_downloader.py         # Dataset downloader (NEW)
│   ├── resource_monitor.py           # Resource monitoring (NEW)
│   └── storage_analyzer.py           # Storage analysis (NEW)
├── docker/
│   ├── docker-compose-milvus.yml     # Milvus standalone
│   └── docker-compose-weaviate.yml   # Weaviate
├── results/                           # Benchmark results
├── main.py                            # Basic benchmark
├── run_advanced_analysis.py          # Advanced analysis
├── run_concurrent_test.py            # Concurrent testing
├── run_real_data_benchmark.py        # Real data benchmark (NEW)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── ADVANCED_USAGE.md                 # Detailed usage guide
```

---

## 🔧 Advanced Usage

### Custom Benchmark Configuration

Edit `main.py` to customize:
```python
DATASET_SIZE = 1000000  # 1M vectors
DIMENSION = 384         # 384D vectors
N_QUERIES = 1000        # 1000 test queries
```

### Index Tuning

Milvus HNSW parameters:
```python
milvus_loader.create_index(
    index_type='HNSW',
    metric_type='L2',
    params={'M': 16, 'efConstruction': 256}
)
```

### Resource Monitoring

```python
from utils.resource_monitor import ResourceMonitor

with ResourceMonitor(interval=0.5) as monitor:
    # Run benchmark
    results = benchmark.run()

# Get resource stats
stats = monitor.get_stats()
print(f"Peak CPU: {stats['cpu']['max']:.1f}%")
print(f"Peak Memory: {stats['memory_rss_mb']['max']:.1f} MB")
```

### Storage Analysis

```python
from utils.storage_analyzer import StorageAnalyzer

analyzer = StorageAnalyzer()
analyzer.analyze_milvus_storage(raw_data_size_mb=512)
analyzer.analyze_weaviate_storage(raw_data_size_mb=512)
analyzer.print_comparison()
```

For more advanced usage, see [ADVANCED_USAGE.md](ADVANCED_USAGE.md).

---

## 📈 Key Metrics Explained

### Latency Metrics
- **P50 (Median)**: 50% of queries complete faster
- **P95**: 95% of queries complete faster (important for SLAs)
- **P99**: 99% of queries complete faster (tail latency)

### Throughput
- **QPS**: Queries Per Second (higher is better)

### Accuracy
- **Recall@K**: Percentage of true top-K neighbors found
  - 1.0 = perfect (100% accuracy)
  - 0.95 = found 95% of true neighbors

### Resource Usage
- **CPU**: Processor utilization during operations
- **Memory**: RAM usage (RSS = actual memory used)
- **Disk I/O**: Read/write throughput

---

## 🎓 Use Cases

**Choose Milvus if:**
- Maximum performance at scale (10M+ vectors)
- GPU acceleration is important
- Building production ML/AI systems
- Need fine-grained index control

**Choose Weaviate if:**
- GraphQL API support needed
- Hybrid search (vector + keyword) required
- Easier setup/management preferred
- Built-in ML model integration needed

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional vector databases (Qdrant, Pinecone, etc.)
- More real-world datasets
- GPU-accelerated index testing
- Distributed/cluster benchmarks

---

## 📝 License

This project is for educational and research purposes.

---

## 🔗 References

- [Milvus Documentation](https://milvus.io/docs)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [ANN Benchmarks](http://ann-benchmarks.com/)
- [SIFT Dataset](http://corpus-texmex.irisa.fr/)

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: December 2025
