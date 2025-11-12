"""
Advanced Analysis Runner
Run comprehensive performance analysis with latency distributions, scalability tests, and recall calculations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from analysis.performance_analyzer import PerformanceAnalyzer
from utils.recall_calculator import RecallCalculator
from data.generators.vector_generator import VectorDataGenerator
from data.loaders.milvus_loader import MilvusLoader
from data.loaders.weaviate_loader import WeaviateLoader
from queries.query_generator import QueryGenerator
from queries.milvus_queries import MilvusQueryExecutor
from queries.weaviate_queries import WeaviateQueryExecutor

def run_latency_analysis():
    """Analyze latency distributions from existing results"""
    print("\n" + "="*60)
    print("LATENCY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Find the most recent CSV results file
    results_dir = Path('results')
    csv_files = sorted(results_dir.glob('comparison_*.csv'))
    
    if not csv_files:
        print("No result files found. Run main.py first!")
        return
    
    latest_csv = csv_files[-1]
    print(f"Analyzing: {latest_csv.name}")
    
    # Load results
    df = pd.read_csv(latest_csv)
    
    # Extract latency data from actual CSV structure
    latencies_dict = {}
    
    # Generate synthetic latency distributions based on actual statistics
    # This creates realistic latency patterns from P50, P95, P99 data
    for db in ['Milvus', 'Weaviate']:
        db_rows = df[df['Database'] == db]
        if not db_rows.empty:
            # Use actual P50 from the CSV (column name is 'P50 (ms)')
            mean_lat = db_rows['P50 (ms)'].mean()
            p95_lat = db_rows['P95 (ms)'].mean()
            
            # Create realistic distribution
            # Use lognormal to simulate typical latency patterns
            std_lat = (p95_lat - mean_lat) / 1.645  # approx std from P95
            latencies_dict[db] = np.random.lognormal(
                np.log(mean_lat), 
                std_lat / mean_lat, 
                1000
            )
    
    # Create analysis
    analyzer = PerformanceAnalyzer()
    output_path = results_dir / f'latency_distribution_{latest_csv.stem.split("_")[-1]}.png'
    analyzer.analyze_latency_distribution(latencies_dict, str(output_path))
    
    print(f"\n✓ Latency analysis complete!")
    print(f"  Saved to: {output_path}")


def run_recall_test():
    """Test recall accuracy for both databases"""
    print("\n" + "="*60)
    print("RECALL@K ACCURACY TEST")
    print("="*60)
    
    # Configuration
    DATASET_SIZE = 10000  # Smaller for recall computation
    DIMENSION = 128
    N_QUERIES = 100
    K_VALUES = [1, 5, 10, 20, 50, 100]
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_SIZE} vectors")
    print(f"  Dimension: {DIMENSION}")
    print(f"  Queries: {N_QUERIES}")
    print(f"  K values: {K_VALUES}")
    
    # Generate data
    print("\n[1/5] Generating data...")
    generator = VectorDataGenerator()
    ids, vectors, metadata = generator.generate_with_metadata(DATASET_SIZE, DIMENSION)
    
    # Generate queries
    print("[2/5] Generating queries...")
    query_gen = QueryGenerator()
    query_vectors = query_gen.generate_query_vectors(N_QUERIES, DIMENSION)
    
    # Compute ground truth
    print("[3/5] Computing ground truth (brute force)...")
    recall_calc = RecallCalculator(metric='l2')
    ground_truth = recall_calc.compute_ground_truth(
        query_vectors, 
        vectors, 
        k=max(K_VALUES)
    )
    
    # Test Milvus
    print("\n[4/5] Testing Milvus recall...")
    milvus_loader = MilvusLoader()
    milvus_loader.connect()
    milvus_loader.create_collection("recall_test", DIMENSION)
    milvus_loader.load_data(ids, vectors, metadata, batch_size=1000)
    milvus_loader.create_index(index_type='HNSW', metric_type='L2')
    milvus_loader.load_collection()
    
    milvus_executor = MilvusQueryExecutor(milvus_loader.collection)
    
    milvus_recalls = {}
    for k in K_VALUES:
        retrieved = []
        for query in query_vectors:
            results = milvus_executor.search(query, k=k)
            retrieved.append([r['id'] for r in results])
        
        recall, _ = recall_calc.calculate_recall(retrieved, ground_truth, k=k)
        milvus_recalls[k] = recall
        print(f"  Recall@{k}: {recall:.4f}")
    
    # Test Weaviate
    print("\n[5/5] Testing Weaviate recall...")
    weaviate_loader = WeaviateLoader()
    weaviate_loader.connect()
    weaviate_loader.create_schema(DIMENSION)
    weaviate_loader.load_data(ids, vectors, metadata, batch_size=100)
    
    weaviate_executor = WeaviateQueryExecutor(weaviate_loader.client, 'BenchmarkVector')
    
    weaviate_recalls = {}
    for k in K_VALUES:
        retrieved = []
        for query in query_vectors:
            results = weaviate_executor.search(query, k=k)
            retrieved.append([r['id'] for r in results])
        
        recall, _ = recall_calc.calculate_recall(retrieved, ground_truth, k=k)
        weaviate_recalls[k] = recall
        print(f"  Recall@{k}: {recall:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("RECALL COMPARISON")
    print("="*60)
    print(f"\n{'K':<10} {'Milvus':<15} {'Weaviate':<15} {'Winner':<15}")
    print("-" * 60)
    for k in K_VALUES:
        m_recall = milvus_recalls[k]
        w_recall = weaviate_recalls[k]
        winner = "Milvus" if m_recall > w_recall else "Weaviate" if w_recall > m_recall else "Tie"
        print(f"{k:<10} {m_recall:<15.4f} {w_recall:<15.4f} {winner:<15}")
    
    # Save results
    recall_df = pd.DataFrame({
        'K': K_VALUES,
        'Milvus_Recall': [milvus_recalls[k] for k in K_VALUES],
        'Weaviate_Recall': [weaviate_recalls[k] for k in K_VALUES]
    })
    
    output_file = f'results/recall_comparison_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
    recall_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")


def run_scalability_analysis():
    """Test performance across different dataset sizes"""
    print("\n" + "="*60)
    print("SCALABILITY ANALYSIS")
    print("="*60)
    
    # Different dataset sizes to test
    test_configs = [
        {'size': 10000, 'dimension': 128, 'name': '10K'},
        {'size': 50000, 'dimension': 128, 'name': '50K'},
        {'size': 100000, 'dimension': 128, 'name': '100K'},
        {'size': 500000, 'dimension': 128, 'name': '500K'},
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']} vectors")
        print(f"{'='*60}")
        
        size = config['size']
        dimension = config['dimension']
        
        # Generate data
        print("Generating data...")
        generator = VectorDataGenerator()
        ids, vectors, metadata = generator.generate_with_metadata(size, dimension)
        
        # Generate queries
        query_gen = QueryGenerator()
        query_vectors = query_gen.generate_query_vectors(100, dimension)
        
        # Test Milvus
        print("\nTesting Milvus...")
        milvus_loader = MilvusLoader()
        milvus_loader.connect()
        milvus_loader.create_collection(f"scale_test_{size}", dimension)
        milvus_loader.load_data(ids, vectors, metadata, batch_size=10000)
        milvus_loader.create_index(index_type='HNSW', metric_type='L2')
        milvus_loader.load_collection()
        
        milvus_executor = MilvusQueryExecutor(milvus_loader.collection)
        
        # Measure performance
        import time
        start = time.time()
        latencies = []
        for query in query_vectors:
            q_start = time.time()
            milvus_executor.search(query, k=10)
            latencies.append((time.time() - q_start) * 1000)
        total_time = time.time() - start
        
        milvus_results = {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'qps': len(query_vectors) / total_time,
            'memory_mb': milvus_loader.metrics.get('memory_used_mb', 0)
        }
        
        # Test Weaviate
        print("Testing Weaviate...")
        weaviate_loader = WeaviateLoader()
        weaviate_loader.connect()
        weaviate_loader.create_schema(dimension)
        weaviate_loader.load_data(ids, vectors, metadata, batch_size=100)
        
        weaviate_executor = WeaviateQueryExecutor(weaviate_loader.client, 'BenchmarkVector')
        
        start = time.time()
        latencies = []
        for query in query_vectors:
            q_start = time.time()
            weaviate_executor.search(query, k=10)
            latencies.append((time.time() - q_start) * 1000)
        total_time = time.time() - start
        
        weaviate_results = {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'qps': len(query_vectors) / total_time,
            'memory_mb': weaviate_loader.metrics.get('memory_used_mb', 0)
        }
        
        all_results[config['name']] = {
            'dataset_size': size,
            'Milvus': milvus_results,
            'Weaviate': weaviate_results
        }
        
        print(f"\nResults for {config['name']}:")
        print(f"  Milvus   - P50: {milvus_results['p50']:.2f}ms, QPS: {milvus_results['qps']:.2f}")
        print(f"  Weaviate - P50: {weaviate_results['p50']:.2f}ms, QPS: {weaviate_results['qps']:.2f}")
    
    # Generate scalability plots
    analyzer = PerformanceAnalyzer()
    output_path = f'results/scalability_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
    analyzer.compare_scalability(all_results, output_path)
    
    print(f"\n✓ Scalability analysis complete!")
    print(f"  Saved to: {output_path}")


def generate_comprehensive_report():
    """Generate final comprehensive report"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    # Load all result files
    results_dir = Path('results')
    csv_files = list(results_dir.glob('comparison_*.csv'))
    
    if not csv_files:
        print("No result files found!")
        return
    
    # Aggregate results from all CSV files
    all_results = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Process each test in the CSV
        for _, row in df.iterrows():
            test_name = row['Test']
            db = row['Database']
            
            # Create nested structure for analyzer
            if test_name not in all_results:
                all_results[test_name] = {}
            
            # Map CSV columns to expected format
            all_results[test_name][db] = {
                'p50_latency_ms': row['P50 (ms)'],
                'p95_latency_ms': row['P95 (ms)'],
                'p99_latency_ms': row['P99 (ms)'],
                'mean_latency_ms': row['Mean (ms)'],
                'qps': row['QPS']
            }
    
    # Generate report
    analyzer = PerformanceAnalyzer()
    output_path = f'results/comprehensive_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt'
    analyzer.generate_summary_report(all_results, output_path)
    
    print(f"\n✓ Report generated: {output_path}")


def main():
    """Main menu for advanced analysis"""
    
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS & UTILITIES")
    print("="*60)
    print("\nAvailable analyses:")
    print("  1. Latency Distribution Analysis")
    print("  2. Recall@K Accuracy Test")
    print("  3. Scalability Analysis (multiple dataset sizes)")
    print("  4. Generate Comprehensive Report")
    print("  5. Run ALL analyses")
    print("  0. Exit")
    
    choice = input("\nSelect analysis (0-5): ").strip()
    
    if choice == '1':
        run_latency_analysis()
    elif choice == '2':
        run_recall_test()
    elif choice == '3':
        run_scalability_analysis()
    elif choice == '4':
        generate_comprehensive_report()
    elif choice == '5':
        run_latency_analysis()
        run_recall_test()
        run_scalability_analysis()
        generate_comprehensive_report()
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
