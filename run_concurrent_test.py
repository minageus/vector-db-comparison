"""
Concurrent Load Testing
Test database performance under concurrent load with multiple clients
"""

import numpy as np
from data.generators.vector_generator import VectorDataGenerator
from data.loaders.milvus_loader import MilvusLoader
from data.loaders.weaviate_loader import WeaviateLoader
from queries.query_generator import QueryGenerator
from queries.milvus_queries import MilvusQueryExecutor
from queries.weaviate_queries import WeaviateQueryExecutor
from utils.concurrent_tester import ConcurrentTester
import pandas as pd

def run_concurrent_benchmark():
    """Run concurrent load tests on both databases"""
    
    print("="*60)
    print("CONCURRENT LOAD TEST")
    print("="*60)
    
    # Configuration
    DATASET_SIZE = 100000
    DIMENSION = 128
    N_QUERIES = 100
    TEST_DURATION = 10  # seconds (reduced for faster testing)
    WARMUP_DURATION = 2  # seconds warmup before measuring
    CONCURRENCY_LEVELS = [1, 5, 10]  # Reduced levels for initial testing
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_SIZE} vectors")
    print(f"  Dimension: {DIMENSION}")
    print(f"  Test duration: {TEST_DURATION}s per level")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    
    # Setup data
    print("\n[1/4] Generating data...")
    generator = VectorDataGenerator()
    ids, vectors, metadata = generator.generate_with_metadata(DATASET_SIZE, DIMENSION)
    
    query_gen = QueryGenerator()
    query_vectors = query_gen.generate_query_vectors(N_QUERIES, DIMENSION)
    
    # Setup Milvus
    print("\n[2/4] Setting up Milvus...")
    milvus_loader = MilvusLoader()
    milvus_loader.connect()
    milvus_loader.create_collection("concurrent_test", DIMENSION)
    milvus_loader.load_data(ids, vectors, metadata, batch_size=10000)
    milvus_loader.create_index(index_type='HNSW', metric_type='L2')
    milvus_loader.load_collection()
    
    milvus_executor = MilvusQueryExecutor(milvus_loader.collection)
    
    # Setup Weaviate
    print("\n[3/4] Setting up Weaviate...")
    weaviate_loader = WeaviateLoader()
    weaviate_loader.connect()
    weaviate_loader.create_schema(DIMENSION)
    weaviate_loader.load_data(ids, vectors, metadata, batch_size=100)
    
    weaviate_executor = WeaviateQueryExecutor(weaviate_loader.client, 'BenchmarkVector')
    
    print("\n⚠ Note: Concurrent testing may take several minutes...")
    
    # Run tests
    print("\n[4/4] Running concurrent tests...")
    
    all_results = []
    
    for n_clients in CONCURRENCY_LEVELS:
        print(f"\n{'='*60}")
        print(f"Testing with {n_clients} concurrent clients")
        print(f"{'='*60}")
        
        # Test Milvus
        print(f"\nMilvus ({n_clients} clients):")
        tester = ConcurrentTester(n_clients=n_clients)
        
        def milvus_search(query):
            return milvus_executor.search(query, k=10)
        
        milvus_result = tester.run_load_test(
            milvus_search,
            query_vectors,
            duration_seconds=TEST_DURATION,
            warmup_seconds=WARMUP_DURATION
        )
        
        # Test Weaviate
        print(f"\nWeaviate ({n_clients} clients):")
        tester = ConcurrentTester(n_clients=n_clients)
        
        def weaviate_search(query):
            return weaviate_executor.search(query, k=10)
        
        weaviate_result = tester.run_load_test(
            weaviate_search,
            query_vectors,
            duration_seconds=TEST_DURATION,
            warmup_seconds=WARMUP_DURATION
        )
        
        # Store results
        all_results.append({
            'concurrency': n_clients,
            'database': 'Milvus',
            'qps': milvus_result.qps,
            'p50_ms': milvus_result.p50 * 1000,
            'p95_ms': milvus_result.p95 * 1000,
            'p99_ms': milvus_result.p99 * 1000,
            'failed_requests': milvus_result.failed_requests,
            'success_rate': milvus_result.successful_requests / milvus_result.total_requests
        })
        
        all_results.append({
            'concurrency': n_clients,
            'database': 'Weaviate',
            'qps': weaviate_result.qps,
            'p50_ms': weaviate_result.p50 * 1000,
            'p95_ms': weaviate_result.p95 * 1000,
            'p99_ms': weaviate_result.p99 * 1000,
            'failed_requests': weaviate_result.failed_requests,
            'success_rate': weaviate_result.successful_requests / weaviate_result.total_requests
        })
        
        # Print comparison
        print(f"\n{'='*60}")
        print(f"Comparison at {n_clients} clients:")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Milvus':<20} {'Weaviate':<20}")
        print("-" * 60)
        print(f"{'QPS':<20} {milvus_result.qps:<20.2f} {weaviate_result.qps:<20.2f}")
        print(f"{'P50 Latency (ms)':<20} {milvus_result.p50*1000:<20.2f} {weaviate_result.p50*1000:<20.2f}")
        print(f"{'P95 Latency (ms)':<20} {milvus_result.p95*1000:<20.2f} {weaviate_result.p95*1000:<20.2f}")
        print(f"{'P99 Latency (ms)':<20} {milvus_result.p99*1000:<20.2f} {weaviate_result.p99*1000:<20.2f}")
        print(f"{'Failed Requests':<20} {milvus_result.failed_requests:<20} {weaviate_result.failed_requests:<20}")
    
    # Save results
    df = pd.DataFrame(all_results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/concurrent_test_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("CONCURRENT LOAD TEST SUMMARY")
    print("="*60)
    print("\n" + df.to_string(index=False))
    print(f"\n✓ Results saved to: {output_file}")
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # QPS vs Concurrency
    for db in ['Milvus', 'Weaviate']:
        db_data = df[df['database'] == db]
        axes[0, 0].plot(db_data['concurrency'], db_data['qps'], 
                       marker='o', label=db, linewidth=2)
    axes[0, 0].set_xlabel('Concurrent Clients')
    axes[0, 0].set_ylabel('QPS')
    axes[0, 0].set_title('Throughput vs Concurrency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # P50 Latency vs Concurrency
    for db in ['Milvus', 'Weaviate']:
        db_data = df[df['database'] == db]
        axes[0, 1].plot(db_data['concurrency'], db_data['p50_ms'], 
                       marker='o', label=db, linewidth=2)
    axes[0, 1].set_xlabel('Concurrent Clients')
    axes[0, 1].set_ylabel('P50 Latency (ms)')
    axes[0, 1].set_title('Median Latency vs Concurrency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # P95 Latency vs Concurrency
    for db in ['Milvus', 'Weaviate']:
        db_data = df[df['database'] == db]
        axes[1, 0].plot(db_data['concurrency'], db_data['p95_ms'], 
                       marker='o', label=db, linewidth=2)
    axes[1, 0].set_xlabel('Concurrent Clients')
    axes[1, 0].set_ylabel('P95 Latency (ms)')
    axes[1, 0].set_title('95th Percentile Latency vs Concurrency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success Rate vs Concurrency
    for db in ['Milvus', 'Weaviate']:
        db_data = df[df['database'] == db]
        axes[1, 1].plot(db_data['concurrency'], db_data['success_rate'] * 100, 
                       marker='o', label=db, linewidth=2)
    axes[1, 1].set_xlabel('Concurrent Clients')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Success Rate vs Concurrency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([95, 100.5])
    
    plt.tight_layout()
    plot_file = f'results/concurrent_test_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_file}")
    
    # Cleanup
    print("\nCleaning up connections...")
    try:
        if weaviate_loader and weaviate_loader.client:
            weaviate_loader.client.close()
            print("✓ Weaviate connection closed")
    except Exception as e:
        print(f"Note: {e}")


if __name__ == "__main__":
    try:
        run_concurrent_benchmark()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
