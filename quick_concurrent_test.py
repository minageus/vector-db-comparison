"""
Quick Concurrent Test
A faster version for quick verification
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

def quick_concurrent_test():
    """Quick concurrent test with smaller dataset"""
    
    print("="*60)
    print("QUICK CONCURRENT LOAD TEST")
    print("="*60)
    
    # Smaller configuration for quick testing
    DATASET_SIZE = 10000  # Much smaller
    DIMENSION = 128
    N_QUERIES = 50
    TEST_DURATION = 5  # Just 5 seconds
    WARMUP_DURATION = 1
    CONCURRENCY_LEVELS = [1, 5]  # Just 2 levels
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_SIZE} vectors (quick test)")
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
    milvus_loader.create_collection("quick_concurrent_test", DIMENSION)
    milvus_loader.load_data(ids, vectors, metadata, batch_size=1000)
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
    
    # Run tests
    print("\n[4/4] Running quick tests...")
    
    all_results = []
    
    for n_clients in CONCURRENCY_LEVELS:
        print(f"\n{'='*60}")
        print(f"Testing with {n_clients} concurrent client(s)")
        print(f"{'='*60}")
        
        # Test Milvus
        print(f"\nMilvus ({n_clients} client(s)):")
        tester = ConcurrentTester(n_clients=n_clients)
        
        def milvus_search(query):
            return milvus_executor.search(query, k=10)
        
        try:
            milvus_result = tester.run_load_test(
                milvus_search,
                query_vectors,
                duration_seconds=TEST_DURATION,
                warmup_seconds=WARMUP_DURATION
            )
            
            all_results.append({
                'concurrency': n_clients,
                'database': 'Milvus',
                'qps': milvus_result.qps,
                'p50_ms': milvus_result.p50 * 1000,
                'p95_ms': milvus_result.p95 * 1000,
                'failed': milvus_result.failed_requests,
                'success_rate': milvus_result.successful_requests / max(milvus_result.total_requests, 1) * 100
            })
        except Exception as e:
            print(f"⚠ Milvus test failed: {e}")
        
        # Test Weaviate
        print(f"\nWeaviate ({n_clients} client(s)):")
        tester = ConcurrentTester(n_clients=n_clients)
        
        def weaviate_search(query):
            return weaviate_executor.search(query, k=10)
        
        try:
            weaviate_result = tester.run_load_test(
                weaviate_search,
                query_vectors,
                duration_seconds=TEST_DURATION,
                warmup_seconds=WARMUP_DURATION
            )
            
            all_results.append({
                'concurrency': n_clients,
                'database': 'Weaviate',
                'qps': weaviate_result.qps,
                'p50_ms': weaviate_result.p50 * 1000,
                'p95_ms': weaviate_result.p95 * 1000,
                'failed': weaviate_result.failed_requests,
                'success_rate': weaviate_result.successful_requests / max(weaviate_result.total_requests, 1) * 100
            })
        except Exception as e:
            print(f"⚠ Weaviate test failed: {e}")
        
        # Print comparison
        if len(all_results) >= 2:
            print(f"\n{'='*60}")
            print(f"Quick Comparison at {n_clients} client(s):")
            print(f"{'='*60}")
            print(f"{'Metric':<20} {'Milvus':<20} {'Weaviate':<20}")
            print("-" * 60)
            m_idx = -2
            w_idx = -1
            print(f"{'QPS':<20} {all_results[m_idx]['qps']:<20.2f} {all_results[w_idx]['qps']:<20.2f}")
            print(f"{'P50 Latency (ms)':<20} {all_results[m_idx]['p50_ms']:<20.2f} {all_results[w_idx]['p50_ms']:<20.2f}")
            print(f"{'P95 Latency (ms)':<20} {all_results[m_idx]['p95_ms']:<20.2f} {all_results[w_idx]['p95_ms']:<20.2f}")
            print(f"{'Failed Requests':<20} {all_results[m_idx]['failed']:<20} {all_results[w_idx]['failed']:<20}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'results/quick_concurrent_test_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("QUICK TEST SUMMARY")
        print("="*60)
        print("\n" + df.to_string(index=False))
        print(f"\n✓ Results saved to: {output_file}")
    
    # Cleanup
    print("\nCleaning up...")
    try:
        weaviate_loader.client.close()
        print("✓ Weaviate connection closed")
    except:
        pass


if __name__ == "__main__":
    try:
        quick_concurrent_test()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
