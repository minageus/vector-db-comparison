"""
QUICK REFERENCE - Advanced Analysis Tools
==========================================

BASIC USAGE
-----------
1. Run interactive menu:
   python run_advanced_analysis.py

2. Run concurrent load test:
   python run_concurrent_test.py

3. Run basic benchmark (required first):
   python main.py


ANALYSIS OPTIONS
----------------
When you run run_advanced_analysis.py, you get:

[1] Latency Distribution Analysis
    → Creates histogram, CDF, box plot, percentile charts
    → Output: results/latency_distribution_*.png
    
[2] Recall@K Accuracy Test  
    → Measures search result accuracy (ground truth comparison)
    → Tests K values: 1, 5, 10, 20, 50, 100
    → Output: results/recall_comparison_*.csv
    
[3] Scalability Analysis
    → Tests 10K, 50K, 100K, 500K vectors
    → Measures: latency, QPS, memory, efficiency
    → Output: results/scalability_analysis_*.png
    
[4] Comprehensive Report
    → Text-based winner determination and recommendations
    → Output: results/comprehensive_report_*.txt
    
[5] Run ALL analyses
    → Executes options 1-4 in sequence


CONCURRENT LOAD TESTING
------------------------
python run_concurrent_test.py

Tests: 1, 5, 10, 25, 50 concurrent clients
Duration: 30 seconds per level
Outputs:
  • results/concurrent_test_*.csv
  • results/concurrent_test_*.png


KEY METRICS
-----------
Latency:
  P50  = Median (50% faster than this)
  P95  = 95th percentile
  P99  = 99th percentile
  
Throughput:
  QPS  = Queries Per Second
  
Accuracy:
  Recall@K = % of true top-K neighbors found
  
Resources:
  Memory = RAM usage (MB)
  Storage = Disk usage (MB)


TYPICAL WORKFLOW
----------------
1. python main.py
   → Basic benchmark
   
2. python run_advanced_analysis.py → Option 2
   → Verify accuracy with Recall test
   
3. python run_concurrent_test.py
   → Test under concurrent load
   
4. python run_advanced_analysis.py → Option 5
   → Generate all analyses and report


FILE OUTPUTS
------------
main.py:
  • comparison_*.csv
  • latency_comparison_*.png
  • qps_comparison_*.png

run_advanced_analysis.py:
  • latency_distribution_*.png
  • recall_comparison_*.csv
  • scalability_analysis_*.png
  • comprehensive_report_*.txt

run_concurrent_test.py:
  • concurrent_test_*.csv
  • concurrent_test_*.png


CUSTOMIZATION
-------------
Edit configurations in the script files:

DATASET_SIZE    - Number of vectors
DIMENSION       - Vector dimensions
N_QUERIES       - Number of test queries
TEST_DURATION   - Load test duration (seconds)
K_VALUES        - Values for Recall@K test


TROUBLESHOOTING
---------------
Error: "No result files found"
→ Run: python main.py first

Error: Out of memory
→ Reduce DATASET_SIZE

Error: Tests too slow
→ Reduce TEST_DURATION or dataset sizes

Error: Weaviate connection
→ Check: docker ps | Select-String "weaviate"
→ Restart: docker-compose -f docker/docker-compose-weaviate.yml up -d


UNDERSTANDING RESULTS
---------------------
Winner Criteria:
  Latency    → Lower is better
  QPS        → Higher is better
  Recall@K   → Higher is better (max 1.0)
  Memory     → Lower is better

Typical Performance:
  Good latency:  < 10ms P50
  Good QPS:      > 100 QPS
  Good recall:   > 0.95 @ K=10


QUICK EXAMPLES
--------------
# Just test accuracy
python run_advanced_analysis.py
→ Select 2

# Full analysis suite
python run_advanced_analysis.py
→ Select 5

# Stress test with concurrency
python run_concurrent_test.py

# View existing results
dir results


GETTING HELP
------------
• Read: ADVANCED_USAGE.md (full documentation)
• Check: Individual .py files for docstrings
• View: results/ folder for example outputs
"""

if __name__ == "__main__":
    print(__doc__)
