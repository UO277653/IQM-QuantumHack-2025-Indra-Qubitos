#!/bin/bash
# Example scripts for running battlefield tests

echo "=========================================="
echo "BATTLEFIELD TESTER - EXAMPLE RUNS"
echo "=========================================="
echo ""

# Example 1: Quick random test (100 battles)
echo "Example 1: Random Composition Test (100 battles)"
echo "Command: python battlefield_tester.py --test 1 --battles 100"
echo "This will test random team compositions and generate statistics"
echo ""

# Example 2: Balance test (160 battles - 20 per composition type)
echo "Example 2: Balance Test (160 battles)"
echo "Command: python battlefield_tester.py --test 2 --battles 160"
echo "This will test 8 predefined compositions with 20 battles each"
echo ""

# Example 3: Performance test (500 battles)
echo "Example 3: Performance Test (500 battles)"
echo "Command: python battlefield_tester.py --test 3 --battles 500"
echo "This will run extensive performance analysis"
echo ""

# Example 4: Large-scale test (1000 battles)
echo "Example 4: Large-scale Random Test (1000 battles)"
echo "Command: python battlefield_tester.py --test 1 --battles 1000"
echo "This will generate comprehensive statistics with 1000 battles"
echo ""

# Example 5: Custom grid size
echo "Example 5: Custom Grid Size Test"
echo "Command: python battlefield_tester.py --test 1 --battles 100 --grid-width 6 --grid-height 6"
echo "This will test on a larger 6x6 battlefield"
echo ""

# Example 6: Quick test with limited turns
echo "Example 6: Quick Test with Turn Limit"
echo "Command: python battlefield_tester.py --test 1 --battles 200 --max-turns 20"
echo "This will run battles with a maximum of 20 turns"
echo ""

echo "=========================================="
echo "To run any example, copy the command and execute it in your terminal"
echo "=========================================="
