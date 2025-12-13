# Battlefield Testing Framework

Comprehensive testing suite for the QuantumBattlefield simulation with extensive battle generation and statistical analysis.

## Overview

The Battlefield Testing Framework allows you to run thousands of automated battles with different configurations to:
- Test game balance
- Analyze team composition effectiveness
- Measure performance metrics
- Generate comprehensive visualizations and statistics

## Features

- **3 Different Test Modes** for comprehensive coverage
- **Random Battle Generation** with configurable parameters
- **Extensive Statistical Analysis** with detailed metrics
- **Beautiful Visualizations** including:
  - Victory distribution pie charts
  - Battle duration histograms
  - Survivor distribution analysis
  - Position heatmaps
  - Win rate trends
  - Performance metrics

## Installation

### Requirements

```bash
pip install matplotlib seaborn numpy tqdm
```

### Files Required

- `battlefield.py` - Main battlefield simulation
- `battlefield_tester.py` - Testing framework (this tool)
- `quantum_library.py` - Quantum move calculation library

## Usage

### Basic Command Structure

```bash
python battlefield_tester.py --test <TEST_TYPE> --battles <NUM_BATTLES>
```

### Test Types

#### Test 1: Random Composition Test
Generate battles with completely random unit compositions to explore all possible team configurations.

```bash
python battlefield_tester.py --test 1 --battles 1000
```

**What it does:**
- Generates random team compositions for each battle
- Tests a wide variety of unit combinations
- Good for discovering edge cases and unexpected behaviors

#### Test 2: Balance Test
Test predefined balanced compositions to analyze game balance.

```bash
python battlefield_tester.py --test 2 --battles 500
```

**What it does:**
- Tests specific predefined compositions including:
  - Identical balanced teams
  - All soldiers vs balanced team
  - All archers vs balanced team
  - All knights vs balanced team
  - High strength vs high health
  - Long range vs short range
  - Many weak vs few strong units
  - Diverse team compositions
- Analyzes which strategies/compositions have advantages

#### Test 3: Performance Test
Measure comprehensive battle statistics and performance metrics.

```bash
python battlefield_tester.py --test 3 --battles 2000
```

**What it does:**
- Uses a mix of random and balanced compositions
- Focuses on performance metrics
- Measures execution time and battle efficiency
- Good for benchmarking and optimization analysis

### Advanced Options

```bash
python battlefield_tester.py --test 1 --battles 1000 --max-turns 50 --grid-width 4 --grid-height 4
```

**Parameters:**
- `--test` (required): Test type [1, 2, or 3]
- `--battles` (default: 100): Number of battles to simulate
- `--max-turns` (default: 50): Maximum turns per battle before declaring a draw
- `--grid-width` (default: 4): Battlefield width
- `--grid-height` (default: 4): Battlefield height

## Output

### Console Output

The framework prints:
1. Progress bars for battle simulation
2. Detailed summary statistics including:
   - Total battles simulated
   - Win/loss/draw counts and percentages
   - Average battle length
   - Execution time metrics
   - Survivor statistics

### Generated Files

#### 1. `battlefield_test_results.png`
Comprehensive visualization with 9 subplots:
- Victory distribution (pie chart)
- Battle duration distribution
- Survivors distribution
- Turns per battle histogram
- Win rate by team size
- Average survivors by winner
- Battle outcome timeline
- Final position heatmap (Quantum)
- Final position heatmap (Classical)

#### 2. `battlefield_detailed_statistics.png`
Detailed analysis plots:
- Initial size vs survivors correlation
- Turn distribution by winner
- Performance: turns vs execution time
- Summary statistics table

## Example Workflow

### Quick Test (100 battles)
```bash
python battlefield_tester.py --test 1 --battles 100
```

### Medium Test (500 battles)
```bash
python battlefield_tester.py --test 2 --battles 500
```

### Extensive Test (5000 battles)
```bash
python battlefield_tester.py --test 3 --battles 5000
```

## Understanding the Results

### Victory Distribution
Shows the percentage of wins for each team. Ideally should be close to 50/50 if teams are balanced.

### Battle Duration
Longer battles might indicate balanced teams, while very short battles might indicate one-sided matchups.

### Survivor Analysis
Higher average survivors for the winning team suggests dominance. Lower survivors suggest close battles.

### Position Heatmaps
Shows where winning teams tend to end up on the battlefield. Useful for understanding tactical positioning.

### Win Rate Trends
- Positive slope: Quantum team has advantage with larger teams
- Negative slope: Classical team has advantage with larger teams
- Flat line: Team size doesn't affect outcome

## Customization

### Adding Custom Compositions

Edit the `get_balanced_compositions()` method in `battlefield_tester.py`:

```python
# Add your custom composition
custom_comp = {
    'soldier': (count, strength, health, range),
    'knight': (count, strength, health, range),
    'archer': (count, strength, health, range)
}
compositions.append((custom_comp, other_comp, "Description"))
```

### Modifying Visualizations

All plotting methods are in the `BattlefieldTester` class:
- `_plot_victory_distribution()`
- `_plot_duration_distribution()`
- `_plot_survivors_distribution()`
- etc.

Modify these methods to customize visualizations.

## Performance Tips

### For Large-Scale Testing (10,000+ battles):

1. **Use Test 1 or 3** (they're more efficient than Test 2)
2. **Lower max-turns** if battles are running too long:
   ```bash
   python battlefield_tester.py --test 1 --battles 10000 --max-turns 30
   ```
3. **Monitor memory** - each battle stores results

### Expected Performance:
- 100 battles: ~5-10 seconds
- 1000 battles: ~1-2 minutes
- 10000 battles: ~10-20 minutes

(Times vary based on hardware and battle complexity)

## Troubleshooting

### ImportError: No module named 'quantum_library'
Make sure `quantum_library.py` exists in the same directory. The current version is a mock implementation.

### ImportError: No module named 'matplotlib'
Install required packages:
```bash
pip install matplotlib seaborn numpy tqdm
```

### Battles are too slow
- Reduce `--max-turns`
- Reduce `--battles`
- Check if quantum_library implementation is efficient

### No output files
Check that you have write permissions in the current directory.

## Code Structure

### BattleResult Class
Stores individual battle results with all relevant metrics.

### BattlefieldTester Class
Main testing framework with methods:
- `generate_random_composition()` - Create random teams
- `get_balanced_compositions()` - Get predefined compositions
- `run_single_battle()` - Execute one battle
- `test_random_compositions()` - Run Test 1
- `test_balanced_compositions()` - Run Test 2
- `test_performance_metrics()` - Run Test 3
- `generate_comprehensive_plots()` - Create visualizations
- `print_summary_statistics()` - Display results

## Future Enhancements

Potential additions:
- Export results to CSV/JSON
- Interactive visualizations (Plotly)
- Statistical significance testing
- Machine learning analysis of winning strategies
- Real-time battle visualization
- Parallel battle execution for faster testing
- Custom metric tracking
- Tournament-style multi-round testing

## Contributing

To add new test types:
1. Add a new method `test_your_name()` to `BattlefieldTester`
2. Update the argument parser in `main()`
3. Add documentation here

## License

This is part of the IQM Quantum Hack 2025 - Indra Challenge project.

## Contact

For issues or questions, please refer to the main repository documentation.

---

**Happy Testing!** 🎮⚔️🔬
