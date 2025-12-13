"""
Battlefield Testing Framework
Extensive testing suite for QuantumBattlefield with random battle generation and comprehensive analytics.

Usage:
    python battlefield_tester.py --test <test_type> --battles <num_battles>

Test Types:
    1 - Random Composition Test: Test with completely random unit compositions
    2 - Balance Test: Test predefined compositions to analyze balance
    3 - Performance Test: Measure battle statistics and performance metrics
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
from tqdm import tqdm

from battlefield import QuantumBattlefield, Soldier


@dataclass
class BattleResult:
    """Store results from a single battle."""
    battle_id: int
    winner: str
    turns: int
    quantum_survivors: int
    classical_survivors: int
    quantum_composition: dict
    classical_composition: dict
    quantum_initial_count: int
    classical_initial_count: int
    duration_seconds: float
    final_positions: List[Tuple[float, float, str]] = field(default_factory=list)


class BattlefieldTester:
    """Comprehensive testing framework for QuantumBattlefield."""

    def __init__(self, num_battles: int = 100, max_turns: int = 50, grid_size: Tuple[int, int] = (4, 4)):
        """
        Initialize the tester.

        Args:
            num_battles: Number of battles to simulate
            max_turns: Maximum turns per battle before declaring a draw
            grid_size: (width, height) of the battlefield
        """
        self.num_battles = num_battles
        self.max_turns = max_turns
        self.grid_size = grid_size
        self.results: List[BattleResult] = []

        # Available unit types with their stats
        self.unit_types = ['soldier', 'knight', 'archer']

    def generate_random_composition(self, max_units: int = 8) -> dict:
        """
        Generate a random team composition.

        Args:
            max_units: Maximum number of units in the team

        Returns:
            Dictionary with composition format: {'unit_type': (count, strength, health, range)}
        """
        composition = {}
        total_units = random.randint(3, max_units)

        # Distribute units randomly among types
        units_per_type = [0, 0, 0]
        for _ in range(total_units):
            units_per_type[random.randint(0, 2)] += 1

        for unit_type, count in zip(self.unit_types, units_per_type):
            if count > 0:
                strength = random.randint(1, 3)
                health = random.randint(1, 3)
                range_dist = random.randint(1, 3)
                composition[unit_type] = (count, strength, health, range_dist)

        return composition

    def get_balanced_compositions(self) -> List[Tuple[dict, dict, str]]:
        """
        Get a list of predefined balanced compositions for testing.

        Returns:
            List of tuples: (quantum_comp, classical_comp, description)
        """
        compositions = []

        # Test 1: Identical balanced teams
        balanced = {
            'soldier': (2, 2, 2, 1),
            'knight': (1, 3, 3, 1),
            'archer': (1, 2, 1, 3)
        }
        compositions.append((balanced.copy(), balanced.copy(), "Identical Balanced"))

        # Test 2: All soldiers vs balanced
        all_soldiers = {'soldier': (4, 2, 2, 1)}
        compositions.append((all_soldiers, balanced.copy(), "All Soldiers vs Balanced"))

        # Test 3: All archers vs balanced
        all_archers = {'archer': (4, 2, 2, 3)}
        compositions.append((all_archers, balanced.copy(), "All Archers vs Balanced"))

        # Test 4: All knights vs balanced
        all_knights = {'knight': (4, 3, 3, 1)}
        compositions.append((all_knights, balanced.copy(), "All Knights vs Balanced"))

        # Test 5: High strength vs high health
        high_strength = {
            'soldier': (2, 3, 1, 1),
            'knight': (2, 3, 2, 1)
        }
        high_health = {
            'soldier': (2, 1, 3, 1),
            'knight': (2, 2, 3, 1)
        }
        compositions.append((high_strength, high_health, "High Strength vs High Health"))

        # Test 6: Long range vs short range
        long_range = {'archer': (4, 2, 2, 3)}
        short_range = {'knight': (4, 3, 3, 1)}
        compositions.append((long_range, short_range, "Long Range vs Short Range"))

        # Test 7: Many weak units vs few strong units
        many_weak = {'soldier': (6, 1, 1, 1)}
        few_strong = {'knight': (2, 3, 3, 1)}
        compositions.append((many_weak, few_strong, "Many Weak vs Few Strong"))

        # Test 8: Diverse team
        diverse1 = {
            'soldier': (1, 2, 2, 1),
            'knight': (1, 3, 2, 1),
            'archer': (2, 2, 2, 2)
        }
        diverse2 = {
            'soldier': (2, 2, 2, 1),
            'knight': (1, 2, 3, 1),
            'archer': (1, 2, 2, 3)
        }
        compositions.append((diverse1, diverse2, "Diverse Teams"))

        return compositions

    def run_single_battle(self, battle_id: int, quantum_comp: dict, classical_comp: dict) -> BattleResult:
        """
        Run a single battle and collect results.

        Args:
            battle_id: Unique identifier for this battle
            quantum_comp: Quantum team composition
            classical_comp: Classical team composition

        Returns:
            BattleResult object with battle statistics
        """
        start_time = time.time()

        # Create battlefield
        battlefield = QuantumBattlefield(width=self.grid_size[0], height=self.grid_size[1])

        # Initialize teams
        battlefield.initialize_team('Quantum', quantum_comp, start_x_range=(0, 1))
        battlefield.initialize_team('Classical', classical_comp, start_x_range=(2, 3))

        # Record initial counts
        quantum_initial = sum(count for count, _, _, _ in quantum_comp.values())
        classical_initial = sum(count for count, _, _, _ in classical_comp.values())

        # Run battle
        while not battlefield.is_battle_over() and battlefield.turn < self.max_turns:
            battlefield.step()

        # Collect results
        winner = battlefield.get_winner()
        if winner is None:
            winner = "Draw"

        quantum_survivors, classical_survivors = battlefield.get_survivor_count()

        # Collect final positions
        final_positions = [(s.x, s.y, s.team) for s in battlefield.soldiers]

        duration = time.time() - start_time

        return BattleResult(
            battle_id=battle_id,
            winner=winner,
            turns=battlefield.turn,
            quantum_survivors=quantum_survivors,
            classical_survivors=classical_survivors,
            quantum_composition=quantum_comp,
            classical_composition=classical_comp,
            quantum_initial_count=quantum_initial,
            classical_initial_count=classical_initial,
            duration_seconds=duration,
            final_positions=final_positions
        )

    def test_random_compositions(self):
        """Test 1: Generate and test random compositions."""
        print(f"\n{'='*60}")
        print(f"TEST 1: RANDOM COMPOSITION TEST")
        print(f"Running {self.num_battles} battles with random compositions...")
        print(f"{'='*60}\n")

        for i in tqdm(range(self.num_battles), desc="Simulating battles"):
            quantum_comp = self.generate_random_composition()
            classical_comp = self.generate_random_composition()

            result = self.run_single_battle(i, quantum_comp, classical_comp)
            self.results.append(result)

    def test_balanced_compositions(self):
        """Test 2: Test predefined balanced compositions."""
        print(f"\n{'='*60}")
        print(f"TEST 2: BALANCE TEST")
        print(f"Testing predefined compositions to analyze balance...")
        print(f"{'='*60}\n")

        compositions = self.get_balanced_compositions()
        battles_per_comp = self.num_battles // len(compositions)

        battle_id = 0
        for quantum_comp, classical_comp, description in compositions:
            print(f"\nTesting: {description}")
            for _ in tqdm(range(battles_per_comp), desc=f"  {description}"):
                result = self.run_single_battle(battle_id, quantum_comp, classical_comp)
                self.results.append(result)
                battle_id += 1

    def test_performance_metrics(self):
        """Test 3: Focus on performance and statistical metrics."""
        print(f"\n{'='*60}")
        print(f"TEST 3: PERFORMANCE & STATISTICS TEST")
        print(f"Running {self.num_battles} battles with performance tracking...")
        print(f"{'='*60}\n")

        # Use a mix of random and balanced compositions
        balanced_comps = self.get_balanced_compositions()

        for i in tqdm(range(self.num_battles), desc="Simulating battles"):
            if i % 3 == 0:
                # Use balanced composition
                quantum_comp, classical_comp, _ = random.choice(balanced_comps)
            else:
                # Use random composition
                quantum_comp = self.generate_random_composition()
                classical_comp = self.generate_random_composition()

            result = self.run_single_battle(i, quantum_comp, classical_comp)
            self.results.append(result)

    def generate_comprehensive_plots(self):
        """Generate comprehensive visualization plots."""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print(f"{'='*60}\n")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 14))

        # 1. Victory Distribution (Pie Chart)
        ax1 = plt.subplot(3, 3, 1)
        self._plot_victory_distribution(ax1)

        # 2. Battle Duration Distribution
        ax2 = plt.subplot(3, 3, 2)
        self._plot_duration_distribution(ax2)

        # 3. Survivors Distribution
        ax3 = plt.subplot(3, 3, 3)
        self._plot_survivors_distribution(ax3)

        # 4. Turns per Battle
        ax4 = plt.subplot(3, 3, 4)
        self._plot_turns_distribution(ax4)

        # 5. Win Rate by Initial Team Size
        ax5 = plt.subplot(3, 3, 5)
        self._plot_winrate_by_team_size(ax5)

        # 6. Average Survivors by Winner
        ax6 = plt.subplot(3, 3, 6)
        self._plot_survivors_by_winner(ax6)

        # 7. Battle Outcome Timeline
        ax7 = plt.subplot(3, 3, 7)
        self._plot_battle_timeline(ax7)

        # 8. Final Position Heatmap - Quantum
        ax8 = plt.subplot(3, 3, 8)
        self._plot_position_heatmap(ax8, 'Quantum')

        # 9. Final Position Heatmap - Classical
        ax9 = plt.subplot(3, 3, 9)
        self._plot_position_heatmap(ax9, 'Classical')

        plt.tight_layout()
        plt.savefig('battlefield_test_results.png', dpi=300, bbox_inches='tight')
        print("Saved comprehensive plot: battlefield_test_results.png")
        plt.close()

        # Generate additional detailed plots
        self._generate_detailed_statistics_plot()

    def _plot_victory_distribution(self, ax):
        """Plot victory distribution pie chart."""
        winners = [r.winner for r in self.results]
        winner_counts = defaultdict(int)
        for w in winners:
            winner_counts[w] += 1

        colors = {'Quantum': '#4169E1', 'Classical': '#DC143C', 'Draw': '#808080'}
        labels = list(winner_counts.keys())
        sizes = list(winner_counts.values())
        plot_colors = [colors.get(label, '#CCCCCC') for label in labels]

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=plot_colors, startangle=90)
        ax.set_title('Victory Distribution', fontsize=12, fontweight='bold')

    def _plot_duration_distribution(self, ax):
        """Plot battle duration distribution."""
        durations = [r.duration_seconds * 1000 for r in self.results]  # Convert to ms
        ax.hist(durations, bins=30, color='#2E8B57', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Battle Duration Distribution', fontsize=12, fontweight='bold')
        ax.axvline(np.mean(durations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(durations):.1f}ms')
        ax.legend()

    def _plot_survivors_distribution(self, ax):
        """Plot survivors distribution."""
        quantum_survivors = [r.quantum_survivors for r in self.results]
        classical_survivors = [r.classical_survivors for r in self.results]

        x = np.arange(max(max(quantum_survivors), max(classical_survivors)) + 1)
        quantum_dist = [quantum_survivors.count(i) for i in x]
        classical_dist = [classical_survivors.count(i) for i in x]

        width = 0.35
        ax.bar(x - width/2, quantum_dist, width, label='Quantum', color='#4169E1', alpha=0.7)
        ax.bar(x + width/2, classical_dist, width, label='Classical', color='#DC143C', alpha=0.7)

        ax.set_xlabel('Number of Survivors')
        ax.set_ylabel('Frequency')
        ax.set_title('Survivors Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xticks(x)

    def _plot_turns_distribution(self, ax):
        """Plot distribution of battle turns."""
        turns = [r.turns for r in self.results]
        ax.hist(turns, bins=20, color='#9370DB', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Turns')
        ax.set_ylabel('Frequency')
        ax.set_title('Battle Length Distribution', fontsize=12, fontweight='bold')
        ax.axvline(np.mean(turns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(turns):.1f}')
        ax.legend()

    def _plot_winrate_by_team_size(self, ax):
        """Plot win rate by initial team size."""
        size_wins = defaultdict(lambda: {'Quantum': 0, 'Classical': 0, 'Draw': 0})

        for r in self.results:
            size_diff = r.quantum_initial_count - r.classical_initial_count
            size_wins[size_diff][r.winner] += 1

        sizes = sorted(size_wins.keys())
        quantum_rates = [size_wins[s]['Quantum'] / sum(size_wins[s].values()) * 100 for s in sizes]
        classical_rates = [size_wins[s]['Classical'] / sum(size_wins[s].values()) * 100 for s in sizes]

        ax.plot(sizes, quantum_rates, marker='o', label='Quantum Win Rate', color='#4169E1', linewidth=2)
        ax.plot(sizes, classical_rates, marker='s', label='Classical Win Rate', color='#DC143C', linewidth=2)
        ax.set_xlabel('Quantum Size - Classical Size')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate by Team Size Difference', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(50, color='black', linestyle=':', alpha=0.5)

    def _plot_survivors_by_winner(self, ax):
        """Plot average survivors grouped by winner."""
        winners = ['Quantum', 'Classical', 'Draw']
        quantum_survivors_by_winner = defaultdict(list)
        classical_survivors_by_winner = defaultdict(list)

        for r in self.results:
            quantum_survivors_by_winner[r.winner].append(r.quantum_survivors)
            classical_survivors_by_winner[r.winner].append(r.classical_survivors)

        quantum_means = [np.mean(quantum_survivors_by_winner[w]) if quantum_survivors_by_winner[w] else 0 for w in winners]
        classical_means = [np.mean(classical_survivors_by_winner[w]) if classical_survivors_by_winner[w] else 0 for w in winners]

        x = np.arange(len(winners))
        width = 0.35

        ax.bar(x - width/2, quantum_means, width, label='Quantum Survivors', color='#4169E1', alpha=0.7)
        ax.bar(x + width/2, classical_means, width, label='Classical Survivors', color='#DC143C', alpha=0.7)

        ax.set_xlabel('Battle Winner')
        ax.set_ylabel('Average Survivors')
        ax.set_title('Average Survivors by Winner', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(winners)
        ax.legend()

    def _plot_battle_timeline(self, ax):
        """Plot battle outcomes over time."""
        battle_ids = [r.battle_id for r in self.results]
        quantum_wins = [1 if r.winner == 'Quantum' else 0 for r in self.results]

        # Calculate rolling win rate (window of 20 battles)
        window = min(20, len(quantum_wins))
        rolling_winrate = np.convolve(quantum_wins, np.ones(window)/window, mode='valid')

        ax.plot(range(len(rolling_winrate)), rolling_winrate * 100, color='#4169E1', linewidth=2)
        ax.set_xlabel('Battle ID')
        ax.set_ylabel('Quantum Win Rate (%)')
        ax.set_title(f'Quantum Win Rate Over Time (Rolling Window: {window})', fontsize=12, fontweight='bold')
        ax.axhline(50, color='black', linestyle='--', alpha=0.5, label='50% Win Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_position_heatmap(self, ax, team):
        """Plot heatmap of final positions for winning team."""
        heatmap = np.zeros((self.grid_size[1], self.grid_size[0]))

        for r in self.results:
            if r.winner == team:
                for x, y, t in r.final_positions:
                    if t == team:
                        heatmap[int(y), int(x)] += 1

        color = 'Blues' if team == 'Quantum' else 'Reds'
        sns.heatmap(heatmap, annot=True, fmt='.0f', cmap=color, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title(f'{team} Final Positions (Wins Only)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

    def _generate_detailed_statistics_plot(self):
        """Generate additional detailed statistics plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Correlation: Initial Size vs Survivors
        ax1 = axes[0, 0]
        quantum_initial = [r.quantum_initial_count for r in self.results]
        quantum_survivors = [r.quantum_survivors for r in self.results]
        ax1.scatter(quantum_initial, quantum_survivors, alpha=0.5, color='#4169E1', label='Quantum')

        classical_initial = [r.classical_initial_count for r in self.results]
        classical_survivors = [r.classical_survivors for r in self.results]
        ax1.scatter(classical_initial, classical_survivors, alpha=0.5, color='#DC143C', label='Classical')

        ax1.set_xlabel('Initial Team Size')
        ax1.set_ylabel('Final Survivors')
        ax1.set_title('Initial Size vs Survivors', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Turn Distribution by Winner
        ax2 = axes[0, 1]
        quantum_win_turns = [r.turns for r in self.results if r.winner == 'Quantum']
        classical_win_turns = [r.turns for r in self.results if r.winner == 'Classical']
        draw_turns = [r.turns for r in self.results if r.winner == 'Draw']

        ax2.hist([quantum_win_turns, classical_win_turns, draw_turns],
                bins=15, label=['Quantum Wins', 'Classical Wins', 'Draws'],
                color=['#4169E1', '#DC143C', '#808080'], alpha=0.6)
        ax2.set_xlabel('Number of Turns')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Battle Length by Winner', fontsize=12, fontweight='bold')
        ax2.legend()

        # 3. Execution Time Statistics
        ax3 = axes[1, 0]
        durations = [r.duration_seconds * 1000 for r in self.results]
        turns = [r.turns for r in self.results]
        ax3.scatter(turns, durations, alpha=0.5, color='#2E8B57')
        ax3.set_xlabel('Number of Turns')
        ax3.set_ylabel('Execution Time (ms)')
        ax3.set_title('Performance: Turns vs Execution Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_data = [
            ['Metric', 'Value'],
            ['Total Battles', f'{len(self.results)}'],
            ['Quantum Wins', f'{sum(1 for r in self.results if r.winner == "Quantum")}'],
            ['Classical Wins', f'{sum(1 for r in self.results if r.winner == "Classical")}'],
            ['Draws', f'{sum(1 for r in self.results if r.winner == "Draw")}'],
            ['Avg Turns', f'{np.mean([r.turns for r in self.results]):.2f}'],
            ['Avg Duration (ms)', f'{np.mean([r.duration_seconds * 1000 for r in self.results]):.2f}'],
            ['Avg Quantum Survivors', f'{np.mean([r.quantum_survivors for r in self.results]):.2f}'],
            ['Avg Classical Survivors', f'{np.mean([r.classical_survivors for r in self.results]):.2f}'],
        ]

        table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4169E1')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('battlefield_detailed_statistics.png', dpi=300, bbox_inches='tight')
        print("Saved detailed statistics plot: battlefield_detailed_statistics.png")
        plt.close()

    def print_summary_statistics(self):
        """Print detailed summary statistics."""
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}\n")

        total = len(self.results)
        quantum_wins = sum(1 for r in self.results if r.winner == 'Quantum')
        classical_wins = sum(1 for r in self.results if r.winner == 'Classical')
        draws = sum(1 for r in self.results if r.winner == 'Draw')

        print(f"Total Battles Simulated: {total}")
        print(f"Quantum Wins: {quantum_wins} ({quantum_wins/total*100:.2f}%)")
        print(f"Classical Wins: {classical_wins} ({classical_wins/total*100:.2f}%)")
        print(f"Draws: {draws} ({draws/total*100:.2f}%)")
        print()

        avg_turns = np.mean([r.turns for r in self.results])
        std_turns = np.std([r.turns for r in self.results])
        print(f"Average Battle Length: {avg_turns:.2f} ± {std_turns:.2f} turns")
        print(f"Shortest Battle: {min(r.turns for r in self.results)} turns")
        print(f"Longest Battle: {max(r.turns for r in self.results)} turns")
        print()

        avg_duration = np.mean([r.duration_seconds for r in self.results]) * 1000
        print(f"Average Execution Time: {avg_duration:.2f} ms/battle")
        print(f"Total Simulation Time: {sum(r.duration_seconds for r in self.results):.2f} seconds")
        print()

        avg_q_survivors = np.mean([r.quantum_survivors for r in self.results])
        avg_c_survivors = np.mean([r.classical_survivors for r in self.results])
        print(f"Average Quantum Survivors: {avg_q_survivors:.2f}")
        print(f"Average Classical Survivors: {avg_c_survivors:.2f}")
        print()

        if quantum_wins > 0:
            avg_q_win_survivors = np.mean([r.quantum_survivors for r in self.results if r.winner == 'Quantum'])
            print(f"Avg Quantum Survivors (when winning): {avg_q_win_survivors:.2f}")

        if classical_wins > 0:
            avg_c_win_survivors = np.mean([r.classical_survivors for r in self.results if r.winner == 'Classical'])
            print(f"Avg Classical Survivors (when winning): {avg_c_win_survivors:.2f}")

        print(f"\n{'='*60}\n")


def main():
    """Main entry point for the testing framework."""
    parser = argparse.ArgumentParser(
        description='Battlefield Testing Framework - Extensive battle simulation and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  1 - Random Composition Test: Generate battles with completely random unit compositions
  2 - Balance Test: Test predefined compositions to analyze game balance
  3 - Performance Test: Measure battle statistics and performance metrics

Examples:
  python battlefield_tester.py --test 1 --battles 1000
  python battlefield_tester.py --test 2 --battles 500
  python battlefield_tester.py --test 3 --battles 2000
        """
    )

    parser.add_argument('--test', type=int, choices=[1, 2, 3], required=True,
                       help='Test type to run (1, 2, or 3)')
    parser.add_argument('--battles', type=int, default=100,
                       help='Number of battles to simulate (default: 100)')
    parser.add_argument('--max-turns', type=int, default=50,
                       help='Maximum turns per battle (default: 50)')
    parser.add_argument('--grid-width', type=int, default=4,
                       help='Battlefield width (default: 4)')
    parser.add_argument('--grid-height', type=int, default=4,
                       help='Battlefield height (default: 4)')

    args = parser.parse_args()

    # Initialize tester
    tester = BattlefieldTester(
        num_battles=args.battles,
        max_turns=args.max_turns,
        grid_size=(args.grid_width, args.grid_height)
    )

    print("\n" + "="*60)
    print("BATTLEFIELD TESTING FRAMEWORK")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of Battles: {args.battles}")
    print(f"  - Max Turns per Battle: {args.max_turns}")
    print(f"  - Grid Size: {args.grid_width}x{args.grid_height}")
    print(f"  - Test Type: {args.test}")

    # Run selected test
    if args.test == 1:
        tester.test_random_compositions()
    elif args.test == 2:
        tester.test_balanced_compositions()
    elif args.test == 3:
        tester.test_performance_metrics()

    # Generate outputs
    tester.print_summary_statistics()
    tester.generate_comprehensive_plots()

    print("\nTesting complete!")
    print("Generated files:")
    print("  - battlefield_test_results.png")
    print("  - battlefield_detailed_statistics.png")


if __name__ == "__main__":
    main()
