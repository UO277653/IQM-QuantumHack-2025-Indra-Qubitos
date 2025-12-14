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
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Disable toolbar in interactive window
matplotlib.rcParams['toolbar'] = 'None'
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
    initial_config: str = 'default'  # 'face_to_face', 'surrounded', 'random', or 'default'
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

    def run_single_battle(self, battle_id: int, quantum_comp: dict, classical_comp: dict,
                         initial_config: str = 'default') -> BattleResult:
        """
        Run a single battle and collect results.

        Args:
            battle_id: Unique identifier for this battle
            quantum_comp: Quantum team composition
            classical_comp: Classical team composition
            initial_config: Initial configuration type ('face_to_face', 'surrounded', 'random', or 'default')

        Returns:
            BattleResult object with battle statistics
        """
        start_time = time.time()

        # Create battlefield
        battlefield = QuantumBattlefield(width=self.grid_size[0], height=self.grid_size[1])

        # Record initial counts
        quantum_initial = sum(count for count, _, _, _ in quantum_comp.values())
        classical_initial = sum(count for count, _, _, _ in classical_comp.values())

        # Initialize teams based on configuration type
        if initial_config == 'default':
            battlefield.initialize_team('Quantum', quantum_comp, start_x_range=(0, 1))
            battlefield.initialize_team('Classical', classical_comp, start_x_range=(2, 3))
        else:
            # Get positions for the specific configuration
            quantum_positions = battlefield.get_initial_configuration_positions(
                initial_config, 'Quantum', quantum_initial
            )
            classical_positions = battlefield.get_initial_configuration_positions(
                initial_config, 'Classical', classical_initial
            )
            battlefield.initialize_team('Quantum', quantum_comp, positions=quantum_positions)
            battlefield.initialize_team('Classical', classical_comp, positions=classical_positions)

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
            initial_config=initial_config,
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

    def test_initial_configurations(self):
        """Test 4: Test different initial configurations (face to face, surrounded, random)."""
        print(f"\n{'='*60}")
        print(f"TEST 4: INITIAL CONFIGURATIONS TEST")
        print(f"Testing different initial battlefield configurations...")
        print(f"{'='*60}\n")

        configurations = ['face_to_face', 'surrounded', 'random']
        battles_per_config = self.num_battles // len(configurations)

        battle_id = 0
        for config in configurations:
            config_name = config.replace('_', ' ').title()
            print(f"\nTesting configuration: {config_name}")

            # Use a mix of random and balanced compositions
            balanced_comps = self.get_balanced_compositions()

            for i in tqdm(range(battles_per_config), desc=f"  {config_name}"):
                if i % 3 == 0:
                    # Use balanced composition
                    quantum_comp, classical_comp, _ = random.choice(balanced_comps)
                else:
                    # Use random composition
                    quantum_comp = self.generate_random_composition()
                    classical_comp = self.generate_random_composition()

                result = self.run_single_battle(battle_id, quantum_comp, classical_comp, initial_config=config)
                self.results.append(result)
                battle_id += 1

    def generate_comprehensive_plots(self):
        """Generate comprehensive visualization plots."""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print(f"{'='*60}\n")

        # Set modern style
        sns.set_style("darkgrid")
        plt.rcParams['figure.facecolor'] = '#f8f9fa'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

        # Create figure with subplots - 2x3 grid (6 plots)
        fig = plt.figure(figsize=(20, 11))
        fig.patch.set_facecolor('#f8f9fa')

        # Add main title
        fig.suptitle('QUANTUM BATTLEFIELD ANALYSIS DASHBOARD',
                     fontsize=20, fontweight='bold', y=0.98, color='#2c3e50',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#e2e8f0', edgecolor='#cbd5e1', linewidth=2))

        # 1. Victory Distribution (Pie Chart)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_victory_distribution(ax1)

        # 2. Win Rate by Initial Team Size
        ax2 = plt.subplot(2, 3, 2)
        self._plot_winrate_by_team_size(ax2)

        # 3. Battle Outcome Timeline
        ax3 = plt.subplot(2, 3, 3)
        self._plot_battle_timeline(ax3)

        # 4. Win Rate by Initial Configuration
        ax4 = plt.subplot(2, 3, 4)
        self._plot_winrate_by_initial_config(ax4)

        # 5. Turn Distribution by Configuration
        ax5 = plt.subplot(2, 3, 5)
        self._plot_turns_by_config(ax5)

        # 6. Summary Statistics Table
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_statistics_table(ax6)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.subplots_adjust(hspace=0.30, wspace=0.25)  # Add more space between subplots
        plt.savefig('battlefield_test_results.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
        print("Saved comprehensive plot: battlefield_test_results.png")
        print("Displaying plot...")
        plt.show()  # Show the plot on screen
        plt.close()

    def _plot_victory_distribution(self, ax):
        """Plot victory distribution pie chart."""
        winners = [r.winner for r in self.results]
        winner_counts = defaultdict(int)
        for w in winners:
            winner_counts[w] += 1

        # Modern color palette
        colors = {
            'Quantum': '#3b82f6',    # Modern blue
            'Classical': '#ef4444',   # Modern red
            'Draw': '#64748b'         # Modern gray
        }
        labels = list(winner_counts.keys())
        sizes = list(winner_counts.values())
        plot_colors = [colors.get(label, '#94a3b8') for label in labels]

        # Add explode effect for visual impact
        explode = [0.05 if label == 'Quantum' else 0.03 for label in labels]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=plot_colors,
            startangle=90,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )

        # Make percentage text white for better contrast
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)

        ax.set_title('VICTORY DISTRIBUTION', fontsize=14, fontweight='bold', pad=15, color='#1e293b')



    def _plot_winrate_by_team_size(self, ax):
        """Plot win rate by initial team size."""
        size_wins = defaultdict(lambda: {'Quantum': 0, 'Classical': 0, 'Draw': 0})

        for r in self.results:
            size_diff = r.quantum_initial_count - r.classical_initial_count
            size_wins[size_diff][r.winner] += 1

        sizes = sorted(size_wins.keys())
        quantum_rates = [size_wins[s]['Quantum'] / sum(size_wins[s].values()) * 100 for s in sizes]
        classical_rates = [size_wins[s]['Classical'] / sum(size_wins[s].values()) * 100 for s in sizes]

        # Plot with modern styling
        ax.plot(sizes, quantum_rates, marker='o', label='Quantum',
                color='#3b82f6', linewidth=3, markersize=8,
                markerfacecolor='#3b82f6', markeredgewidth=2, markeredgecolor='white')
        ax.plot(sizes, classical_rates, marker='s', label='Classical',
                color='#ef4444', linewidth=3, markersize=8,
                markerfacecolor='#ef4444', markeredgewidth=2, markeredgecolor='white')

        ax.set_xlabel('Team Size Difference (Quantum - Classical)', fontsize=11, fontweight='bold', color='#334155')
        ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold', color='#334155')
        ax.set_title('WIN RATE BY TEAM SIZE', fontsize=14, fontweight='bold', pad=15, color='#1e293b')

        # Improved legend
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=10)

        # Better grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        ax.axhline(50, color='#64748b', linestyle='--', alpha=0.6, linewidth=2, label='50% baseline')

        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')


    def _plot_battle_timeline(self, ax):
        """Plot battle outcomes over time."""
        battle_ids = [r.battle_id for r in self.results]
        quantum_wins = [1 if r.winner == 'Quantum' else 0 for r in self.results]

        # Calculate rolling win rate (window of 20 battles)
        window = min(20, len(quantum_wins))
        rolling_winrate = np.convolve(quantum_wins, np.ones(window)/window, mode='valid')

        # Plot with gradient effect
        x = range(len(rolling_winrate))
        y = rolling_winrate * 100

        ax.plot(x, y, color='#3b82f6', linewidth=3, label='Quantum Win Rate', zorder=3)
        ax.fill_between(x, y, 50, where=(np.array(y) >= 50), alpha=0.2, color='#3b82f6', interpolate=True)
        ax.fill_between(x, y, 50, where=(np.array(y) < 50), alpha=0.2, color='#ef4444', interpolate=True)

        ax.set_xlabel('Battle Sequence', fontsize=11, fontweight='bold', color='#334155')
        ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold', color='#334155')
        ax.set_title(f'BATTLE OUTCOME TIMELINE (Window: {window})', fontsize=14, fontweight='bold', pad=15, color='#1e293b')

        # Reference line
        ax.axhline(50, color='#64748b', linestyle='--', alpha=0.6, linewidth=2, label='Equal win rate')

        # Improved legend
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=10)

        # Better grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)

        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')

        # Set y-axis limits for better visualization
        ax.set_ylim([0, 100])

    def _plot_winrate_by_initial_config(self, ax):
        """Plot win rate by initial configuration."""
        # Collect wins by configuration
        config_wins = defaultdict(lambda: {'Quantum': 0, 'Classical': 0, 'Draw': 0, 'total': 0})

        for r in self.results:
            config = r.initial_config
            config_wins[config][r.winner] += 1
            config_wins[config]['total'] += 1

        # Prepare data for plotting
        configs = sorted(config_wins.keys())
        config_labels = [c.replace('_', ' ').title() if c != 'default' else 'Default' for c in configs]
        quantum_rates = []
        classical_rates = []

        for config in configs:
            total = config_wins[config]['total']
            if total > 0:
                quantum_rates.append(config_wins[config]['Quantum'] / total * 100)
                classical_rates.append(config_wins[config]['Classical'] / total * 100)
            else:
                quantum_rates.append(0)
                classical_rates.append(0)

        # Create grouped bar chart
        x = np.arange(len(config_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, quantum_rates, width, label='Quantum',
                      color='#3b82f6', edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, classical_rates, width, label='Classical',
                      color='#ef4444', edgecolor='white', linewidth=1.5)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Initial Configuration', fontsize=11, fontweight='bold', color='#334155')
        ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold', color='#334155')
        ax.set_title('WIN RATE BY INITIAL CONFIGURATION', fontsize=14, fontweight='bold', pad=15, color='#1e293b')
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=15, ha='right')
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=10)

        # Better grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')
        ax.axhline(50, color='#64748b', linestyle='--', alpha=0.6, linewidth=2)

        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.set_ylim([0, 100])

    def _plot_turns_by_config(self, ax):
        """Plot average battle turns by initial configuration."""
        # Collect turns by configuration
        config_turns = defaultdict(list)

        for r in self.results:
            config = r.initial_config
            config_turns[config].append(r.turns)

        # Prepare data for box plot
        configs = sorted(config_turns.keys())
        config_labels = [c.replace('_', ' ').title() if c != 'default' else 'Default' for c in configs]
        turns_data = [config_turns[config] for config in configs]

        # Create violin plot for better distribution visualization
        parts = ax.violinplot(turns_data, positions=range(len(configs)),
                             showmeans=True, showmedians=True)

        # Color the violin plots
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.6)
            pc.set_edgecolor('white')
            pc.set_linewidth(1.5)

        # Style the lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('#1e293b')
                vp.set_linewidth(1.5)

        # Add mean values as text
        for i, config in enumerate(configs):
            mean_val = np.mean(config_turns[config])
            ax.text(i, mean_val, f'{mean_val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

        ax.set_xlabel('Initial Configuration', fontsize=11, fontweight='bold', color='#334155')
        ax.set_ylabel('Battle Duration (Turns)', fontsize=11, fontweight='bold', color='#334155')
        ax.set_title('BATTLE DURATION BY CONFIGURATION', fontsize=14, fontweight='bold', pad=15, color='#1e293b')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=15, ha='right')

        # Better grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')

        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')

    def _plot_summary_statistics_table(self, ax):
        """Plot summary statistics table."""
        ax.axis('off')

        quantum_wins = sum(1 for r in self.results if r.winner == "Quantum")
        classical_wins = sum(1 for r in self.results if r.winner == "Classical")
        draws = sum(1 for r in self.results if r.winner == "Draw")
        total = len(self.results)

        stats_data = [
            ['Metric', 'Value'],
            ['Total Battles', f'{total}'],
            ['Quantum Wins', f'{quantum_wins} ({quantum_wins/total*100:.1f}%)'],
            ['Classical Wins', f'{classical_wins} ({classical_wins/total*100:.1f}%)'],
            ['Draws', f'{draws} ({draws/total*100:.1f}%)'],
            ['Avg Turns', f'{np.mean([r.turns for r in self.results]):.2f}'],
        ]

        table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.55, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.8)

        # Style header row with gradient-like effect
        for i in range(2):
            table[(0, i)].set_facecolor('#1e293b')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
            table[(0, i)].set_edgecolor('#cbd5e1')

        # Alternate row colors for better readability
        for i in range(1, len(stats_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8fafc')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
                table[(i, j)].set_edgecolor('#cbd5e1')
                table[(i, j)].set_text_props(fontsize=11)

                # Highlight key metrics
                if 'Quantum Wins' in stats_data[i][0]:
                    table[(i, j)].set_text_props(color='#3b82f6', weight='bold', fontsize=11)
                elif 'Classical Wins' in stats_data[i][0]:
                    table[(i, j)].set_text_props(color='#ef4444', weight='bold', fontsize=11)

        ax.set_title('SUMMARY STATISTICS', fontsize=14, fontweight='bold', pad=20, color='#1e293b')


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
  4 - Initial Configuration Test: Test different initial battlefield configurations (face to face, surrounded, random)

Examples:
  python battlefield_tester.py --test 1 --battles 1000
  python battlefield_tester.py --test 2 --battles 500
  python battlefield_tester.py --test 3 --battles 2000
  python battlefield_tester.py --test 4 --battles 900
        """
    )

    parser.add_argument('--test', type=int, choices=[1, 2, 3, 4], required=True,
                       help='Test type to run (1, 2, 3, or 4)')
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
    elif args.test == 4:
        tester.test_initial_configurations()

    # Generate outputs
    tester.print_summary_statistics()
    tester.generate_comprehensive_plots()

    print("\nTesting complete!")
    print("="*60)
    print("Generated visualization file:")
    print("  - battlefield_test_results.png")
    print("="*60)


if __name__ == "__main__":
    main()
