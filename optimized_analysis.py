#!/usr/bin/env python3
"""
Call Center Staffing Analysis Tool
==================================

Analyzes call center performance to determine optimal staffing levels.
Focuses on key business metrics including cost optimization, wait times,
and service level agreements.


"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from mm_c_queue_simulation import run_multiple_servers_experiment, CustomerServiceSimulation
import time

# Configuration
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

class CallCenterAnalyzer:
    """Main analysis class for call center staffing optimization"""
    
    def __init__(self, arrival_rate=35.0, service_rate=4.5, agent_cost=28.0):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.agent_cost_hourly = agent_cost
        self.simulation_time = 2000  # hours - extended for higher accuracy
        self.results = {}
        
        # Calculate reasonable server range based on traffic intensity
        min_theoretical = max(8, int(arrival_rate / service_rate) + 1)
        max_reasonable = min(25, int(arrival_rate / service_rate * 2.5))
        
        # Use denser sampling around critical points for better accuracy
        self.server_range = (
            list(range(min_theoretical, min_theoretical + 10)) +  # Critical range
            list(range(min_theoretical + 10, max_reasonable, 3)) +  # Operational range
            [max_reasonable]  # Upper bound
        )
        
        print(f"Analysis Range: {len(self.server_range)} configurations")
        print(f"Testing {min(self.server_range)} to {max(self.server_range)} agents")
    
    def run_analysis(self):
        """Execute the complete simulation analysis"""
        print("\n" + "="*80)
        print("CALL CENTER ANALYSIS")
        print("="*80)
        print(f"System Parameters: {self.arrival_rate} calls/hr | {self.service_rate} calls/hr/agent | €{self.agent_cost_hourly}/hr/agent")
        
        start_time = time.time()
        
        # Run simulations - using optimal replications for statistical confidence
        self.results = run_multiple_servers_experiment(
            arrival_rate=self.arrival_rate,
            service_rate=self.service_rate,
            server_range=self.server_range,
            sim_time=self.simulation_time,
            replications=5  # Optimal balance between accuracy and speed
        )
        
        # Calculate business metrics
        self._calculate_business_metrics()
        
        runtime = time.time() - start_time
        print(f"Analysis completed in {runtime:.1f} seconds")
        
        # Generate insights and visualizations
        self._find_optimal_configurations()
        self._create_essential_visualizations()
        self._generate_executive_summary()
        
        return self.results
    
    def _calculate_business_metrics(self):
        """Calculate key business metrics efficiently"""
        for agents in self.results:
            r = self.results[agents]
            
            # Financial metrics
            r['monthly_cost'] = agents * self.agent_cost_hourly * 24 * 30
            r['cost_per_call'] = (agents * self.agent_cost_hourly) / self.arrival_rate
            
            # Performance metrics (in minutes for business clarity)
            r['wait_minutes'] = r['avg_wait_time'] * 60
            
            # SLA compliance (binary for clear decision making)
            r['meets_30s'] = r['wait_minutes'] <= 0.5
            r['meets_1min'] = r['wait_minutes'] <= 1.0
            r['meets_2min'] = r['wait_minutes'] <= 2.0
            
            # Business efficiency score
            r['efficiency'] = r['server_utilization'] / (r['traffic_intensity'] + 0.01)
    
    def _find_optimal_configurations(self):
        """Find key staffing recommendations"""
        agents_list = sorted(self.results.keys())
        
        # Find minimum agents for each SLA
        self.optimal_30s = min([a for a in agents_list if self.results[a]['meets_30s']], default=None)
        self.optimal_1min = min([a for a in agents_list if self.results[a]['meets_1min']], default=None)
        self.optimal_2min = min([a for a in agents_list if self.results[a]['meets_2min']], default=None)
        
        # Find most cost-efficient (best utilization with acceptable wait)
        acceptable = [a for a in agents_list if self.results[a]['wait_minutes'] <= 2.0]
        self.most_efficient = max(acceptable, key=lambda a: self.results[a]['efficiency']) if acceptable else None
        
        # Find balanced recommendation (good performance, reasonable cost)
        balanced_candidates = [a for a in agents_list if 
                              self.results[a]['wait_minutes'] <= 1.5 and 
                              self.results[a]['server_utilization'] >= 0.6]
        self.balanced = min(balanced_candidates, key=lambda a: self.results[a]['monthly_cost']) if balanced_candidates else None
    
    def _create_essential_visualizations(self):
        """Create only the most important visualizations"""
        
        agents_list = sorted(self.results.keys())
        
        # Create individual charts - shown one at a time for better analysis
        
        wait_times = [self.results[a]['wait_minutes'] for a in agents_list]
        
        # 1. Wait Time Analysis - Individual Plot
        plt.figure(figsize=(14, 10))
        plt.plot(agents_list, wait_times, 'b-o', linewidth=4, markersize=8, alpha=0.9)
        plt.fill_between(agents_list, wait_times, alpha=0.3, color='blue')
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=3, label='30s SLA Target')
        plt.axhline(y=1.0, color='orange', linestyle='--', linewidth=3, label='1min SLA Target')
        plt.axhline(y=2.0, color='green', linestyle='--', linewidth=3, label='2min SLA Target')
        
        # Add annotation for recommended configuration
        if self.optimal_1min:
            plt.annotate(f'RECOMMENDED\n{self.optimal_1min} agents\n€{self.results[self.optimal_1min]["monthly_cost"]/1000:.0f}K/month\n{self.results[self.optimal_1min]["wait_minutes"]:.2f} min wait\n{self.results[self.optimal_1min]["server_utilization"]*100:.1f}% utilization', 
                        xy=(self.optimal_1min, self.results[self.optimal_1min]['wait_minutes']),
                        xytext=(self.optimal_1min + 3, self.results[self.optimal_1min]['wait_minutes'] * 3),
                        arrowprops=dict(arrowstyle='->', color='red', lw=4),
                        fontweight='bold', ha='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
        
        # Add efficiency sweet spot annotation
        if self.most_efficient and self.most_efficient != self.optimal_1min:
            plt.annotate(f'Most Efficient\n{self.most_efficient} agents\n{self.results[self.most_efficient]["wait_minutes"]:.2f} min wait\n{self.results[self.most_efficient]["server_utilization"]*100:.1f}% utilization', 
                        xy=(self.most_efficient, self.results[self.most_efficient]['wait_minutes']),
                        xytext=(self.most_efficient - 2, self.results[self.most_efficient]['wait_minutes'] * 1.5),
                        arrowprops=dict(arrowstyle='->', color='green', lw=3),
                        fontweight='bold', ha='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
        
        plt.title(f'Wait Time vs Staffing Level\nCall Center Analysis: {self.arrival_rate} calls/hr, {self.service_rate} calls/hr/agent', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Agents', fontsize=14)
        plt.ylabel('Average Wait Time (minutes)', fontsize=14)
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Cost Analysis - Individual Plot
        plt.figure(figsize=(14, 10))
        monthly_costs = [self.results[a]['monthly_cost'] for a in agents_list]
        plt.plot(agents_list, monthly_costs, 'g-s', linewidth=4, markersize=8, alpha=0.9)
        plt.fill_between(agents_list, monthly_costs, alpha=0.3, color='green')
        
        # Highlight key configurations with detailed annotations
        if self.optimal_1min:
            plt.scatter([self.optimal_1min], [self.results[self.optimal_1min]['monthly_cost']], 
                       color='red', s=200, zorder=5, label='1min SLA Optimal', marker='*', edgecolors='black', linewidth=2)
            plt.annotate(f'RECOMMENDED\n{self.optimal_1min} agents\n€{self.results[self.optimal_1min]["monthly_cost"]/1000:.0f}K/month\n{self.results[self.optimal_1min]["wait_minutes"]:.2f} min wait\n{self.results[self.optimal_1min]["server_utilization"]*100:.1f}% utilization', 
                        xy=(self.optimal_1min, self.results[self.optimal_1min]['monthly_cost']),
                        xytext=(self.optimal_1min + 1, self.results[self.optimal_1min]['monthly_cost'] * 1.15),
                        arrowprops=dict(arrowstyle='->', color='red', lw=4),
                        fontweight='bold', ha='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
        
        if self.balanced and self.balanced != self.optimal_1min:
            plt.scatter([self.balanced], [self.results[self.balanced]['monthly_cost']], 
                       color='orange', s=150, zorder=5, label='Balanced Choice', marker='D', edgecolors='black', linewidth=1)
            plt.annotate(f'BALANCED\n{self.balanced} agents\n€{self.results[self.balanced]["monthly_cost"]/1000:.0f}K/month\n{self.results[self.balanced]["wait_minutes"]:.2f} min wait\n{self.results[self.balanced]["server_utilization"]*100:.1f}% utilization', 
                        xy=(self.balanced, self.results[self.balanced]['monthly_cost']),
                        xytext=(self.balanced - 1, self.results[self.balanced]['monthly_cost'] * 0.85),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=3),
                        fontweight='bold', ha='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='orange', alpha=0.7))
        
        # Add minimum cost point
        min_cost_agents = min(agents_list)
        plt.scatter([min_cost_agents], [self.results[min_cost_agents]['monthly_cost']], 
                   color='green', s=150, zorder=5, label='Minimum Cost', marker='s', edgecolors='black', linewidth=1)
        plt.annotate(f'MIN COST\n{min_cost_agents} agents\n€{self.results[min_cost_agents]["monthly_cost"]/1000:.0f}K/month\n{self.results[min_cost_agents]["wait_minutes"]:.2f} min wait\n{self.results[min_cost_agents]["server_utilization"]*100:.1f}% utilization', 
                    xy=(min_cost_agents, self.results[min_cost_agents]['monthly_cost']),
                    xytext=(min_cost_agents + 1, self.results[min_cost_agents]['monthly_cost'] * 0.75),
                    arrowprops=dict(arrowstyle='->', color='green', lw=3),
                    fontweight='bold', ha='center', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
        
        plt.title(f'Monthly Staffing Cost Analysis\nCall Center: {self.arrival_rate} calls/hr, €{self.agent_cost_hourly}/hr/agent', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Agents', fontsize=14)
        plt.ylabel('Monthly Cost (€)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 3. Utilization Analysis - Individual Plot
        plt.figure(figsize=(14, 10))
        utilizations = [self.results[a]['server_utilization'] * 100 for a in agents_list]
        colors = ['red' if wt > 2 else 'orange' if wt > 1 else 'lightblue' if wt > 0.5 else 'green' 
                 for wt in wait_times]
        
        scatter = plt.scatter(agents_list, utilizations, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1)
        plt.axhline(y=80, color='orange', linestyle='--', linewidth=3, alpha=0.8, label='High Utilization (80%)')
        plt.axhline(y=70, color='green', linestyle=':', linewidth=3, alpha=0.7, label='Efficient Range (70%)')
        plt.axhline(y=60, color='blue', linestyle=':', linewidth=2, alpha=0.6, label='Conservative (60%)')
        
        # Highlight optimal utilization points
        if self.optimal_1min:
            optimal_util = self.results[self.optimal_1min]['server_utilization'] * 100
            plt.scatter([self.optimal_1min], [optimal_util], 
                       color='red', s=200, zorder=10, marker='*', edgecolors='black', linewidth=2)
            plt.annotate(f'RECOMMENDED\n{self.optimal_1min} agents\n{optimal_util:.1f}% utilization\n€{self.results[self.optimal_1min]["monthly_cost"]/1000:.0f}K/month\n{self.results[self.optimal_1min]["wait_minutes"]:.2f} min wait', 
                        xy=(self.optimal_1min, optimal_util),
                        xytext=(self.optimal_1min + 1.5, optimal_util + 8),
                        arrowprops=dict(arrowstyle='->', color='red', lw=4),
                        fontweight='bold', ha='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
        
        # Find and highlight the sweet spot (70-80% utilization with good performance)
        sweet_spot_candidates = [a for a in agents_list 
                               if 70 <= self.results[a]['server_utilization'] * 100 <= 80 
                               and self.results[a]['wait_minutes'] <= 1.5]
        if sweet_spot_candidates and sweet_spot_candidates[0] != self.optimal_1min:
            sweet_spot = sweet_spot_candidates[0]  # Take the first one
            sweet_util = self.results[sweet_spot]['server_utilization'] * 100
            plt.scatter([sweet_spot], [sweet_util], 
                       color='gold', s=150, zorder=9, marker='D', edgecolors='black', linewidth=2)
            plt.annotate(f'SWEET SPOT\n{sweet_spot} agents\n{sweet_util:.1f}% utilization\n{self.results[sweet_spot]["wait_minutes"]:.2f} min wait', 
                        xy=(sweet_spot, sweet_util),
                        xytext=(sweet_spot - 1.5, sweet_util - 10),
                        arrowprops=dict(arrowstyle='->', color='gold', lw=3),
                        fontweight='bold', ha='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='gold', alpha=0.8))
        
        plt.title(f'Agent Utilization by Performance\nCall Center: {self.arrival_rate} calls/hr', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Agents', fontsize=14)
        plt.ylabel('Utilization (%)', fontsize=14)
        plt.legend(['High Utilization (80%)', 'Efficient Range (70%)', 'Conservative (60%)', 'Excellent (<30s)', 'Good (<1min)', 'Fair (1-2min)', 'Poor (>2min)'], fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()
        
        # 4. SLA Achievement Summary - Individual Plot
        plt.figure(figsize=(14, 10))
        sla_data = {
            '30s SLA': sum(1 for a in agents_list if self.results[a]['meets_30s']),
            '1min SLA': sum(1 for a in agents_list if self.results[a]['meets_1min']),
            '2min SLA': sum(1 for a in agents_list if self.results[a]['meets_2min'])
        }
        
        bars = plt.bar(sla_data.keys(), sla_data.values(), 
                      color=['red', 'orange', 'green'], alpha=0.8, edgecolor='black', linewidth=2)
        plt.title(f'SLA Achievement Summary\nConfigurations Meeting Service Level Targets', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Number of Configurations Meeting SLA', fontsize=14)
        plt.xlabel('Service Level Agreement Target', fontsize=14)
        
        # Add percentage and count labels on bars
        for bar, count in zip(bars, sla_data.values()):
            percentage = (count / len(agents_list)) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count} configs\n({percentage:.0f}%)', ha='center', fontweight='bold', fontsize=12)
        
        # Add recommended configuration highlight
        if self.optimal_1min:
            plt.text(0.5, max(sla_data.values()) * 0.8, 
                    f'RECOMMENDED: {self.optimal_1min} agents\nMeets 1-minute SLA target\n€{self.results[self.optimal_1min]["monthly_cost"]/1000:.0f}K/month', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        # Show recommendations comparison
        self._create_recommendations_chart()
    
    def _create_recommendations_chart(self):
        """Create a focused recommendations visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Recommendation comparison
        recommendations = []
        costs = []
        wait_times = []
        labels = []
        
        if self.optimal_2min:
            recommendations.append(self.optimal_2min)
            costs.append(self.results[self.optimal_2min]['monthly_cost'])
            wait_times.append(self.results[self.optimal_2min]['wait_minutes'])
            labels.append('2min SLA\n(Basic)')
        
        if self.optimal_1min:
            recommendations.append(self.optimal_1min)
            costs.append(self.results[self.optimal_1min]['monthly_cost'])
            wait_times.append(self.results[self.optimal_1min]['wait_minutes'])
            labels.append('1min SLA\n(Standard)')
        
        if self.optimal_30s:
            recommendations.append(self.optimal_30s)
            costs.append(self.results[self.optimal_30s]['monthly_cost'])
            wait_times.append(self.results[self.optimal_30s]['wait_minutes'])
            labels.append('30s SLA\n(Premium)')
        
        # Cost comparison
        bars1 = ax1.bar(labels, costs, color=['green', 'orange', 'red'], alpha=0.7)
        ax1.set_title('Monthly Cost by Service Level', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Monthly Cost (€)')
        
        # Add cost labels
        for bar, cost in zip(bars1, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                    f'€{cost:,.0f}', ha='center', fontweight='bold')
        
        # Performance comparison
        bars2 = ax2.bar(labels, wait_times, color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_title('Wait Time by Service Level', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Wait Time (minutes)')
        
        # Add wait time labels
        for bar, wait in zip(bars2, wait_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(wait_times)*0.02,
                    f'{wait:.2f}min', ha='center', fontweight='bold')
        
        plt.suptitle('STAFFING RECOMMENDATIONS COMPARISON', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _generate_executive_summary(self):
        """Generate summary of key findings and recommendations"""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        print("KEY RECOMMENDATIONS:")
        print("-" * 40)
        
        if self.optimal_1min:
            r = self.results[self.optimal_1min]
            print(f"RECOMMENDED STAFFING: {self.optimal_1min} agents")
            print(f"   - Monthly Cost: €{r['monthly_cost']:,.0f}")
            print(f"   - Average Wait: {r['wait_minutes']:.1f} minutes")
            print(f"   - Agent Utilization: {r['server_utilization']*100:.1f}%")
            print(f"   - Meets 1-minute SLA with good cost efficiency")
        
        print("\nALTERNATIVE OPTIONS:")
        print("-" * 40)
        
        if self.optimal_2min and self.optimal_2min != self.optimal_1min:
            r = self.results[self.optimal_2min]
            savings = self.results[self.optimal_1min]['monthly_cost'] - r['monthly_cost']
            print(f"BUDGET OPTION: {self.optimal_2min} agents")
            print(f"   - Monthly savings: €{savings:,.0f}")
            print(f"   - Wait time: {r['wait_minutes']:.1f} minutes")
            print(f"   - Trade-off: Longer waits for cost savings")
        
        if self.optimal_30s and self.optimal_30s != self.optimal_1min:
            r = self.results[self.optimal_30s]
            extra_cost = r['monthly_cost'] - self.results[self.optimal_1min]['monthly_cost']
            print(f"PREMIUM OPTION: {self.optimal_30s} agents")
            print(f"   - Extra monthly cost: €{extra_cost:,.0f}")
            print(f"   - Wait time: {r['wait_minutes']:.1f} minutes")
            print(f"   - Benefit: Premium customer experience")
        
        # Business impact analysis
        if self.optimal_1min:
            base_config = self.results[self.optimal_1min]
            calls_per_month = self.arrival_rate * 24 * 30
            print(f"\nBUSINESS IMPACT (Recommended Configuration):")
            print("-" * 50)
            print(f"   - Monthly call volume: {calls_per_month:,.0f} calls")
            print(f"   - Cost per call: €{base_config['cost_per_call']:.2f}")
            print(f"   - Customer satisfaction: High (1-min response)")
            print(f"   - Operational efficiency: {base_config['server_utilization']*100:.1f}% utilization")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - Ready for implementation")
        print("="*80)

def run_analysis():
    """Main function to execute call center analysis"""
    # Initialize analyzer with typical call center parameters
    analyzer = CallCenterAnalyzer(
        arrival_rate=25.0,    # calls per hour
        service_rate=4.0,     # calls per hour per agent
        agent_cost=30.0       # euros per hour per agent
    )
    
    # Run the complete analysis
    results = analyzer.run_analysis()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = run_analysis() 