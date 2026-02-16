#!/usr/bin/env python3
"""
Comprehensive Call Center Performance Analysis
==============================================

Advanced business intelligence tool for call center optimization providing
detailed analysis across multiple operational dimensions including performance
metrics, cost optimization, and service level compliance.

This module extends the core M/M/c simulation to deliver enterprise-level
analytics for strategic decision making in call center management.

Features:
- Large-scale configuration analysis (5-100 agents)
- Multi-dimensional performance dashboards
- Cost-benefit optimization analysis
- Service level agreement tracking
- Statistical performance modeling
- Executive reporting and recommendations

"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from mm_c_queue_simulation import run_multiple_servers_experiment, CustomerServiceSimulation

# Optional seaborn import for enhanced styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available, using matplotlib defaults")

# Set style for better visualizations
try:
    if HAS_SEABORN:
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    else:
        plt.style.use('ggplot')
except:
    try:
        if HAS_SEABORN:
            plt.style.use('seaborn')
            sns.set_palette("husl")
        else:
            plt.style.use('classic')
    except:
        pass  # Use default style

def run_comprehensive_analysis():
    """Run comprehensive call center analysis with extensive server range"""
    
    print("=" * 100)
    print("COMPREHENSIVE CALL CENTER PERFORMANCE ANALYSIS")
    print("Testing Large-Scale Server Configurations (5-100 servers)")
    print("=" * 100)
    
    # Enhanced system parameters for realistic call center - Updated with optimal values
    arrival_rate = 35.0      # 35 calls per hour (optimal medium-high volume)
    service_rate = 4.5       # 4.5 calls per hour per agent (realistic service time)
    simulation_time = 2000   # 2000 hours (optimal accuracy vs speed)
    
    print(f"System Configuration:")
    print(f"  - Arrival rate: {arrival_rate} calls/hour")
    print(f"  - Service rate: {service_rate} calls/hour/agent")
    print(f"  - Simulation time: {simulation_time} hours")
    print(f"  - Agent cost: €28/hour")
    print(f"  - Target SLAs: 30s, 1min, 2min")
    print("=" * 100)
    
    # Define comprehensive server range - Updated for optimal parameters
    server_ranges = {
        'critical': list(range(8, 15)),           # Critical understaffed range
        'operational': list(range(15, 25, 2)),    # Normal operational range  
        'optimal': list(range(25, 35, 2)),        # Optimal range
        'extended': list(range(35, 51, 3))        # Extended analysis range
    }
    
    all_servers = []
    for range_name, servers in server_ranges.items():
        all_servers.extend(servers)
        print(f"  {range_name.title()}: {len(servers)} configurations ({min(servers)}-{max(servers)} servers)")
    
    print(f"\nTotal configurations to test: {len(all_servers)}")
    print("=" * 100)
    
    # Run comprehensive experiment
    print("Running simulations... This may take several minutes.")
    results = run_multiple_servers_experiment(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        server_range=all_servers,
        sim_time=simulation_time,
        replications=5  # Optimal balance for statistical confidence
    )
    
    # Enhanced metrics calculation
    print("Calculating enhanced metrics...")
    for c in results:
        r = results[c]
        
        # Financial metrics
        r['agent_cost_hourly'] = c * 28.0  # €28 per agent per hour (optimal rate)
        r['agent_cost_daily'] = r['agent_cost_hourly'] * 24
        r['agent_cost_monthly'] = r['agent_cost_daily'] * 30
        
        # Performance metrics in minutes
        r['avg_wait_minutes'] = r['avg_wait_time'] * 60
        r['max_wait_minutes'] = r.get('max_wait_time', 0) * 60
        
        # Service Level Agreements
        r['sla_30s'] = 100.0 if r['avg_wait_minutes'] <= 0.5 else 0.0
        r['sla_1min'] = 100.0 if r['avg_wait_minutes'] <= 1.0 else 0.0
        r['sla_2min'] = 100.0 if r['avg_wait_minutes'] <= 2.0 else 0.0
        
        # Efficiency and capacity metrics
        r['calls_per_agent'] = arrival_rate / c
        r['capacity_utilization'] = r['server_utilization'] / r['traffic_intensity'] if r['traffic_intensity'] > 0 else 0
        r['cost_per_call'] = r['agent_cost_hourly'] / arrival_rate if arrival_rate > 0 else 0
        
        # Performance categories
        if r['avg_wait_minutes'] <= 0.5:
            r['performance_category'] = 'Excellent'
        elif r['avg_wait_minutes'] <= 1.0:
            r['performance_category'] = 'Good'
        elif r['avg_wait_minutes'] <= 2.0:
            r['performance_category'] = 'Acceptable'
        elif r['avg_wait_minutes'] <= 5.0:
            r['performance_category'] = 'Poor'
        else:
            r['performance_category'] = 'Unacceptable'
    
    # Create comprehensive visualization dashboard
    create_comprehensive_dashboard(results, arrival_rate, service_rate)
    
    # Generate detailed analysis report
    generate_analysis_report(results, arrival_rate, service_rate)
    
    return results

def create_comprehensive_dashboard(results, arrival_rate, service_rate):
    """Create separate individual infographics for better readability"""
    
    server_counts = sorted(results.keys())
    
    # Set up consistent styling for all plots
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # 1. Wait Time Analysis (Log Scale) - Individual Plot
    plt.figure(figsize=(14, 10))
    wait_times = [results[c]['avg_wait_minutes'] for c in server_counts]
    plt.plot(server_counts, wait_times, 'b-', linewidth=4, alpha=0.9, marker='o', markersize=8)
    plt.fill_between(server_counts, wait_times, alpha=0.3, color='blue')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=3, label='30s SLA Target')
    plt.axhline(y=1.0, color='orange', linestyle='--', linewidth=3, label='1min SLA Target')
    plt.axhline(y=2.0, color='green', linestyle='--', linewidth=3, label='2min SLA Target')
    plt.title(f'Average Wait Time vs Number of Agents (Log Scale)\nCall Center Performance: {arrival_rate} calls/hour', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of Agents', fontsize=14)
    plt.ylabel('Wait Time (minutes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')
    
    # Add annotations for key points
    optimal_30s = min([c for c in server_counts if results[c]['avg_wait_minutes'] <= 0.5], default=None)
    optimal_1min = min([c for c in server_counts if results[c]['avg_wait_minutes'] <= 1.0], default=None)
    if optimal_30s:
        plt.annotate(f'30s SLA OPTIMAL\n{optimal_30s} agents\n€{results[optimal_30s]["agent_cost_monthly"]/1000:.0f}K/month\n{results[optimal_30s]["avg_wait_minutes"]:.2f} min wait', 
                    xy=(optimal_30s, results[optimal_30s]['avg_wait_minutes']),
                    xytext=(optimal_30s + 8, results[optimal_30s]['avg_wait_minutes'] * 3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.8))
    
    if optimal_1min and optimal_1min != optimal_30s:
        plt.annotate(f'1min SLA OPTIMAL\n{optimal_1min} agents\n€{results[optimal_1min]["agent_cost_monthly"]/1000:.0f}K/month\n{results[optimal_1min]["avg_wait_minutes"]:.2f} min wait', 
                    xy=(optimal_1min, results[optimal_1min]['avg_wait_minutes']),
                    xytext=(optimal_1min + 5, results[optimal_1min]['avg_wait_minutes'] * 2),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=3),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 2. Wait Time Analysis (Linear Scale) - Individual Plot
    plt.figure(figsize=(14, 10))
    plt.plot(server_counts, wait_times, 'b-', linewidth=4, alpha=0.9, marker='o', markersize=8)
    plt.fill_between(server_counts, wait_times, alpha=0.3, color='blue')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=3, label='30s SLA Target')
    plt.axhline(y=1.0, color='orange', linestyle='--', linewidth=3, label='1min SLA Target')
    plt.axhline(y=2.0, color='green', linestyle='--', linewidth=3, label='2min SLA Target')
    plt.title(f'Average Wait Time vs Number of Agents (Linear Scale)\nCall Center Performance: {arrival_rate} calls/hour', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of Agents', fontsize=14)
    plt.ylabel('Wait Time (minutes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, max(5, max(wait_times[:len(server_counts)//2])))  # Focus on reasonable wait times
    
    # Add same annotations for linear scale
    if optimal_30s:
        plt.annotate(f'30s SLA OPTIMAL\n{optimal_30s} agents\n€{results[optimal_30s]["agent_cost_monthly"]/1000:.0f}K/month', 
                    xy=(optimal_30s, results[optimal_30s]['avg_wait_minutes']),
                    xytext=(optimal_30s + 3, results[optimal_30s]['avg_wait_minutes'] + 1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.8))
    
    if optimal_1min and optimal_1min != optimal_30s:
        plt.annotate(f'1min SLA OPTIMAL\n{optimal_1min} agents\n€{results[optimal_1min]["agent_cost_monthly"]/1000:.0f}K/month', 
                    xy=(optimal_1min, results[optimal_1min]['avg_wait_minutes']),
                    xytext=(optimal_1min + 2, results[optimal_1min]['avg_wait_minutes'] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=3),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 2. Cost Analysis - Separate Plot
    plt.figure(figsize=(14, 10))
    monthly_costs = [results[c]['agent_cost_monthly'] for c in server_counts]
    daily_costs = [results[c]['agent_cost_daily'] for c in server_counts]
    hourly_costs = [results[c]['agent_cost_hourly'] for c in server_counts]
    
    plt.subplot(2, 2, 1)
    plt.bar(server_counts[::3], [monthly_costs[i] for i in range(0, len(monthly_costs), 3)], 
            alpha=0.8, color='red', width=2)
    plt.title('Monthly Staffing Cost', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Monthly Cost (€)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(server_counts, hourly_costs, 'g-', linewidth=3, marker='s', markersize=6)
    plt.title('Hourly Staffing Cost', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Hourly Cost (€)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    cost_per_call = [results[c]['cost_per_call'] for c in server_counts]
    plt.plot(server_counts, cost_per_call, 'purple', linewidth=3, marker='D', markersize=6)
    plt.title('Cost per Call', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Cost per Call (€)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Cost efficiency analysis
    efficiency = [results[c]['server_utilization'] / (c * 30 / 1000) for c in server_counts]  # Util per €1000
    plt.plot(server_counts, efficiency, 'orange', linewidth=3, marker='h', markersize=6)
    plt.title('Cost Efficiency (Utilization per €1000)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Efficiency Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('COST ANALYSIS DASHBOARD', fontsize=20, fontweight='bold', y=0.98)
    plt.show()
    
    # 3. Utilization Analysis - Separate Plot
    plt.figure(figsize=(14, 10))
    utilizations = [results[c]['server_utilization'] * 100 for c in server_counts]
    
    plt.subplot(2, 2, 1)
    colors = ['red' if u > 90 else 'orange' if u > 80 else 'lightgreen' if u > 70 else 'green' for u in utilizations]
    plt.scatter(server_counts, utilizations, c=colors, s=100, alpha=0.8, edgecolors='black')
    plt.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Target 80%')
    plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Max 90%')
    plt.title('Agent Utilization by Configuration', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Utilization (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    # Utilization histogram
    plt.hist(utilizations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='Target 80%')
    plt.title('Distribution of Utilization Rates', fontsize=16, fontweight='bold')
    plt.xlabel('Utilization (%)')
    plt.ylabel('Number of Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Traffic intensity
    traffic_intensities = [results[c]['traffic_intensity'] for c in server_counts]
    plt.plot(server_counts, traffic_intensities, 'brown', linewidth=3, marker='D', markersize=6)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Stability Limit')
    plt.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='Recommended Max')
    plt.title('Traffic Intensity (ρ)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Traffic Intensity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Utilization vs Traffic Intensity
    plt.scatter(traffic_intensities, utilizations, c=server_counts, cmap='viridis', s=80, alpha=0.7)
    plt.plot([0, 1], [0, 100], 'r--', alpha=0.5, label='Perfect Efficiency')
    plt.title('Utilization vs Traffic Intensity', fontsize=16, fontweight='bold')
    plt.xlabel('Traffic Intensity (ρ)')
    plt.ylabel('Utilization (%)')
    plt.colorbar(label='Number of Agents')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle('UTILIZATION ANALYSIS DASHBOARD', fontsize=20, fontweight='bold', y=0.98)
    plt.show()
    
    # 4. Queue Length Analysis - Separate Plot
    plt.figure(figsize=(14, 10))
    queue_lengths = [results[c]['avg_queue_length'] for c in server_counts]
    
    plt.subplot(2, 2, 1)
    plt.semilogy(server_counts, queue_lengths, 'purple', linewidth=4, marker='o', markersize=6)
    plt.title('Average Queue Length (Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Queue Length (log scale)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Linear scale for smaller queue lengths
    reasonable_queues = [min(q, 20) for q in queue_lengths]  # Cap at 20 for visibility
    plt.plot(server_counts, reasonable_queues, 'purple', linewidth=4, marker='o', markersize=6)
    plt.fill_between(server_counts, reasonable_queues, alpha=0.3, color='purple')
    plt.title('Queue Length (Capped at 20)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('Queue Length')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Queue length distribution
    plt.hist(queue_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Queue Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Average Queue Length')
    plt.ylabel('Number of Configurations')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Queue length vs wait time correlation
    plt.scatter(queue_lengths, wait_times, c=server_counts, cmap='plasma', s=80, alpha=0.7)
    plt.title('Queue Length vs Wait Time', fontsize=16, fontweight='bold')
    plt.xlabel('Average Queue Length')
    plt.ylabel('Wait Time (minutes)')
    plt.colorbar(label='Number of Agents')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('QUEUE LENGTH ANALYSIS', fontsize=20, fontweight='bold', y=0.98)
    plt.show()
    
    # 5. Service Level Agreement (SLA) Analysis - Separate Plot
    plt.figure(figsize=(14, 10))
    
    sla_30s = [results[c]['sla_30s'] for c in server_counts]
    sla_1min = [results[c]['sla_1min'] for c in server_counts]
    sla_2min = [results[c]['sla_2min'] for c in server_counts]
    
    plt.subplot(2, 2, 1)
    plt.plot(server_counts, sla_30s, 'r-o', linewidth=3, markersize=8, label='30s SLA')
    plt.plot(server_counts, sla_1min, 'orange', linewidth=3, marker='s', markersize=8, label='1min SLA')
    plt.plot(server_counts, sla_2min, 'g-^', linewidth=3, markersize=8, label='2min SLA')
    plt.title('SLA Achievement by Agent Count', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Agents')
    plt.ylabel('SLA Achievement (%)')
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.subplot(2, 2, 2)
    # SLA transition points
    transition_30s = min([c for c in server_counts if results[c]['sla_30s'] == 100.0], default=None)
    transition_1min = min([c for c in server_counts if results[c]['sla_1min'] == 100.0], default=None)
    transition_2min = min([c for c in server_counts if results[c]['sla_2min'] == 100.0], default=None)
    
    transitions = []
    labels = []
    colors = []
    if transition_2min:
        transitions.append(transition_2min)
        labels.append('2min SLA')
        colors.append('green')
    if transition_1min:
        transitions.append(transition_1min)
        labels.append('1min SLA')
        colors.append('orange')
    if transition_30s:
        transitions.append(transition_30s)
        labels.append('30s SLA')
        colors.append('red')
    
    if transitions:
        plt.bar(labels, transitions, color=colors, alpha=0.7)
        plt.title('Minimum Agents for SLA Achievement', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Agents Required')
        plt.grid(True, alpha=0.3)
        
        # Add cost annotations
        for i, (label, agents) in enumerate(zip(labels, transitions)):
            cost = results[agents]['agent_cost_monthly']
            plt.text(i, agents + max(transitions) * 0.05, f'€{cost:,.0f}/month', 
                    ha='center', fontweight='bold', fontsize=10)
    
    plt.subplot(2, 2, 3)
    # Performance categories pie chart
    categories = {}
    for c in server_counts:
        cat = results[c]['performance_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    colors_cat = {'Excellent': 'green', 'Good': 'lightgreen', 'Acceptable': 'yellow', 
                  'Poor': 'orange', 'Unacceptable': 'red'}
    plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%',
            colors=[colors_cat.get(cat, 'gray') for cat in categories.keys()],
            startangle=90)
    plt.title('Performance Category Distribution', fontsize=16, fontweight='bold')
    
    plt.subplot(2, 2, 4)
    # SLA cost analysis
    if transitions:
        costs_sla = [results[agents]['agent_cost_monthly'] for agents in transitions]
        plt.plot(labels, costs_sla, 'bo-', linewidth=3, markersize=10)
        plt.title('Monthly Cost by SLA Level', fontsize=16, fontweight='bold')
        plt.ylabel('Monthly Cost (€)')
        plt.grid(True, alpha=0.3)
        
        # Add percentage annotations
        for i, (label, cost) in enumerate(zip(labels, costs_sla)):
            if i > 0:
                pct_increase = ((cost - costs_sla[i-1]) / costs_sla[i-1]) * 100
                plt.text(i, cost + max(costs_sla) * 0.05, f'+{pct_increase:.0f}%', 
                        ha='center', fontweight='bold', color='red', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('SERVICE LEVEL AGREEMENT (SLA) ANALYSIS', fontsize=20, fontweight='bold', y=0.98)
    plt.show()
    
    # 6. Cost vs Performance Trade-off Analysis - Separate Plot
    plt.figure(figsize=(14, 10))
    
    costs = [results[c]['agent_cost_hourly'] for c in server_counts]
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(costs, wait_times, c=server_counts, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    plt.title('Cost vs Wait Time Trade-off', fontsize=16, fontweight='bold')
    plt.xlabel('Hourly Cost (€)')
    plt.ylabel('Wait Time (minutes)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Number of Agents')
    
    # Add annotations for key points
    if transition_1min:
        cost_1min = results[transition_1min]['agent_cost_hourly']
        wait_1min = results[transition_1min]['avg_wait_minutes']
        plt.annotate(f'1min SLA\n{transition_1min} agents', 
                    xy=(cost_1min, wait_1min),
                    xytext=(cost_1min + 200, wait_1min * 2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, fontweight='bold', ha='center')
    
    plt.subplot(2, 2, 2)
    # Efficiency frontier
    utilizations = [results[c]['server_utilization'] * 100 for c in server_counts]
    plt.scatter(costs, utilizations, c=wait_times, cmap='RdYlGn_r', s=100, alpha=0.7, edgecolors='black')
    plt.title('Cost vs Utilization (colored by wait time)', fontsize=16, fontweight='bold')
    plt.xlabel('Hourly Cost (€)')
    plt.ylabel('Utilization (%)')
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(label='Wait Time (minutes)')
    
    plt.subplot(2, 2, 3)
    # ROI analysis (cost per minute saved)
    if len(server_counts) > 1:
        cost_differences = [costs[i] - costs[i-1] for i in range(1, len(costs))]
        wait_improvements = [wait_times[i-1] - wait_times[i] for i in range(1, len(wait_times))]
        roi = [cd / wi if wi > 0 else float('inf') for cd, wi in zip(cost_differences, wait_improvements)]
        
        plt.plot(server_counts[1:], roi, 'purple', linewidth=3, marker='o', markersize=6)
        plt.title('Cost per Minute Wait Time Saved', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Agents')
        plt.ylabel('Additional Cost per Minute Saved (€/min)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Pareto frontier analysis
    normalized_cost = [(c - min(costs)) / (max(costs) - min(costs)) for c in costs]
    normalized_wait = [(w - min(wait_times)) / (max(wait_times) - min(wait_times)) for w in wait_times]
    pareto_score = [nc + nw for nc, nw in zip(normalized_cost, normalized_wait)]
    
    plt.scatter(normalized_cost, normalized_wait, c=pareto_score, cmap='RdYlBu_r', s=100, alpha=0.7, edgecolors='black')
    plt.title('Cost-Performance Pareto Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Normalized Cost (0=min, 1=max)')
    plt.ylabel('Normalized Wait Time (0=min, 1=max)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Pareto Score (lower=better)')
    
    # Find and annotate Pareto optimal points
    min_pareto_idx = pareto_score.index(min(pareto_score))
    plt.annotate(f'Optimal\n{server_counts[min_pareto_idx]} agents', 
                xy=(normalized_cost[min_pareto_idx], normalized_wait[min_pareto_idx]),
                xytext=(0.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.suptitle('COST vs PERFORMANCE OPTIMIZATION', fontsize=20, fontweight='bold', y=0.98)
    plt.show()
    
    print("All individual infographics have been displayed separately!")
    print("Each chart focuses on specific metrics for detailed analysis.")

def generate_analysis_report(results, arrival_rate, service_rate):
    """Generate a detailed text analysis report"""
    
    server_counts = sorted(results.keys())
    
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS REPORT")
    print("=" * 100)
    
    # Find key configurations
    optimal_30s = min([c for c in server_counts if results[c]['sla_30s'] == 100.0], default=None)
    optimal_1min = min([c for c in server_counts if results[c]['sla_1min'] == 100.0], default=None)
    optimal_2min = min([c for c in server_counts if results[c]['sla_2min'] == 100.0], default=None)
    
    # Find most cost-efficient configuration (best utilization with acceptable performance)
    acceptable_configs = [c for c in server_counts if results[c]['avg_wait_minutes'] <= 2.0]
    if acceptable_configs:
        most_efficient = max(acceptable_configs, key=lambda c: results[c]['server_utilization'])
    else:
        most_efficient = None
    
    print("KEY FINDINGS:")
    print("-" * 50)
    
    if optimal_30s:
        r = results[optimal_30s]
        print(f"30-second SLA Achievement:")
        print(f"  • Minimum agents required: {optimal_30s}")
        print(f"  • Monthly cost: €{r['agent_cost_monthly']:,.0f}")
        print(f"  • Agent utilization: {r['server_utilization']*100:.1f}%")
        print(f"  • Average wait time: {r['avg_wait_minutes']:.1f} minutes")
        print()
    
    if optimal_1min:
        r = results[optimal_1min]
        print(f"1-minute SLA Achievement:")
        print(f"  • Minimum agents required: {optimal_1min}")
        print(f"  • Monthly cost: €{r['agent_cost_monthly']:,.0f}")
        print(f"  • Agent utilization: {r['server_utilization']*100:.1f}%")
        print(f"  • Average wait time: {r['avg_wait_minutes']:.1f} minutes")
        print()
    
    if optimal_2min:
        r = results[optimal_2min]
        print(f"2-minute SLA Achievement:")
        print(f"  • Minimum agents required: {optimal_2min}")
        print(f"  • Monthly cost: €{r['agent_cost_monthly']:,.0f}")
        print(f"  • Agent utilization: {r['server_utilization']*100:.1f}%")
        print(f"  • Average wait time: {r['avg_wait_minutes']:.1f} minutes")
        print()
    
    if most_efficient:
        r = results[most_efficient]
        print(f"Most Cost-Efficient Configuration:")
        print(f"  • Agents: {most_efficient}")
        print(f"  • Monthly cost: €{r['agent_cost_monthly']:,.0f}")
        print(f"  • Agent utilization: {r['server_utilization']*100:.1f}%")
        print(f"  • Average wait time: {r['avg_wait_minutes']:.1f} minutes")
        print(f"  • Cost per call: €{r['cost_per_call']:.2f}")
        print()
    
    # Performance summary table
    print("PERFORMANCE SUMMARY TABLE:")
    print("-" * 100)
    print(f"{'Agents':^8} | {'Wait(min)':^10} | {'Util(%)':^8} | {'Cost/Month':^12} | {'SLA-30s':^8} | {'SLA-1m':^8} | {'Performance':^12}")
    print("-" * 100)
    
    for c in server_counts[::5]:  # Show every 5th configuration to avoid clutter
        r = results[c]
        print(f"{c:^8} | {r['avg_wait_minutes']:^10.2f} | {r['server_utilization']*100:^8.1f} | "
              f"€{r['agent_cost_monthly']:^11,.0f} | {r['sla_30s']:^8.0f} | {r['sla_1min']:^8.0f} | "
              f"{r['performance_category']:^12}")
    
    print("=" * 100)
    print("ANALYSIS COMPLETE - Comprehensive dashboard saved as visualization")
    print("=" * 100)

if __name__ == "__main__":
    results = run_comprehensive_analysis() 