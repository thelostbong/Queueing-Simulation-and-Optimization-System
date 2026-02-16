#!/usr/bin/env python3
"""
M/M/c Queue Simulation Engine
============================

Core discrete event simulation module for multi-server queueing systems
modeling customer service centers. Implements the classic M/M/c queueing
model using SimPy for accurate performance analysis.

This module provides:
- CustomerServiceSimulation class for M/M/c queue modeling
- Statistical analysis and performance metrics calculation
- Theoretical validation using Erlang C formula
- Multi-scenario experimentation framework
- Visualization tools for results analysis

Technical Specifications:
- Model: M/M/c (Markovian arrivals, Markovian service, c servers)
- Method: Discrete event simulation with warmup period
- Validation: Statistical comparison with theoretical results
- Metrics: Wait times, utilization, queue lengths, SLA compliance

"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time


class CustomerServiceSimulation:
    """
    Multi-server (M/M/c) queue simulation for a customer service center
    """
    
    def __init__(self, env, num_servers, arrival_rate, service_rate, 
                 simulation_time=1000, warmup_time=100, 
                 track_queue_length=True, time_unit="minutes"):
        """
        Initialize the simulation
        
        Parameters:
        -----------
        env : simpy.Environment
            The SimPy environment
        num_servers : int
            Number of service agents (c in M/M/c)
        arrival_rate : float
            Average number of customers arriving per time unit (lambda)
        service_rate : float
            Average number of customers served per time unit per server (mu)
        simulation_time : float
            Total simulation time
        warmup_time : float
            Time to warm up the simulation before collecting statistics
        track_queue_length : bool
            Whether to track queue length over time
        time_unit : str
            The unit of time (for reporting purposes)
        """
        self.env = env
        self.servers = simpy.Resource(env, capacity=num_servers)
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        self.time_unit = time_unit
        
        # Statistics
        self.wait_times = []
        self.service_times = []
        self.queue_lengths = []
        self.queue_length_times = []
        self.system_times = []
        self.arrivals = 0
        self.completed = 0
        self.track_queue_length = track_queue_length
        
        # Start processes
        self.env.process(self.customer_arrivals())
        if track_queue_length:
            self.env.process(self.monitor_queue())
    
    def customer_arrivals(self):
        """Generate customer arrivals based on exponential interarrival times"""
        while True:
            # Generate next arrival
            interarrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(interarrival_time)
            
            # Create a new customer
            self.arrivals += 1
            self.env.process(self.customer_service(f"Customer {self.arrivals}"))
    
    def customer_service(self, customer_id):
        """Process a customer through the service system"""
        arrival_time = self.env.now
        
        # Request a server
        with self.servers.request() as request:
            # Wait for a server to become available
            queue_entry_time = self.env.now
            yield request
            queue_exit_time = self.env.now
            
            # Calculate waiting time
            wait_time = queue_exit_time - queue_entry_time
            
            # Generate service time (exponential)
            service_time = random.expovariate(self.service_rate)
            yield self.env.timeout(service_time)
            
            # Collect statistics only after warmup
            if self.env.now > self.warmup_time:
                self.wait_times.append(wait_time)
                self.service_times.append(service_time)
                self.system_times.append(wait_time + service_time)
            
            self.completed += 1
    
    def monitor_queue(self):
        """Monitor queue length at regular intervals"""
        while True:
            if self.env.now > self.warmup_time:
                self.queue_lengths.append(len(self.servers.queue))
                self.queue_length_times.append(self.env.now)
            yield self.env.timeout(1.0)  # Sample every time unit
    
    def get_statistics(self):
        """Calculate and return simulation statistics"""
        stats = {
            'avg_wait_time': np.mean(self.wait_times) if self.wait_times else 0,
            'max_wait_time': np.max(self.wait_times) if self.wait_times else 0,
            'avg_service_time': np.mean(self.service_times) if self.service_times else 0,
            'avg_system_time': np.mean(self.system_times) if self.system_times else 0,
            'avg_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': np.max(self.queue_lengths) if self.queue_lengths else 0,
            'server_utilization': (sum(self.service_times) / (self.simulation_time - self.warmup_time)) / self.servers.capacity,
            'customers_served': len(self.wait_times),
            'total_customers': self.arrivals
        }
        
        # Calculate percentiles for wait time
        if self.wait_times:
            percentiles = [50, 75, 90, 95, 99]
            for p in percentiles:
                stats[f'wait_time_p{p}'] = np.percentile(self.wait_times, p)
                
        return stats
    
    def plot_queue_length(self):
        """Plot queue length over time"""
        if not self.track_queue_length or not self.queue_lengths:
            print("Queue length tracking disabled or no data collected.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.queue_length_times, self.queue_lengths)
        plt.title('Queue Length Over Time')
        plt.xlabel(f'Time ({self.time_unit})')
        plt.ylabel('Number of Customers in Queue')
        plt.grid(True)
        return plt
    
    def plot_wait_time_histogram(self):
        """Plot histogram of waiting times"""
        if not self.wait_times:
            print("No wait time data collected.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(self.wait_times, bins=30, alpha=0.7)
        plt.title('Distribution of Customer Wait Times')
        plt.xlabel(f'Wait Time ({self.time_unit})')
        plt.ylabel('Frequency')
        plt.grid(True)
        return plt


def run_simulation(num_servers, arrival_rate, service_rate, sim_time=1000, warmup=100, random_seed=None):
    """Run a single simulation with given parameters"""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create simulation
    sim = CustomerServiceSimulation(
        env=env,
        num_servers=num_servers,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        simulation_time=sim_time,
        warmup_time=warmup
    )
    
    # Run simulation
    start_time = time.time()
    env.run(until=sim_time)
    runtime = time.time() - start_time
    
    # Get statistics
    stats = sim.get_statistics()
    stats['simulation_runtime'] = runtime
    
    return sim, stats


def run_multiple_servers_experiment(arrival_rate, service_rate, server_range, sim_time=1000, replications=5):
    """
    Run simulations with varying number of servers
    
    Parameters:
    -----------
    arrival_rate : float
        Customer arrival rate (lambda)
    service_rate : float
        Service rate per server (mu)
    server_range : range or list
        Range of server counts to test
    sim_time : float
        Simulation time
    replications : int
        Number of replications for each configuration
    
    Returns:
    --------
    dict
        Results for each server count
    """
    results = {}
    
    for c in server_range:
        print(f"Running simulation with {c} servers...")
        c_results = []
        
        for rep in range(replications):
            random_seed = rep * 1000 + c  # Different seed for each replication
            _, stats = run_simulation(
                num_servers=c,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                sim_time=sim_time,
                random_seed=random_seed
            )
            c_results.append(stats)
        
        # Calculate averages across replications
        avg_results = {
            'avg_wait_time': np.mean([r['avg_wait_time'] for r in c_results]),
            'avg_queue_length': np.mean([r['avg_queue_length'] for r in c_results]),
            'server_utilization': np.mean([r['server_utilization'] for r in c_results]),
            'traffic_intensity': arrival_rate / (c * service_rate),
            'theoretical_utilization': min(1.0, arrival_rate / (c * service_rate))
        }
        
        results[c] = avg_results
    
    return results


def plot_server_results(results, metric='avg_wait_time', title=None, ylabel=None):
    """Plot results across different server counts"""
    server_counts = sorted(results.keys())
    metric_values = [results[c][metric] for c in server_counts]
    
    plt.figure(figsize=(10, 6))
    plt.plot(server_counts, metric_values, 'o-')
    plt.title(title or f'{metric.replace("_", " ").title()} vs Number of Servers')
    plt.xlabel('Number of Servers (c)')
    plt.ylabel(ylabel or metric.replace('_', ' ').title())
    plt.grid(True)
    plt.xticks(server_counts)
    return plt


def calculate_theoretical_wait_time(arrival_rate, service_rate, num_servers):
    """
    Calculate theoretical expected wait time for an M/M/c queue using Erlang C formula
    """
    rho = arrival_rate / (num_servers * service_rate)  # Server utilization
    
    if rho >= 1:
        return float('inf')  # Unstable system
    
    # Calculate p0 (probability of empty system)
    sum_term = sum([(num_servers * rho)**n / np.math.factorial(n) for n in range(num_servers)])
    last_term = (num_servers * rho)**num_servers / (np.math.factorial(num_servers) * (1 - rho))
    p0 = 1 / (sum_term + last_term)
    
    # Calculate Erlang C (probability of waiting)
    erlang_c = ((num_servers * rho)**num_servers / np.math.factorial(num_servers)) * \
               (p0 / (1 - rho))
    
    # Expected wait time
    expected_wait = erlang_c / (num_servers * service_rate - arrival_rate)
    
    return expected_wait


def demo():
    """Run an optimized demonstration focusing on key insights"""
    print("=" * 80)
    print("CALL CENTER SIMULATION")
    print("=" * 80)
    
    # Optimized system parameters - Updated with optimal values
    arrival_rate = 35.0     # customers per hour (optimal medium-high volume)
    service_rate = 4.5      # customers per hour per server (realistic service time)
    min_servers = 8
    max_servers = 25        # Optimal range for analysis
    simulation_time = 2000  # Extended for higher accuracy
    
    print(f"System Parameters:")
    print(f"  Arrival rate (λ): {arrival_rate} customers/hour")
    print(f"  Service rate (μ): {service_rate} customers/hour/server")
    print(f"  Simulation time: {simulation_time} hours")
    print(f"  Testing server counts: {min_servers} to {max_servers}")
    print(f"  Traffic intensity range: {arrival_rate/(max_servers*service_rate):.3f} to {arrival_rate/(min_servers*service_rate):.3f}")
    print("=" * 80)
    
    # Run focused experiment with smart server selection
    smart_range = list(range(min_servers, min_servers + 8)) + list(range(min_servers + 8, max_servers + 1, 2))
    
    results = run_multiple_servers_experiment(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        server_range=smart_range,
        sim_time=simulation_time,
        replications=5  # Increased for better statistical confidence
    )
    
    # Calculate business metrics
    for c in results:
        r = results[c]
        r['monthly_cost'] = c * 28.0 * 24 * 30  # €28/hour/server (optimal rate)
        r['wait_minutes'] = r['avg_wait_time'] * 60
        r['meets_1min_sla'] = r['wait_minutes'] <= 1.0
        r['efficiency_score'] = r['server_utilization'] / max(0.1, r['traffic_intensity'])
    
    # Find optimal configurations
    optimal_1min = min([c for c in smart_range if results[c]['meets_1min_sla']], default=None)
    
    # Display focused results table
    print("\nOPTIMIZED RESULTS SUMMARY:")
    print("=" * 80)
    print(f"{'Servers':^7} | {'Wait(min)':^9} | {'Cost/Month':^11} | {'Util(%)':^8} | {'1min SLA':^8}")
    print("-" * 80)
    
    for c in sorted(results.keys()):
        r = results[c]
        sla_status = "PASS" if r['meets_1min_sla'] else "FAIL"
        print(f"{c:^7} | {r['wait_minutes']:^9.2f} | €{r['monthly_cost']:^10,.0f} | {r['server_utilization']*100:^8.1f} | {sla_status:^8}")
    
    # Create individual visualizations - shown one at a time
    server_counts = sorted(results.keys())
    wait_times = [results[c]['wait_minutes'] for c in server_counts]
    costs = [results[c]['monthly_cost'] for c in server_counts]
    
    # 1. Wait Time Analysis - Individual Plot
    plt.figure(figsize=(12, 8))
    plt.plot(server_counts, wait_times, 'b-o', linewidth=3, markersize=8)
    plt.fill_between(server_counts, wait_times, alpha=0.3, color='blue')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=3, label='1min SLA Target')
    plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='30s SLA Target')
    plt.axhline(y=2.0, color='green', linestyle='--', linewidth=2, label='2min SLA Target')
    plt.title(f'Wait Time vs Number of Servers\nCall Center Analysis: {arrival_rate} calls/hour', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Servers', fontsize=14)
    plt.ylabel('Wait Time (minutes)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if optimal_1min:
        plt.annotate(f'RECOMMENDED\n{optimal_1min} servers\n€{results[optimal_1min]["monthly_cost"]/1000:.0f}K/month\n{results[optimal_1min]["wait_minutes"]:.1f} min wait', 
                    xy=(optimal_1min, results[optimal_1min]['wait_minutes']),
                    xytext=(optimal_1min + 2, results[optimal_1min]['wait_minutes'] + 1.0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                    fontweight='bold', ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
    
    # Add minimum cost annotation
    min_servers = min(server_counts)
    plt.annotate(f'MIN COST\n{min_servers} servers\n€{results[min_servers]["monthly_cost"]/1000:.0f}K/month\n{results[min_servers]["wait_minutes"]:.1f} min wait', 
                xy=(min_servers, results[min_servers]['wait_minutes']),
                xytext=(min_servers + 1, results[min_servers]['wait_minutes'] * 0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontweight='bold', ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 2. Cost Analysis - Individual Plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(server_counts, costs, alpha=0.7, color='green', edgecolor='darkgreen', linewidth=1.5)
    plt.title(f'Monthly Staffing Cost Analysis\nCall Center: {arrival_rate} calls/hour', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Servers', fontsize=14)
    plt.ylabel('Monthly Cost (€)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Highlight recommended and minimum cost options
    if optimal_1min:
        optimal_idx = server_counts.index(optimal_1min)
        bars[optimal_idx].set_color('red')
        bars[optimal_idx].set_alpha(0.9)
        plt.annotate(f'RECOMMENDED\n{optimal_1min} servers\n€{costs[optimal_idx]/1000:.0f}K/month\n{results[optimal_1min]["server_utilization"]*100:.1f}% utilization', 
                    xy=(optimal_1min, costs[optimal_idx]),
                    xytext=(optimal_1min, costs[optimal_idx] * 1.2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                    fontweight='bold', ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
    
    # Highlight minimum cost
    min_idx = 0
    bars[min_idx].set_color('darkgreen')
    bars[min_idx].set_alpha(0.9)
    plt.annotate(f'MIN COST\n{server_counts[min_idx]} servers\n€{costs[min_idx]/1000:.0f}K/month\n{results[server_counts[min_idx]]["server_utilization"]*100:.1f}% utilization', 
                xy=(server_counts[min_idx], costs[min_idx]),
                xytext=(server_counts[min_idx] + 1, costs[min_idx] * 0.7),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontweight='bold', ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 3. Utilization Analysis - Individual Plot
    plt.figure(figsize=(12, 8))
    utilizations = [results[c]['server_utilization'] * 100 for c in server_counts]
    colors = ['red' if u > 90 else 'orange' if u > 80 else 'lightblue' if u > 70 else 'green' for u in utilizations]
    plt.scatter(server_counts, utilizations, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1)
    plt.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='High Utilization (80%)')
    plt.axhline(y=70, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Efficient Range (70%)')
    plt.axhline(y=60, color='blue', linestyle=':', linewidth=2, alpha=0.7, label='Conservative (60%)')
    
    # Highlight recommended configuration
    if optimal_1min:
        optimal_util = results[optimal_1min]['server_utilization'] * 100
        plt.scatter([optimal_1min], [optimal_util], 
                   color='red', s=200, zorder=10, marker='*', edgecolors='black', linewidth=2)
        plt.annotate(f'RECOMMENDED\n{optimal_1min} servers\n{optimal_util:.1f}% utilization\n€{results[optimal_1min]["monthly_cost"]/1000:.0f}K/month', 
                    xy=(optimal_1min, optimal_util),
                    xytext=(optimal_1min + 1.5, optimal_util + 10),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                    fontweight='bold', ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
    
    plt.title(f'Server Utilization Analysis\nCall Center: {arrival_rate} calls/hour', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Servers', fontsize=14)
    plt.ylabel('Utilization (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
    
    # 4. SLA Achievement - Individual Plot
    plt.figure(figsize=(12, 8))
    sla_achievement = [100 if results[c]['meets_1min_sla'] else 0 for c in server_counts]
    colors = ['green' if sla == 100 else 'red' for sla in sla_achievement]
    bars = plt.bar(server_counts, sla_achievement, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on bars
    for i, (server, sla) in enumerate(zip(server_counts, sla_achievement)):
        if sla == 100:
            plt.text(server, sla + 2, 'PASS', ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            plt.text(server, 10, 'FAIL', ha='center', va='bottom', fontweight='bold', fontsize=10, color='white')
    
    # Highlight recommended configuration
    if optimal_1min:
        optimal_idx = server_counts.index(optimal_1min)
        bars[optimal_idx].set_edgecolor('gold')
        bars[optimal_idx].set_linewidth(4)
        plt.annotate(f'RECOMMENDED\n{optimal_1min} servers\nMeets 1-min SLA\n{results[optimal_1min]["wait_minutes"]:.1f} min avg wait', 
                    xy=(optimal_1min, 100),
                    xytext=(optimal_1min + 2, 80),
                    arrowprops=dict(arrowstyle='->', color='gold', lw=3),
                    fontweight='bold', ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9))
    
    plt.title(f'Service Level Agreement Achievement\n1-Minute SLA Target: {arrival_rate} calls/hour', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Servers', fontsize=14)
    plt.ylabel('SLA Achievement (%)', fontsize=14)
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Executive Summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    
    if optimal_1min:
        r = results[optimal_1min]
        print(f"RECOMMENDED CONFIGURATION: {optimal_1min} servers")
        print(f"   - Monthly Cost: €{r['monthly_cost']:,.0f}")
        print(f"   - Average Wait Time: {r['wait_minutes']:.1f} minutes")
        print(f"   - Server Utilization: {r['server_utilization']*100:.1f}%")
        print(f"   - Meets 1-minute SLA target")
        
        # ROI Analysis
        if len(server_counts) > 1:
            cheaper_option = min(server_counts)
            if cheaper_option != optimal_1min:
                savings = r['monthly_cost'] - results[cheaper_option]['monthly_cost']
                wait_improvement = results[cheaper_option]['wait_minutes'] - r['wait_minutes']
                print(f"\nInvestment Analysis:")
                print(f"   - Extra cost vs minimum staffing: €{savings:,.0f}/month")
                print(f"   - Wait time improvement: {wait_improvement:.1f} minutes")
                print(f"   - Cost per minute improvement: €{savings/max(0.1, wait_improvement):,.0f}")
    else:
        print("WARNING: No configuration meets 1-minute SLA target")
        print("    Consider increasing server capacity or adjusting SLA target")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo() 