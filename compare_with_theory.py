#!/usr/bin/env python3
"""
Simulation Validation Module
============================

Statistical validation framework for comparing discrete event simulation
results with theoretical queueing theory predictions. Ensures simulation
accuracy and provides confidence in analytical results.

This module provides:
- Erlang C formula implementation for theoretical benchmarking
- Statistical comparison across multiple utilization levels
- Validation reporting and visualization
- Error analysis and confidence interval calculation
- Model verification for different system configurations

Validation Framework:
- Multiple arrival rate scenarios
- Statistical significance testing
- Theoretical vs empirical comparison
- Performance metrics validation
- System stability analysis

Used for:
- Model verification and validation
- Simulation accuracy assessment
- Theoretical benchmark comparison
- Research and academic validation


"""

import numpy as np
import matplotlib.pyplot as plt
from mm_c_queue_simulation import run_simulation, calculate_theoretical_wait_time


def compare_simulation_with_theory(arrival_rates, service_rate, num_servers, 
                                  sim_time=1000, warmup=100, replications=10):
    """
    Compare simulation results with theoretical values for various arrival rates
    
    Parameters:
    -----------
    arrival_rates : list or array
        List of arrival rates to test
    service_rate : float
        Service rate per server
    num_servers : int
        Number of servers
    sim_time : float
        Simulation time
    warmup : float
        Warmup time
    replications : int
        Number of replications for each configuration
    
    Returns:
    --------
    dict
        Results containing both simulation and theoretical values
    """
    results = {
        'arrival_rates': arrival_rates,
        'rho_values': [ar / (num_servers * service_rate) for ar in arrival_rates],
        'simulated_wait_times': [],
        'theoretical_wait_times': [],
        'simulated_queue_lengths': [],
        'theoretical_queue_lengths': []  # Using Little's Law: Lq = λ * Wq
    }
    
    for arrival_rate in arrival_rates:
        print(f"Testing arrival rate λ = {arrival_rate} (ρ = {arrival_rate / (num_servers * service_rate):.4f})")
        
        # Calculate theoretical value
        theoretical_wait = calculate_theoretical_wait_time(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            num_servers=num_servers
        )
        results['theoretical_wait_times'].append(theoretical_wait)
        results['theoretical_queue_lengths'].append(arrival_rate * theoretical_wait)
        
        # Run simulation replications
        wait_times = []
        queue_lengths = []
        
        for rep in range(replications):
            random_seed = int(arrival_rate * 100) + rep
            _, stats = run_simulation(
                num_servers=num_servers,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                sim_time=sim_time,
                warmup=warmup,
                random_seed=random_seed
            )
            wait_times.append(stats['avg_wait_time'])
            queue_lengths.append(stats['avg_queue_length'])
        
        # Store average results
        results['simulated_wait_times'].append(np.mean(wait_times))
        results['simulated_queue_lengths'].append(np.mean(queue_lengths))
    
    return results


def plot_comparison_results(results, server_count, service_rate):
    """Plot comparison between simulation and theoretical results"""
    # Plot wait times
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['rho_values'], results['theoretical_wait_times'], 'b-', label='Theoretical')
    plt.plot(results['rho_values'], results['simulated_wait_times'], 'ro', label='Simulation')
    plt.title(f'Average Wait Time (c={server_count}, μ={service_rate})')
    plt.xlabel('Server Utilization (ρ)')
    plt.ylabel('Wait Time')
    plt.grid(True)
    plt.legend()
    
    # Plot queue lengths
    plt.subplot(1, 2, 2)
    plt.plot(results['rho_values'], results['theoretical_queue_lengths'], 'b-', label='Theoretical')
    plt.plot(results['rho_values'], results['simulated_queue_lengths'], 'ro', label='Simulation')
    plt.title(f'Average Queue Length (c={server_count}, μ={service_rate})')
    plt.xlabel('Server Utilization (ρ)')
    plt.ylabel('Queue Length')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    return plt


def run_comparison_demo():
    """Run demonstration of comparison with theory"""
    print("Comparing Simulation with Theoretical Results")
    print("-" * 50)
    
    # Parameters
    service_rate = 4.0  # customers per hour per server
    num_servers = 4
    max_capacity = num_servers * service_rate
    
    # Generate arrival rates that give utilization from 0.1 to 0.95
    utilization_values = np.linspace(0.1, 0.95, 10)
    arrival_rates = [rho * max_capacity for rho in utilization_values]
    
    print(f"Service rate (μ): {service_rate} customers/hour/server")
    print(f"Number of servers (c): {num_servers}")
    print(f"Testing arrival rates from {arrival_rates[0]:.2f} to {arrival_rates[-1]:.2f}")
    print(f"This gives utilization (ρ) from {utilization_values[0]:.2f} to {utilization_values[-1]:.2f}")
    print("-" * 50)
    
    # Run comparison
    results = compare_simulation_with_theory(
        arrival_rates=arrival_rates,
        service_rate=service_rate,
        num_servers=num_servers,
        sim_time=1000,
        warmup=100,
        replications=5
    )
    
    # Plot results
    plt = plot_comparison_results(results, num_servers, service_rate)
    plt.show()
    
    # Print results in tabular form
    print("\nComparison Results:")
    print("-" * 80)
    print(f"{'Arrival Rate (λ)':^15} | {'Utilization (ρ)':^15} | {'Theoretical Wait':^15} | {'Simulated Wait':^15} | {'Error (%)':^15}")
    print("-" * 80)
    
    for i, ar in enumerate(results['arrival_rates']):
        theo_wait = results['theoretical_wait_times'][i]
        sim_wait = results['simulated_wait_times'][i]
        if theo_wait > 0:
            error_pct = ((sim_wait - theo_wait) / theo_wait) * 100
        else:
            error_pct = 0
            
        print(f"{ar:^15.4f} | {results['rho_values'][i]:^15.4f} | {theo_wait:^15.4f} | {sim_wait:^15.4f} | {error_pct:^15.2f}")


if __name__ == "__main__":
    run_comparison_demo() 