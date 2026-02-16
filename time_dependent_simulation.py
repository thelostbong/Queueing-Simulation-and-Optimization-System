#!/usr/bin/env python3
"""
Time-Dependent Call Center Simulation
=====================================

Advanced simulation module for modeling call centers with time-varying
arrival patterns. Extends the basic M/M/c model to handle realistic
daily, weekly, and seasonal demand fluctuations.

This module enables analysis of:
- Peak hour demand management
- Daily staffing schedule optimization
- Seasonal capacity planning
- Time-based performance metrics
- Dynamic resource allocation

Key Capabilities:
- Flexible arrival rate functions (hourly, daily, weekly patterns)
- Time-segmented performance analysis
- Arrival pattern visualization and analysis
- Integration with staffing optimization algorithms
- Statistical validation across time periods

Applications:
- Call center capacity planning
- Workforce scheduling optimization
- Service level management
- Demand forecasting validation


"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque


class TimeDependentCustomerServiceSimulation:
    """
    Multi-server (M/M/c) queue simulation with time-dependent arrival rates
    Simulates a customer service center with arrival rates that vary by time of day
    """
    
    def __init__(self, env, num_servers, arrival_rate_function, service_rate, 
                 simulation_time=1000, warmup_time=100, time_unit="hours"):
        """
        Initialize the simulation
        
        Parameters:
        -----------
        env : simpy.Environment
            The SimPy environment
        num_servers : int
            Number of service agents (c in M/M/c)
        arrival_rate_function : callable
            Function that returns arrival rate based on current time
        service_rate : float
            Average number of customers served per time unit per server (mu)
        simulation_time : float
            Total simulation time
        warmup_time : float
            Time to warm up the simulation before collecting statistics
        time_unit : str
            The unit of time (for reporting purposes)
        """
        self.env = env
        self.servers = simpy.Resource(env, capacity=num_servers)
        self.arrival_rate_function = arrival_rate_function
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
        self.arrivals_by_hour = {}
        self.completed_by_hour = {}
        self.wait_times_by_hour = {}
        
        # Start processes
        self.env.process(self.customer_arrivals())
        self.env.process(self.monitor_queue())
    
    def get_hour_bin(self, time):
        """Get the hour bin for a given time"""
        # Assuming time is in hours, get the integer hour or hour % 24 for day cycle
        return int(time) % 24
    
    def customer_arrivals(self):
        """Generate customer arrivals based on time-varying rates"""
        while True:
            # Get current arrival rate based on time
            current_time = self.env.now
            current_rate = self.arrival_rate_function(current_time)
            
            if current_rate <= 0:
                # No arrivals during this period
                yield self.env.timeout(0.1)  # Check again after a short time
                continue
                
            # Generate next arrival with exponential interarrival time
            interarrival_time = random.expovariate(current_rate)
            yield self.env.timeout(interarrival_time)
            
            # Create a new customer
            self.arrivals += 1
            hour_bin = self.get_hour_bin(self.env.now)
            self.arrivals_by_hour[hour_bin] = self.arrivals_by_hour.get(hour_bin, 0) + 1
            self.env.process(self.customer_service(f"Customer {self.arrivals}"))
    
    def customer_service(self, customer_id):
        """Process a customer through the service system"""
        arrival_time = self.env.now
        arrival_hour = self.get_hour_bin(arrival_time)
        
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
                
                # Track wait times by arrival hour
                if arrival_hour not in self.wait_times_by_hour:
                    self.wait_times_by_hour[arrival_hour] = []
                self.wait_times_by_hour[arrival_hour].append(wait_time)
                
                self.service_times.append(service_time)
                self.system_times.append(wait_time + service_time)
                
                # Track completions by hour
                completion_hour = self.get_hour_bin(self.env.now)
                self.completed_by_hour[completion_hour] = self.completed_by_hour.get(completion_hour, 0) + 1
    
    def monitor_queue(self):
        """Monitor queue length at regular intervals"""
        while True:
            if self.env.now > self.warmup_time:
                self.queue_lengths.append(len(self.servers.queue))
                self.queue_length_times.append(self.env.now)
            yield self.env.timeout(0.25)  # Sample 4 times per hour
    
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
            'total_arrivals': self.arrivals,
            'arrivals_by_hour': dict(sorted(self.arrivals_by_hour.items())),
            'completed_by_hour': dict(sorted(self.completed_by_hour.items())),
            'avg_wait_by_hour': {
                hour: np.mean(waits) for hour, waits in sorted(self.wait_times_by_hour.items())
            }
        }
        
        # Calculate percentiles for wait time
        if self.wait_times:
            percentiles = [50, 75, 90, 95, 99]
            for p in percentiles:
                stats[f'wait_time_p{p}'] = np.percentile(self.wait_times, p)
                
        # Primary targets (recommended)
        stats['meets_30s'] = stats['avg_wait_time'] <= 0.5   # Premium service
        stats['meets_1min'] = stats['avg_wait_time'] <= 1.0  # Standard target
        stats['meets_2min'] = stats['avg_wait_time'] <= 2.0  # Acceptable threshold
        
        return stats
    
    def plot_queue_length(self):
        """Plot queue length over time"""
        if not self.queue_lengths:
            print("No queue length data collected.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.queue_length_times, self.queue_lengths)
        plt.title('Queue Length Over Time')
        plt.xlabel(f'Time ({self.time_unit})')
        plt.ylabel('Number of Customers in Queue')
        plt.grid(True)
        return plt
    
    def plot_arrivals_by_hour(self):
        """Plot arrivals by hour"""
        if not self.arrivals_by_hour:
            print("No arrival data collected.")
            return
        
        hours = range(24)
        arrivals = [self.arrivals_by_hour.get(hour, 0) for hour in hours]
        
        plt.figure(figsize=(12, 6))
        plt.bar(hours, arrivals, alpha=0.7)
        plt.title('Customer Arrivals by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Arrivals')
        plt.xticks(hours)
        plt.grid(True, axis='y')
        return plt
    
    def plot_wait_times_by_hour(self):
        """Plot average wait times by hour"""
        if not self.wait_times_by_hour:
            print("No wait time data collected.")
            return
        
        hours = sorted(self.wait_times_by_hour.keys())
        avg_waits = [np.mean(self.wait_times_by_hour[hour]) for hour in hours]
        
        plt.figure(figsize=(12, 6))
        plt.bar(hours, avg_waits, alpha=0.7)
        plt.title('Average Wait Time by Arrival Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel(f'Average Wait Time ({self.time_unit})')
        plt.xticks(range(24))
        plt.grid(True, axis='y')
        return plt


def create_arrival_rate_function(base_rate, peak_hours, peak_multiplier):
    """
    Create a function that returns arrival rate based on time of day
    
    Parameters:
    -----------
    base_rate : float
        Base arrival rate (lambda) during regular hours
    peak_hours : list of tuples
        List of (start_hour, end_hour) representing peak periods
    peak_multiplier : float
        Multiplier for arrival rate during peak hours
    
    Returns:
    --------
    callable
        Function that takes time and returns current arrival rate
    """
    def arrival_rate(time):
        hour = time % 24  # Get hour of day (0-23)
        
        # Check if current hour is in a peak period
        for start_hour, end_hour in peak_hours:
            if start_hour <= hour < end_hour:
                return base_rate * peak_multiplier
        
        # Check if it's night time (very low traffic)
        if 0 <= hour < 6:  # Midnight to 6am
            return base_rate * 0.2
            
        # Regular hours
        return base_rate
    
    return arrival_rate


def run_time_dependent_simulation(num_servers, base_arrival_rate, service_rate, 
                                 peak_hours, peak_multiplier,
                                 sim_days=7, warmup_days=1):
    """
    Run a simulation with time-dependent arrival rates
    
    Parameters:
    -----------
    num_servers : int
        Number of servers
    base_arrival_rate : float
        Base arrival rate (customers per hour)
    service_rate : float
        Service rate per server (customers per hour)
    peak_hours : list of tuples
        List of (start_hour, end_hour) representing peak periods
    peak_multiplier : float
        Multiplier for arrival rate during peak hours
    sim_days : int
        Simulation duration in days
    warmup_days : int
        Warmup period in days
    
    Returns:
    --------
    simulation, stats
    """
    # Create arrival rate function
    arrival_func = create_arrival_rate_function(
        base_rate=base_arrival_rate,
        peak_hours=peak_hours,
        peak_multiplier=peak_multiplier
    )
    
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create simulation
    sim = TimeDependentCustomerServiceSimulation(
        env=env,
        num_servers=num_servers,
        arrival_rate_function=arrival_func,
        service_rate=service_rate,
        simulation_time=2000,  # ~83 days of operation
        warmup_time=warmup_days * 24,   # Convert days to hours
        time_unit="hours"
    )
    
    # Run simulation
    start_time = time.time()
    env.run(until=sim_days * 24)
    runtime = time.time() - start_time
    
    # Get statistics
    stats = sim.get_statistics()
    stats['simulation_runtime_seconds'] = runtime
    
    return sim, stats


def demo_time_dependent():
    """Run demonstration of time-dependent simulation"""
    print("Starting Time-Dependent Customer Service Queue Simulation")
    print("-" * 60)
    
    # System parameters
    arrival_rate = 35.0  # Sweet spot for medium-volume business
    service_rate = 4.5  # Efficient but realistic service time
    num_servers = 4
    
    # Define peak hours (e.g., morning rush and afternoon rush)
    peak_hours = [(9, 12), (13, 16)]  # 9am-12pm and 1pm-4pm
    peak_multiplier = 2.0             # Twice as many arrivals during peak hours
    
    print(f"Parameters:")
    print(f"  Base arrival rate (λ): {arrival_rate} customers/hour")
    print(f"  Service rate (μ): {service_rate} customers/hour/server")
    print(f"  Number of servers (c): {num_servers}")
    print(f"  Peak hours: {peak_hours}")
    print(f"  Peak multiplier: {peak_multiplier}x")
    print(f"  Simulation duration: 7 days (with 1 day warmup)")
    print("-" * 60)
    
    # Run simulation
    sim, stats = run_time_dependent_simulation(
        num_servers=num_servers,
        base_arrival_rate=arrival_rate,
        service_rate=service_rate,
        peak_hours=peak_hours,
        peak_multiplier=peak_multiplier,
        sim_days=7,
        warmup_days=1
    )
    
    # Display basic results
    print("\nResults Summary:")
    print("-" * 60)
    print(f"  Average wait time: {stats['avg_wait_time']:.4f} hours")
    print(f"  90th percentile wait time: {stats['wait_time_p90']:.4f} hours")
    print(f"  Maximum wait time: {stats['max_wait_time']:.4f} hours")
    print(f"  Average queue length: {stats['avg_queue_length']:.4f} customers")
    print(f"  Maximum queue length: {stats['max_queue_length']} customers")
    print(f"  Server utilization: {stats['server_utilization'] * 100:.2f}%")
    print(f"  Total customers served: {stats['customers_served']}")
    print(f"  Simulation runtime: {stats['simulation_runtime_seconds']:.2f} seconds")
    
    # Plot results
    plt1 = sim.plot_queue_length()
    plt2 = sim.plot_arrivals_by_hour()
    plt3 = sim.plot_wait_times_by_hour()
    
    # Show all plots
    plt.tight_layout()
    plt.show()
    
    # Display hourly statistics
    print("\nHourly Statistics:")
    print("-" * 60)
    print(f"Hour | Arrivals | Avg Wait Time")
    print("-" * 60)
    
    for hour in range(24):
        arrivals = stats['arrivals_by_hour'].get(hour, 0)
        avg_wait = stats['avg_wait_by_hour'].get(hour, 0)
        print(f"{hour:4} | {arrivals:8} | {avg_wait:12.4f}")


if __name__ == "__main__":
    demo_time_dependent() 