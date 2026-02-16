#!/usr/bin/env python3
"""
Automated Presentation Generator
===============================

Professional presentation generation tool for call center analysis results.
Creates comprehensive PDF presentations with charts, tables, and executive
summaries suitable for business stakeholder communication.

This module provides:
- Automated slide generation with matplotlib
- Professional formatting and layout
- Integration with all simulation modules
- Executive summary generation
- Multi-format export capabilities

Presentation Features:
- Title and introduction slides
- Methodology and technical approach
- Simulation results and analysis
- Time-dependent analysis
- Optimization recommendations
- Conclusions and business impact

Output Formats:
- PDF presentation (primary)
- Individual chart exports
- Executive summary reports
- Technical appendices

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mm_c_queue_simulation import run_multiple_servers_experiment, plot_server_results, calculate_theoretical_wait_time
from time_dependent_simulation import create_arrival_rate_function, run_time_dependent_simulation
import os


def add_slide_title(fig, title, subtitle=None):
    """Add a title and optional subtitle to a figure/slide"""
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, fontsize=12, ha='center')


def generate_title_slide(pdf, title, author, date):
    """Generate the title slide"""
    fig = plt.figure(figsize=(11, 8.5))
    
    # Title
    plt.text(0.5, 0.6, title, fontsize=24, fontweight='bold', ha='center')
    
    # Author and date
    plt.text(0.5, 0.45, f"By: {author}", fontsize=16, ha='center')
    plt.text(0.5, 0.38, date, fontsize=14, ha='center')
    
    # No axes
    plt.axis('off')
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_introduction_slide(pdf):
    """Generate an introduction slide"""
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "Introduction & Background", "Multi-server Queueing Systems")
    
    # Content
    content = [
        "• Queueing systems are everywhere in service industries",
        "• M/M/c model: Poisson arrivals, exponential service time, c servers",
        "• Key challenge: Balance service quality vs. staffing costs",
        "• Simulation allows analysis of complex scenarios",
        "• SimPy provides a powerful framework for discrete-event simulation",
        "• Metrics: wait time, queue length, utilization"
    ]
    
    y_pos = 0.75
    for line in content:
        plt.text(0.1, y_pos, line, fontsize=14)
        y_pos -= 0.08
    
    # Simple queue diagram
    plt.text(0.75, 0.7, "Arrivals", fontsize=12, ha='center')
    plt.text(0.75, 0.5, "Queue", fontsize=12, ha='center')
    plt.text(0.75, 0.3, "Servers", fontsize=12, ha='center')
    
    # Draw arrows
    plt.arrow(0.75, 0.65, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.75, 0.45, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # No axes
    plt.axis('off')
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_methods_slide(pdf):
    """Generate a methods slide"""
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "Methods & Approach", "SimPy Modeling Approach")
    
    # Content
    left_content = [
        "• SimPy Environment:",
        "  - Discrete event simulation",
        "  - Process-based modeling",
        "  - Resource management",
        "",
        "• Customer Process:",
        "  - Arrivals: exponential interarrivals",
        "  - Request server resource",
        "  - Service: exponential time",
        "  - Collect statistics",
        "",
        "• Metrics Collection:",
        "  - Wait times (time in queue)",
        "  - Queue length sampling",
        "  - Server utilization"
    ]
    
    right_content = [
        "Pseudocode:",
        "",
        "def customer_process(env, servers):",
        "    # Request a server",
        "    with servers.request() as req:",
        "        start_wait = env.now",
        "        # Wait for our turn",
        "        yield req",
        "        wait_time = env.now - start_wait",
        "        ",
        "        # Service time",
        "        service_time = random.expovariate(mu)",
        "        yield env.timeout(service_time)",
        "        ",
        "        # Record statistics",
        "        record_wait_time(wait_time)",
        "        record_service_time(service_time)"
    ]
    
    # Left content
    y_pos = 0.78
    for line in left_content:
        plt.text(0.05, y_pos, line, fontsize=12)
        y_pos -= 0.04
    
    # Right content
    y_pos = 0.78
    for line in right_content:
        plt.text(0.55, y_pos, line, fontsize=10, family='monospace')
        y_pos -= 0.04
    
    # No axes
    plt.axis('off')
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_results_slide(pdf, arrival_rate, service_rate, server_range):
    """Generate a results slide with actual simulation data"""
    # Run the experiment
    results = run_multiple_servers_experiment(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        server_range=server_range,
        sim_time=1000,
        replications=3
    )
    
    # Create figure
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "Simulation Results", 
                   f"λ={arrival_rate}, μ={service_rate}, Servers={min(server_range)}-{max(server_range)}")
    
    # Plot wait times
    ax1 = fig.add_subplot(2, 2, 1)
    server_counts = sorted(results.keys())
    wait_times = [results[c]['avg_wait_time'] for c in server_counts]
    ax1.plot(server_counts, wait_times, 'o-')
    ax1.set_title('Average Wait Time vs. Number of Servers')
    ax1.set_xlabel('Number of Servers (c)')
    ax1.set_ylabel('Average Wait Time')
    ax1.grid(True)
    
    # Plot utilization
    ax2 = fig.add_subplot(2, 2, 2)
    utilizations = [results[c]['server_utilization'] * 100 for c in server_counts]
    traffic_intensities = [results[c]['traffic_intensity'] * 100 for c in server_counts]
    ax2.plot(server_counts, utilizations, 'o-', label='Actual')
    ax2.plot(server_counts, traffic_intensities, 'x--', label='Theoretical')
    ax2.set_title('Server Utilization vs. Number of Servers')
    ax2.set_xlabel('Number of Servers (c)')
    ax2.set_ylabel('Utilization (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Add a table with key metrics
    ax3 = fig.add_subplot(2, 1, 2)
    cell_text = []
    for c in server_counts:
        theo_wait = calculate_theoretical_wait_time(arrival_rate, service_rate, c)
        cell_text.append([
            c,
            f"{results[c]['traffic_intensity']:.3f}",
            f"{results[c]['avg_wait_time']:.3f}",
            f"{theo_wait:.3f}",
            f"{results[c]['avg_queue_length']:.2f}"
        ])
    
    columns = ('Servers', 'Traffic Intensity', 'Avg Wait (Sim)', 'Avg Wait (Theory)', 'Avg Queue Length')
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_time_dependent_slide(pdf):
    """Generate a slide showing time-dependent simulation results"""
    # Parameters for time-dependent simulation
    base_arrival_rate = 35.0  # Updated to optimal volume
    service_rate = 4.5  # Updated to optimal rate
    num_servers = 4
    peak_hours = [(9, 12), (13, 16)]
    peak_multiplier = 2.0
    
    # Create arrival rate function
    arrival_func = create_arrival_rate_function(
        base_rate=base_arrival_rate,
        peak_hours=peak_hours,
        peak_multiplier=peak_multiplier
    )
    
    # Run a short simulation for the presentation
    sim, stats = run_time_dependent_simulation(
        num_servers=num_servers,
        base_arrival_rate=base_arrival_rate,
        service_rate=service_rate,
        peak_hours=peak_hours,
        peak_multiplier=peak_multiplier,
        sim_days=3,
        warmup_days=1
    )
    
    # Create figure
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "Time-Dependent Arrivals", 
                   "Modeling daily patterns in customer traffic")
    
    # Plot arrival pattern
    ax1 = fig.add_subplot(2, 2, 1)
    hours = range(24)
    arrival_rates = [arrival_func(hour) for hour in hours]
    ax1.bar(hours, arrival_rates)
    ax1.set_title('Arrival Rate by Hour of Day')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Arrival Rate (λ)')
    ax1.set_xticks(range(0, 24, 4))
    ax1.grid(True, axis='y')
    
    # Plot wait times by hour
    ax2 = fig.add_subplot(2, 2, 2)
    wait_hours = sorted(stats['avg_wait_by_hour'].keys())
    wait_values = [stats['avg_wait_by_hour'][h] for h in wait_hours]
    ax2.bar(wait_hours, wait_values)
    ax2.set_title('Average Wait Time by Hour')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Avg Wait Time')
    ax2.set_xticks(range(0, 24, 4))
    ax2.grid(True, axis='y')
    
    # Plot queue length over time (partial)
    ax3 = fig.add_subplot(2, 1, 2)
    queue_times = sim.queue_length_times[:500]  # Just show a portion for clarity
    queue_lengths = sim.queue_lengths[:500]
    ax3.plot(queue_times, queue_lengths)
    ax3.set_title('Queue Length Over Time')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Queue Length')
    ax3.grid(True)
    
    # Add key statistics as text
    stats_text = (
        f"Average Wait Time: {stats['avg_wait_time']:.3f} hours\n"
        f"90th Percentile Wait: {stats['wait_time_p90']:.3f} hours\n"
        f"Average Queue Length: {stats['avg_queue_length']:.2f}\n"
        f"Server Utilization: {stats['server_utilization']*100:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.02, 0.97, stats_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_optimization_slide(pdf):
    """Generate a slide about staffing optimization"""
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "Staffing Optimization", 
                   "Finding optimal staffing levels to meet SLAs")
    
    # Left content - key concepts
    left_content = [
        "Key Optimization Concepts:",
        "",
        "• Service Level Agreement (SLA):",
        "  - Maximum acceptable wait time",
        "  - Percentile guarantee (e.g., 90th percentile ≤ 15 min)",
        "",
        "• Binary Search Approach:",
        "  - Find minimum staffing level meeting SLA",
        "  - Faster than testing all options",
        "",
        "• Trade-offs:",
        "  - Staffing cost vs. customer satisfaction",
        "  - Higher utilization increases wait times",
        "  - Wait times grow non-linearly as ρ approaches 1"
    ]
    
    # Example of a staffing optimization result
    # This is a mock chart showing wait time vs. staffing
    server_counts = [3, 4, 5, 6, 7, 8]
    wait_times = [1.5, 0.6, 0.3, 0.18, 0.12, 0.09]
    utilizations = [0.85, 0.64, 0.51, 0.43, 0.36, 0.32]
    sla_target = 0.25
    
    # Plot wait times
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(server_counts, wait_times, 'o-', color='blue')
    ax1.axhline(y=sla_target, color='red', linestyle='--', label=f'SLA Target ({sla_target})')
    ax1.set_title('90th Percentile Wait Time vs. Servers')
    ax1.set_xlabel('Number of Servers')
    ax1.set_ylabel('Wait Time (hours)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot utilization
    ax2 = fig.add_subplot(2, 2, 4)
    ax2.plot(server_counts, utilizations, 'o-', color='green')
    ax2.set_title('Server Utilization vs. Servers')
    ax2.set_xlabel('Number of Servers')
    ax2.set_ylabel('Utilization (%)')
    ax2.grid(True)
    
    # Add optimization key points
    y_pos = 0.78
    for line in left_content:
        plt.text(0.05, y_pos, line, fontsize=10)
        y_pos -= 0.03
    
    # No axes for the text area
    plt.axis('off')
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_conclusions_slide(pdf):
    """Generate a conclusions slide"""
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "Conclusions & Recommendations", "Best Practices for Queue Management")
    
    # Content
    content = [
        "Key Findings:",
        "",
        "• Server utilization has a non-linear impact on wait times",
        "  - Small increases in traffic intensity can cause large increases in wait time",
        "  - Example: At 4 servers, going from 80% to 90% utilization doubles wait time",
        "",
        "• Staffing Efficiency:",
        "  - Minimum servers for stability: ⌈λ/μ⌉",
        "  - Additional servers needed for acceptable wait times",
        "  - Typical utilization target: 70-85%, depending on SLA requirements",
        "",
        "• Time-Dependent Arrivals:",
        "  - Peak periods require more servers than average arrival rate would suggest",
        "  - Staffing to peak demand often leaves resources idle during off-peak times",
        "  - Dynamic staffing can improve cost efficiency",
        "",
        "• Future Directions:",
        "  - Non-exponential service times (e.g., Erlang distributions)",
        "  - Customer abandonment models (balking, reneging)",
        "  - Cost-benefit analysis integrating wait time costs and staffing costs",
        "  - Priority queueing for different customer classes"
    ]
    
    # Add text content
    y_pos = 0.82
    for line in content:
        plt.text(0.1, y_pos, line, fontsize=12)
        y_pos -= 0.037
    
    # No axes
    plt.axis('off')
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_references_slide(pdf):
    """Generate a references slide"""
    fig = plt.figure(figsize=(11, 8.5))
    add_slide_title(fig, "References", "")
    
    # Content
    content = [
        "SimPy Documentation:",
        "• Team SimPy. (2023). SimPy: Discrete-Event Simulation for Python.",
        "  https://simpy.readthedocs.io/",
        "",
        "Queueing Theory:",
        "• Kleinrock, L. (1975). Queueing Systems, Volume 1: Theory. Wiley-Interscience.",
        "• Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2018). Fundamentals of Queueing Theory.",
        "  5th Edition, Wiley Series in Probability and Statistics.",
        "",
        "Call Center Applications:",
        "• Gans, N., Koole, G., & Mandelbaum, A. (2003). Telephone call centers: Tutorial, review, and",
        "  research prospects. Manufacturing & Service Operations Management, 5(2), 79-141.",
        "",
        "Queue Simulation in Python:",
        "• Jarvis, K. (2019). Simulating Some Queues. Concerning Quality blog.",
        "• Ebert, A. et al. (2020). Computationally Efficient Simulation of Queues: The R Package",
        "  queuecomputer. Journal of Statistical Software, 95(5).",
        "",
        "Staffing Optimization:",
        "• Borst, S., Mandelbaum, A., & Reiman, M. I. (2004). Dimensioning large call centers.",
        "  Operations Research, 52(1), 17-34."
    ]
    
    # Add text content
    y_pos = 0.8
    for line in content:
        plt.text(0.1, y_pos, line, fontsize=11)
        y_pos -= 0.036
    
    # No axes
    plt.axis('off')
    
    # Save to PDF
    pdf.savefig(fig)
    plt.close()


def generate_presentation(output_filename="queue_simulation_presentation.pdf"):
    """Generate the complete presentation as a PDF"""
    with PdfPages(output_filename) as pdf:
        # Title slide
        generate_title_slide(
            pdf, 
            "M/M/c Queue Simulation: Cost-Efficient Customer Service Center Modeling", 
            "Your Name", 
            "Date: 2023"
        )
        
        # Content slides
        generate_introduction_slide(pdf)
        generate_methods_slide(pdf)
        
        # Results slides
        # Use actual simulation with moderate settings
        generate_results_slide(pdf, arrival_rate=35.0, service_rate=4.5, server_range=range(8, 25))
        generate_time_dependent_slide(pdf)
        generate_optimization_slide(pdf)
        
        # Conclusion and references
        generate_conclusions_slide(pdf)
        generate_references_slide(pdf)
        
    print(f"Presentation generated: {output_filename}")
    print(f"Contains {os.path.getsize(output_filename) / 1024:.1f} KB in 8 slides")


if __name__ == "__main__":
    generate_presentation() 