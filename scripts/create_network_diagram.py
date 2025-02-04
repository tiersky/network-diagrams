import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from plotly.subplots import make_subplots

# We'll store each node's "final lag" in a cache so we don't recalc multiple times.
lag_cache = {}

def get_final_lag(node, graph):
    """
    Returns the sum of 'Lag with Target' from 'node' all the way to 'Consideration'
    (i.e., if node -> child1 -> child2 -> ... -> Consideration).
    If multiple children exist, we pick the maximum path sum. 
    (Adjust to min() if you prefer a shortest path.)
    """
    # If we've computed this node's final lag before, return it
    if node in lag_cache:
        return lag_cache[node]
    
    # If no outgoing edges, check if this is 'Consideration'
    if graph.out_degree(node) == 0:
        # If it's the final node, final lag is 0
        if node == "Consideration":
            lag_cache[node] = 0
            return 0
        else:
            # If there's a sink that's not Consideration, define as 0 or special case
            lag_cache[node] = 0
            return 0

    # The node has outgoing edges, so we sum (edge lag + child's final lag),
    # and take the path that yields the maximum total.
    best_sum = 0  # or float('-inf') if you have larger negative or zero lags
    for child in graph.successors(node):
        edge_lag = graph[node][child]["lag"]   # single-hop lag
        child_sum = get_final_lag(child, graph)
        path_sum = edge_lag + child_sum
        if path_sum > best_sum:
            best_sum = path_sum

    # Memoize the result
    lag_cache[node] = best_sum
    return best_sum

def create_interactive_network(G, final_lags, node_positions, correlations):
    """
    Create an interactive network diagram using plotly with improved layout
    """
    # Calculate min and max correlations for scaling
    correlation_values = list(correlations.values())
    min_corr = min(correlation_values)
    max_corr = max(correlation_values)
    
    # Function to scale correlation to width
    def scale_width(correlation):
        return 1 + 9 * (correlation - min_corr) / (max_corr - min_corr)
    
    # Create edge traces with curved paths
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        
        # Calculate control points for curved path
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # Add some curvature based on the direction
        curve_height = 0.3 * (x1 - x0)
        
        # Create curved path using bezier curve
        path_x = [x0, mid_x, x1]
        path_y = [y0, mid_y + curve_height, y1]
        
        # Get correlation and width
        key = f"{edge[0]}|||{edge[1]}"
        corr = correlations.get(key, 0)
        width = scale_width(corr)
        
        edge_trace = go.Scatter(
            x=path_x,
            y=path_y,
            line=dict(
                width=width,
                color='#888',
                shape='spline'  # Makes the line curved
            ),
            hoverinfo='text',
            mode='lines',
            text=f'Lag: {edge[2]["lag"]}<br>Correlation: {corr:.2f}',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_trace = go.Scatter(
        x=[pos[0] for pos in node_positions.values()],
        y=[pos[1] for pos in node_positions.values()],
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition="middle center",  # Center text in circles
        textfont=dict(
            size=12,
            color='black'
        ),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=40,  # Increased size for better text fit
            color='white',
            line=dict(
                color=list(final_lags.values()),
                width=2
            ),
            colorbar=dict(
                title='Final Lag',
                thickness=15,
                orientation='h',  # Horizontal colorbar
                yanchor='bottom',
                y=-0.2,  # Position below the graph
                xanchor='center',
                x=0.5
            )
        )
    )

    # Create row labels
    all_lags = list(final_lags.values())
    min_lag = min(all_lags)
    max_lag = max(all_lags)
    interval_size = (max_lag - min_lag) / 6

    row_labels = []
    for i in range(6):
        lower = min_lag + (i * interval_size)
        upper = min_lag + ((i + 1) * interval_size)
        x_pos = i
        
        label_trace = go.Scatter(
            x=[x_pos],
            y=[max([y for x, y in node_positions.values()]) + 1],
            mode='text',
            text=[f'Lag: {lower:.1f}-{upper:.1f}'],
            textposition='top center',
            showlegend=False
        )
        row_labels.append(label_trace)

    # Create the figure with responsive layout
    fig = go.Figure(
        data=[*edge_traces, node_trace, *row_labels],
        layout=go.Layout(
            title='Network Diagram',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=100, l=20, r=20, t=40),  # Increased bottom margin for colorbar
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            autosize=True,  # Make it responsive
            height=800,  # Default height
            width=None  # Width will adjust automatically
        )
    )

    # Add responsive layout configuration
    fig.update_layout(
        # This ensures the layout responds to window size
        autosize=True,
        # Maintain aspect ratio while resizing
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )

    return fig

def export_to_html(fig, data, G):
    """Export the figure to HTML with the filter"""
    # Create the HTML file with the figure
    fig.write_html("network_diagram.html", 
                  include_plotlyjs=True,
                  config={'displayModeBar': True,
                         'displaylogo': False})

def calculate_cumulative_lags(G, df, final_lags):
    """
    Calculate cumulative lags using the Cumulative Lag column from the DataFrame
    """
    # Find the final target
    all_features = set(df['Feature'].unique())
    all_targets = set(df['Target'].unique())
    final_target = list(all_targets - all_features)[0]
    
    # Calculate cumulative lags directly from the DataFrame
    cumulative_lags = {}
    
    print("\nCalculating cumulative lags:")
    
    for node in G.nodes():
        if node != final_target:
            # Get the cumulative lag directly from the DataFrame
            node_data = df[df['Feature'] == node]
            if not node_data.empty:
                cum_lag = node_data['Cumulative Lag'].iloc[0]
                cumulative_lags[node] = cum_lag
                print(f"{node} = {cum_lag} months (Cumulative Lag)")
            else:
                cumulative_lags[node] = 0
                print(f"{node} = 0 months (No data found)")
    
    # Set final target lag to 0
    cumulative_lags[final_target] = 0
    
    print("\nFinal cumulative lags:", cumulative_lags)
    
    return cumulative_lags

def create_sankey_network(G, final_lags, correlations, df, final_target):
    # Create intervals
    max_lag = max(final_lags.values())
    interval_size = 5  # months per interval
    num_intervals = (max_lag // interval_size) + 1
    
    # Create initial intervals (in reverse order - larger intervals first)
    intervals = []
    for i in range(num_intervals - 1, -1, -1):  # Changed to count down
        start = i * interval_size
        end = (i + 1) * interval_size - 1
        if i == num_intervals - 1:
            end = max_lag
        intervals.append((num_intervals - 1 - i, f"{start}-{end} months", (start, end)))
    
    # Group nodes by intervals
    interval_groups = [[] for _ in range(len(intervals))]
    for node, lag in final_lags.items():
        for i, (_, _, (start, end)) in enumerate(intervals):
            if start <= lag <= end:
                interval_groups[i].append(node)
                break
    
    # Merge empty intervals with neighbors
    final_intervals = []
    final_groups = []
    i = 0
    while i < len(intervals):
        if len(interval_groups[i]) == 0:
            # Find nearest non-empty neighbor
            left_dist = float('inf')
            right_dist = float('inf')
            left_idx = i - 1
            right_idx = i + 1
            
            # Check left neighbor
            while left_idx >= 0:
                if len(interval_groups[left_idx]) > 0:
                    left_dist = i - left_idx
                    break
                left_idx -= 1
            
            # Check right neighbor
            while right_idx < len(intervals):
                if len(interval_groups[right_idx]) > 0:
                    right_dist = right_idx - i
                    break
                right_idx += 1
            
            # Merge with closest non-empty neighbor
            if left_dist <= right_dist and left_idx >= 0:
                # Merge with left neighbor
                start = intervals[left_idx][2][0]
                end = intervals[i][2][1]
                final_intervals[-1] = (len(final_intervals)-1, f"{start}-{end} months", (start, end))
                final_groups[-1].extend(interval_groups[i])
            elif right_idx < len(intervals):
                # Merge with right neighbor
                start = intervals[i][2][0]
                end = intervals[right_idx][2][1]
                final_intervals.append((len(final_intervals), f"{start}-{end} months", (start, end)))
                final_groups.append(interval_groups[i] + interval_groups[right_idx])
                i = right_idx  # Skip the merged interval
            i += 1
        else:
            final_intervals.append(intervals[i])
            final_groups.append(interval_groups[i])
            i += 1
    
    # Update intervals and groups
    intervals = final_intervals
    interval_groups = final_groups
    
    print("\nFinal intervals and their nodes:")
    for i, (_, label, _) in enumerate(intervals):
        print(f"\nInterval {label}:")
        print(interval_groups[i])
    
    # Calculate positions for better alignment
    num_sections = len(intervals)
    section_width = 1.0 / num_sections
    
    # Create node positions with improved spacing and alignment
    nodes = []
    node_x = []
    node_y = []
    node_indices = {}
    
    # Create node positions with improved spacing
    for i in range(len(intervals)):
        interval_nodes = sorted(interval_groups[i])
        num_nodes = len(interval_nodes)
        
        if num_nodes > 0:
            spacing = 0.8 / (num_nodes + 1)  # Vertical spacing between nodes
            section_center = (i * section_width) + (section_width / 2)  # Center of each section
            
            for j, node in enumerate(interval_nodes):
                nodes.append(node)
                node_indices[node] = len(nodes) - 1
                node_x.append(section_center)  # Place nodes at section center
                node_y.append((j + 1) * spacing)
    
    # Add final target node
    nodes.append(final_target)
    node_indices[final_target] = len(nodes) - 1
    node_x.append(1.0)  # Keep final target at the very end
    node_y.append(0.5)
    
    # Calculate interval boundaries for perfect alignment
    interval_boundaries = []
    for i in range(num_sections + 1):  # One more boundary than sections
        x_pos = i * section_width
        interval_boundaries.append(x_pos)
    
    # Create edges (links)
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # Add edges
    for edge in G.edges():
        source = edge[0]
        target = edge[1]
        key = f"{source}|||{target}"
        correlation = correlations.get(key, 0)
        
        sources.append(node_indices[source])
        targets.append(node_indices[target])
        values.append(abs(correlation) * 100)
        
        if source in ["Ad Awareness", "Awareness", "Attention"]:
            link_colors.append("rgba(245, 133, 218, 0.4)")  # Visibility color
        elif source in ["Buzz", "WOM Exposure"]:
            link_colors.append("rgba(255, 184, 78, 0.4)")  # Vibrancy color
        elif source in ["Quality", "Value", "Impression", "Reputation", "Recommend"]:
            link_colors.append("rgba(6, 184, 162, 0.4)")    # Variability color
        else:
            link_colors.append("rgba(180, 180, 180, 0.4)")    # YouGov color
    
    # Create node colors
    node_colors = []
    for n in nodes:
        if n in ["Ad Awareness", "Awareness"]:
            node_colors.append("rgba(245, 133, 218, 0.8)")  # Visibility
        elif n in ["Buzz", "WOM Exposure"]:
            node_colors.append("rgba(255, 184, 78, 0.8)")   # Vibrancy
        elif n in ["Quality", "Value", "Impression", "Reputation", "Recommend"]:
            node_colors.append("rgba(6, 184, 162, 0.8)")    # Variability
        else:
            node_colors.append("rgba(180, 180, 180, 0.8)")    # YouGov
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement = "freeform",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = nodes,
            color = node_colors,
            x = node_x,
            y = node_y
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = link_colors
        )
    )])
    
    # Add interval boundary lines
    for x_pos in interval_boundaries:
        fig.add_shape(
            type="line",
            x0=x_pos,
            x1=x_pos,
            y0=0,
            y1=1,
            line=dict(
                color="gray",
                width=1,
                dash="dash"
            ),
            layer="below"
        )
    
    # Add interval labels centered between boundaries
    for i, (_, label, _) in enumerate(intervals):
        fig.add_annotation(
            x=(i * section_width) + (section_width / 2),  # Center between boundaries
            y=1.05,  # Adjusted y position
            text=f"TIME<br>{label}",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=12, color="black", weight="bold"),
            align="center",
            yanchor='bottom'  # Added anchor point
        )
    
    # Add FINAL TARGET label on the right
    fig.add_annotation(
        x=1.1,  # Position slightly beyond the right edge
        y=1.05,  # Matched height with interval labels
        text="FINAL TARGET",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=12, color="black", weight="bold"),
        align="left",
        yanchor='bottom'  # Added anchor point
    )
    
    # Update layout
    fig.update_layout(
        font_size=12,
        autosize=True,
        height=900,
        width=1500,
        margin=dict(
            t=200,  # Increased top margin
            l=25,
            r=150,
            b=25
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_network_diagram(G, final_lags, correlations):
    # Find the final target
    def find_final_target(df):
        all_features = set(df['Feature'].unique())
        all_targets = set(df['Target'].unique())
        # The final target will be the one target that never appears as a feature
        final_target = list(all_targets - all_features)[0]
        return final_target

    # Calculate dynamic intervals
    def calculate_dynamic_intervals(final_lags, final_target):
        # Get all lag values except for the final target
        lag_values = [v for k, v in final_lags.items() if k != final_target]
        
        if not lag_values:
            return []
            
        # Calculate min and max lags
        min_lag = min(lag_values)
        max_lag = max(lag_values)
        
        # Calculate interval size (divide into 5 groups)
        interval_size = (max_lag - min_lag) / 5
        
        # Create intervals
        intervals = []
        for i in range(5):
            start = max_lag - (i + 1) * interval_size
            end = max_lag - i * interval_size
            # Round to nearest integer
            start = round(start)
            end = round(end)
            intervals.append((str(i), f"{start}-{end} months"))
        
        # Add final target interval
        intervals.append(("5", "Final Target"))
        
        return intervals

    # Find the final target
    final_target = find_final_target(df)  # df should be passed to the function
    
    # Get dynamic intervals
    intervals = calculate_dynamic_intervals(final_lags, final_target)
    
    # Rest of your existing code...
    
    # Update interval labels
    for i, (_, label) in enumerate(intervals):
        fig.add_annotation(
            x=i/5,
            y=1.05,
            text=f"TIME<br>{label}",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=12, color="black", weight="bold"),
            align="center"
        )

    return fig

def main():
    # Read CSV
    df = pd.read_csv("monthly_data_Full_Brand_Pattern_including_EK_AE.csv_relationships.csv")
    print("Loading data...")

    # Build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        feature = row["Feature"]
        target = row["Target"]
        lag = row["Lag with Target"]
        correlation = row["Correlation"]
        G.add_node(feature)
        G.add_node(target)
        G.add_edge(feature, target, lag=lag, correlation=correlation)

    # Calculate final lags
    final_lags = {}
    for node in G.nodes():
        try:
            paths = list(nx.all_simple_paths(G, node, "Consideration"))
            if paths:
                path = paths[0]
                total_lag = 0
                for i in range(len(path)-1):
                    if G.has_edge(path[i], path[i+1]):
                        total_lag += G[path[i]][path[i+1]]["lag"]
                final_lags[node] = total_lag
            else:
                final_lags[node] = 0
        except Exception as e:
            print(f"Warning: Could not calculate path for node {node}: {e}")
            final_lags[node] = 0

    # Add correlation data
    correlations = {}
    for _, row in df.iterrows():
        feature = row["Feature"]
        target = row["Target"]
        correlation = row["Correlation"]
        key = f"{feature}|||{target}"
        correlations[key] = correlation

    print(f"Processed {len(G.nodes())} nodes and {len(G.edges())} edges")
    print("Sample final lags:", dict(list(final_lags.items())[:3]))
    print("Sample correlations:", dict(list(correlations.items())[:3]))

    # Create Sankey diagram
    fig = create_sankey_network(G, final_lags, correlations, df, final_target)
    
    # Export to HTML
    network_data = {
        'nodes': list(G.nodes()), 
        'edges': [(u, v, str(d)) for u, v, d in G.edges(data=True)],
        'final_lags': final_lags,
        'correlations': correlations
    }
    export_to_html(fig, network_data, G)
    
    print("Created network_diagram.html")

if __name__ == "__main__":
    main()

