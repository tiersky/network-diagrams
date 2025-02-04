# generate_diagram.py
import pandas as pd
import networkx as nx
from create_network_diagram import create_sankey_network
import os

def process_csv_file(csv_file):
    """Process a single CSV file and return the figure"""
    try:
        # Read CSV
        df = pd.read_csv(f"data/{csv_file}")
        
        print(f"\nProcessing {csv_file}")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        
        # Create network graph
        G = nx.DiGraph()
        correlations = {}
        final_lags = {}
        
        # Find final target (node that appears only in Target column, never in Feature column)
        all_sources = set(df['Feature'].unique())
        all_targets = set(df['Target'].unique())
        final_targets = all_targets - all_sources
        
        if len(final_targets) > 0:
            final_target = list(final_targets)[0]
            print(f"Found final target: {final_target}")
        else:
            print(f"Warning: No unique final target found in {csv_file}, using last target")
            final_target = df['Target'].iloc[-1]
        
        # Process each row
        for _, row in df.iterrows():
            source = row['Feature']
            target = row['Target']
            lag = row['Lag with Target']
            correlation = row['Correlation']
            
            G.add_edge(source, target)
            key = f"{source}|||{target}"
            correlations[key] = correlation
            final_lags[source] = row['Cumulative Lag']
        
        # Create visualization
        print(f"  Creating visualization...")
        fig = create_sankey_network(G, final_lags, correlations, df, final_target)
        if fig is not None:
            print(f"  ✓ Successfully processed {csv_file}")
            return fig
            
    except Exception as e:
        print(f"  ✗ Error processing {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Get all CSV files in data directory
    data_dir = "data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the data directory!")
        return
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}\n")
    
    # Process each CSV file
    figures = {}
    total = len(csv_files)
    for i, csv_file in enumerate(csv_files, 1):
        print(f"Processing file {i}/{total}: {csv_file}")
        fig = process_csv_file(csv_file)
        if fig is not None:
            figures[csv_file] = fig
    
    if not figures:
        print("\nNo figures were successfully created!")
        return
    
    print(f"\nCreating combined HTML file...")
    create_combined_html(figures)
    print(f"✓ Created network_diagrams.html with {len(figures)} visualizations!")

def create_combined_html(figures):
    """Create HTML file with dropdown selector for multiple figures"""
    
    # Create dropdown HTML
    dropdown_html = """
    <div style="padding: 20px;">
        <label for="csvSelector" style="font-size: 16px; margin-right: 10px;">Select Dataset:</label>
        <select id="csvSelector" onchange="updateFigure()" style="padding: 5px; font-size: 14px;">
    """
    
    for csv_name in figures.keys():
        dropdown_html += f'<option value="{csv_name}">{csv_name}</option>'
    
    dropdown_html += "</select></div>"
    
    # Create div for figures
    figures_html = ""
    for csv_name, fig in figures.items():
        display = "none" if csv_name != list(figures.keys())[0] else "block"
        figures_html += f'<div id="{csv_name}" class="figure-div" style="display: {display}">'
        figures_html += fig.to_html(full_html=False, include_plotlyjs=False)
        figures_html += "</div>"
    
    # Create final HTML
    html_content = f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
            .figure-div {{ padding: 20px; }}
        </style>
        <script>
            function updateFigure() {{
                var selector = document.getElementById('csvSelector');
                var selected = selector.value;
                
                // Hide all figures
                var figures = document.getElementsByClassName('figure-div');
                for(var i = 0; i < figures.length; i++) {{
                    figures[i].style.display = 'none';
                }}
                
                // Show selected figure
                document.getElementById(selected).style.display = 'block';
            }}
        </script>
    </head>
    <body>
        {dropdown_html}
        {figures_html}
    </body>
    </html>
    """
    
    with open("network_diagrams.html", "w", encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
