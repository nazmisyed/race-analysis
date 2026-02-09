import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re

# Set page configuration
st.set_page_config(page_title="Race Analysis Dashboard", layout="wide")

# Constants for colors
PODIUM_COLORS = {
    1: '#FFD700',  # Gold
    2: '#C0C0C0',  # Silver
    3: '#CD7F32'   # Bronze
}
DEFAULT_COLOR = '#1f77b4'  # Blue
HIGHLIGHT_COLOR = '#FF0000'  # Red for searched person

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    if pd.isna(seconds):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@st.cache_data
def load_race_data():
    """Load all processed CSV files from the Dataset folder"""
    dataset_path = Path('Dataset')
    data_dict = {}
    
    # Find all processed CSV files (avoiding double-processed files)
    csv_files = list(dataset_path.glob('*_processed.csv'))
    # Filter out double-processed files
    csv_files = [f for f in csv_files if not f.name.endswith('_processed_processed.csv')]
    
    for csv_file in csv_files:
        # Parse filename: Asia Triathlon Cup {year}_{date}_{category}_processed.csv
        filename = csv_file.stem  # Remove .csv
        filename = filename.replace('_processed', '')  # Remove _processed suffix
        
        # Extract components using regex
        match = re.match(r'(.+)_(\d{8})_(.+)', filename)
        if match:
            event_name = match.group(1)
            date_str = match.group(2)
            category = match.group(3)
            
            # Format date as YYYY-MM-DD
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            event_date = f"{event_name} ({formatted_date})"
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Store in dictionary
            if event_date not in data_dict:
                data_dict[event_date] = {}
            data_dict[event_date][category] = df
    
    return data_dict

def create_scatterplot(df, highlighted_name=None, stat_type='mean'):
    """Create scatter plot with swim vs run times"""
    fig = go.Figure()
    
    # Calculate statistics based on selection
    if stat_type == 'median':
        stat_swim = df['Swim_seconds'].median()
        stat_run = df['Run_seconds'].median()
        stat_label = 'Median'
    else:
        stat_swim = df['Swim_seconds'].mean()
        stat_run = df['Run_seconds'].mean()
        stat_label = 'Avg'
    
    # Prepare data for plotting
    for idx, row in df.iterrows():
        position = row['Pos']
        name = row['Name']
        
        # Determine color
        if highlighted_name and name.strip().lower() == highlighted_name.strip().lower():
            color = HIGHLIGHT_COLOR
            size = 15
            symbol = 'star'
        elif position in PODIUM_COLORS:
            color = PODIUM_COLORS[position]
            size = 12
            symbol = 'circle'
        else:
            color = DEFAULT_COLOR
            size = 8
            symbol = 'circle'
        
        # Create hover text
        hover_text = (
            f"<b>{name}</b><br>"
            f"Category: {row['Category']}<br>"
            f"Position: {position}<br>"
            f"Swim: {row['Swim']} ({row['Swim_seconds']}s)<br>"
            f"T1: {row['T1']} ({row['T1_seconds']}s)<br>"
            f"Run: {row['Run']} ({row['Run_seconds']}s)<br>"
            f"Total: {row['Time']}"
        )
        
        fig.add_trace(go.Scatter(
            x=[row['Swim_seconds']],
            y=[row['Run_seconds']],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                symbol=symbol,
                line=dict(width=2, color='white')
            ),
            name=f"{position}. {name}",
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    # Add statistics lines
    fig.add_hline(y=stat_run, line_dash="dot", line_color="red", 
                  annotation_text=f"{stat_label} Run: {stat_run:.0f}s ({seconds_to_time(stat_run)})", 
                  annotation_position="right")
    fig.add_vline(x=stat_swim, line_dash="dot", line_color="green", 
                  annotation_text=f"{stat_label} Swim: {stat_swim:.0f}s ({seconds_to_time(stat_swim)})", 
                  annotation_position="top")
    
    # Update layout
    fig.update_layout(
        title="Swim vs Run Performance",
        xaxis_title="Swim Time (seconds)",
        yaxis_title="Run Time (seconds)",
        hovermode='closest',
        height=600,
        template='plotly_white'
    )
    
    return fig

def main():
    st.title("🏊‍♂️🏃‍♂️ Race Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    try:
        data_dict = load_race_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    if not data_dict:
        st.warning("No race data found. Please ensure processed CSV files are in the Dataset folder.")
        return
    
    # Sidebar for selections
    st.sidebar.header("📋 Select Event & Categories")
    
    # Event date dropdown
    event_dates = sorted(data_dict.keys(), reverse=True)
    selected_event = st.sidebar.selectbox("Select Event Date", event_dates)
    
    # Category checkboxes
    if selected_event:
        categories = sorted(data_dict[selected_event].keys())
        st.sidebar.subheader("Select Categories")
        
        selected_categories = []
        for category in categories:
            if st.sidebar.checkbox(category, value=True, key=category):
                selected_categories.append(category)
        
        # Statistics selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Statistics Display")
        stat_type = st.sidebar.radio(
            "Show on graph:",
            options=['mean', 'median'],
            format_func=lambda x: x.capitalize()
        )
        
        # Name search
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Search Participant")
        search_name = st.sidebar.text_input("Enter name to highlight", "")
        
        # Combine data from selected categories
        if selected_categories:
            combined_df = pd.concat([
                data_dict[selected_event][cat].assign(Category=cat)
                for cat in selected_categories
            ], ignore_index=True)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Participants", len(combined_df))
            with col2:
                avg_swim = combined_df['Swim_seconds'].mean()
                st.metric("Avg Swim Time", f"{avg_swim:.0f}s ({seconds_to_time(avg_swim)})")
            with col3:
                avg_t1 = combined_df['T1_seconds'].mean()
                st.metric("Avg Transition Time", f"{avg_t1:.0f}s ({seconds_to_time(avg_t1)})")
            with col4:
                avg_run = combined_df['Run_seconds'].mean()
                st.metric("Avg Run Time", f"{avg_run:.0f}s ({seconds_to_time(avg_run)})")
            
            st.markdown("---")
            
            # Create two columns for table and plot
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("📊 Race Results")
                
                # Format the dataframe for display
                display_df = combined_df[[
                    'Pos', 'Bib No', 'Name', 'Country', 'Category',
                    'Swim', 'T1', 'Run', 'Time'
                ]].copy()
                
                # Sort by position
                display_df = display_df.sort_values('Pos').reset_index(drop=True)
                
                # Highlight searched name
                if search_name:
                    def highlight_row(row):
                        if search_name.strip().lower() in row['Name'].strip().lower():
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = display_df.style.apply(highlight_row, axis=1)
                    st.dataframe(styled_df, height=600, width='stretch')
                else:
                    st.dataframe(display_df, height=600, width='stretch')
            
            with col_right:
                st.subheader("📈 Swim vs Run Analysis")
                
                # Create and display scatter plot
                fig = create_scatterplot(combined_df, search_name if search_name else None, stat_type)
                st.plotly_chart(fig, width='stretch')
            
            # Legend
            st.markdown("---")
            st.markdown("### 🎨 Legend")
            legend_cols = st.columns(5)
            with legend_cols[0]:
                st.markdown("🥇 **1st Place** - Gold")
            with legend_cols[1]:
                st.markdown("🥈 **2nd Place** - Silver")
            with legend_cols[2]:
                st.markdown("🥉 **3rd Place** - Bronze")
            with legend_cols[3]:
                st.markdown("🔵 **Other** - Blue")
            with legend_cols[4]:
                st.markdown("⭐ **Highlighted** - Red Star")
        else:
            st.info("Please select at least one category to view results.")
    else:
        st.info("Please select an event date.")

if __name__ == "__main__":
    main()
