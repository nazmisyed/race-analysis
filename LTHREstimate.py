import streamlit as st
import pandas as pd
import numpy as np
from fitparse import FitFile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io


def parse_fit_file(fit_file):
    """
    Parse a .FIT file and extract all available data fields.
    
    Args:
        fit_file: Uploaded .FIT file
        
    Returns:
        pandas.DataFrame: DataFrame with all available data columns
    """
    try:
        # Create FitFile object
        fitfile = FitFile(fit_file)
        
        # Extract all available data
        all_data = []
        
        for record in fitfile.get_messages('record'):
            data_point = {}
            
            for field in record:
                # Common fields we want to capture
                if field.name in [
                    'timestamp', 'heart_rate', 'cadence', 'speed', 'distance',
                    'power', 'altitude', 'temperature', 'position_lat', 'position_long',
                    'enhanced_speed', 'enhanced_altitude', 'grade', 'calories',
                    'accumulated_power', 'left_right_balance', 'gps_accuracy',
                    'vertical_oscillation', 'stance_time_percent', 'stance_time',
                    'activity_type', 'left_torque_effectiveness', 'right_torque_effectiveness',
                    'left_pedal_smoothness', 'right_pedal_smoothness', 'combined_pedal_smoothness',
                    'time_from_course', 'cycle_length', 'total_cycles', 'compressed_speed_distance',
                    'resistance', 'time_in_hr_zone', 'time_in_speed_zone', 'time_in_cadence_zone',
                    'time_in_power_zone', 'repetition_num', 'min_heart_rate', 'max_heart_rate',
                    'avg_heart_rate', 'max_speed', 'avg_speed', 'total_calories', 'fat_calories',
                    'avg_cadence', 'max_cadence', 'avg_power', 'max_power', 'total_ascent',
                    'total_descent', 'training_stress_score', 'intensity_factor', 'normalized_power',
                    'left_right_balance_100', 'step_length', 'avg_vertical_oscillation',
                    'avg_stance_time_percent', 'avg_stance_time', 'fractional_cadence'
                ]:
                    data_point[field.name] = field.value
            
            # Only add records that have a timestamp (core requirement)
            if 'timestamp' in data_point:
                all_data.append(data_point)
        
        if not all_data:
            return None
            
        df = pd.DataFrame(all_data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Clean up GPS coordinates if they exist
        if 'position_lat' in df.columns:
            # Convert from semicircles to degrees
            df['position_lat'] = df['position_lat'] * (180 / 2**31)
        if 'position_long' in df.columns:
            df['position_long'] = df['position_long'] * (180 / 2**31)
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing FIT file: {str(e)}")
        return None


def calculate_lthr_from_last_20_minutes(df):
    """
    Calculate LTHR from the average heart rate of the last 20 minutes.
    
    Args:
        df: DataFrame with heart rate data
        
    Returns:
        tuple: (average_hr_last_20min, lthr, total_duration_minutes)
    """
    if df is None or len(df) == 0:
        return None, None, None
    
    # Check if heart rate data is available
    if 'heart_rate' not in df.columns or df['heart_rate'].isna().all():
        return None, None, None
    
    # Calculate total duration
    start_time = df['timestamp'].iloc[0]
    end_time = df['timestamp'].iloc[-1]
    total_duration = end_time - start_time
    total_duration_minutes = total_duration.total_seconds() / 60
    
    # Get last 20 minutes of data
    last_20_min_start = end_time - timedelta(minutes=20)
    last_20_min_data = df[df['timestamp'] >= last_20_min_start]
    
    if len(last_20_min_data) == 0:
        # If activity is less than 20 minutes, use all data
        last_20_min_data = df
    
    # Calculate average heart rate for the last 20 minutes (excluding NaN values)
    avg_hr_last_20min = last_20_min_data['heart_rate'].dropna().mean()
    
    if pd.isna(avg_hr_last_20min):
        return None, None, None
    
    # Calculate LTHR (95% of average HR from last 20 minutes)
    lthr = avg_hr_last_20min * 0.95
    
    return avg_hr_last_20min, lthr, total_duration_minutes


def calculate_heart_rate_zones(lthr):
    """
    Calculate heart rate zones based on LTHR.
    
    Args:
        lthr: Lactate Threshold Heart Rate
        
    Returns:
        dict: Dictionary with zone names and ranges
    """
    zones = {
        'Z1': {
            'name': 'Zone 1 (Active Recovery)',
            'range': (int(lthr * 0.62), int(lthr * 0.77)),
            'percentage': '62-77% of LTHR'
        },
        'Z2': {
            'name': 'Zone 2 (Aerobic Base)',
            'range': (int(lthr * 0.77), int(lthr * 0.86)),
            'percentage': '77-86% of LTHR'
        },
        'Z3': {
            'name': 'Zone 3 (Tempo)',
            'range': (int(lthr * 0.86), int(lthr * 0.91)),
            'percentage': '86-91% of LTHR'
        },
        'Z4': {
            'name': 'Zone 4 (Lactate Threshold)',
            'range': (int(lthr * 0.91), int(lthr * 0.95)),
            'percentage': '91-95% of LTHR'
        },
        'Z5': {
            'name': 'Zone 5 (VO2 Max)',
            'range': (int(lthr * 0.95), int(lthr * 1.03)),
            'percentage': '95-103% of LTHR'
        }
    }
    
    return zones


def create_heart_rate_plot(df, zones, lthr):
    """
    Create an interactive plot of heart rate data with zones.
    
    Args:
        df: DataFrame with heart rate data
        zones: Heart rate zones dictionary
        lthr: Lactate Threshold Heart Rate
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    # Add heart rate line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['heart_rate'],
        mode='lines',
        name='Heart Rate',
        line=dict(color='red', width=2)
    ))
    
    # Add LTHR line
    fig.add_hline(
        y=lthr,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"LTHR: {lthr:.0f} bpm"
    )
    
    # Color coding for zones
    zone_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
    
    for i, (zone_key, zone_info) in enumerate(zones.items()):
        fig.add_hrect(
            y0=zone_info['range'][0],
            y1=zone_info['range'][1],
            fillcolor=zone_colors[i],
            opacity=0.2,
            line_width=0,
            annotation_text=zone_key,
            annotation_position="top left"
        )
    
    fig.update_layout(
        title='Heart Rate Analysis with Training Zones',
        xaxis_title='Time',
        yaxis_title='Heart Rate (bpm)',
        hovermode='x unified',
        height=500
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="LTHR Estimator",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("❤️ LTHR Estimator & Heart Rate Zone Calculator")
    st.markdown("Upload your .FIT file to calculate your Lactate Threshold Heart Rate (LTHR) and training zones.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a .FIT file",
        type=['fit'],
        help="Upload a .FIT file from your training device (Garmin, Polar, etc.)"
    )
    
    if uploaded_file is not None:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Parse the file
            status_text.text("Parsing FIT file...")
            progress_bar.progress(25)
            
            df = parse_fit_file(uploaded_file)
            
            if df is not None and len(df) > 0:
                progress_bar.progress(50)
                status_text.text("Calculating LTHR...")
                
                # Calculate LTHR
                avg_hr_last_20min, lthr, total_duration_minutes = calculate_lthr_from_last_20_minutes(df)
                
                if lthr is not None:
                    progress_bar.progress(75)
                    status_text.text("Calculating heart rate zones...")
                    
                    # Calculate zones
                    zones = calculate_heart_rate_zones(lthr)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Analysis Results")
                        
                        # Activity summary
                        st.metric("Total Activity Duration", f"{total_duration_minutes:.1f} minutes")
                        st.metric("Average HR (Last 20 min)", f"{avg_hr_last_20min:.0f} bpm")
                        st.metric("LTHR (95% of avg)", f"{lthr:.0f} bpm")
                        
                        # Heart rate zones
                        st.subheader("🎯 Training Zones")
                        
                        zone_colors = ['#E3F2FD', '#E8F5E8', '#FFF9C4', '#FFE0B2', '#FFCDD2']
                        
                        for i, (zone_key, zone_info) in enumerate(zones.items()):
                            with st.container():
                                st.markdown(
                                    f"""
                                    <div style="background-color: {zone_colors[i]}; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #2196F3;">
                                        <strong>{zone_key}: {zone_info['name']}</strong><br>
                                        <span style="font-size: 18px; font-weight: bold;">{zone_info['range'][0]} - {zone_info['range'][1]} bpm</span><br>
                                        <small>{zone_info['percentage']}</small>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    
                    with col2:
                        st.subheader("📈 Heart Rate Chart")
                        
                        # Create and display the plot
                        fig = create_heart_rate_plot(df, zones, lthr)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data summary
                    st.subheader("📋 Data Summary")
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.metric("Min Heart Rate", f"{df['heart_rate'].min():.0f} bpm")
                    with col4:
                        st.metric("Max Heart Rate", f"{df['heart_rate'].max():.0f} bpm")
                    with col5:
                        st.metric("Average Heart Rate", f"{df['heart_rate'].mean():.0f} bpm")
                    
                    # Export data as downloadable CSV
                    st.subheader("💾 Export Data")
                    
                    # Show available data fields
                    available_fields = list(df.columns)
                    st.write(f"**Available data fields:** {len(available_fields)} fields detected")
                    
                    # Create field categories for better organization
                    field_categories = {
                        "Core Data": {
                            "fields": ["timestamp", "heart_rate", "speed", "distance", "cadence", "power"],
                            "description": "Essential training metrics"
                        },
                        "GPS & Location": {
                            "fields": ["position_lat", "position_long", "altitude", "enhanced_altitude", "grade"],
                            "description": "Location and elevation data"
                        },
                        "Environmental": {
                            "fields": ["temperature", "gps_accuracy"],
                            "description": "Environmental conditions"
                        },
                        "Power Metrics": {
                            "fields": ["accumulated_power", "normalized_power", "left_right_balance", "left_torque_effectiveness", "right_torque_effectiveness", "left_pedal_smoothness", "right_pedal_smoothness", "combined_pedal_smoothness"],
                            "description": "Advanced power meter data"
                        },
                        "Running Dynamics": {
                            "fields": ["vertical_oscillation", "stance_time_percent", "stance_time", "step_length", "avg_vertical_oscillation", "avg_stance_time_percent", "avg_stance_time"],
                            "description": "Running form metrics"
                        },
                        "Advanced Metrics": {
                            "fields": ["calories", "training_stress_score", "intensity_factor", "total_cycles", "cycle_length", "fractional_cadence"],
                            "description": "Calculated performance metrics"
                        }
                    }
                    
                    # Create tabs for different export options
                    tab1, tab2, tab3 = st.tabs(["📊 Training Zones", "📈 Custom Data Export", "🔍 Data Preview"])
                    
                    with tab1:
                        # Export zones
                        zones_df = pd.DataFrame([
                            {
                                'Zone': zone_key,
                                'Name': zone_info['name'],
                                'Min_HR': zone_info['range'][0],
                                'Max_HR': zone_info['range'][1],
                                'Percentage': zone_info['percentage']
                            }
                            for zone_key, zone_info in zones.items()
                        ])
                        
                        zones_csv = zones_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Training Zones",
                            data=zones_csv,
                            file_name=f"heart_rate_zones_lthr_{lthr:.0f}.csv",
                            mime="text/csv",
                            help="Download the calculated heart rate zones as CSV"
                        )
                        
                        st.write("**Training Zones Preview:**")
                        st.dataframe(zones_df, use_container_width=True)
                    
                    with tab2:
                        st.write("**Select data fields to include in export:**")
                        
                        # Quick selection buttons
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Select All", key="select_all"):
                                for field in available_fields:
                                    st.session_state[f"field_{field}"] = True
                        with col2:
                            if st.button("Select None", key="select_none"):
                                for field in available_fields:
                                    st.session_state[f"field_{field}"] = False
                        with col3:
                            if st.button("Select Core Only", key="select_core"):
                                for field in available_fields:
                                    st.session_state[f"field_{field}"] = field in field_categories["Core Data"]["fields"]
                        
                        # Initialize session state for checkboxes
                        if "export_initialized" not in st.session_state:
                            # Default to core data fields
                            for field in available_fields:
                                st.session_state[f"field_{field}"] = field in field_categories["Core Data"]["fields"]
                            st.session_state["export_initialized"] = True
                        
                        # Create checkboxes organized by category
                        selected_fields = []
                        
                        for category_name, category_info in field_categories.items():
                            # Check if any fields from this category are available
                            available_category_fields = [f for f in category_info["fields"] if f in available_fields]
                            
                            if available_category_fields:
                                st.write(f"**{category_name}** - {category_info['description']}")
                                
                                # Create columns for checkboxes (3 per row)
                                cols = st.columns(3)
                                for i, field in enumerate(available_category_fields):
                                    with cols[i % 3]:
                                        if st.checkbox(field.replace('_', ' ').title(), 
                                                     key=f"field_{field}",
                                                     value=st.session_state.get(f"field_{field}", False)):
                                            selected_fields.append(field)
                                
                                st.write("")  # Add spacing
                        
                        # Handle any fields not in categories
                        uncategorized_fields = [f for f in available_fields 
                                              if not any(f in cat["fields"] for cat in field_categories.values())]
                        
                        if uncategorized_fields:
                            st.write("**Other Available Fields**")
                            cols = st.columns(3)
                            for i, field in enumerate(uncategorized_fields):
                                with cols[i % 3]:
                                    if st.checkbox(field.replace('_', ' ').title(), 
                                                 key=f"field_{field}",
                                                 value=st.session_state.get(f"field_{field}", False)):
                                        selected_fields.append(field)
                        
                        # Collect all selected fields from session state
                        selected_fields = [field for field in available_fields 
                                         if st.session_state.get(f"field_{field}", False)]
                        
                        if selected_fields:
                            st.write(f"**Selected: {len(selected_fields)} fields**")
                            
                            # Prepare export dataframe
                            export_df = df[selected_fields].copy()
                            
                            # Add calculated fields if relevant fields are selected
                            if 'heart_rate' in selected_fields and lthr is not None:
                                def classify_zone(hr):
                                    if pd.isna(hr):
                                        return 'No Data'
                                    for zone_key, zone_info in zones.items():
                                        if zone_info['range'][0] <= hr <= zone_info['range'][1]:
                                            return zone_key
                                    return 'Above Z5' if hr > zones['Z5']['range'][1] else 'Below Z1'
                                
                                export_df['hr_zone'] = export_df['heart_rate'].apply(classify_zone)
                                export_df['lthr'] = lthr
                            
                            if 'timestamp' in selected_fields:
                                export_df['time_elapsed_seconds'] = (export_df['timestamp'] - export_df['timestamp'].iloc[0]).dt.total_seconds()
                                export_df['time_elapsed_minutes'] = export_df['time_elapsed_seconds'] / 60
                            
                            # Export button
                            custom_csv = export_df.to_csv(index=False)
                            st.download_button(
                                label=f"📈 Download Selected Data ({len(selected_fields)} fields)",
                                data=custom_csv,
                                file_name=f"custom_fit_data_{df['timestamp'].iloc[0].strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help=f"Download {len(selected_fields)} selected data fields"
                            )
                            
                            # Show preview of selected data
                            st.write("**Preview of selected data:**")
                            st.dataframe(export_df.head(10), use_container_width=True)
                            
                        else:
                            st.warning("Please select at least one field to export.")
                    
                    with tab3:
                        st.write("**Complete Data Overview:**")
                        
                        # Show data summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Data Fields", len(available_fields))
                        with col3:
                            duration_minutes = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 60
                            st.metric("Duration", f"{duration_minutes:.1f} min")
                        
                        # Show field availability
                        st.write("**Field Availability:**")
                        field_stats = []
                        for field in available_fields:
                            non_null_count = df[field].notna().sum()
                            percentage = (non_null_count / len(df)) * 100
                            field_stats.append({
                                'Field': field.replace('_', ' ').title(),
                                'Available Records': non_null_count,
                                'Coverage': f"{percentage:.1f}%",
                                'Data Type': str(df[field].dtype)
                            })
                        
                        field_stats_df = pd.DataFrame(field_stats)
                        st.dataframe(field_stats_df, use_container_width=True)
                        
                        # Show sample data
                        st.write("**Sample Data (first 10 records):**")
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                else:
                    st.error("Could not calculate LTHR. Please check your FIT file.")
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.error("No heart rate data found in the FIT file.")
                progress_bar.empty()
                status_text.empty()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a .FIT file to get started.")
        
        st.subheader("ℹ️ How it works:")
        st.markdown("""
        1. **Upload** your .FIT file from your training device
        2. **Analysis** extracts heart rate data from the last 20 minutes
        3. **LTHR Calculation** takes 95% of the average heart rate from those 20 minutes
        4. **Zone Calculation** creates 5 training zones based on your LTHR % based on Garmin formula:
           - **Zone 1** (62-77%): Active Recovery
           - **Zone 2** (77-86%): Easy
           - **Zone 3** (86-91%): Aerobic
           - **Zone 4** (91-95%): Lactate Threshold
           - **Zone 5** (95-103%): VO2 Max
        """)
        
        st.subheader("📝 Requirements:")
        st.markdown("""
        - .FIT file from your training device (Garmin, Polar, Suunto, etc.)
        - Activity should be at least 20 minutes for accurate LTHR calculation. Ideal is 5k all out race
        - Heart rate data must be recorded during the activity
        """)


if __name__ == "__main__":
    main()
