"""
Combined growth chart component: Height and Velocity charts as subplots with synchronized hovering.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Set, Optional
from src.config import PLOT_HEIGHT, PLOT_COLORS, CHART_CONFIG
from src.data.growth_standards import get_height_zscore_bounds, get_velocity_zscore_bounds


def render_combined_charts(
    patient_data: pd.DataFrame,
    sex: str,
    error_indices: Set[int] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    """
    Render combined height and velocity charts as subplots with synchronized hovering.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data with calculated fields
    sex : str
        Patient sex ('M' or 'F') for growth standards
    error_indices : Set[int], optional
        Set of indices marked as errors
    selected_index : int, optional
        Currently selected point index
        
    Returns
    -------
    go.Figure
        Plotly figure object with subplots
    """
    if error_indices is None:
        error_indices = set()
    
    # Create figure with grid layout for synchronized hovering
    fig = go.Figure()
    
    # ============ HEIGHT CHART (Row 1) ============
    
    # Prepare height data
    ages_days = patient_data['age_in_days'].values
    ages_years = patient_data['age_years'].values
    heights = patient_data['height_in'].values
    height_zscores = patient_data['height_zscore'].values

    # Separate valid and missing height measurements so visits without height stay visible
    height_valid_mask = patient_data['height_in'].notna().to_numpy()
    height_null_mask = ~height_valid_mask
    valid_height_positions = np.flatnonzero(height_valid_mask)
    null_height_positions = np.flatnonzero(height_null_mask)
    visit_order_map = {int(pos): order for order, pos in enumerate(valid_height_positions, start=1)}

    def _format_visit_label(index_value: int) -> str:
        """Return a visit label that skips missing height measurements."""
        label = visit_order_map.get(int(index_value))
        return f"Visit {label}" if label is not None else "Visit"
    
    # Get height growth standard bounds
    age_range = np.linspace(ages_days.min(), ages_days.max(), 100)
    height_bounds = get_height_zscore_bounds(age_range, sex)
    
    # Add height ribbon (±5 z-score bounds) - Green boundaries
    fig.add_trace(go.Scatter(
        x=height_bounds['age_days'],
        y=height_bounds['upper_bound'],
        mode='lines',
        name='+5z (WHO/CDC)',
        line=dict(color='green', width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip',
        legendgroup='height',
        xaxis='x',
        yaxis='y',
    ))
    
    fig.add_trace(go.Scatter(
        x=height_bounds['age_days'],
        y=height_bounds['lower_bound'],
        mode='lines',
        name='-5z (WHO/CDC)',
        line=dict(color='green', width=2, dash='dot'),
        fill='tonexty',
        fillcolor=PLOT_COLORS['ribbon_fill'],
        showlegend=False,
        hoverinfo='skip',
        legendgroup='height',
        xaxis='x',
        yaxis='y',
    ))
    
    # Add height median line - Green dashed
    fig.add_trace(go.Scatter(
        x=height_bounds['age_days'],
        y=height_bounds['median'],
        mode='lines',
        name='Growth Standard Median (±5z)',
        line=dict(color='green', width=2, dash='dash'),
        showlegend=True,
        hoverinfo='skip',
        legendgroup='height',
        xaxis='x',
        yaxis='y',
    ))
    
    # Prepare masks for error and normal points
    error_mask = np.array([i in error_indices for i in range(len(patient_data))])
    valid_error_mask = np.array([int(pos) in error_indices for pos in valid_height_positions]) if valid_height_positions.size > 0 else np.array([])
    
    # Get absolute and percent changes for height chart
    abs_changes_height = patient_data['absolute_change'].values
    pct_changes_height = patient_data['percent_change'].values
    
    # Add ALL height measurements with connected line - Blue line with markers
    # This ensures the line stays connected even when points are marked as errors
    if valid_height_positions.size > 0:
        hover_text_all = []
        for pos in valid_height_positions:
            visit_label = visit_order_map.get(int(pos))
            visit_text = f"Visit {visit_label}" if visit_label is not None else "Visit"
            header_prefix = "♦️ ERROR - " if int(pos) in error_indices else ""
            text_parts = [
                f"<b>{header_prefix}{visit_text}</b>",
                f"Age: {ages_days[pos]:.0f} days",
                f"Height: {heights[pos]:.2f} inches"
            ]
            if not np.isnan(abs_changes_height[pos]):
                text_parts.append(
                    f"Height change: {abs_changes_height[pos]:+.2f} in ({pct_changes_height[pos]:+.1f}%)"
                )
            if not np.isnan(height_zscores[pos]):
                text_parts.append(f"Height-for-age z-score: {height_zscores[pos]:.2f}")
            hover_text_all.append("<br>".join(text_parts))

        fig.add_trace(go.Scatter(
            x=ages_days[valid_height_positions],
            y=heights[valid_height_positions],
            mode='lines+markers',
            name='Height',
            line=dict(color='blue', width=3),
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(color='blue', width=1)
            ),
            customdata=valid_height_positions.astype(int),
            hovertext=hover_text_all,
            hoverinfo='text',
            showlegend=True,
            legendgroup='height',
            xaxis='x',
            yaxis='y',
        ))
    
    # Add error height markers as overlay - Red diamond markers with z-score labels
    # This is a separate toggleable trace that can be hidden/shown independently
    if valid_height_positions.size > 0 and valid_error_mask.any():
        error_positions = valid_height_positions[valid_error_mask]
        fig.add_trace(go.Scatter(
            x=ages_days[error_positions],
            y=heights[error_positions],
            mode='markers+text',
            name='Error Height Markers',
            marker=dict(
                size=11,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            text=[
                f"z={height_zscores[pos]:.2f}" if not np.isnan(height_zscores[pos]) else ""
                for pos in error_positions
            ],
            textposition='top center',
            customdata=error_positions.astype(int),
            hoverinfo='skip',
            showlegend=True,
            legendgroup='height_errors',
            xaxis='x',
            yaxis='y',
        ))

    null_marker_y = None
    if null_height_positions.size > 0:
        candidate_arrays = []
        if valid_height_positions.size > 0:
            candidate_arrays.append(heights[valid_height_positions])
        for key in ('lower_bound', 'median'):
            bounds_array = np.asarray(height_bounds.get(key, np.array([])))
            if bounds_array.size > 0:
                candidate_arrays.append(bounds_array)
        candidate_values = None
        if candidate_arrays:
            candidate_values = np.concatenate(candidate_arrays)
            candidate_values = candidate_values[~np.isnan(candidate_values)]
        if candidate_values is None or candidate_values.size == 0:
            null_marker_y = 0.0
        else:
            null_marker_y = float(np.nanmin(candidate_values))
        fig.add_trace(go.Scatter(
            x=ages_days[null_height_positions],
            y=[null_marker_y] * null_height_positions.size,
            mode='markers',
            name='Missing Height',
            marker=dict(size=8, color='gray', symbol='x', line=dict(width=2)),
            hovertemplate='<b>Missing Height</b><br>Age: %{x:.0f} days<extra></extra>',
            showlegend=True,
            legendgroup='height_missing',
            xaxis='x',
            yaxis='y',
        ))
    
    # Add selected height point highlight (visual overlay only, no tooltip)
    if (selected_index is not None and 0 <= selected_index < len(patient_data)
            and height_valid_mask[selected_index]):
        selected_age_d = ages_days[selected_index]
        selected_height = heights[selected_index]
        
        fig.add_trace(go.Scatter(
            x=[selected_age_d],
            y=[selected_height],
            mode='markers',
            name='Selected',
            marker=dict(
                size=14,
                color=PLOT_COLORS['selected'],
                symbol='circle',
                line=dict(color='white', width=3)
            ),
            hoverinfo='skip',  # Skip hover to avoid duplicate tooltip
            showlegend=False,
            legendgroup='height',
            xaxis='x',
            yaxis='y',
        ))
    
    # ============ VELOCITY CHART (Row 2) ============
    
    # Filter out first point (no velocity)
    velocity_data = patient_data[patient_data['velocity'].notna()].copy()
    
    if not velocity_data.empty:
        # Prepare velocity data
        vel_ages_days = velocity_data['age_in_days'].values
        vel_ages_years = velocity_data['age_years'].values
        velocities = velocity_data['velocity'].values
        velocity_zscores = velocity_data['velocity_zscore'].values
        abs_changes = velocity_data['absolute_change'].values
        pct_changes = velocity_data['percent_change'].values
        original_indices = velocity_data.index.values
        
        # Also prepare adjacent velocity data (if available)
        adjacent_velocities = velocity_data['height_adjacent_velocity'].values if 'height_adjacent_velocity' in velocity_data.columns else None
        
        # Get velocity growth standard bounds
        vel_age_range = np.linspace(vel_ages_days.min(), vel_ages_days.max(), 100)
        velocity_bounds = get_velocity_zscore_bounds(vel_age_range, sex)
        
        # Add velocity ribbon - Green boundaries
        fig.add_trace(go.Scatter(
            x=velocity_bounds['age_days'],
            y=velocity_bounds['upper_bound'],
            mode='lines',
            name='+3 SD (WHO/CDC)',
            line=dict(color='green', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
            legendgroup='velocity',
            xaxis='x',
            yaxis='y2',
        ))
        
        fig.add_trace(go.Scatter(
            x=velocity_bounds['age_days'],
            y=velocity_bounds['lower_bound'],
            mode='lines',
            name='-3 SD (WHO/CDC)',
            line=dict(color='green', width=2, dash='dot'),
            fill='tonexty',
            fillcolor=PLOT_COLORS['ribbon_fill'],
            showlegend=False,
            hoverinfo='skip',
            legendgroup='velocity',
            xaxis='x',
            yaxis='y2',
        ))
        
        # Add velocity median line - Green dashed
        fig.add_trace(go.Scatter(
            x=velocity_bounds['age_days'],
            y=velocity_bounds['median'],
            mode='lines',
            name='Median',
            line=dict(color='green', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
            legendgroup='velocity',
            xaxis='x',
            yaxis='y2',
        ))
        
        # Prepare masks for velocity errors
        vel_error_mask = np.array([idx in error_indices
                                     for idx in original_indices])
        
        # Add ALL velocity measurements with connected line
        # This ensures the line stays connected even when points are marked as errors
        vel_hover_text_all = []
        for idx, age_yr, age_d, vel, z in zip(
            original_indices,
            vel_ages_years,
            vel_ages_days,
            velocities,
            velocity_zscores
        ):
            visit_text = _format_visit_label(idx)
            prefix = "♦️ ERROR - " if idx in error_indices else ""
            text_parts = [
                f"<b>{prefix}{visit_text}</b>",
                f"Age: {age_d:.0f} days",
                f"Min-interval Adjusted Velocity: {vel:.6f} inches/day",
                f"Min-interval Adjusted Velocity z-score: {z:.2f}"
            ]
            vel_hover_text_all.append("<br>".join(text_parts))
        
        fig.add_trace(go.Scatter(
            x=vel_ages_days,
            y=velocities,
            mode='lines+markers',
            name='Min-interval Adjusted Velocity',
            line=dict(color='orange', width=3),
            marker=dict(
                size=6,
                color='lemonchiffon',
                line=dict(color='orange', width=1)
            ),
            customdata=original_indices,
            hovertext=vel_hover_text_all,
            hoverinfo='text',
            showlegend=True,
            legendgroup='velocity_adjusted',
            xaxis='x',
            yaxis='y2',
        ))
        
        # Add adjacent velocity line (consecutive points)
        if adjacent_velocities is not None and len(adjacent_velocities) > 0:
            # Calculate z-scores for adjacent velocities
            from src.data.growth_standards import calculate_velocity_for_age_zscore
            adj_vel_zscores = [
                calculate_velocity_for_age_zscore(adj_vel, age_d, sex)
                if not np.isnan(adj_vel) else np.nan
                for age_d, adj_vel in zip(
                    vel_ages_days,
                    adjacent_velocities
                )
            ]
            
            adj_vel_hover_text = []
            for idx, age_d, adj_vel, adj_z in zip(
                original_indices,
                vel_ages_days,
                adjacent_velocities,
                adj_vel_zscores
            ):
                visit_text = _format_visit_label(idx)
                prefix = "♦️ ERROR - " if idx in error_indices else ""
                text_parts = [
                    f"<b>{prefix}{visit_text}</b>",
                    f"Age: {age_d:.0f} days",
                    f"Adjacent Velocity: {adj_vel:.6f} inches/day",
                    f"Adjacent Velocity z-score: {adj_z:.2f}"
                ]
                adj_vel_hover_text.append("<br>".join(text_parts))
            
            fig.add_trace(go.Scatter(
                x=vel_ages_days,
                y=adjacent_velocities,
                mode='lines+markers',
                name='Adjacent Velocity',
                line=dict(color='mediumpurple', width=3),
                marker=dict(
                    size=6,
                    color='lavender',
                    line=dict(color='mediumpurple', width=1)
                ),
                customdata=original_indices,
                hovertext=adj_vel_hover_text,
                hoverinfo='text',
                showlegend=True,
                legendgroup='velocity_adjacent',
                xaxis='x',
                yaxis='y2',
            ))
        
        # Add error velocity markers as overlay - Red diamond markers with z-score labels
        # This is a separate toggleable trace that can be hidden/shown independently
        if vel_error_mask.any():
            # Error markers for Min-interval Adjusted Velocity (orange line)
            vel_hover_text_errors = []
            for idx, age_yr, age_d, vel, z, adj_vel in zip(
                original_indices[vel_error_mask],
                vel_ages_years[vel_error_mask],
                vel_ages_days[vel_error_mask],
                velocities[vel_error_mask],
                velocity_zscores[vel_error_mask],
                adjacent_velocities[vel_error_mask] if adjacent_velocities is not None else [np.nan] * vel_error_mask.sum()
            ):
                visit_text = _format_visit_label(idx)
                text_parts = [
                    f"<b>♦️ ERROR - {visit_text}</b>",
                    f"Age: {age_d:.0f} days",
                    f"Min-interval Adjusted Velocity: {vel:.6f} inches/day",
                    f"Min-interval Adjusted Velocity z-score: {z:.2f}"
                ]
                if adjacent_velocities is not None and not np.isnan(adj_vel):
                    text_parts.append(f"Adjacent Velocity: {adj_vel:.6f} inches/day")
                vel_hover_text_errors.append("<br>".join(text_parts))
            
            fig.add_trace(go.Scatter(
                x=vel_ages_days[vel_error_mask],
                y=velocities[vel_error_mask],
                mode='markers+text',
                name='Error Min-interval Markers',
                marker=dict(
                    size=11,
                    color='red',
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                text=[f"z={z:.2f}" for z in velocity_zscores[vel_error_mask]],
                textposition='top center',
                customdata=original_indices[vel_error_mask],
                hoverinfo='skip',
                showlegend=False,
                legendgroup='velocity_adjusted',
                xaxis='x',
                yaxis='y2',
            ))
            
            # Error markers for Adjacent Velocity
            if adjacent_velocities is not None:
                from src.data.growth_standards import calculate_velocity_for_age_zscore
                adj_vel_hover_text_errors = []
                for idx, age_d, adj_vel in zip(
                    original_indices[vel_error_mask],
                    vel_ages_days[vel_error_mask],
                    adjacent_velocities[vel_error_mask]
                ):
                    visit_text = _format_visit_label(idx)
                    adj_z = calculate_velocity_for_age_zscore(adj_vel, age_d, sex) if not np.isnan(adj_vel) else np.nan
                    text_parts = [
                        f"<b>♦️ ERROR - {visit_text}</b>",
                        f"Age: {age_d:.0f} days",
                        f"Adjacent Velocity: {adj_vel:.6f} inches/day"
                    ]
                    if not np.isnan(adj_z):
                        text_parts.append(f"Adjacent Velocity z-score: {adj_z:.2f}")
                    adj_vel_hover_text_errors.append("<br>".join(text_parts))
                
                fig.add_trace(go.Scatter(
                    x=vel_ages_days[vel_error_mask],
                    y=adjacent_velocities[vel_error_mask],
                    mode='markers+text',
                    name='Error Adjacent Markers',
                    marker=dict(
                        size=11,
                        color='red',
                        symbol='diamond',
                        line=dict(color='black', width=1)
                    ),
                    text=[f"z={adj_z:.2f}" if not np.isnan(adj_z) else "" 
                          for adj_z in [calculate_velocity_for_age_zscore(adj_vel, age_d, sex) 
                                       if not np.isnan(adj_vel) else np.nan 
                                       for age_d, adj_vel in zip(vel_ages_days[vel_error_mask], 
                                                                  adjacent_velocities[vel_error_mask])]],
                    textposition='bottom center',
                    customdata=original_indices[vel_error_mask],
                    hoverinfo='skip',
                    showlegend=False,
                    legendgroup='velocity_adjacent',
                    xaxis='x',
                    yaxis='y2',
                ))
        
        # Add selected velocity point highlight (visual overlay only, no tooltip)
        if selected_index is not None and selected_index in original_indices:
            pos = np.where(original_indices == selected_index)[0]
            if len(pos) > 0:
                pos = pos[0]
                sel_age_d = vel_ages_days[pos]
                sel_velocity = velocities[pos]
                
                # Selected marker on Min-interval Adjusted Velocity (orange line)
                fig.add_trace(go.Scatter(
                    x=[sel_age_d],
                    y=[sel_velocity],
                    mode='markers',
                    name='Selected',
                    marker=dict(
                        size=14,
                        color=PLOT_COLORS['selected'],
                        symbol='circle',
                        line=dict(color='white', width=3)
                    ),
                    hoverinfo='skip',  # Skip hover to avoid duplicate tooltip
                    showlegend=False,
                    legendgroup='velocity_adjusted',
                    xaxis='x',
                    yaxis='y2',
                ))
                
                # Selected marker on Adjacent Velocity (purple line) if available
                if adjacent_velocities is not None and not np.isnan(adjacent_velocities[pos]):
                    sel_adj_velocity = adjacent_velocities[pos]
                    
                    fig.add_trace(go.Scatter(
                        x=[sel_age_d],
                        y=[sel_adj_velocity],
                        mode='markers',
                        name='Selected',
                        marker=dict(
                            size=14,
                            color=PLOT_COLORS['selected'],
                            symbol='circle',
                            line=dict(color='white', width=3)
                        ),
                        hoverinfo='skip',  # Skip hover to avoid duplicate tooltip
                        showlegend=False,
                        legendgroup='velocity_adjacent',
                        xaxis='x',
                        yaxis='y2',
                    ))
    
    # Update layout with grid and synchronized hovering
    patient_id = patient_data['patient_id'].iloc[0] if 'patient_id' in patient_data.columns else "Unknown"
    min_age = ages_days.min()
    max_age = ages_days.max()

    age_span = max_age - min_age
    x_pad = max(5, age_span * 0.05) if age_span > 0 else 5
    x_min = max(0, min_age - x_pad)
    x_max = max_age + x_pad

    height_axis_range = None
    height_components = []
    if valid_height_positions.size > 0:
        height_components.append(heights[valid_height_positions])
    for key in ('lower_bound', 'median', 'upper_bound'):
        bounds_array = np.asarray(height_bounds.get(key, np.array([])))
        if bounds_array.size > 0:
            height_components.append(bounds_array)
    if null_marker_y is not None:
        height_components.append(np.array([null_marker_y]))
    if height_components:
        height_values = np.concatenate(height_components)
        height_values = height_values[~np.isnan(height_values)]
        if height_values.size > 0:
            height_min = float(np.nanmin(height_values))
            height_max = float(np.nanmax(height_values))
            height_span = height_max - height_min
            height_pad = max(0.5, height_span * 0.05) if height_span > 0 else 0.5
            height_axis_range = [height_min - height_pad, height_max + height_pad]

    # For velocity chart, we'll use autorange to dynamically adjust when traces are hidden
    # This allows the y-axis to auto-adjust based on visible traces
    velocity_axis_range = None
    
    fig.update_layout(
        height=1000,  # Taller for two stacked charts
        hovermode='x',  # Hover based on x-axis position
        hoversubplots='axis',  # Synchronized hover across subplots
        grid=dict(rows=2, columns=1, pattern='independent'),  # Grid layout like the example
        showlegend=True,
        dragmode='zoom',  # Default to zoom box instead of pan on initial render
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.48,  # Position between the two charts (middle area)
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        # Configure xaxis (shared for both)
        xaxis=dict(
            title="Age (days)",
            autorange=True,  # Auto-adjust range based on visible traces when toggling legend
            automargin=True,  # Automatically adjust margins
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            domain=[0, 1]
        ),
        # Configure yaxis for height chart
        yaxis=dict(
            title="Height (inches)",
            range=height_axis_range,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            domain=[0.55, 1]
        ),
        # Configure yaxis2 for velocity chart
        yaxis2=dict(
            title="Velocity (inches/day)",
            autorange=True,  # Auto-adjust range based on visible traces when toggling legend
            automargin=True,  # Automatically adjust margins
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            domain=[0, 0.45]
        ),
    )
    
    return fig
