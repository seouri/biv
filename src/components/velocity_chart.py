"""
Velocity chart component: Growth velocity vs Age with CDC/WHO standards.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Set, Optional
from src.config import PLOT_HEIGHT, PLOT_COLORS, CHART_CONFIG
from src.data.growth_standards import get_velocity_zscore_bounds


def render_velocity_chart(
    patient_data: pd.DataFrame,
    sex: str,
    outlier_indices: Set[int] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    """
    Render interactive velocity chart (velocity vs age).
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data with calculated fields including velocity
    sex : str
        Patient sex ('M' or 'F') for growth standards
    outlier_indices : Set[int], optional
        Set of indices marked as outliers
    selected_index : int, optional
        Currently selected point index
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if outlier_indices is None:
        outlier_indices = set()
    
    # Filter out first point (no velocity)
    velocity_data = patient_data[patient_data['velocity'].notna()].copy()
    
    if velocity_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No velocity data available<br>(requires at least 2 measurements)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=PLOT_HEIGHT)
        return fig
    
    # Prepare data
    ages_days = velocity_data['age_in_days'].values
    ages_years = velocity_data['age_years'].values
    velocities = velocity_data['velocity'].values
    velocity_zscores = velocity_data['velocity_zscore'].values
    abs_changes = velocity_data['absolute_change'].values
    pct_changes = velocity_data['percent_change'].values
    original_indices = velocity_data.index.values
    
    # Get velocity growth standard bounds
    age_range = np.linspace(ages_days.min(), ages_days.max(), 100)
    bounds = get_velocity_zscore_bounds(age_range, sex)
    
    # Create figure
    fig = go.Figure()
    
    # Add CDC/WHO velocity ribbon (±5 z-score bounds) - Green boundaries
    fig.add_trace(go.Scatter(
        x=bounds['age_days'],
        y=bounds['upper_bound'],
        mode='lines',
        name='+5 SD',
        line=dict(color='green', width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    fig.add_trace(go.Scatter(
        x=bounds['age_days'],
        y=bounds['lower_bound'],
        mode='lines',
        name='-5 SD',
        line=dict(color='green', width=2, dash='dot'),
        fill='tonexty',
        fillcolor=PLOT_COLORS['ribbon_fill'],
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Add median line - Green dashed
    fig.add_trace(go.Scatter(
        x=bounds['age_days'],
        y=bounds['median'],
        mode='lines',
        name='Median',
        line=dict(color='green', width=2, dash='dash'),
        showlegend=True,
        hoverinfo='skip',
    ))
    
    # Separate data into normal, outlier, and selected points
    normal_mask = np.array([idx not in outlier_indices and idx != selected_index 
                           for idx in original_indices])
    outlier_mask = np.array([idx in outlier_indices and idx != selected_index 
                            for idx in original_indices])
    
    # Add normal measurements - Blue line with markers
    if normal_mask.any():
        hover_text = [
            f"<b>Visit {idx+1}</b><br>" +
            f"Age: {age_d:.0f} days<br>" +
            f"Velocity: {vel:.6f} inches/day<br>" +
            f"Change: {abs_ch:+.2f} in ({pct_ch:+.1f}%)<br>" +
            f"Velocity z-score: {z:.2f}"
            for idx, age_yr, age_d, vel, abs_ch, pct_ch, z in zip(
                original_indices[normal_mask],
                ages_years[normal_mask],
                ages_days[normal_mask],
                velocities[normal_mask],
                abs_changes[normal_mask],
                pct_changes[normal_mask],
                velocity_zscores[normal_mask]
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=ages_days[normal_mask],
            y=velocities[normal_mask],
            mode='lines+markers',
            name='Velocity',
            line=dict(color='blue', width=3),
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(color='blue', width=1)
            ),
            customdata=original_indices[normal_mask],  # Store original indices
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=True,
        ))
    
    # Add outlier measurements (red diamond markers with z-score labels)
    if outlier_mask.any():
        hover_text_outliers = [
            f"<b>♦️ OUTLIER - Visit {idx+1}</b><br>" +
            f"Age: {age_d:.0f} days<br>" +
            f"Velocity: {vel:.6f} inches/day<br>" +
            f"Change: {abs_ch:+.2f} in ({pct_ch:+.1f}%)<br>" +
            f"Velocity z-score: {z:.2f}"
            for idx, age_yr, age_d, vel, abs_ch, pct_ch, z in zip(
                original_indices[outlier_mask],
                ages_years[outlier_mask],
                ages_days[outlier_mask],
                velocities[outlier_mask],
                abs_changes[outlier_mask],
                pct_changes[outlier_mask],
                velocity_zscores[outlier_mask]
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=ages_days[outlier_mask],
            y=velocities[outlier_mask],
            mode='markers+text',
            name='Outlier Velocity',
            marker=dict(
                size=11,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            text=[f"z={z:.2f}" for z in velocity_zscores[outlier_mask]],
            textposition='top center',
            customdata=original_indices[outlier_mask],
            hovertext=hover_text_outliers,
            hoverinfo='text',
            showlegend=True,
        ))
    
    # Add selected point (highlighted)
    if selected_index is not None and selected_index in original_indices:
        # Find position in velocity_data
        pos = np.where(original_indices == selected_index)[0]
        if len(pos) > 0:
            pos = pos[0]
            selected_age_yr = ages_years[pos]
            selected_age_d = ages_days[pos]
            selected_velocity = velocities[pos]
            selected_zscore = velocity_zscores[pos]
            selected_abs_change = abs_changes[pos]
            selected_pct_change = pct_changes[pos]
            
            is_outlier = selected_index in outlier_indices
            label = "♦️ OUTLIER - " if is_outlier else ""
            
            fig.add_trace(go.Scatter(
                x=[selected_age_d],
                y=[selected_velocity],
                mode='markers',
                name='Selected',
                marker=dict(
                    size=14,
                    color=PLOT_COLORS['selected'],
                    symbol='circle',
                    line=dict(color='white', width=3)
                ),
                hovertext=f"<b>{label}SELECTED - Visit {selected_index+1}</b><br>" +
                         f"Age: {selected_age_d:.0f} days<br>" +
                         f"Velocity: {selected_velocity:.6f} inches/day<br>" +
                         f"Change: {selected_abs_change:+.2f} in ({selected_pct_change:+.1f}%)<br>" +
                         f"Velocity z-score: {selected_zscore:.2f}",
                hoverinfo='text',
                showlegend=True,
            ))
    
    # Update layout
    patient_id = patient_data['patient_id'].iloc[0] if 'patient_id' in patient_data.columns else "Unknown"
    min_age = ages_days.min()
    max_age = ages_days.max()
    
    fig.update_layout(
        height=500,
        title=dict(
            text=f"Age vs. Velocity - Patient {patient_id}",
            x=0.5,
            font=dict(size=16, color="black")
        ),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Age (days)",
            range=[min_age, max_age],
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
        ),
        yaxis=dict(
            title="Velocity (inches/day)",
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
        ),
    )
    
    return fig
