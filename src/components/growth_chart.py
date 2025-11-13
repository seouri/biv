"""
Growth chart component: Height vs Age with CDC/WHO growth standards.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Set, Optional
from src.config import PLOT_HEIGHT, PLOT_COLORS, CHART_CONFIG
from src.data.growth_standards import get_height_zscore_bounds


def render_growth_chart(
    patient_data: pd.DataFrame,
    sex: str,
    outlier_indices: Set[int] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    """
    Render interactive growth chart (height vs age).
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data with calculated fields
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
    
    # Prepare data
    ages_days = patient_data['age_in_days'].values
    ages_years = patient_data['age_years'].values
    heights = patient_data['height_in'].values
    height_zscores = patient_data['height_zscore'].values
    
    # Get growth standard bounds
    age_range = np.linspace(ages_days.min(), ages_days.max(), 100)
    bounds = get_height_zscore_bounds(age_range, sex)
    
    # Create figure
    fig = go.Figure()
    
    # Add CDC/WHO ribbon (±3 z-score bounds) - Green boundaries
    fig.add_trace(go.Scatter(
        x=bounds['age_days'],
        y=bounds['upper_bound'],
        mode='lines',
        name='+3 SD (WHO/CDC)',
        line=dict(color='green', width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    fig.add_trace(go.Scatter(
        x=bounds['age_days'],
        y=bounds['lower_bound'],
        mode='lines',
        name='-3 SD (WHO/CDC)',
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
    normal_mask = np.array([i not in outlier_indices and i != selected_index 
                           for i in range(len(patient_data))])
    outlier_mask = np.array([i in outlier_indices and i != selected_index 
                            for i in range(len(patient_data))])
    
    # Add normal measurements - Blue line with markers
    if normal_mask.any():
        hover_text = [
            f"<b>Visit {i+1}</b><br>" +
            f"Age: {age_d:.0f} days<br>" +
            f"Height: {ht:.2f} inches<br>" +
            f"Height-for-age z-score: {z:.2f}"
            for i, (age_yr, age_d, ht, z) in enumerate(
                zip(ages_years[normal_mask], ages_days[normal_mask], 
                    heights[normal_mask], height_zscores[normal_mask])
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=ages_days[normal_mask],
            y=heights[normal_mask],
            mode='lines+markers',
            name='Height',
            line=dict(color='blue', width=3),
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(color='blue', width=1)
            ),
            customdata=np.where(normal_mask)[0],  # Store original indices
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=True,
        ))
    
    # Add outlier measurements (red diamond markers with z-score labels)
    if outlier_mask.any():
        hover_text_outliers = [
            f"<b>♦️ OUTLIER - Visit {i+1}</b><br>" +
            f"Age: {age_d:.0f} days<br>" +
            f"Height: {ht:.2f} inches<br>" +
            f"Height-for-age z-score: {z:.2f}"
            for i, (age_yr, age_d, ht, z) in enumerate(
                zip(ages_years[outlier_mask], ages_days[outlier_mask],
                    heights[outlier_mask], height_zscores[outlier_mask])
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=ages_days[outlier_mask],
            y=heights[outlier_mask],
            mode='markers+text',
            name='Outlier Height',
            marker=dict(
                size=11,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            text=[f"z={z:.2f}" for z in height_zscores[outlier_mask]],
            textposition='top center',
            customdata=np.where(outlier_mask)[0],
            hovertext=hover_text_outliers,
            hoverinfo='text',
            showlegend=True,
        ))
    
    # Add selected point (highlighted)
    if selected_index is not None and 0 <= selected_index < len(patient_data):
        selected_age_yr = ages_years[selected_index]
        selected_age_d = ages_days[selected_index]
        selected_height = heights[selected_index]
        selected_zscore = height_zscores[selected_index]
        
        is_outlier = selected_index in outlier_indices
        label = "♦️ OUTLIER - " if is_outlier else ""
        
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
            hovertext=f"<b>{label}SELECTED - Visit {selected_index+1}</b><br>" +
                     f"Age: {selected_age_d:.0f} days<br>" +
                     f"Height: {selected_height:.2f} inches<br>" +
                     f"Height-for-age z-score: {selected_zscore:.2f}",
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
            text=f"Age vs. Height - Patient {patient_id}",
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
            title="Height (inches)",
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
        ),
    )
    
    return fig


def create_vertical_line_shape(x_position: float, y_range: tuple) -> dict:
    """
    Create a vertical line shape for synchronized hovering.
    
    Parameters
    ----------
    x_position : float
        X-coordinate for the vertical line (age in years)
    y_range : tuple
        (min, max) y-values for the line
        
    Returns
    -------
    dict
        Plotly shape dictionary
    """
    return {
        'type': 'line',
        'x0': x_position,
        'x1': x_position,
        'y0': y_range[0],
        'y1': y_range[1],
        'line': dict(
            color='gray',
            width=1,
            dash='dot',
        ),
    }
