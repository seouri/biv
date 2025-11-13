"""Interactive data table component for patient measurements."""

import json
from typing import Optional, Set

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from src.config import TABLE_COLUMNS
from src.utils.visit_helpers import compute_height_visit_metadata


def render_data_table(
    patient_data: pd.DataFrame,
    selected_index: Optional[int] = None,
    error_indices: Optional[Set[int]] = None,
    show_missing_heights: bool = True,
) -> Optional[int]:
    """Render the interactive visit table with synchronized selection behavior."""

    if error_indices is None:
        error_indices = set()

    if patient_data.empty:
        st.info("No visit data available for this patient.")
        return None

    def _to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    _valid_indices, _missing_indices, visit_number_map = compute_height_visit_metadata(patient_data)
    selected_index_int = _to_int(selected_index) if selected_index is not None else None
    error_index_set = {_to_int(idx) for idx in error_indices}

    display_data = patient_data.copy()

    if "height_in" in display_data.columns and not show_missing_heights:
        display_data = display_data[display_data["height_in"].notna()]

    if display_data.empty:
        st.info(
            "No visits with height measurements to display. Enable \"Show visits missing height\" to include visits without height."
        )
        return None

    patient_id_value: Optional[str] = None
    if "patient_id" in display_data.columns and not display_data["patient_id"].empty:
        patient_id_value = str(display_data["patient_id"].iloc[0])

    if "height_in" in display_data.columns:
        height_presence = display_data["height_in"].notna()
    else:
        height_presence = pd.Series([True] * len(display_data), index=display_data.index)

    has_height_map = {_to_int(idx): bool(val) for idx, val in height_presence.items()}

    status_icons = []
    visit_numbers = []
    row_ids = []
    has_height_flags = []
    error_flags = []

    for raw_idx in display_data.index:
        idx = _to_int(raw_idx)
        has_height = has_height_map.get(idx, True)
        is_error = idx in error_index_set

        row_ids.append(idx)
        has_height_flags.append(has_height)
        error_flags.append(is_error)

        icons = []
        if idx == selected_index_int and has_height:
            icons.append("👉")
        if is_error:
            icons.append("♦️")
        if not has_height:
            icons.append("？")
        status_icons.append(" ".join(icons))

        if has_height:
            visit_num = visit_number_map.get(idx, "")
            visit_numbers.append(str(visit_num) if visit_num != "" else "—")
        else:
            visit_numbers.append("—")

    display_columns = [col for col in TABLE_COLUMNS if col in display_data.columns]
    grid_display = display_data[display_columns].copy()
    grid_display.insert(0, "Visit #", visit_numbers)
    grid_display.insert(0, "Status", status_icons)

    # Format numeric columns for display
    if "age_in_days" in grid_display.columns:
        grid_display["age_in_days"] = grid_display["age_in_days"].round(0).astype(int)
    if "height_in" in grid_display.columns:
        grid_display["height_in"] = grid_display["height_in"].round(2)
    if "weight_oz" in grid_display.columns:
        grid_display["weight_oz"] = grid_display["weight_oz"].round(1)
    if "head_circ_cm" in grid_display.columns:
        grid_display["head_circ_cm"] = grid_display["head_circ_cm"].round(1)
    if "bmi" in grid_display.columns:
        grid_display["bmi"] = grid_display["bmi"].round(2)

    grid_data = grid_display.copy()
    grid_data.insert(0, "_row_id", row_ids)
    grid_data["_has_height"] = has_height_flags
    grid_data["_is_error"] = error_flags

    # Provide legend
    st.caption("👉 = Selected | ♦️ = Marked as Error | ？ = Missing height")
    st.info("👇 Click on a row to select a visit. Rows with missing height cannot be selected.")

    # Per-patient widget key for isolated selection state
    table_widget_key = "patient_data_table"
    if patient_id_value:
        table_widget_key = f"{table_widget_key}_{patient_id_value}"

    selection_state_key = f"{table_widget_key}_last_reported"
    last_reported_index = st.session_state.get(selection_state_key)

    st.markdown(
        """
        <style>
        /* Column borders - apply to all cells */
        .ag-theme-streamlit .ag-cell {
            border-right: 1px solid #ddd !important;
        }
        .ag-theme-streamlit .ag-cell:not(:last-child) {
            border-right: 1px solid #ddd !important;
        }
        .ag-theme-streamlit .ag-row .ag-cell {
            border-right: 1px solid #ddd !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    builder = GridOptionsBuilder.from_dataframe(grid_data)
    builder.configure_default_column(
        resizable=True, 
        filter=False, 
        sortable=False,
        cellStyle={'border-right': '1px solid #ddd'}
    )
    builder.configure_selection("single", use_checkbox=False)
    builder.configure_column("_row_id", hide=True)
    builder.configure_column("_has_height", hide=True)
    builder.configure_column("_is_error", hide=True)
    builder.configure_column("Status", pinned="left", width=50, suppressSizeToFit=True)
    builder.configure_column("Visit #", width=50, suppressSizeToFit=True)

    # Define row styling function using JsCode for proper color application
    get_row_style = JsCode("""
        function(params) {
            if (!params.data) return {};
            
            const isSelected = params.node.isSelected();
            const hasHeight = params.data._has_height;
            const isError = params.data._is_error;
            
            // Priority order: selected error > error > selected > missing height
            if (isSelected && isError) {
                return {
                    'background-color': '#ff4444',
                    'color': 'white',
                    'font-weight': '600'
                };
            }
            if (isError) {
                return {
                    'background-color': '#FFE5E6',
                };
            }
            if (isSelected) {
                return {
                    'background-color': '#FEF5E5',
                    'font-weight': '600'
                };
            }
            if (!hasHeight) {
                return {
                    'background-color': '#F6F6F6',
                    'color': '#666'
                };
            }
            return {};
        }
    """)

    row_class_rules = {
        "row-missing-height": "data._has_height === false",
        "row-error": "data._is_error === true",
    }

    builder.configure_grid_options(
        rowClassRules=row_class_rules,
        getRowStyle=get_row_style,
        suppressRowHoverHighlight=True,
        suppressScrollOnNewData=True,
        maintainColumnOrder=True,
        getRowId=JsCode("function(params) { return params.data._row_id?.toString(); }"),
        isRowSelectable=JsCode("function(node) { return !!(node.data && node.data._has_height); }"),
    )

    selected_row_id = None
    if selected_index_int is not None:
        for idx in row_ids:
            if idx == selected_index_int:
                selected_row_id = str(idx)
                break

    selected_row_js_value = json.dumps(selected_row_id) if selected_row_id is not None else "null"

    select_row_js_code = (
        f"""
        function(params) {{
            var api = params.api;
            if (!api) {{ return; }}
            var selectedId = {selected_row_js_value};
            if (selectedId === null) {{
                api.deselectAll();
                return;
            }}
            var selectedIdStr = selectedId.toString();
            setTimeout(function() {{
                var node = api.getRowNode(selectedIdStr);
                if (node) {{
                    api.deselectAll();
                    node.setSelected(true);
                    api.ensureNodeVisible(node, 'middle');
                }}
            }}, 0);
        }}
        """
    )

    select_row_handler = JsCode(select_row_js_code)

    builder.configure_grid_options(
        onGridReady=select_row_handler,
        onFirstDataRendered=select_row_handler,
        onModelUpdated=select_row_handler,
        onRowDataUpdated=select_row_handler,
        onSelectionChanged=JsCode(
            """function(params) { 
                const selected = params.api.getSelectedNodes(); 
                if (selected && selected.length) { 
                    params.api.ensureNodeVisible(selected[0], 'middle'); 
                }
                // Force redraw all rows to update styling
                params.api.redrawRows();
            }"""
        ),
    )

    grid_options = builder.build()

    grid_response = AgGrid(
        grid_data,
        gridOptions=grid_options,
        height=500,
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=False,
        key=table_widget_key,
    )

    selected_rows = grid_response.get("selected_rows")

    if isinstance(selected_rows, pd.DataFrame):
        if selected_rows.empty:
            return None
        selected_rows = selected_rows.to_dict("records")

    if not selected_rows:
        return None

    clicked_index = selected_rows[0].get("_row_id")
    if clicked_index is None:
        return None

    clicked_index_int = _to_int(clicked_index)
    if not has_height_map.get(clicked_index_int, True):
        return None

    if last_reported_index is not None and clicked_index_int == last_reported_index:
        return None

    st.session_state[selection_state_key] = clicked_index_int

    if selected_index_int is not None and clicked_index_int == selected_index_int:
        return None

    return clicked_index_int


def render_table_summary(patient_data: pd.DataFrame, error_indices: Set[int]) -> None:
    """
    Render summary statistics for the table.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data
    error_indices : Set[int]
        Set of error indices
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Visits", len(patient_data))
    
    with col2:
        st.metric("Errors Marked", len(error_indices))
    
    with col3:
        age_range = patient_data['age_years'].min(), patient_data['age_years'].max()
        st.metric("Age Range", f"{age_range[0]:.1f} - {age_range[1]:.1f} yrs")
    
    with col4:
        height_range = patient_data['height_in'].min(), patient_data['height_in'].max()
        st.metric("Height Range", f"{height_range[0]:.1f} - {height_range[1]:.1f} in")


def format_table_for_export(
    patient_data: pd.DataFrame,
    error_indices: Set[int],
    point_comments: dict
) -> pd.DataFrame:
    """
    Format patient data table for export with error labels.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data
    error_indices : Set[int]
        Set of error indices
    point_comments : dict
        Dictionary mapping indices to comments
        
    Returns
    -------
    pd.DataFrame
        Formatted table ready for export
    """
    export_data = patient_data.copy()
    
    # Add error flag
    export_data['is_error'] = export_data.index.isin(error_indices)
    
    # Add comments
    export_data['comment'] = export_data.index.map(
        lambda idx: point_comments.get(idx, '')
    )
    
    return export_data
