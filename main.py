"""
Entry point for the Growth Error Labeling Dashboard.

This file imports and runs the main Streamlit application.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure Streamlit page layout each rerun
st.set_page_config(
    page_title="Growth Error Labeling Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import and run the main app
from src.app import main

if __name__ == "__main__":
    main()
