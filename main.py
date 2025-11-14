"""
Entry point for the Growth Error Labeling Dashboard.

This file imports and runs the main Streamlit application with authentication.
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

# Import authentication and app
from src.auth import is_authenticated, render_login_page, get_current_user, logout
from src.app import main

if __name__ == "__main__":
    # Check if user is authenticated
    if not is_authenticated():
        # Show login page
        render_login_page()
    else:
        # Show logout button in the sidebar and run main app
        with st.sidebar:
            st.markdown("---")
            current_user = get_current_user()
            st.success(f"👤 Logged in as: **{current_user}**")
            if st.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()
            st.markdown("---")
        
        # Run the main application
        main()
