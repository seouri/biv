"""
Authentication module for the Growth Error Labeling Dashboard.

Provides simple username/password authentication with user-specific data storage.
"""

import streamlit as st
from typing import Optional, Dict

# User credentials - in production, these should be stored securely
# For now, using simple password scheme
USERS: Dict[str, str] = {
    "user1": "growthai",
    "user2": "growthai",
    "user3": "growthai",
    "user4": "growthai",
    "user5": "growthai",
}


def check_credentials(username: str, password: str) -> bool:
    """
    Verify user credentials.
    
    Parameters
    ----------
    username : str
        Username to check
    password : str
        Password to verify
        
    Returns
    -------
    bool
        True if credentials are valid
    """
    return username in USERS and USERS[username] == password


def get_current_user() -> Optional[str]:
    """
    Get the currently logged-in user.
    
    Returns
    -------
    Optional[str]
        Username if logged in, None otherwise
    """
    return st.session_state.get("authenticated_user")


def is_authenticated() -> bool:
    """
    Check if a user is currently authenticated.
    
    Returns
    -------
    bool
        True if user is logged in
    """
    return st.session_state.get("authenticated", False)


def login(username: str, password: str) -> bool:
    """
    Attempt to log in a user.
    
    Parameters
    ----------
    username : str
        Username
    password : str
        Password
        
    Returns
    -------
    bool
        True if login successful
    """
    if check_credentials(username, password):
        st.session_state["authenticated"] = True
        st.session_state["authenticated_user"] = username
        return True
    return False


def logout() -> None:
    """Log out the current user and clear all session state."""
    # Clear authentication
    st.session_state["authenticated"] = False
    st.session_state["authenticated_user"] = None
    
    # Clear all other session state to prevent data leakage
    keys_to_clear = [key for key in st.session_state.keys() 
                     if key not in ["authenticated", "authenticated_user"]]
    for key in keys_to_clear:
        del st.session_state[key]


def render_login_page() -> bool:
    """
    Render the login page.
    
    Returns
    -------
    bool
        True if user successfully logged in
    """
    st.title("🔐 Growth Error Labeling Dashboard")
    st.markdown("### Please log in to continue")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Log In", use_container_width=True)
            
            if submit:
                if username and password:
                    if login(username, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
    
    return False
