import streamlit as st

# Set page configuration
st.set_page_config(page_title="Deep Learning Models", page_icon="🤖", layout="wide")

# Custom CSS for Sidebar Styling
st.markdown("""
    <style>
        /* Sidebar Background Color */
        [data-testid="stSidebar"] {
            background-color: lightgoldenrodyellow !important;
        }

        /* Sidebar Navigation Menu Styling */
        [data-testid="stSidebarNav"] a {
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }

        [data-testid="stSidebarNav"] a:hover {
            color: darkblue !important;
            background-color: white !important;
            border-radius: 10px;
        }

        /* Logo Styling (Smaller Size) */
        [data-testid="stSidebarNav"]::before {
            content: "";
            display: block;
            margin: 10px auto;
            width: 60px;  /* Small Logo */
            height: 60px;
            background-image: url('https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg'); /* Replace with your logo URL */
            background-size: cover;
            border-radius: 50%;
        }
    </style>
""", unsafe_allow_html=True)



# Main Content
st.title("🏆 Welcome to the Deep Learning Models App")
st.write("Use the sidebar to navigate between **LSTM** and **GRU** models.")
show_sidebar = True  # Change this dynamically if needed
if show_sidebar:
    st.sidebar.title("Sidebar Content")

