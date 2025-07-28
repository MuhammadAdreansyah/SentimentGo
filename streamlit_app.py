"""
GoRide Sentiment Analysis - Main Application Entry Point

Aplikasi analisis sentimen untuk ulasan GoRide dengan fitur:
- Autentikasi pengguna (login/register/logout)
- Dashboard ringkasan analisis
- Analisis data mendalam
- Prediksi sentimen real-time

Author: SentimenGo Team
Version: 2.0
"""

import streamlit as st

# ===================================================================
# CONFIGURATION & IMPORTS
# ===================================================================

# ===================================================================
# CONFIGURATION & IMPORTS
# ===================================================================

# Setup basic Streamlit config first
import streamlit as st

def get_login_status():
    """Mendapatkan status login user dari session state"""
    try:
        return st.session_state.get('logged_in', False)
    except Exception:
        return False

# Konfigurasi halaman Streamlit
logged_in = get_login_status()
st.set_page_config(
    page_title="GoRide Sentiment Analysis",
    page_icon="🛵" if logged_in else "🔐",
    layout="centered",
    initial_sidebar_state="expanded" if logged_in else "collapsed"
)

# Lazy imports - hanya import ketika diperlukan
def get_auth_module():
    """Lazy import untuk modul auth"""
    from ui.auth import auth
    return auth

def get_dashboard_module():
    """Lazy import untuk modul dashboard"""
    from ui.tools.Dashboard_Ringkasan import render_dashboard
    return render_dashboard

def get_analysis_module():
    """Lazy import untuk modul analisis"""
    from ui.tools.Analisis_Data import render_data_analysis
    return render_data_analysis

def get_prediction_module():
    """Lazy import untuk modul prediksi"""
    from ui.tools.Prediksi_Sentimen import render_sentiment_prediction
    return render_sentiment_prediction



# ===================================================================
# PAGE FUNCTIONS
# ===================================================================

def login_page():
    """
    Halaman autentikasi (login/register/forgot password)
    """
    # Handle redirect setelah login sukses
    if st.session_state.get('login_success', False):
        st.markdown('<meta http-equiv="refresh" content="0;url=/" />', unsafe_allow_html=True)
        return
    
    auth = get_auth_module()
    auth.main()

def logout_page():
    """
    Proses logout dan redirect ke halaman utama
    """
    auth = get_auth_module()
    auth.logout()
    st.markdown(
        '<meta http-equiv="refresh" content="0;url=/?logout=1" />',
        unsafe_allow_html=True
    )

def dashboard_page():
    """
    Halaman Dashboard Ringkasan
    """
    render_dashboard = get_dashboard_module()
    render_dashboard()

def data_analysis_page():
    """
    Halaman Analisis Data
    """
    render_data_analysis = get_analysis_module()
    render_data_analysis()

def sentiment_prediction_page():
    """
    Halaman Prediksi Sentimen
    """
    render_sentiment_prediction = get_prediction_module()
    render_sentiment_prediction()

# ===================================================================
# NAVIGATION SETUP
# ===================================================================

# Definisi halaman-halaman untuk navigation
logout_pg = st.Page(
    logout_page, 
    title="Logout", 
    icon=":material/logout:"
)

dashboard_pg = st.Page(
    dashboard_page, 
    title="Dashboard Ringkasan", 
    icon=":material/dashboard:", 
    default=True
)

data_analysis_pg = st.Page(
    data_analysis_page, 
    title="Analisis Data", 
    icon=":material/analytics:"
)

prediction_pg = st.Page(
    sentiment_prediction_page, 
    title="Prediksi Sentimen", 
    icon=":material/psychology:"
)

# ===================================================================
# MODEL PREPARATION FUNCTIONS
# ===================================================================

def check_and_setup_models():
    """
    Memeriksa dan menyiapkan model jika diperlukan
    Returns: bool - True jika model siap, False jika perlu setup
    """
    from ui.utils import quick_model_check, render_model_preparation_page
    
    models_ready, compatibility_msg, data_count = quick_model_check()
    
    if models_ready:
        # Model sudah siap
        st.session_state['ready_for_tools'] = True
        st.session_state['models_prepared'] = True
        st.toast(f"✅ Model siap digunakan! Dataset: {data_count:,} ulasan", icon="📊")
        return True
    else:
        # Model belum siap, tampilkan halaman persiapan
        render_model_preparation_page()
        return False

# ===================================================================
# MAIN APPLICATION WORKFLOW
# ===================================================================

def main():
    """
    Workflow utama aplikasi:
    1. Inisialisasi session state
    2. Autentikasi pengguna
    3. Setup model dan data
    4. Routing ke tools yang sesuai
    """
    
    # Inisialisasi dan sinkronisasi session state
    auth = get_auth_module()
    auth.sync_login_state()
    auth.initialize_session_state()
    
    # Tampilkan notifikasi login sukses
    if st.session_state.get('login_success', False):
        user_email = st.session_state.get('user_email', 'User')
        st.toast(f"✅ {user_email} berhasil login!", icon="🎉")
        st.session_state['login_success'] = False
    
    # Routing berdasarkan status login
    if st.session_state.get('logged_in', False):
        # User sudah login
        if st.session_state.get('ready_for_tools', False):
            # Model sudah siap, tampilkan tools utama
            navigation = st.navigation({
                "🚀 Tools": [dashboard_pg, data_analysis_pg, prediction_pg],
                "👤 Akun": [logout_pg],
            })
            navigation.run()
        else:
            # Cek dan setup model jika diperlukan
            if check_and_setup_models():
                st.rerun()
    else:
        # User belum login, tampilkan halaman autentikasi
        login_page()

# ===================================================================
# APPLICATION ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()