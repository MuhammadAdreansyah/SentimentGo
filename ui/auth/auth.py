"""
Sistem Autentikasi Streamlit dengan Integrasi Firebase

Aplikasi ini menyediakan sistem autentikasi lengkap dengan fitur:
- Autentikasi email/kata sandi
- Login OAuth Google  
- Registrasi pengguna baru
- Reset kata sandi
- Manajemen sesi yang aman
- Verifikasi email
- Rate limiting dan keamanan

Author: SentimenGo App Team
Created: 2024
Last Modified: 2025-07-06
"""

import streamlit as st
import streamlit.components.v1 as components
import re
import os
import asyncio
import httpx
import time
import base64
import logging
import secrets
import firebase_admin
import pyrebase
from firebase_admin import credentials, auth, firestore
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlencode
from streamlit_cookies_controller import CookieController

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
SESSION_TIMEOUT = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 300  # 5 minutes
EMAIL_VERIFICATION_LIMIT = 50  # per hour
REMEMBER_ME_DURATION = 30 * 24 * 60 * 60  # 30 days
LAST_EMAIL_DURATION = 90 * 24 * 60 * 60  # 90 days

# Initialize cookie controller
cookie_controller = CookieController()

def detect_environment() -> Tuple[bool, str]:
    """Detect if running on Streamlit Cloud or local development
    
    Returns:
        Tuple[bool, str]: (is_streamlit_cloud, environment_description)
    """
    # Emergency override for troubleshooting
    if st.secrets.get("STREAMLIT_CLOUD_OVERRIDE") == "true":
        return True, "Streamlit Cloud (Manual Override)"
    
    # Method 1: Primary detection - STREAMLIT_SERVER_HEADLESS
    if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true':
        return True, "Streamlit Cloud (HEADLESS=true)"
    
    # Method 2: Secondary detection - STREAMLIT_CLOUD variable
    if os.getenv('STREAMLIT_CLOUD'):
        return True, "Streamlit Cloud (CLOUD_VAR=true)"
    
    # Method 3: Port-based detection
    server_port = os.getenv('STREAMLIT_SERVER_PORT', '8501')
    if server_port != '8501':
        return True, f"Streamlit Cloud (PORT={server_port})"
    
    # Method 4: Host-based detection
    host = os.getenv('HOST', '').lower()
    if 'streamlit.app' in host or '.streamlit.app' in host:
        return True, f"Streamlit Cloud (HOST={host})"
    
    # Method 5: Browser gather stats (Streamlit Cloud specific)
    if os.getenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS') == 'false':
        return True, "Streamlit Cloud (STATS=false)"
    
    # Method 6: URL-based detection (fallback)
    try:
        # Check if we can access streamlit context
        import streamlit.runtime.scriptrunner as sr
        ctx = sr.get_script_run_ctx()
        if ctx and hasattr(ctx, 'session_id'):
            # This is an indicator we might be in cloud
            session_id = str(ctx.session_id)
            if len(session_id) > 20:  # Cloud sessions typically have longer IDs
                return True, "Streamlit Cloud (Session Detection)"
    except:
        pass
    
    # Default: Local development
    return False, "Local Development"

def get_redirect_uri() -> str:
    """Smart Environment Detection for Streamlit Cloud vs Local Development
    
    Uses multiple reliable detection methods WITHOUT forcing production mode.
    Works seamlessly in both environments.
    """
    try:
        import os
        
        # Get environment variables
        streamlit_headless = os.getenv('STREAMLIT_SERVER_HEADLESS')
        streamlit_cloud = os.getenv('STREAMLIT_CLOUD') 
        
        # Debug logging
        logger.info(f"🔍 Environment Detection - STREAMLIT_SERVER_HEADLESS: {streamlit_headless}")
        logger.info(f"🔍 Environment Detection - STREAMLIT_CLOUD: {streamlit_cloud}")
        
        # Emergency override - ONLY for Streamlit Cloud if auto-detection fails
        cloud_override = st.secrets.get("STREAMLIT_CLOUD_OVERRIDE")
        if cloud_override == "true":
            redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
            logger.info(f"🚨 Environment: STREAMLIT CLOUD (MANUAL OVERRIDE) - Using: {redirect_uri}")
            return redirect_uri
        
        # Method 1: RELIABLE Streamlit Cloud Detection
        # STREAMLIT_SERVER_HEADLESS is set to 'true' ONLY in Streamlit Cloud
        if streamlit_headless == 'true':
            redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
            logger.info(f"☁️ Environment: STREAMLIT CLOUD (Headless Mode) - Using: {redirect_uri}")
            return redirect_uri
        
        # Method 2: STREAMLIT_CLOUD environment variable
        # This is set in Streamlit Cloud environment
        if streamlit_cloud:
            redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
            logger.info(f"☁️ Environment: STREAMLIT CLOUD (Cloud Env) - Using: {redirect_uri}")
            return redirect_uri
            
        # Method 3: URL/Domain-based detection - ENHANCED for Streamlit Cloud
        # Check if running on streamlit.app domain by examining the current context
        try:
            # Check system environment for cloud indicators
            import sys
            python_path = sys.executable
            
            # Streamlit Cloud has specific Python path patterns
            if "/mount/src" in python_path or "streamlit_cloud" in python_path.lower():
                redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
                logger.info(f"☁️ Environment: STREAMLIT CLOUD (Python Path Detection) - Using: {redirect_uri}")
                return redirect_uri
                
        except Exception as e:
            logger.debug(f"Python path detection failed: {e}")
            
        # Method 4: Check platform and hostname patterns for cloud detection
        try:
            import platform
            
            # Check hostname patterns
            hostname = platform.node()
            logger.debug(f"System hostname: {hostname}")
            
            # Cloud hostnames are usually not localhost and have specific patterns
            if hostname and not hostname.startswith('localhost') and not hostname.startswith('127.0.0.1'):
                # Additional check - if hostname looks like cloud infrastructure
                if len(hostname) > 8 and ('-' in hostname or hostname.isalnum()):
                    redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
                    logger.info(f"☁️ Environment: STREAMLIT CLOUD (Hostname Pattern: {hostname}) - Using: {redirect_uri}")
                    return redirect_uri
                    
        except Exception as e:
            logger.debug(f"Platform detection failed: {e}")
            
        # Method 5: Port-based detection for local development confirmation
        try:
            import socket
            hostname = socket.gethostname()
            
            # If we're explicitly on localhost/127.0.0.1, it's definitely local
            if hostname in ['localhost', '127.0.0.1'] or hostname.startswith('localhost'):
                redirect_uri = st.secrets.get("REDIRECT_URI_DEVELOPMENT", "http://localhost:8501/oauth2callback")
                logger.info(f"💻 Environment: LOCAL DEVELOPMENT (Hostname: {hostname}) - Using: {redirect_uri}")
                return redirect_uri
        except Exception as hostname_error:
            logger.debug(f"Hostname detection failed: {hostname_error}")
        
        # DEFAULT: If we reach here and no local indicators found, assume production
        # This is safer for deployment as it prevents OAuth redirect errors
        redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
        logger.info(f"☁️ Environment: STREAMLIT CLOUD (Default/Fallback) - Using: {redirect_uri}")
        return redirect_uri
        
    except Exception as main_error:
        logger.error(f"❌ Environment detection failed: {main_error}")
        
        # Emergency fallback - assume production for safety in cloud deployment
        redirect_uri = st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
        logger.warning(f"⚠️ Using production URI as emergency fallback: {redirect_uri}")
        return redirect_uri
        return "http://localhost:8501/oauth2callback"

def debug_environment_variables() -> Dict[str, Any]:
    """Debug function untuk melihat environment variables Streamlit Cloud
    
    Returns:
        Dict berisi informasi environment variables yang relevan untuk Streamlit Cloud
    """
    import os
    
    debug_info = {
        "detected_platform": "unknown",
        "environment_variables": {},
        "redirect_uri": None
    }
    
    # Streamlit Cloud specific environment variables
    streamlit_env_vars = [
        'STREAMLIT_SERVER_HEADLESS',  # Primary indicator
        'STREAMLIT_CLOUD',            # Secondary indicator
        'STREAMLIT_SERVER_PORT',      # Port information
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS',  # Usage stats setting
    ]
    
    # Collect Streamlit-specific environment variables
    for var in streamlit_env_vars:
        value = os.getenv(var)
        if value is not None:  # Include even if empty string
            debug_info["environment_variables"][var] = value
    
    # Determine platform specifically for Streamlit Cloud
    if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true':
        debug_info["detected_platform"] = "Streamlit Cloud (Primary Detection)"
    elif os.getenv('STREAMLIT_CLOUD'):
        debug_info["detected_platform"] = "Streamlit Cloud (Secondary Detection)"
    else:
        debug_info["detected_platform"] = "Local Development"
    
    # Get redirect URI
    debug_info["redirect_uri"] = get_redirect_uri()
    
    # Add deployment info for Streamlit Cloud
    if debug_info["detected_platform"].startswith("Streamlit Cloud"):
        debug_info["deployment_info"] = {
            "app_url": "https://sentimentgo.streamlit.app",
            "oauth_callback": "https://sentimentgo.streamlit.app/oauth2callback",
            "environment": "production"
        }
    else:
        debug_info["deployment_info"] = {
            "app_url": "http://localhost:8501",
            "oauth_callback": "http://localhost:8501/oauth2callback",
            "environment": "development"
        }
    
    return debug_info

def get_firebase_config() -> Dict[str, Any]:
    """Dapatkan konfigurasi Firebase yang terstruktur"""
    try:
        if "firebase" not in st.secrets:
            return {}
        
        service_account = dict(st.secrets["firebase"])
        firebase_api_key = st.secrets.get("FIREBASE_API_KEY", "")
        
        return {
            "apiKey": firebase_api_key,
            "authDomain": f"{service_account.get('project_id', '')}.firebaseapp.com",
            "projectId": service_account.get('project_id', ''),
            "databaseURL": f"https://{service_account.get('project_id', '')}-default-rtdb.firebaseio.com",
            "storageBucket": f"{service_account.get('project_id', '')}.appspot.com"
        }
    except Exception as e:
        logger.error(f"Failed to get Firebase config: {e}")
        return {}

def is_config_valid() -> bool:
    """Check apakah konfigurasi valid untuk operasi"""
    return bool(
        st.secrets.get("GOOGLE_CLIENT_ID") and 
        st.secrets.get("GOOGLE_CLIENT_SECRET") and 
        (st.secrets.get("REDIRECT_URI_PRODUCTION") or st.secrets.get("REDIRECT_URI_DEVELOPMENT")) and 
        st.secrets.get("FIREBASE_API_KEY")
    )

def validate_google_oauth_config() -> Tuple[bool, str]:
    """Validate Google OAuth configuration specifically
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    missing_configs = []
    
    if not st.secrets.get("GOOGLE_CLIENT_ID"):
        missing_configs.append("GOOGLE_CLIENT_ID")
    
    if not st.secrets.get("GOOGLE_CLIENT_SECRET"):
        missing_configs.append("GOOGLE_CLIENT_SECRET")
    
    # Check for appropriate redirect URI based on environment
    is_cloud, _ = detect_environment()
    if is_cloud:
        if not st.secrets.get("REDIRECT_URI_PRODUCTION"):
            missing_configs.append("REDIRECT_URI_PRODUCTION")
    else:
        if not st.secrets.get("REDIRECT_URI_DEVELOPMENT"):
            missing_configs.append("REDIRECT_URI_DEVELOPMENT")
    
    if missing_configs:
        return False, f"Konfigurasi Google OAuth tidak lengkap. Hilang: {', '.join(missing_configs)}"
    
    return True, "Konfigurasi Google OAuth valid"

def initialize_session_state() -> None:
    """Inisialisasi state sesi dengan nilai default"""
    default_values = {
        'logged_in': False,
        'login_attempts': 0,
        'firebase_initialized': False,
        'auth_type': '🔒 Masuk',
        'user_email': None,
        'remember_me': False,
        'login_time': None
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def verify_environment() -> bool:
    """Verifikasi bahwa semua variabel lingkungan yang diperlukan telah diset"""
    if not is_config_valid():
        logger.error("Configuration validation failed")
        return False
    
    # Periksa apakah Firebase config ada
    if "firebase" not in st.secrets:
        logger.error("Firebase configuration missing")
        return False
    
    return True

def sync_login_state() -> None:
    """Sinkronisasi status login dari cookie ke session_state dengan error handling yang lebih baik"""
    try:
        # Tunggu sejenak untuk memastikan cookie controller siap
        if not hasattr(cookie_controller, 'get'):
            logger.warning("Cookie controller not ready, skipping sync")
            return
        
        is_logged_in_cookie = cookie_controller.get('is_logged_in')
        user_email_cookie = cookie_controller.get('user_email')
        remember_me_cookie = cookie_controller.get('remember_me')
        
        # Validasi data cookie sebelum sync
        if is_logged_in_cookie == 'True' and user_email_cookie:
            # Validasi format email dari cookie
            is_valid_email, _ = validate_email_format(user_email_cookie)
            if is_valid_email:
                st.session_state['logged_in'] = True
                st.session_state['user_email'] = user_email_cookie
                if remember_me_cookie == 'True':
                    st.session_state['remember_me'] = True
                logger.info(f"Login state synced from cookies for user: {user_email_cookie}")
            else:
                logger.warning(f"Invalid email format in cookie: {user_email_cookie}")
                # Clear invalid cookies
                clear_remember_me_cookies()
                st.session_state['logged_in'] = False
        else:
            st.session_state['logged_in'] = False
            
    except Exception as e:
        logger.error(f"Error syncing login state: {e}")
        st.session_state['logged_in'] = False
        # Clear potentially corrupted cookies
        try:
            clear_remember_me_cookies()
        except:
            pass

def set_remember_me_cookies(email: str, remember: bool = False) -> None:
    """Set cookies untuk fungsionalitas 'ingat saya'"""
    try:
        if remember:
            # Set cookies dengan masa berlaku yang dikonfigurasi
            cookie_controller.set('is_logged_in', 'True', max_age=REMEMBER_ME_DURATION)
            cookie_controller.set('user_email', email, max_age=REMEMBER_ME_DURATION)
            cookie_controller.set('remember_me', 'True', max_age=REMEMBER_ME_DURATION)
            cookie_controller.set('last_email', email, max_age=LAST_EMAIL_DURATION)
        else:
            # Set session cookies (berakhir saat browser ditutup)
            cookie_controller.set('is_logged_in', 'True')
            cookie_controller.set('user_email', email)
            cookie_controller.set('remember_me', 'False')
            
    except Exception as e:
        logger.error(f"Error setting cookies: {e}")

def get_remembered_email() -> str:
    """Dapatkan email terakhir yang diingat untuk kemudahan pengguna"""
    try:
        remembered_email = cookie_controller.get('last_email') or ""
        # Validasi email yang diingat
        if remembered_email:
            is_valid, _ = validate_email_format(remembered_email)
            if is_valid:
                return remembered_email
            else:
                logger.warning(f"Invalid remembered email format: {remembered_email}")
                # Clear invalid email
                try:
                    cookie_controller.remove('last_email')
                except:
                    pass
                return ""
        return ""
    except Exception as e:
        logger.error(f"Error getting remembered email: {e}")
        return ""

def is_app_ready() -> bool:
    """Check apakah aplikasi sudah siap untuk digunakan"""
    return (
        st.session_state.get('firebase_initialized', False) and
        st.session_state.get('firebase_auth') is not None and
        st.session_state.get('firestore') is not None
    )

def clear_remember_me_cookies() -> None:
    """Bersihkan semua cookies terkait autentikasi"""
    try:
        cookie_controller.remove('is_logged_in')
        cookie_controller.remove('user_email')
        cookie_controller.remove('remember_me')
    except Exception as e:
        logger.error(f"Error clearing cookies: {e}")

def initialize_feedback_containers():
    """Initialize feedback containers untuk layout consistency"""
    feedback_placeholder = st.empty()
    with feedback_placeholder.container():
        progress_container = st.empty()
        message_container = st.empty()
    return feedback_placeholder, progress_container, message_container

# =============================================================================
# CENTRALIZED ERROR HANDLING
# =============================================================================

def handle_firebase_error(error: Exception, context: str = "") -> Tuple[str, str]:
    """Centralized Firebase error handling dengan pesan yang konsisten"""
    error_str = str(error).upper()
    
    # Firebase Authentication Errors
    if "INVALID_EMAIL" in error_str:
        return "Format email tidak valid", "Format email tidak valid. Periksa kembali alamat email Anda."
    elif "USER_NOT_FOUND" in error_str or "INVALID_LOGIN_CREDENTIALS" in error_str:
        if context == "login":
            return "Email tidak terdaftar", "Email tidak terdaftar dalam sistem kami. Silakan daftar terlebih dahulu."
        else:
            return "User tidak ditemukan", "User tidak ditemukan dalam sistem."
    elif "WRONG_PASSWORD" in error_str or "INVALID_PASSWORD" in error_str:
        return "Kata sandi salah", "Kata sandi salah. Silakan coba lagi atau reset kata sandi."
    elif "USER_DISABLED" in error_str:
        return "Akun dinonaktifkan", "Akun Anda telah dinonaktifkan. Hubungi administrator."
    elif "EMAIL_EXISTS" in error_str or "EMAIL_ALREADY_IN_USE" in error_str:
        return "Email sudah terdaftar", "Email ini sudah terdaftar. Silakan gunakan email lain atau login."
    elif "WEAK_PASSWORD" in error_str:
        return "Kata sandi terlalu lemah", "Kata sandi terlalu lemah. Gunakan minimal 8 karakter dengan kombinasi huruf besar, kecil, angka dan simbol."
    elif "TOO_MANY_REQUESTS" in error_str:
        return "Terlalu banyak percobaan", "Terlalu banyak percobaan. Tunggu beberapa menit sebelum mencoba lagi."
    elif "NETWORK_REQUEST_FAILED" in error_str:
        return "Koneksi bermasalah", "Koneksi internet bermasalah. Periksa koneksi Anda dan coba lagi."
    elif "QUOTA_EXCEEDED" in error_str:
        return "Batas tercapai", "Batas pengiriman email Firebase tercapai. Coba lagi nanti."
    else:
        return f"{context.title()} gagal", f"{context.title()} gagal: {str(error)}"

def show_error_with_context(error: Exception, context: str, progress_container: Any = None, message_container: Any = None) -> None:
    """Tampilkan error dengan konteks dan UI feedback yang konsisten"""
    toast_msg, detailed_msg = handle_firebase_error(error, context)
    
    # Clear progress jika ada
    if progress_container:
        try:
            progress_container.empty()
        except:
            pass
    
    # Tampilkan pesan error
    try:
        if message_container:
            message_container.error(f"❌ {detailed_msg}")
        else:
            st.error(f"❌ {detailed_msg}")
    except:
        st.write(f"❌ {detailed_msg}")
    
    # Tampilkan toast
    show_error_toast(toast_msg)
    
    # Log error
    logger.error(f"{context.title()} failed: {str(error)}")

def handle_validation_errors(errors: list, progress_container: Any = None, message_container: Any = None) -> None:
    """Handle validation errors dengan display yang konsisten"""
    if not errors:
        return
    
    if progress_container:
        progress_container.empty()
    
    if message_container:
        message_container.error("❌ Validasi data gagal:")
        for error in errors:
            message_container.error(error)
    else:
        st.error("❌ Validasi data gagal:")
        for error in errors:
            st.error(error)
    
    show_error_toast("Data tidak valid")
    logger.warning(f"Validation failed: {'; '.join(errors)}")

def validate_email_format(email: str) -> Tuple[bool, str]:
    """Validasi format email"""
    if not email:
        return False, "Email tidak boleh kosong"
    
    # Pola email dasar
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False, "Format email tidak valid. Contoh: nama@domain.com"
    
    if len(email) > 254:
        return False, "Email terlalu panjang (maksimal 254 karakter)"
    
    local_part, domain = email.rsplit('@', 1)
    if len(local_part) > 64:
        return False, "Bagian lokal email terlalu panjang (maksimal 64 karakter)"
    
    if '..' in email:
        return False, "Email tidak boleh mengandung titik berturut-turut"
    
    if local_part.startswith('.') or local_part.endswith('.'):
        return False, "Email tidak boleh diawali atau diakhiri dengan titik"
    
    return True, ""

def validate_name_format(name: str, field_name: str) -> Tuple[bool, str]:
    """Validasi format nama"""
    if not name:
        return False, f"{field_name} tidak boleh kosong"
    
    if len(name) < 2:
        return False, f"{field_name} minimal 2 karakter"
    
    if len(name) > 50:
        return False, f"{field_name} maksimal 50 karakter"
    
    # Izinkan huruf, spasi, dan karakter nama umum
    name_pattern = r'^[a-zA-Z\s\'-]+$'
    if not re.match(name_pattern, name):
        return False, f"{field_name} hanya boleh mengandung huruf, spasi, apostrof, dan tanda hubung"
    
    return True, ""

def validate_password(password: str) -> Tuple[bool, str]:
    """Validasi persyaratan kekuatan kata sandi"""
    if len(password) < 8:
        return False, "Kata sandi minimal 8 karakter"
    if not any(c.isupper() for c in password):
        return False, "Kata sandi harus mengandung huruf besar"
    if not any(c.islower() for c in password):
        return False, "Kata sandi harus mengandung huruf kecil"  
    if not any(c.isdigit() for c in password):
        return False, "Kata sandi harus mengandung angka"
    return True, ""

def check_rate_limit(user_email: str) -> bool:
    """Periksa apakah pengguna telah melebihi batas laju untuk percobaan login"""
    now = datetime.now()
    rate_limit_key = f'ratelimit_{user_email}'
    attempts = st.session_state.get(rate_limit_key, [])

    # Hapus percobaan di luar jendela
    valid_attempts = [
        attempt for attempt in attempts 
        if (now - attempt) < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]

    if len(valid_attempts) >= MAX_LOGIN_ATTEMPTS:
        return False

    valid_attempts.append(now)
    st.session_state[rate_limit_key] = valid_attempts
    return True

def check_session_timeout() -> bool:
    """Periksa apakah sesi pengguna telah kedaluwarsa"""
    if 'login_time' in st.session_state and st.session_state['login_time']:
        elapsed = (datetime.now() - st.session_state['login_time']).total_seconds()
        if elapsed > SESSION_TIMEOUT:
            logout()
            return False
    return True

def check_email_verification_quota() -> Tuple[bool, str]:
    """Periksa kuota verifikasi email untuk mencegah spam"""
    try:
        now = datetime.now()
        quota_key = 'email_verification_attempts'
        attempts = st.session_state.get(quota_key, [])
        
        # Hapus upaya yang lebih dari 1 jam
        valid_attempts = [
            attempt for attempt in attempts 
            if (now - attempt) < timedelta(hours=1)
        ]
        
        if len(valid_attempts) >= EMAIL_VERIFICATION_LIMIT:
            return False, "Batas pengiriman email tercapai. Silakan coba lagi dalam 1 jam."
        
        valid_attempts.append(now)
        st.session_state[quota_key] = valid_attempts
        return True, ""
        
    except Exception as e:
        logger.error(f"Error checking email quota: {e}")
        return False, "Error checking email quota"

# =============================================================================
# FIREBASE FUNCTIONS
# =============================================================================

def initialize_firebase() -> Tuple[Optional[Any], Optional[Any]]:
    """Inisialisasi Firebase Admin SDK dan Pyrebase"""
    try:
        # Cek apakah Firebase sudah diinisialisasi sebelumnya
        if st.session_state.get('firebase_initialized', False):
            firebase_auth = st.session_state.get('firebase_auth')
            firestore_client = st.session_state.get('firestore')
            
            if firebase_auth and firestore_client:
                logger.info("Using existing Firebase initialization")
                return firebase_auth, firestore_client
            else:
                logger.warning("Firebase objects invalid, reinitializing...")
                st.session_state['firebase_initialized'] = False

        # Verifikasi environment dan konfigurasi
        if not verify_environment():
            logger.error("Environment verification failed")
            return None, None
            
        # Periksa konfigurasi Firebase
        if "firebase" not in st.secrets:
            logger.critical("Firebase configuration not found in secrets")
            st.error("🔥 Konfigurasi Firebase tidak ditemukan!")
            return None, None
        
        # Ambil konfigurasi service account
        service_account = dict(st.secrets["firebase"])
        
        # Periksa field yang diperlukan
        required_fields = ["project_id", "client_email", "private_key"]
        missing_fields = [field for field in required_fields if field not in service_account]
        
        if missing_fields:
            logger.critical(f"Missing Firebase config fields: {missing_fields}")
            st.error(f"🔥 Konfigurasi Firebase tidak lengkap! Field yang diperlukan: {', '.join(missing_fields)}")
            return None, None
        
        # Inisialisasi Firebase Admin SDK
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully")
        
        # Konfigurasi Pyrebase
        config = get_firebase_config()
        if not config:
            logger.error("Failed to get Firebase configuration")
            return None, None
        
        # Inisialisasi Pyrebase
        pb = pyrebase.initialize_app(config)
        firebase_auth = pb.auth()
        logger.info("Pyrebase initialized successfully")
        
        # Inisialisasi Firestore client
        firestore_client = firestore.client()
        logger.info("Firestore client initialized successfully")
        
        # Simpan ke session state
        st.session_state['firebase_auth'] = firebase_auth
        st.session_state['firestore'] = firestore_client
        st.session_state['firebase_initialized'] = True
        
        logger.info("Firebase initialized successfully")
        return firebase_auth, firestore_client
        
    except Exception as e:
        logger.critical(f"Firebase initialization failed: {str(e)}")
        st.error(f"Gagal menginisialisasi Firebase: {str(e)}")
        return None, None

def send_email_verification_safe(firebase_auth: Any, id_token: str, email: str) -> Tuple[bool, str]:
    """Kirim verifikasi email dengan penanganan error yang komprehensif"""
    try:
        # Periksa kuota terlebih dahulu
        can_send, quota_message = check_email_verification_quota()
        if not can_send:
            return False, quota_message
        
        # Kirim verifikasi email
        firebase_auth.send_email_verification(id_token)
        logger.info(f"Email verification sent to: {email}")
        return True, "Email verifikasi berhasil dikirim"
        
    except Exception as e:
        logger.error(f"Failed to send email verification to {email}: {e}")
        toast_msg, detailed_msg = handle_firebase_error(e, "email_verification")
        return False, detailed_msg

def verify_user_exists(user_email: str, firestore_client: Any) -> bool:
    """Verifikasi bahwa pengguna ada dan memiliki data yang valid di Firestore"""
    try:
        firebase_user = auth.get_user_by_email(user_email)
        user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
        
        if user_doc.exists:
            logger.info(f"User {user_email} verified successfully")
            return True
        
        logger.warning(f"User {user_email} has no Firestore data")
        return False

    except auth.UserNotFoundError:
        logger.warning(f"User {user_email} not found in Firebase Auth")
        return False
    except Exception as e:
        logger.error(f"Error verifying user {user_email}: {str(e)}")
        return False

# =============================================================================
# GOOGLE OAUTH FUNCTIONS
# =============================================================================

def generate_popup_oauth_html(google_url: str) -> str:
    """Generate HTML dengan JavaScript untuk Google OAuth pop-up yang advanced"""
    return f"""
    <div id="oauth-container" style="text-align: center; padding: 20px;">
        <div id="oauth-status" style="
            padding: 15px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border-radius: 10px; 
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        ">
            <h3 style="margin: 0 0 10px 0;">🔐 Google OAuth Login</h3>
            <p id="status-text" style="margin: 0;">Siap untuk membuka jendela login Google</p>
        </div>
        
        <button id="popup-login-btn" onclick="startGoogleLogin()" style="
            background: linear-gradient(135deg, #4285f4 0%, #1a73e8 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(66, 133, 244, 0.3);
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-2px)'" 
           onmouseout="this.style.transform='translateY(0px)'">
            🚀 Buka Login Google
        </button>
        
        <div id="manual-link" style="margin-top: 15px; display: none;">
            <p style="color: #666; font-size: 14px;">Jendela pop-up diblokir? Klik link di bawah:</p>
            <a href="{google_url}" target="_blank" style="
                color: #1a73e8; 
                text-decoration: none; 
                font-weight: bold;
                padding: 8px 16px;
                border: 2px solid #1a73e8;
                border-radius: 20px;
                display: inline-block;
                transition: all 0.3s ease;
            " onmouseover="this.style.backgroundColor='#1a73e8'; this.style.color='white'"
               onmouseout="this.style.backgroundColor='transparent'; this.style.color='#1a73e8'">
                🔗 Manual Login Link
            </a>
        </div>
    </div>

    <script>
        let oauthPopup = null;
        let loginCheckInterval = null;
        let popupCheckInterval = null;
        
        // Status update function
        function updateStatus(text, isError = false, isSuccess = false) {{
            const statusDiv = document.getElementById('oauth-status');
            const statusText = document.getElementById('status-text');
            
            statusText.textContent = text;
            
            if (isError) {{
                statusDiv.style.background = 'linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%)';
            }} else if (isSuccess) {{
                statusDiv.style.background = 'linear-gradient(135deg, #51cf66 0%, #40c057 100%)';
            }} else {{
                statusDiv.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            }}
        }}
        
        // Start Google login process
        function startGoogleLogin() {{
            updateStatus('🔄 Membuka jendela login Google...');
            
            const button = document.getElementById('popup-login-btn');
            button.disabled = true;
            button.textContent = '⏳ Membuka Pop-up...';
            
            // Pop-up window specifications
            const popupWidth = 500;
            const popupHeight = 600;
            const left = (screen.width - popupWidth) / 2;
            const top = (screen.height - popupHeight) / 2;
            
            const popupFeatures = `width=${{popupWidth}},height=${{popupHeight}},left=${{left}},top=${{top}},scrollbars=yes,resizable=yes,status=no,location=no,toolbar=no,menubar=no`;
            
            try {{
                // Open pop-up window
                oauthPopup = window.open('{google_url}', 'googleOAuth', popupFeatures);
                
                if (!oauthPopup || oauthPopup.closed || typeof oauthPopup.closed == 'undefined') {{
                    // Pop-up blocked
                    updateStatus('❌ Pop-up diblokir! Gunakan link manual di bawah.', true);
                    document.getElementById('manual-link').style.display = 'block';
                    resetButton();
                    return;
                }}
                
                updateStatus('✅ Jendela login terbuka. Silakan login di jendela tersebut.');
                
                // Start monitoring the pop-up
                startPopupMonitoring();
                
            }} catch (error) {{
                console.error('Error opening popup:', error);
                updateStatus('❌ Gagal membuka jendela login. Gunakan link manual.', true);
                document.getElementById('manual-link').style.display = 'block';
                resetButton();
            }}
        }}
        
        // Monitor pop-up window
        function startPopupMonitoring() {{
            let checkCount = 0;
            const maxChecks = 300; // 5 minutes maximum
            
            popupCheckInterval = setInterval(() => {{
                checkCount++;
                
                try {{
                    // Check if popup is closed
                    if (oauthPopup.closed) {{
                        clearInterval(popupCheckInterval);
                        handlePopupClosed();
                        return;
                    }}
                    
                    // Try to access popup URL (will succeed after OAuth redirect)
                    try {{
                        const popupUrl = oauthPopup.location.href;
                        
                        // Check if we're back to our callback URL
                        if (popupUrl.includes('/oauth2callback') || popupUrl.includes('?code=')) {{
                            clearInterval(popupCheckInterval);
                            handleOAuthSuccess();
                            return;
                        }}
                    }} catch (e) {{
                        // Cross-origin error is expected during OAuth flow
                        // Continue monitoring
                    }}
                    
                    // Timeout check
                    if (checkCount >= maxChecks) {{
                        clearInterval(popupCheckInterval);
                        updateStatus('⏰ Login timeout. Silakan coba lagi.', true);
                        if (oauthPopup && !oauthPopup.closed) {{
                            oauthPopup.close();
                        }}
                        resetButton();
                    }}
                    
                }} catch (error) {{
                    // Popup might be closed or inaccessible
                    clearInterval(popupCheckInterval);
                    handlePopupClosed();
                }}
            }}, 1000); // Check every second
        }}
        
        // Handle successful OAuth
        function handleOAuthSuccess() {{
            updateStatus('🎉 Login berhasil! Mengalihkan ke dashboard...', false, true);
            
            if (oauthPopup && !oauthPopup.closed) {{
                oauthPopup.close();
            }}
            
            // Wait a moment then refresh the parent page
            setTimeout(() => {{
                window.location.reload();
            }}, 2000);
        }}
        
        // Handle popup closed
        function handlePopupClosed() {{
            updateStatus('❌ Jendela login ditutup. Silakan coba lagi atau gunakan link manual.', true);
            document.getElementById('manual-link').style.display = 'block';
            resetButton();
        }}
        
        // Reset button state
        function resetButton() {{
            const button = document.getElementById('popup-login-btn');
            button.disabled = false;
            button.textContent = '🚀 Buka Login Google';
        }}
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {{
            if (oauthPopup && !oauthPopup.closed) {{
                oauthPopup.close();
            }}
            if (popupCheckInterval) {{
                clearInterval(popupCheckInterval);
            }}
        }});
        
        // Listen for messages from popup (alternative method)
        window.addEventListener('message', (event) => {{
            if (event.data && event.data.type === 'OAUTH_SUCCESS') {{
                handleOAuthSuccess();
            }} else if (event.data && event.data.type === 'OAUTH_ERROR') {{
                updateStatus('❌ Login gagal: ' + (event.data.error || 'Unknown error'), true);
                resetButton();
            }}
        }});
    </script>
    """

def get_google_authorization_url() -> str:
    """Hasilkan URL otorisasi Google OAuth dengan cakupan yang diperlukan"""
    base_url = 'https://accounts.google.com/o/oauth2/v2/auth'
    params = {
        'client_id': st.secrets.get("GOOGLE_CLIENT_ID", ""),
        'redirect_uri': get_redirect_uri(),
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    return f"{base_url}?{urlencode(params)}"

async def exchange_google_token(code: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Tukar kode otorisasi Google untuk informasi pengguna"""
    async with httpx.AsyncClient() as client:
        token_url = 'https://oauth2.googleapis.com/token'
        payload = {
            'client_id': st.secrets.get("GOOGLE_CLIENT_ID", ""),
            'client_secret': st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': get_redirect_uri()
        }

        try:
            response = await client.post(token_url, data=payload)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get('access_token')
            
            if not access_token:
                logger.error("No access token received from Google")
                return None, None
            
            # Get user info
            userinfo_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
            headers = {'Authorization': f'Bearer {access_token}'}
            
            user_response = await client.get(userinfo_url, headers=headers)
            user_response.raise_for_status()
            
            user_info = user_response.json()
            logger.info(f"Google OAuth successful for: {user_info.get('email', 'unknown')}")
            
            return access_token, user_info
            
        except Exception as e:
            logger.error(f"Google token exchange failed: {str(e)}")
            return None, None
    """Create a callback page that communicates with parent window"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OAuth Callback</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .callback-container {
                text-align: center;
                background: rgba(255,255,255,0.1);
                padding: 2rem;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            }
            .loading {
                font-size: 2rem;
                margin-bottom: 1rem;
            }
            .message {
                font-size: 1.2rem;
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        <div class="callback-container">
            <div class="loading">⏳</div>
            <div class="message">Memproses login...</div>
        </div>
        
        <script>
            // Extract code from URL
            const urlParams = new URLSearchParams(window.location.search);
            const code = urlParams.get('code');
            const error = urlParams.get('error');
            
            if (error) {
                // Send error to parent
                if (window.opener) {
                    window.opener.postMessage({
                        type: 'OAUTH_ERROR',
                        error: error
                    }, '*');
                }
                window.close();
            } else if (code) {
                // Send success to parent
                if (window.opener) {
                    window.opener.postMessage({
                        type: 'OAUTH_SUCCESS',
                        code: code
                    }, '*');
                }
                
                // Update UI and close
                document.querySelector('.loading').textContent = '✅';
                document.querySelector('.message').textContent = 'Login berhasil! Menutup jendela...';
                
                setTimeout(() => {
                    window.close();
                }, 1500);
            } else {
                // No code or error - something went wrong
                if (window.opener) {
                    window.opener.postMessage({
                        type: 'OAUTH_ERROR',
                        error: 'No authorization code received'
                    }, '*');
                }
                window.close();
            }
        </script>
    </body>
    </html>
    """
    """Tukar kode otorisasi Google untuk informasi pengguna"""
    async with httpx.AsyncClient() as client:
        token_url = 'https://oauth2.googleapis.com/token'
        payload = {
            'client_id': st.secrets.get("GOOGLE_CLIENT_ID", ""),
            'client_secret': st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': get_redirect_uri()
        }

        try:
            # Tukar kode untuk token
            token_response = await client.post(token_url, data=payload)
            token_data = token_response.json()
            
            if 'access_token' not in token_data:
                logger.error(f"Token exchange failed: {token_data}")
                return None, None
            
            # Gunakan token untuk mendapatkan info pengguna
            user_info_url = f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token_data['access_token']}"
            user_response = await client.get(user_info_url)
            user_info = user_response.json()
            
            if 'email' not in user_info:
                logger.error(f"User info incomplete: {user_info}")
                return None, None
                
            return user_info['email'], user_info

        except Exception as e:
            logger.error(f"Google token exchange error: {e}")
            return None, None

def handle_google_login_callback() -> bool:
    """Tangani callback Google OAuth setelah autentikasi pengguna dengan progress feedback"""
    try:
        if 'code' not in st.query_params:
            return True  # Tidak ada callback Google, lanjutkan normal
            
        code = st.query_params.get('code')
        if not code or not isinstance(code, str):
            st.error("Kode otorisasi Google tidak valid")
            return False

        # Tampilkan progress untuk Google callback processing
        callback_progress = st.empty()
        callback_message = st.empty()
        
        with callback_progress.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            # Step 1: Token exchange
            progress_container.progress(0.2)
            message_container.caption("🔄 Memproses token Google...")
            
            async def async_token_exchange():
                return await exchange_google_token(code)

            user_email, user_info = asyncio.run(async_token_exchange())
            if not user_email or not user_info:
                progress_container.empty()
                message_container.error("❌ Gagal mendapatkan informasi pengguna dari Google")
                show_error_toast("Gagal memproses login Google")
                return False

            # Step 2: Firebase initialization
            progress_container.progress(0.4)
            message_container.caption("🔥 Menginisialisasi Firebase...")
            
            # Verifikasi pengguna ada di sistem
            firebase_auth, firestore_client = initialize_firebase()
            if not firebase_auth or not firestore_client:
                progress_container.empty()
                message_container.error("❌ Gagal menginisialisasi Firebase")
                show_error_toast("Gagal menginisialisasi sistem")
                return False
            
            # Step 3: User verification
            progress_container.progress(0.6)
            message_container.caption("👤 Memverifikasi pengguna...")
            
            try:
                # Cek apakah user sudah terdaftar
                firebase_user = auth.get_user_by_email(user_email)
                user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
                
                if user_doc.exists:
                    # Step 4: Processing login
                    progress_container.progress(0.8)
                    message_container.caption("🔐 Memproses login...")
                    
                    # User ada, cek verifikasi email untuk keamanan ekstra
                    user_data = user_doc.to_dict()
                    is_google_user = user_data.get('auth_provider') == 'google'
                    is_email_verified = user_data.get('email_verified', False)
                    
                    # User Google atau email sudah verified, login berhasil
                    if is_google_user or is_email_verified:
                        # Update status verifikasi untuk user Google jika belum ter-set
                        if is_google_user and not is_email_verified:
                            try:
                                firestore_client.collection('users').document(firebase_user.uid).update({
                                    'email_verified': True,
                                    'last_login': datetime.now().isoformat()
                                })
                                logger.info(f"Updated email verification status for Google user: {user_email}")
                            except Exception as update_error:
                                logger.warning(f"Failed to update email verification for Google user {user_email}: {update_error}")
                        
                        # Step 5: Complete
                        progress_container.progress(1.0)
                        message_container.caption("✅ Login Google berhasil!")
                        
                        st.session_state['logged_in'] = True
                        st.session_state['user_email'] = user_email
                        st.session_state['login_time'] = datetime.now()
                        set_remember_me_cookies(user_email, True)
                        
                        logger.info(f"Google login successful for: {user_email}")
                        
                        # Clear progress dan tampilkan pesan sukses
                        time.sleep(1.0)
                        progress_container.empty()
                        message_container.success("🎉 Login Google berhasil! Selamat datang!")
                        show_success_toast("Login Google berhasil!")
                        
                        time.sleep(1.0)  # Beri waktu untuk membaca pesan
                        callback_progress.empty()
                        st.rerun()
                        return True
                    else:
                        # Email belum diverifikasi untuk user non-Google
                        progress_container.empty()
                        message_container.warning(
                            f"📧 **Email Anda belum diverifikasi!**\n\n"
                            f"Email {user_email} belum diverifikasi. "
                            f"Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim."
                        )
                        show_warning_toast("Email belum diverifikasi")
                        
                        # Simpan error di session state untuk ditampilkan di feedback_placeholder
                        st.session_state['google_auth_verification_error'] = True
                        st.session_state['google_auth_email'] = user_email
                        st.query_params.clear()
                        time.sleep(2.0)
                        callback_progress.empty()
                        return False
                else:
                    # User tidak ada di Firestore, arahkan ke registrasi
                    progress_container.empty()
                    message_container.error(
                        f"**Akun Google Tidak Terdaftar**\n\n"
                        f"Akun Google {user_email} belum terdaftar dalam sistem kami."
                    )
                    message_container.info(
                        "💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                        "atau gunakan akun email yang sudah terdaftar."
                    )
                    show_error_toast(f"Akun Google {user_email} tidak terdaftar")
                    
                    st.session_state['google_auth_error'] = True
                    st.session_state['google_auth_email'] = user_email
                    st.query_params.clear()
                    time.sleep(2.0)
                    callback_progress.empty()
                    return False

            except auth.UserNotFoundError:
                # User tidak ada di Firebase Auth, arahkan ke registrasi
                progress_container.empty()
                message_container.error(
                    f"**Akun Google Tidak Terdaftar**\n\n"
                    f"Akun Google {user_email} belum terdaftar dalam sistem kami."
                )
                message_container.info(
                    "💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                    "atau gunakan akun email yang sudah terdaftar."
                )
                show_error_toast(f"Akun Google {user_email} tidak terdaftar")
                
                st.session_state['google_auth_error'] = True
                st.session_state['google_auth_email'] = user_email
                st.query_params.clear()
                time.sleep(2.0)
                callback_progress.empty()
                return False

    except Exception as e:
        logger.error(f"Google login callback error: {str(e)}")
        if 'callback_progress' in locals():
            callback_progress.empty()
        st.error("❌ Terjadi kesalahan saat memproses login Google")
        show_error_toast("Terjadi kesalahan saat memproses login Google")
        if 'logged_in' in st.session_state:
            st.session_state['logged_in'] = False
        return False

# =============================================================================
# AUTHENTICATION FUNCTIONS
# =============================================================================

def sync_email_verified_to_firestore(firebase_auth, firestore_client, user):
    """Ambil status email_verified dari Firebase Auth dan update ke Firestore."""
    try:
        # Refresh token untuk mendapatkan data terbaru
        firebase_auth.refresh(user['refreshToken'])
        user_info = firebase_auth.get_account_info(user['idToken'])
        email_verified = user_info['users'][0].get('emailVerified', False)
        
        # Update status ke Firestore
        firestore_client.collection('users').document(user['localId']).update({'email_verified': email_verified})
        
        logger.info(f"Email verification status synced for user {user.get('email', 'unknown')}: {email_verified}")
        return email_verified
    except Exception as e:
        logger.error(f"Gagal sync email_verified: {e}")
        return False

def login_user(email: str, password: str, firebase_auth: Any, firestore_client: Any, 
               remember: bool, progress_container: Any, message_container: Any) -> bool:
    """Proses login pengguna dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    # Validasi format email
    is_valid_email, email_message = validate_email_format(email)
    if not is_valid_email:
        progress_container.empty()
        show_error_toast("Format email tidak valid")
        message_container.error(email_message)
        return False
    
    # Cek rate limiting
    if not check_rate_limit(email):
        progress_container.empty()
        show_error_toast("Terlalu banyak percobaan")
        message_container.error("Terlalu banyak percobaan login. Silakan coba lagi nanti.")
        return False
    
    try:
        # Step 1: Validating credentials
        progress_container.progress(0.2)
        message_container.caption("🔐 Memvalidasi kredensial...")
        
        # Coba login dengan Firebase
        user = firebase_auth.sign_in_with_email_and_password(email, password)
        
        # Step 2: Checking email verification status
        progress_container.progress(0.5)
        message_container.caption("⚠️ Memeriksa status verifikasi email...")
        
        # Sync dan cek status verifikasi email dari Firebase Auth
        email_verified = sync_email_verified_to_firestore(firebase_auth, firestore_client, user)
        
        if not email_verified:
            progress_container.empty()
            show_warning_toast("Email belum diverifikasi")
            message_container.warning(
                f"📧 **Email Anda belum diverifikasi!**\n\n"
                f"Email {email} belum diverifikasi. "
                f"Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim. "
                f"Setelah verifikasi, silakan coba login kembali.\n\n"
                f"💡 *Tip: Periksa juga folder spam/junk email*"
            )
            return False
        
        # Step 3: Verifying user data
        progress_container.progress(0.7)
        message_container.caption("👤 Memverifikasi data pengguna...")
        
        # Verifikasi pengguna ada di Firestore
        if not verify_user_exists(email, firestore_client):
            progress_container.empty()
            show_error_toast("Data pengguna tidak ditemukan")
            message_container.error("Data pengguna tidak ditemukan di sistem. Silakan hubungi administrator.")
            return False
        # Step 3: Setting up session
        progress_container.progress(0.9)
        message_container.caption("⚙️ Menyiapkan sesi pengguna...")
        
        # Set status login
        st.session_state['logged_in'] = True
        st.session_state['user_email'] = email
        st.session_state['login_time'] = datetime.now()
        st.session_state['login_attempts'] = 0
        
        # Set cookies
        set_remember_me_cookies(email, remember)
        
        # Step 4: Complete
        progress_container.progress(1.0)
        # message_container.caption("✅ Login berhasil!")
        message_container.success("🎉 Login berhasil! Selamat datang kembali!")
        
        logger.info(f"Login successful for: {email}")
        show_success_toast("Login berhasil! Selamat datang kembali!")
        
        # Clear progress setelah menampilkan pesan sukses, tapi biarkan message tetap
        time.sleep(1.2)  # Beri waktu untuk menampilkan progress completion
        progress_container.empty()
        message_container.empty()
        return True
        
    except Exception as e:
        st.session_state['login_attempts'] = st.session_state.get('login_attempts', 0) + 1
        
        # Enhanced error handling untuk memberikan pesan yang lebih spesifik
        error_str = str(e).upper()
        
        # Khusus untuk INVALID_LOGIN_CREDENTIALS, berikan pesan seperti Google OAuth
        if "INVALID_LOGIN_CREDENTIALS" in error_str:
            progress_container.empty()
            
            # Tampilkan pesan error langsung di message_container
            show_error_toast(f"Email {email} tidak terdaftar dalam sistem kami")
            message_container.error(
                f"**Akun Email Tidak Terdaftar**\n\n"
                f"Email {email} belum terdaftar dalam sistem kami."
            )
            message_container.info(
                f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                f"atau periksa ejaan email Anda."
            )
            
            # Log error
            logger.warning(f"Login failed - email not registered: {email}")
            return False
        else:
            # Gunakan centralized error handling untuk error lainnya
            show_error_with_context(e, "login", progress_container, message_container)
            return False

def register_user(first_name: str, last_name: str, email: str, password: str, 
                 firebase_auth: Any, firestore_client: Any, is_google: bool, 
                 progress_container: Any, message_container: Any) -> Tuple[bool, str]:
    """Proses registrasi pengguna dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    # Step 1: Input validation
    progress_container.progress(0.1)
    message_container.caption("📝 Memvalidasi data input...")
    
    # Validasi input
    validation_errors = []
    
    # Validasi nama
    is_valid_fname, fname_message = validate_name_format(first_name.strip(), "Nama Depan")
    if not is_valid_fname:
        validation_errors.append(f"❌ {fname_message}")
        
    is_valid_lname, lname_message = validate_name_format(last_name.strip(), "Nama Belakang")
    if not is_valid_lname:
        validation_errors.append(f"❌ {lname_message}")
    
    # Validasi email
    is_valid_email, email_message = validate_email_format(email.strip())
    if not is_valid_email:
        validation_errors.append(f"❌ {email_message}")
    
    # Validasi password (hanya untuk registrasi non-Google)
    if not is_google:
        is_valid_password, password_message = validate_password(password)
        if not is_valid_password:
            validation_errors.append(f"❌ {password_message}")
    
    if validation_errors:
        handle_validation_errors(validation_errors, progress_container, message_container)
        return False, "\n".join(validation_errors)
    
    try:
        # Step 2: Checking email availability
        progress_container.progress(0.3)
        message_container.caption("📧 Memeriksa ketersediaan email...")
        
        # Cek apakah email sudah terdaftar
        try:
            existing_user = auth.get_user_by_email(email)
            progress_container.empty()
            message_container.error("❌ Email ini sudah terdaftar. Silakan gunakan email lain atau login dengan akun yang ada.")
            show_error_toast("Email sudah terdaftar")
            return False, "❌ Email ini sudah terdaftar. Silakan gunakan email lain atau login dengan akun yang ada."
        except auth.UserNotFoundError:
            pass  # Email belum terdaftar, lanjutkan
        
        # Step 3: Creating Firebase account
        progress_container.progress(0.5)
        message_container.caption("🔐 Membuat akun Firebase...")
        
        # Buat user di Firebase Auth
        if is_google:
            auto_password = f"Google-{secrets.token_hex(8)}"
            user = firebase_auth.create_user_with_email_and_password(email, auto_password)
        else:
            user = firebase_auth.create_user_with_email_and_password(email, password)
        
        # Step 4: Email verification
        progress_container.progress(0.7)
        if not is_google:
            message_container.caption("📬 Mengirim email verifikasi...")
        else:
            message_container.caption("✅ Memproses akun Google...")
        
        # Kirim email verifikasi untuk registrasi non-Google
        email_verification_sent = False
        if not is_google:
            verification_success, verification_message = send_email_verification_safe(
                firebase_auth, user['idToken'], email
            )
            email_verification_sent = verification_success
        
        # Step 5: Saving user data
        progress_container.progress(0.9)
        message_container.caption("💾 Menyimpan data pengguna...")
        
        # Simpan data user ke Firestore
        user_data = {
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "email": email.strip(),
            "auth_provider": "google" if is_google else "email",
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat(),
            "email_verified": is_google
        }
        
        firestore_client.collection('users').document(user['localId']).set(user_data)
        
        # Step 6: Complete
        progress_container.progress(1.0)
        message_container.caption("✅ Registrasi berhasil!")
        
        if is_google:
            success_message = "🎉 Akun Google berhasil didaftarkan! Anda sekarang dapat login dan menggunakan semua fitur aplikasi."
        else:
            if email_verification_sent:
                success_message = f"✅ Akun berhasil dibuat untuk {email}!\n\n📧 Email verifikasi telah dikirim. Silakan periksa kotak masuk (dan folder spam) untuk mengaktifkan akun Anda."
            else:
                success_message = f"✅ Akun berhasil dibuat untuk {email}! Anda dapat login sekarang dan mulai menggunakan aplikasi."
        
        message_container.success(success_message)
        logger.info(f"Successfully created account for: {email}")
        show_success_toast("Registrasi berhasil")
        
        # Clear progress setelah menampilkan pesan sukses, tapi biarkan message tetap
        time.sleep(1.2)
        progress_container.empty()
        
        return True, success_message
                
    except Exception as e:
        show_error_with_context(e, "register", progress_container, message_container)
        return False, f"❌ Pendaftaran gagal: {str(e)}"

def reset_password(email: str, firebase_auth: Any, progress_container: Any, message_container: Any) -> bool:
    """Proses reset password dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    # Step 1: Input validation
    progress_container.progress(0.2)
    message_container.caption("📝 Memvalidasi alamat email...")
    
    # Validasi format email
    is_valid_email, email_message = validate_email_format(email.strip())
    if not is_valid_email:
        progress_container.empty()
        message_container.error(f"❌ {email_message}")
        show_error_toast("Format email tidak valid")
        return False
    
    # Step 2: Rate limiting check
    progress_container.progress(0.4)
    message_container.caption("🔒 Memeriksa batas permintaan...")
    
    # Cek rate limiting
    if not check_rate_limit(f"reset_{email}"):
        progress_container.empty()
        message_container.error("⚠️ Terlalu banyak percobaan reset password. Silakan tunggu 5 menit sebelum mencoba lagi.")
        show_warning_toast("Terlalu banyak percobaan reset")
        return False
    
    try:
        # Step 3: Checking user existence
        progress_container.progress(0.6)
        message_container.caption("👤 Memeriksa keberadaan akun...")
        
        # Cek apakah user ada
        try:
            auth.get_user_by_email(email)
        except auth.UserNotFoundError:
            progress_container.empty()
            message_container.error("❌ Tidak ada akun yang ditemukan dengan alamat email ini.")
            show_error_toast("❌ Akun tidak ditemukan!")
            return False
        
        # Step 4: Sending reset email
        progress_container.progress(0.8)
        message_container.caption("📧 Mengirim email reset password...")
        
        # Kirim email reset password
        firebase_auth.send_password_reset_email(email)
        logger.info(f"Password reset email sent to: {email}")
        
        # Step 5: Complete
        progress_container.progress(1.0)
        message_container.caption("✅ Email reset berhasil dikirim!")
        
        success_message = f"📧 **Petunjuk reset password telah dikirim ke {email}**\n\nSilakan periksa kotak masuk email Anda (dan folder spam) untuk link reset password.\n\nLink akan aktif selama 1 jam."
        message_container.success(success_message)
        
        show_success_toast("Link reset password berhasil dikirim")
        
        # Clear progress setelah menampilkan pesan sukses, tapi biarkan message tetap
        time.sleep(1.2)
        progress_container.empty()
        return True
        
    except Exception as e:
        show_error_with_context(e, "reset", progress_container, message_container)
        return False

def logout() -> None:
    """Tangani logout pengguna dengan pembersihan sesi"""
    try:
        user_email = st.session_state.get('user_email')
        logger.info(f"Logging out user: {user_email}")
        
        # Simpan objek firebase yang diperlukan
        fb_auth = st.session_state.get('firebase_auth', None)
        fs_client = st.session_state.get('firestore', None)
        fb_initialized = st.session_state.get('firebase_initialized', False)
        
        # Bersihkan session state
        st.session_state.clear()
        
        # Kembalikan objek firebase jika ada
        if fb_auth:
            st.session_state['firebase_auth'] = fb_auth
        if fs_client:
            st.session_state['firestore'] = fs_client
        if fb_initialized:
            st.session_state['firebase_initialized'] = fb_initialized
        
        # Reset status login
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = None
        st.session_state["logout_success"] = True
        
        # Clear URL params
        st.query_params.clear()
        
        # Clear cookies but keep last_email for convenience
        clear_remember_me_cookies()
        
        logger.info("Logout completed successfully")
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        show_error_toast(f"Logout failed: {str(e)}")

def show_toast_notification(message: str, icon: str = "ℹ") -> None:
    """Tampilkan notifikasi toast dengan gaya yang konsisten"""
    try:
        st.toast(message, icon=icon)
    except Exception as e:
        logger.error(f"Failed to show toast: {e}")
        st.info(f"{icon} {message}")

# Toast helper functions
def show_success_toast(message: str) -> None:
    """Tampilkan notifikasi toast sukses"""
    show_toast_notification(message, "✅")

def show_error_toast(message: str) -> None:
    """Tampilkan notifikasi toast error"""
    show_toast_notification(message, "❌")

def show_warning_toast(message: str) -> None:
    """Tampilkan notifikasi toast peringatan"""
    show_toast_notification(message, "⚠️")

def display_auth_tips(auth_type: str) -> None:
    """Tampilkan tips berguna berdasarkan jenis autentikasi"""
    tips = {
        "login": [
            "💡 Gunakan fitur 'Ingat Saya' untuk login otomatis",
            "🔒 Pastikan kata sandi Anda aman dan unik",
            "📱 Gunakan login Google untuk kemudahan akses"
        ],
        "register": [
            "📧 Periksa email spam jika verifikasi tidak diterima",
            "🔐 Gunakan kata sandi yang kuat: 8+ karakter, angka, simbol",
            "✅ Login Google lebih cepat dan aman"
        ],
        "reset": [
            "📧 Link reset berlaku selama 1 jam",
            "🗂️ Periksa folder spam/junk email",
            "⏰ Tunggu 5 menit sebelum meminta link baru"
        ]
    }
    
    if auth_type in tips:
        with st.expander("💡 Tips Berguna", expanded=False):
            for tip in tips[auth_type]:
                st.markdown(f"• {tip}")

# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_login_form(firebase_auth: Any, firestore_client: Any) -> None:
    """Tampilkan dan tangani formulir login"""
    
    # Check app readiness
    app_ready = is_app_ready()
    
    # Initialize feedback containers untuk layout stability
    feedback_placeholder = st.empty()
    progress_container = None
    message_container = None


    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Masuk")

        # Input email dengan nilai yang diingat
        remembered_email = get_remembered_email()
        email = st.text_input(
            "Email",
            value=remembered_email,
            placeholder="email.anda@contoh.com",
            help="Masukkan alamat email terdaftar Anda",
            disabled=not app_ready
        )

        # Validasi email secara real-time (dengan debouncing dan app readiness check)
        if app_ready and email and email.strip() and email != remembered_email and len(email.strip()) > 5:
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                st.error(f"❌ {email_message}")  # Tetap gunakan st.error untuk real-time validation

        # Input password
        password = st.text_input(
            "Kata Sandi",
            type="password",
            placeholder="Masukkan kata sandi Anda",
            help="Masukkan kata sandi yang aman",
            disabled=not app_ready
        )

        # Checkbox remember me
        col1, col2 = st.columns([1, 2])
        with col1:
            remember = st.checkbox(
                "Ingat saya", 
                value=True, 
                help=f"Simpan login selama {REMEMBER_ME_DURATION // (24*60*60)} hari",
                disabled=not app_ready
            )

        # Status app readiness
        if not app_ready:
            st.info("🔄 Sistem sedang mempersiapkan diri... Mohon tunggu sebentar.")

        # Tombol login email
        email_login_clicked = st.form_submit_button(
            "Lanjutkan dengan Email", 
            use_container_width=True, 
            type="primary",
            disabled=not app_ready
        )

        # Divider
        st.markdown("""
            <div class='auth-divider-custom'>
                <div class='divider-line-custom'></div>
                <span class='divider-text-custom'>ATAU</span>
                <div class='divider-line-custom'></div>
            </div>
        """, unsafe_allow_html=True)

        # Tombol login Google
        google_login_clicked = st.form_submit_button(
            "Lanjutkan dengan Google", 
            use_container_width=True, 
            type="primary",
            disabled=not app_ready
        )

        # Placeholder untuk pesan feedback dan progress di bawah tombol Google
        # Gunakan single placeholder dengan containers untuk konsistensi layout
        feedback_placeholder = st.empty()
        
        # Pre-allocate containers untuk mencegah layout shift
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()


    # Tampilkan pesan error Google OAuth jika ada - menggunakan feedback placeholder
    if st.session_state.get('google_auth_error', False):
        email_error = st.session_state.get('google_auth_email', '')
        with feedback_placeholder.container():
            progress_container.empty()  # Clear any existing progress
            message_container.error(f"**Akun Google Tidak Terdaftar**\n\n"
                    f"Akun Google {email_error} belum terdaftar dalam sistem kami.")
            st.info(f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' atau gunakan akun email yang sudah terdaftar.")
        show_error_toast(f"Akun Google {email_error} tidak terdaftar dalam sistem kami.")
        del st.session_state['google_auth_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']

    # Tampilkan pesan error verifikasi Google OAuth jika ada - menggunakan feedback placeholder
    if st.session_state.get('google_auth_verification_error', False):
        email_error = st.session_state.get('google_auth_email', '')
        with feedback_placeholder.container():
            progress_container.empty()  # Clear any existing progress
            message_container.warning(
                f"📧 **Email Anda belum diverifikasi!**\n\n"
                f"Email {email_error} belum diverifikasi. "
                f"Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim. "
                f"Setelah verifikasi, silakan coba login kembali.\n\n"
                f"💡 *Tip: Periksa juga folder spam/junk email*"
            )
        show_warning_toast("Email belum diverifikasi")
        del st.session_state['google_auth_verification_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']


    # Handle tombol login email di luar form
    if email_login_clicked:
        if email and password:
            # Validasi ulang email sebelum proses login
            email_clean = email.strip()
            is_valid_email, email_message = validate_email_format(email_clean)
            if not is_valid_email:
                # Gunakan container yang sudah ada untuk error display
                progress_container.empty()
                message_container.error(f"❌ {email_message}")
                show_error_toast("Format email tidak valid")
                return
            # Pastikan Firebase sudah siap
            if not firebase_auth or not firestore_client:
                progress_container.empty()
                message_container.error("❌ Sistem belum siap. Silakan tunggu beberapa detik dan coba lagi.")
                show_error_toast("Sistem belum siap")
                return
            # Simpan email terakhir untuk kemudahan
            try:
                cookie_controller.set('last_email', email_clean, max_age=LAST_EMAIL_DURATION)
            except Exception as e:
                logger.warning(f"Failed to save last email: {e}")
            
            # Gunakan containers yang sudah di-allocate untuk progress
            progress_container.progress(0.1)
            message_container.caption("🔐 Memulai proses login...")
            # Proses login dengan email yang sudah divalidasi
            try:
                result = login_user(email_clean, password, firebase_auth, firestore_client, remember, progress_container, message_container)
                if result:
                    progress_container.empty()
                    st.rerun()
            except Exception as login_error:
                progress_container.empty()
                logger.error(f"Login process failed: {login_error}")
                error_str = str(login_error).upper()
                if "INVALID_LOGIN_CREDENTIALS" in error_str:
                    show_error_toast(f"Email {email_clean} tidak terdaftar dalam sistem kami")
                    message_container.error(
                        f"**Akun Email Tidak Terdaftar**\n\n"
                        f"Email {email_clean} belum terdaftar dalam sistem kami."
                    )
                    st.info(
                        f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                        f"atau periksa ejaan email Anda."
                    )
                else:
                    show_error_toast("Login gagal")
                    message_container.error(f"❌ Login gagal: {str(login_error)}")
        else:
            # Clear existing content dan tampilkan warning
            progress_container.empty()
            message_container.warning("⚠️ Silakan isi kolom email dan kata sandi.")
            show_warning_toast("Silakan isi kolom email dan kata sandi.")


    # Handle tombol login Google di luar form
    if google_login_clicked:
        # Gunakan containers yang sudah di-allocate
        progress_container.progress(0.1)
        message_container.caption("� Memvalidasi konfigurasi...")
        
        # Validate Google OAuth configuration first
        is_valid_config, config_message = validate_google_oauth_config()
        if not is_valid_config:
            progress_container.empty()
            message_container.error(f"❌ {config_message}")
            show_error_toast(config_message)
            return
        
        progress_container.progress(0.3)
        message_container.caption("� Mengalihkan ke Google OAuth...")
        try:
            google_url = get_google_authorization_url()
            progress_container.progress(0.5)
            message_container.caption("🔗 Membuat URL otorisasi Google...")
            
            # Log URL untuk debugging
            logger.info(f"Generated Google OAuth URL: {google_url}")
            
            progress_container.progress(0.8)
            message_container.caption("✅ Berhasil mengalihkan ke Google...")
            
            # Gunakan meta refresh seperti model lama
            time.sleep(0.5)
            progress_container.empty()
            
            # ✨ ADVANCED POPUP OAUTH IMPLEMENTATION - STAY IN PAGE ✨
            # Clear progress dan switch ke mode advanced pop-up
            time.sleep(0.8)  # Biarkan pengguna melihat progress selesai  
            progress_container.empty()
            message_container.empty()
            
            # Tampilkan divider visual
            st.markdown("---")
            st.markdown("### 🚀 **Advanced Login Pop-up - Stay in Page**")
            
            # Info banner untuk memberitahu pengguna tentang fitur advanced
            st.success("""
                🎯 **Teknik "Stay in Page" Aktif!**  
                • Jendela pop-up Google akan terbuka  
                • Halaman utama tetap terlihat  
                • Setelah login, pop-up menutup otomatis  
                • Anda akan langsung masuk ke dashboard
            """)
            
            # Generate dan tampilkan advanced pop-up OAuth HTML
            popup_oauth_html = generate_popup_oauth_html(google_url)
            components.html(popup_oauth_html, height=480, scrolling=True)
            
            # Panduan penggunaan pop-up
            with st.expander("📋 **Panduan Penggunaan Advanced Pop-up Login**", expanded=False):
                st.markdown("""
                    **🔧 Langkah-langkah:**
                    1. **Klik tombol "🚀 Login dengan Google"** di atas  
                    2. **Jendela pop-up akan terbuka** - jangan tutup halaman ini  
                    3. **Login di jendela pop-up** dengan akun Google Anda  
                    4. **Pop-up akan menutup otomatis** setelah berhasil  
                    5. **Anda akan langsung masuk** ke dashboard aplikasi  
                    
                    **⚠️ Jika pop-up diblokir:**
                    - Browser mungkin memblokir pop-up secara otomatis  
                    - Klik link manual yang akan muncul di atas  
                    - Atau izinkan pop-up untuk domain ini di pengaturan browser  
                    
                    **💡 Tips untuk Pengalaman Terbaik:**
                    - Pastikan pop-up blocker dinonaktifkan  
                    - Jangan refresh halaman saat proses login  
                    - Gunakan browser modern (Chrome, Firefox, Edge)  
                    - Pastikan JavaScript diaktifkan
                """)
            
            # Pesan debug untuk developer
            if st.checkbox("🔧 Tampilkan Info Debug OAuth", value=False):
                debug_info = {
                    "Google OAuth URL": google_url,
                    "Pop-up Blocker Status": "Akan dideteksi otomatis",
                    "Advanced Mode": "Stay in Page - Enabled",
                    "JavaScript Monitoring": "Active"
                }
                st.json(debug_info)
            
        except Exception as e:
            logger.error(f"Google OAuth advanced pop-up failed: {e}")
            progress_container.empty()
            message_container.error("❌ Gagal mengalihkan ke Google. Silakan coba lagi.")
            show_error_toast("❌ Gagal mengalihkan ke Google. Silakan coba lagi.")
    
    # Tampilkan tips untuk login
    display_auth_tips("login")

def display_register_form(firebase_auth: Any, firestore_client: Any) -> None:
    """Tampilkan dan tangani formulir registrasi pengguna"""
    
    google_email = st.session_state.get('google_auth_email', '')
    
    # Inisialisasi data formulir di state sesi
    if 'register_form_data' not in st.session_state:
        st.session_state['register_form_data'] = {
            'first_name': '',
            'last_name': '',
            'email': google_email,
            'terms_accepted': False
        }
    
    # Perbarui email jika google_email diset
    if google_email and st.session_state['register_form_data']['email'] != google_email:
        st.session_state['register_form_data']['email'] = google_email

    with st.form("register_form", clear_on_submit=False):
        st.markdown("### Daftar")

        # Input nama
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input(
                "Nama Depan", 
                value=st.session_state['register_form_data']['first_name'],
                placeholder="John"
            )
                    
        with col2:
            last_name = st.text_input(
                "Nama Belakang", 
                value=st.session_state['register_form_data']['last_name'],
                placeholder="Doe"
            )

        # Input email
        email = st.text_input(
            "Email",
            value=st.session_state['register_form_data']['email'],
            placeholder="email.anda@contoh.com",
            help="Kami akan mengirimkan link verifikasi ke email ini"
        )

        # Input password (hanya untuk non-Google)
        if not google_email:
            col3, col4 = st.columns(2)
            with col3:
                password = st.text_input(
                    "Kata Sandi",
                    type="password",
                    placeholder="Buat kata sandi yang kuat",
                    help="Gunakan 8+ karakter dengan campuran huruf, angka & simbol"
                )
                        
            with col4:
                confirm_password = st.text_input(
                    "Konfirmasi Kata Sandi",
                    type="password",
                    placeholder="Masukkan ulang kata sandi"
                )
        else:
            password = st.text_input(
                "Kata Sandi (Dibuat otomatis untuk akun Google)",
                type="password",
                value=f"Google-{secrets.token_hex(4)}",
                disabled=True
            )
            confirm_password = password
            st.info("Karena Anda mendaftar dengan akun Google, kami akan mengelola kata sandi dengan aman.")

        # Checkbox syarat layanan
        terms = st.checkbox(
            "Saya setuju dengan Syarat Layanan dan Kebijakan Privasi",
            value=st.session_state['register_form_data']['terms_accepted']
        )
        
        button_text = "Daftar dengan Google" if google_email else "Buat Akun"

        register_clicked = st.form_submit_button(button_text, use_container_width=True, type="primary")
        
        # Placeholder untuk pesan feedback dan progress di bawah tombol registrasi
        feedback_placeholder = st.empty()

    # Handle tombol registrasi di luar form
    if register_clicked:
        # Perbarui state sesi dengan nilai formulir saat ini
        st.session_state['register_form_data'].update({
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'terms_accepted': terms
        })
        
        # Validasi dasar
        if not terms:
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan terima Syarat Layanan untuk melanjutkan.")
            show_warning_toast("Silakan terima Syarat Layanan untuk melanjutkan.")
            return

        if not all([first_name, last_name, email, password]):
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan isi semua kolom yang diperlukan.")
            show_error_toast("Silakan isi semua kolom yang diperlukan.")
            return

        if not google_email and password != confirm_password:
            with feedback_placeholder.container():
                st.error("❌ Kata sandi tidak cocok! Silakan periksa kembali.")
            show_error_toast("Kata sandi tidak cocok! Silakan periksa kembali.")
            return
            
        # Proses registrasi dengan progress steps yang konsisten
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            # Progress indicator
            progress_container.progress(0.05)
            message_container.caption("📝 Memulai proses registrasi...")
            
            # Proses registrasi tanpa spinner bawaan
            success, message = register_user(
                first_name or "", last_name or "", email or "", password or "", 
                firebase_auth, firestore_client, bool(google_email),
                progress_container, message_container
            )
            
            if success:
                # Hapus data formulir setelah registrasi berhasil
                if 'register_form_data' in st.session_state:
                    del st.session_state['register_form_data']
                
                # Hapus google auth email jika ada
                if 'google_auth_email' in st.session_state:
                    del st.session_state['google_auth_email']
                
                # Simpan status untuk fitur pengiriman ulang
                st.session_state['last_registration_email'] = email
                
                # Clear progress setelah registrasi berhasil, tapi biarkan message tetap
                time.sleep(1.2)  # Beri waktu untuk membaca pesan
                progress_container.empty()
    
    # Tampilkan tips untuk registrasi
    display_auth_tips("register")

def display_reset_password_form(firebase_auth: Any) -> None:
    """Tampilkan dan tangani formulir reset kata sandi"""
    
    with st.form("reset_form", clear_on_submit=True):
        st.markdown("### Reset Kata Sandi")
        st.info("Masukkan alamat email Anda di bawah ini dan kami akan mengirimkan petunjuk untuk mereset kata sandi Anda.")

        email = st.text_input(
            "Alamat Email",
            placeholder="email.anda@contoh.com",
            help="Masukkan alamat email yang terkait dengan akun Anda"
        )

        # Validasi email real-time
        if email and email.strip():
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                st.error(f"❌ {email_message}")  # Tetap gunakan st.error untuk real-time validation

        reset_clicked = st.form_submit_button("Kirim Link Reset", use_container_width=True, type="primary")
        
        # Placeholder untuk pesan feedback dan progress di bawah tombol reset
        feedback_placeholder = st.empty()

    # Handle tombol reset di luar form
    if reset_clicked:
        if not email or not email.strip():
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan masukkan alamat email Anda.")
            show_warning_toast("Silakan masukkan alamat email Anda.")
            return
            
        # Proses reset password dengan progress steps yang konsisten
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            # Progress indicator
            progress_container.progress(0.1)
            message_container.caption("📧 Memulai proses reset password...")
            
            # Proses reset password tanpa spinner bawaan
            result = reset_password(email.strip(), firebase_auth, progress_container, message_container)
            
            # Clear progress setelah reset password selesai, tapi biarkan message tetap
            if result:
                time.sleep(1.2)  # Beri waktu untuk membaca pesan sukses
                progress_container.empty()
    
    # Tampilkan tips untuk reset password
    display_auth_tips("reset")

def tampilkan_header_sambutan():
    """Menampilkan header sambutan dan logo aplikasi"""
    try:
        logo_path = "ui/icon/logo_app.png"
        with open(logo_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <div class="welcome-header">
                <img src="data:image/png;base64,{img_base64}" alt="SentimenGo Logo" style="width:170px; display:block; margin:0 auto 1rem auto;">
                <div style="text-align:center; font-size:1rem; color:#666; margin-bottom:2rem;">Sistem Analisis Sentimen GoRide</div>
            </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
            <div class="welcome-header">
                <div style='text-align:center; font-size:2.5rem; margin-bottom:1rem; color:#2E8B57;'>🛵 SentimenGo</div>
                <div style='text-align:center; font-size:1.8rem; font-weight:bold; margin-bottom:1rem; color:#333;'>Selamat Datang!</div>
                <div style="text-align:center; font-size:1rem; color:#666; margin-bottom:2rem;">Sistem Analisis Sentimen GoRide</div>
            </div>
        """, unsafe_allow_html=True)

def tampilkan_pilihan_autentikasi(firebase_auth, firestore_client):
    """Menampilkan selectbox pilihan metode autentikasi"""
    # Tampilkan selectbox untuk memilih metode autentikasi
    auth_type = st.selectbox(
        "Pilih metode autentikasi",
        ["🔐 Masuk", "📝 Daftar", "🔑 Reset Kata Sandi"],
        index=0,
        help="Pilih metode autentikasi Anda"
    )
    
    # Tampilkan form sesuai pilihan
    if auth_type == "🔐 Masuk":
        display_login_form(firebase_auth, firestore_client)
    elif auth_type == "📝 Daftar":
        display_register_form(firebase_auth, firestore_client)
    elif auth_type == "🔑 Reset Kata Sandi":
        display_reset_password_form(firebase_auth)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Titik masuk utama aplikasi"""
    try:
        # Inisialisasi
        sync_login_state()
        initialize_session_state()
        
        logger.info("Application started")
        
        # CSS Styles - Optimized with Layout Stability
        st.markdown("""
            <style>
            /* Main layout optimizations */
            html, body { height: 100vh !important; overflow: hidden !important; margin: 0 !important; }
            .main .block-container { padding-top: 1rem !important; max-height: 100vh !important; }
            section.main { height: 100vh !important; display: flex !important; flex-direction: column !important; 
                          justify-content: center !important; align-items: center !important; }
            
            /* Content wrapper with consistent spacing */
            .auth-content-wrapper { width: 100%; max-width: 500px; max-height: 95vh; overflow-y: auto; 
                                   padding: 1rem; display: flex; flex-direction: column; align-items: center; }
            
            /* Form and UI styling with layout stability */
            .welcome-header { text-align: center; margin-bottom: 1rem; }
            .stSelectbox { margin-bottom: 1rem !important; width: 100%; }
            div[data-testid="stForm"] { border: 1px solid #f0f2f6; padding: 1.2rem; border-radius: 10px; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem; width: 100%; }
            .stButton button { width: 100%; border-radius: 20px; height: 2.8rem; font-weight: bold; margin: 0.3rem 0; }
            .stTextInput { margin-bottom: 0.8rem; }
            
            /* Feedback container untuk mencegah layout shift */
            .element-container { min-height: 2rem; }
            .stEmpty > div { min-height: 1px; }
            
            /* Divider styling */
            .auth-divider-custom { display: flex; align-items: center; margin: 1rem 0; }
            .divider-line-custom { flex: 1; height: 1px; background: #e0e0e0; }
            .divider-text-custom { margin: 0 1rem; color: #888; font-weight: 600; letter-spacing: 1px; font-size: 0.9rem; }
            
            /* Scrollbar */
            .auth-content-wrapper::-webkit-scrollbar { width: 4px; }
            .auth-content-wrapper::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }
            
            /* Responsive */
            @media (max-height: 700px) {
                .welcome-header { margin-bottom: 0.5rem; }
                div[data-testid="stForm"] { padding: 1rem; }
                .stButton button { height: 2.5rem; }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Inisialisasi Firebase dengan retry dan status feedback
        firebase_auth, firestore_client = None, None
        initialization_container = st.empty()
        
        with initialization_container.container():
            with st.spinner("🔥 Menginisialisasi Firebase..."):
                firebase_auth, firestore_client = initialize_firebase()
        
        # Clear initialization message setelah selesai
        initialization_container.empty()
        
        # Verifikasi Firebase berhasil diinisialisasi
        if not (firebase_auth and firestore_client):
            logger.error("Firebase initialization failed")
            st.error("🔥 *Kesalahan Konfigurasi Firebase*")
            st.error("""
            *Aplikasi tidak dapat berjalan tanpa konfigurasi Firebase yang valid.*
            
            Silakan pastikan:
            • File .streamlit/secrets.toml tersedia dan lengkap
            • Konfigurasi Firebase service account benar
            • Semua kredensial telah dikonfigurasi dengan benar
            
            Hubungi administrator sistem untuk bantuan konfigurasi.
            """)
            return

        # Cek status login
        if firebase_auth and firestore_client and st.session_state.get('logged_in', False):
            if check_session_timeout():
                user_email = st.session_state.get('user_email')
                if user_email and verify_user_exists(user_email, firestore_client):
                    logger.info(f"User authenticated: {user_email}")
                    return
                else:
                    logger.warning(f"User verification failed: {user_email}")
                    st.error("Masalah autentikasi terdeteksi. Silakan login kembali.")
                    logout()
                    st.rerun()
        # Tampilkan UI autentikasi dalam container yang tepat
        with st.container():
            st.markdown('<div class="auth-content-wrapper">', unsafe_allow_html=True)
            tampilkan_header_sambutan()

            # Handle logout message
            if st.query_params.get("logout") == "1":
                logger.info("User logged out")
                st.toast("Anda telah berhasil logout.", icon="✅")
                st.query_params.clear()
                
            # Handle Google OAuth callback atau tampilkan form autentikasi
            if firebase_auth and firestore_client:
                # Handle Google OAuth callback jika ada
                handle_google_login_callback()

                # Selalu tampilkan pilihan autentikasi jika user belum login
                if not st.session_state.get('logged_in', False):
                    tampilkan_pilihan_autentikasi(firebase_auth, firestore_client)
            else:
                # Firebase tidak tersedia - tampilkan error konfigurasi
                logger.error("Firebase unavailable - configuration error")
                st.error("🔥 *Kesalahan Konfigurasi Firebase*")
                st.error("*Aplikasi tidak dapat berjalan tanpa konfigurasi Firebase yang valid.*")

            # Close the content wrapper
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        st.error("Terjadi kesalahan yang tidak terduga. Silakan coba lagi nanti.")
        st.session_state.clear()
        initialize_session_state()
        st.rerun()

if __name__ == "__main__":
    main()