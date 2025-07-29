# Konfigurasi Google OAuth untuk Streamlit Cloud

## Masalah yang Diperbaiki

1. **Environment Detection**: Deteksi yang lebih robust antara Streamlit Cloud dan Local Development
2. **Redirect URI**: Penggunaan redirect URI yang tepat berdasarkan environment
3. **JavaScript Redirect**: Mengganti meta refresh dengan JavaScript yang lebih kompatibel
4. **Error Handling**: Penanganan error yang lebih baik dengan fallback options

## Environment Variables yang Digunakan untuk Deteksi

### Streamlit Cloud Detection Methods:
1. `STREAMLIT_SERVER_HEADLESS=true` (Primary)
2. `STREAMLIT_CLOUD` (Secondary) 
3. `STREAMLIT_SERVER_PORT` != 8501 (Port-based)
4. `HOST` contains 'streamlit.app' (Host-based)
5. `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` (Stats-based)

## Secrets Configuration

### Untuk secrets.toml (Local Development):
```toml
# Google OAuth Configuration
GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "your-client-secret"

# Redirect URIs
REDIRECT_URI_DEVELOPMENT = "http://localhost:8501/oauth2callback"
REDIRECT_URI_PRODUCTION = "https://sentimentgo.streamlit.app/oauth2callback"

# Firebase Configuration
FIREBASE_API_KEY = "your-firebase-api-key"

[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
```

### Untuk Streamlit Cloud Secrets:
Sama seperti di atas, tetapi masukkan melalui Streamlit Cloud Dashboard > Secrets

## Google Cloud Console Configuration

### OAuth 2.0 Client IDs:
1. **Authorized JavaScript origins**:
   - `http://localhost:8501` (untuk development)
   - `https://sentimentgo.streamlit.app` (untuk production)

2. **Authorized redirect URIs**:
   - `http://localhost:8501/oauth2callback` (untuk development)
   - `https://sentimentgo.streamlit.app/oauth2callback` (untuk production)

## Perubahan Kode Utama

### 1. Environment Detection
```python
def detect_environment() -> Tuple[bool, str]:
    """Detect if running on Streamlit Cloud or local development"""
    # Multiple detection methods for robustness
    if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true':
        return True, "Streamlit Cloud (HEADLESS=true)"
    # ... other methods
```

### 2. JavaScript Redirect (Ganti Meta Refresh)
```python
st.markdown(f"""
<script>
    setTimeout(function() {{
        window.open('{google_url}', '_self');
    }}, 500);
</script>
""", unsafe_allow_html=True)
```

### 3. Fallback Options
- Manual link jika JavaScript gagal
- Debug information untuk troubleshooting
- Validation konfigurasi sebelum redirect

## Testing

### Local Development:
1. Pastikan `REDIRECT_URI_DEVELOPMENT` diset ke `http://localhost:8501/oauth2callback`
2. Test login dengan Google OAuth

### Streamlit Cloud:
1. Pastikan `REDIRECT_URI_PRODUCTION` diset ke `https://sentimentgo.streamlit.app/oauth2callback`
2. Deploy dan test login dengan Google OAuth
3. Periksa logs untuk konfirmasi environment detection

## Troubleshooting

### Jika masih ada error "accounts.google.com refused to connect":
1. Periksa Google Cloud Console - Authorized domains
2. Pastikan redirect URI benar-benar match
3. Gunakan debug function untuk melihat environment variables
4. Periksa browser console untuk JavaScript errors

### Debug Information:
Gunakan `debug_environment_variables()` function untuk melihat:
- Detected platform
- Environment variables
- Redirect URI yang digunakan
- Detection methods results
