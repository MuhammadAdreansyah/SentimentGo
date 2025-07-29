"""
Halaman Prediksi Sentimen Teks GoRide - Modul Utama
===============================================

Modul ini menyediakan antarmuka pengguna untuk prediksi sentimen teks ulasan GoRide
secara real-time dengan dukungan model khusus prediksi dan visualisasi interaktif.

Dependencies:
    - streamlit: Framework untuk antarmuka web
    - pandas: Manipulasi data
    - plotly: Visualisasi interaktif
    - nltk: Natural Language Processing
    - numpy: Komputasi numerik

Fitur Utama:
    - Prediksi sentimen real-time
    - Model khusus untuk prediksi (optimized)
    - Visualisasi tingkat kepercayaan
    - Analisis kata kunci
    - Export hasil prediksi

Author: Mhd Adreansyah
Version: 2.0.0 (Rebuilt)
Date: 2025
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import nltk
import numpy as np
import sys
import os
import base64

# Safe tokenization function with fallback
def safe_word_tokenize(text):
    """Safe word tokenization with fallback for missing NLTK data"""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        try:
            # Try to download missing data
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            return nltk.word_tokenize(text)
        except Exception:
            # Fallback to regex tokenization
            import re
            return re.findall(r'\b\w+\b', text.lower())
import traceback
from typing import Dict, Any, Optional, Tuple

# Setup path untuk import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from ui.auth import auth
from utils import (
    load_sample_data, 
    predict_sentiment, 
    get_or_train_model, 
    load_prediction_model, 
    save_model_and_vectorizer_predict
)


class SentimentPredictionInterface:
    """
    Kelas utama untuk antarmuka prediksi sentimen.
    
    Handles:
        - Model management (khusus vs fallback)
        - User interface rendering
        - Prediction processing
        - Results visualization
    """
    
    def __init__(self):
        """Inisialisasi kelas dengan konfigurasi default."""
        self.data = None
        self.model_info = {
            'use_prediction_model': False,
            'pipeline': None,
            'svm_model': None,
            'tfidf_vectorizer': None,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_mat': None
        }
        self.preprocessing_options = self._get_default_preprocessing_options()
    
    def _get_default_preprocessing_options(self) -> Dict[str, bool]:
        """
        Mengembalikan konfigurasi preprocessing default.
        
        Returns:
            Dict berisi konfigurasi preprocessing yang optimal untuk prediksi
        """
        return {
            'case_folding': True,
            'phrase_standardization': True,
            'cleansing': True,
            'normalize_slang': True,
            'remove_repeated': True,
            'tokenize': True,
            'remove_stopwords': True,
            'stemming': True,
            'rejoin': True
        }
    
    def _initialize_data(self) -> bool:
        """
        Inisialisasi dan validasi data utama.
        
        Returns:
            bool: True jika data berhasil dimuat, False sebaliknya
        """
        try:
            self.data = load_sample_data()
            if self.data.empty:
                st.error("❌ Data tidak tersedia untuk analisis!")
                return False
            return True
        except Exception as e:
            st.error(f"❌ Gagal memuat data: {str(e)}")
            return False
    
    def _load_prediction_model(self) -> bool:
        """
        Memuat model khusus prediksi jika tersedia.
        
        Returns:
            bool: True jika model khusus berhasil dimuat
        """
        try:
            svm_model_predict, tfidf_vectorizer_predict = load_prediction_model()
            
            if svm_model_predict is not None and tfidf_vectorizer_predict is not None:
                st.info("🎯 Menggunakan model khusus prediksi sentimen...")
                
                # Update model info dengan model khusus
                self.model_info.update({
                    'use_prediction_model': True,
                    'pipeline': None,  # Tidak diperlukan untuk model khusus
                    'svm_model': svm_model_predict,
                    'tfidf_vectorizer': tfidf_vectorizer_predict,
                    'accuracy': 0.85,  # Estimasi default
                    'precision': 0.85,
                    'recall': 0.85,
                    'f1': 0.85,
                    'confusion_mat': [[50, 10], [15, 25]]  # Estimasi default
                })
                
                st.toast("🤖 Model prediksi khusus loaded", icon="✅")
                return True
            
            return False
            
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat model khusus: {str(e)}")
            return False
    
    def _load_fallback_model(self) -> bool:
        """
        Memuat model utama sebagai fallback.
        
        Returns:
            bool: True jika model fallback berhasil dimuat
        """
        try:
            if self.data is None:
                st.error("❌ Data tidak tersedia untuk memuat model")
                return False
                
            st.info("🔄 Memuat model utama sebagai fallback...")
            
            pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(
                self.data, 
                self.preprocessing_options
            )
            
            # Update model info dengan model fallback
            self.model_info.update({
                'use_prediction_model': False,
                'pipeline': pipeline,
                'svm_model': svm_model,
                'tfidf_vectorizer': tfidf_vectorizer,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_mat': confusion_mat
            })
            
            st.toast(f"🤖 Model utama loaded sebagai fallback (Akurasi: {accuracy:.1%})", icon="✅")
            return True
            
        except Exception as e:
            error_detail = traceback.format_exc()
            st.error(f"❌ Gagal memuat model fallback: {str(e)}")
            
            with st.expander("🔍 Detail Error (untuk debugging)", expanded=False):
                st.code(error_detail)
            
            return False
    
    def _create_prediction_model(self) -> bool:
        """
        Membuat model prediksi khusus baru.
        
        Returns:
            bool: True jika model berhasil dibuat dan disimpan
        """
        try:
            if self.data is None:
                st.error("❌ Data tidak tersedia untuk membuat model")
                return False
                
            with st.spinner("🔄 Membuat model prediksi khusus..."):
                pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(
                    self.data, 
                    self.preprocessing_options
                )
                
                # Simpan sebagai model prediksi khusus
                save_model_and_vectorizer_predict(pipeline, tfidf_vectorizer)
                
                st.success(f"✅ Model prediksi khusus berhasil dibuat! (Akurasi: {accuracy:.2%})")
                st.info("🔄 Silakan refresh halaman untuk menggunakan model prediksi khusus.")
                st.toast("✅ Model prediksi khusus berhasil dibuat!", icon="🎯")
                
                return True
                
        except Exception as e:
            st.error(f"❌ Gagal membuat model prediksi khusus: {str(e)}")
            return False
    
    def _initialize_models(self) -> bool:
        """
        Inisialisasi model dengan prioritas: khusus > fallback > buat baru.
        
        Returns:
            bool: True jika minimal satu model berhasil dimuat
        """
        # Cek model khusus prediksi terlebih dahulu
        if self._load_prediction_model():
            return True
        
        st.warning("⚠️ Model khusus prediksi tidak ditemukan, menggunakan model utama...")
        
        # Tawarkan opsi untuk membuat model khusus
        if st.button("🎯 Buat Model Prediksi Khusus"):
            if self._create_prediction_model():
                return True
        
        # Fallback ke model utama
        return self._load_fallback_model()
    
    def _render_header(self) -> None:
        """Render header dan informasi halaman."""
        st.title("🔍 Prediksi Sentimen Teks")
        st.subheader("Analisis Sentimen Ulasan GoRide secara Real-time")
        
        # Info penggunaan
        st.write("""
        > **Tips Penggunaan:**
        > - Masukkan ulasan GoRide dalam Bahasa Indonesia.
        > - Model hanya mengenali sentimen POSITIF dan NEGATIF.
        > - Semakin panjang dan jelas ulasan, prediksi akan lebih akurat.
        > - Model tidak mengenali sarkasme, typo berat, atau bahasa campuran.
        """)
    
    def _render_input_section(self) -> Tuple[str, bool]:
        """
        Render bagian input teks dan tombol prediksi.
        
        Returns:
            Tuple[str, bool]: Teks input dan status tombol prediksi
        """
        st.write("### Masukkan teks ulasan:")
        text_input = st.text_area(
            "Ketik ulasan di sini...",
            height=150,
            placeholder="Contoh: Saya puas dengan pelayanan GoRide. Driver ramah dan cepat sampai."
        )
        
        predict_button = st.button("🔍 Prediksi Teks", type="primary")
        
        return text_input, predict_button
    
    def _process_prediction(self, text_input: str) -> Optional[Dict[str, Any]]:
        """
        Memproses prediksi sentimen untuk teks input.
        
        Args:
            text_input: Teks yang akan diprediksi
            
        Returns:
            Dict hasil prediksi atau None jika gagal
        """
        try:
            with st.spinner('Menganalisis teks...'):
                if self.model_info['use_prediction_model']:
                    # Gunakan model khusus prediksi
                    result = predict_sentiment(
                        text_input, 
                        None, 
                        self.preprocessing_options, 
                        self.model_info['use_prediction_model'], 
                        self.model_info['svm_model'], 
                        self.model_info['tfidf_vectorizer']
                    )
                else:
                    # Gunakan model fallback
                    result = predict_sentiment(
                        text_input, 
                        self.model_info['pipeline'], 
                        self.preprocessing_options, 
                        self.model_info['use_prediction_model']
                    )
                
                return result
                
        except Exception as e:
            st.error(f"❌ Gagal memproses prediksi: {str(e)}")
            return None
    
    def _render_gauge_chart(self, confidence: float, gauge_color: str) -> None:
        """
        Render chart gauge untuk tingkat kepercayaan.
        
        Args:
            confidence: Nilai kepercayaan (0-100)
            gauge_color: Warna gauge ('green' atau 'red')
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Tingkat Kepercayaan"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 33], 'color': "#f9e8e8"},
                    {'range': [33, 66], 'color': "#f0f0f0"},
                    {'range': [66, 100], 'color': "#e8f9e8"}
                ]
            },
            number={'suffix': "%", 'valueformat': ".1f"}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_keyword_analysis(self, text_input: str) -> None:
        """
        Render analisis kata kunci yang mempengaruhi prediksi.
        
        Args:
            text_input: Teks input untuk dianalisis
        """
        st.subheader("Kata Kunci yang Mempengaruhi Prediksi")
        
        try:
            # Enhanced NLTK tokenization with fallback
            clean_tokens = safe_word_tokenize(text_input.lower())
            
            if clean_tokens:
                token_df = pd.DataFrame({
                    'Token': clean_tokens[:10],  # Ambil 10 teratas
                    'Present': [1] * min(len(clean_tokens), 10)
                })
                
                fig = px.bar(
                    token_df,
                    x='Present',
                    y='Token',
                    orientation='h',
                    title="Kata Kunci dalam Teks (Top 10)",
                    color='Present',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak cukup kata kunci untuk ditampilkan setelah preprocessing.")
                
        except Exception as e:
            st.warning(f"⚠️ Gagal menganalisis kata kunci: {str(e)}")
    
    def _render_prediction_results(self, text_input: str, result: Dict[str, Any]) -> None:
        """
        Render hasil prediksi dengan visualisasi lengkap.
        
        Args:
            text_input: Teks input yang diprediksi
            result: Hasil prediksi dari model
        """
        prediction = result['sentiment']
        
        # Tentukan confidence, emoji, dan warna berdasarkan prediksi
        if prediction == "POSITIF":
            confidence = result['probabilities']['POSITIF'] * 100
            emoji = "😊"
            gauge_color = "green"
        else:
            confidence = result['probabilities']['NEGATIF'] * 100
            emoji = "😔"
            gauge_color = "red"
        
        # Buat tabs untuk organisasi konten yang lebih baik
        tabs = st.tabs(["Analisis Sentimen", "Ringkasan"])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Hasil Prediksi:")
                if prediction == "POSITIF":
                    st.success(f"Sentimen: {prediction} {emoji}")
                else:
                    st.error(f"Sentimen: {prediction} {emoji}")
                st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
            
            with col2:
                self._render_gauge_chart(confidence, gauge_color)
            
            # Tombol download hasil prediksi
            self._render_download_section(text_input, prediction, confidence)
        
        with tabs[1]:
            self._render_summary_tab(text_input, prediction, confidence, emoji)
    
    def _render_download_section(self, text_input: str, prediction: str, confidence: float) -> None:
        """
        Render bagian download hasil prediksi.
        
        Args:
            text_input: Teks input
            prediction: Hasil prediksi
            confidence: Tingkat kepercayaan
        """
        try:
            pred_df = pd.DataFrame({
                'Teks Ulasan': [text_input],
                'Sentimen': [prediction],
                'Confidence (%)': [f"{confidence:.2f}"],
            })
            
            csv = pred_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi_goride.csv">📥 Download Hasil Prediksi (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"⚠️ Gagal membuat link download: {str(e)}")
    
    def _render_summary_tab(self, text_input: str, prediction: str, confidence: float, emoji: str) -> None:
        """
        Render tab ringkasan analisis.
        
        Args:
            text_input: Teks input
            prediction: Hasil prediksi
            confidence: Tingkat kepercayaan
            emoji: Emoji untuk sentimen
        """
        st.subheader("Ringkasan Analisis")
        
        try:
            summary_data = {
                "Aspek": [
                    "Sentimen Terdeteksi", 
                    "Tingkat Kepercayaan", 
                    "Jumlah Kata", 
                    "Jumlah Karakter"
                ],
                "Nilai": [
                    f"{prediction} {emoji}",
                    f"{confidence:.2f}%",
                    len(safe_word_tokenize(text_input)),
                    len(text_input)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Analisis kata kunci
            self._render_keyword_analysis(text_input)
            
        except Exception as e:
            st.error(f"❌ Gagal membuat ringkasan: {str(e)}")
    
    def _render_footer(self) -> None:
        """Render footer aplikasi."""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
            <p style="margin: 0; font-size: 0.9rem; color: #666;">
                © 2025 GoRide Sentiment Analysis Dashboard • Developed by Mhd Adreansyah
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #888;">
                🎓 Aplikasi ini merupakan bagian dari Tugas Akhir/Skripsi di bawah perlindungan Hak Cipta
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render(self) -> None:
        """
        Method utama untuk render seluruh antarmuka prediksi sentimen.
        
        Workflow:
        1. Sinkronisasi status login
        2. Inisialisasi data dan model
        3. Render UI komponen
        4. Proses prediksi jika ada input
        """
        # Sinkronisasi status login dari cookie ke session_state
        auth.sync_login_state()
        
        # Inisialisasi data
        if not self._initialize_data():
            st.stop()
        
        # Inisialisasi model
        if not self._initialize_models():
            st.error("❌ Gagal menginisialisasi model. Silakan restart aplikasi.")
            st.stop()
        
        # Render header dan info
        self._render_header()
        
        # Render input section
        text_input, predict_button = self._render_input_section()
        
        # Proses prediksi jika ada input dan tombol ditekan
        if text_input and predict_button:
            result = self._process_prediction(text_input)
            if result:
                self._render_prediction_results(text_input, result)
        elif predict_button and not text_input:
            st.error("⚠️ Silakan masukkan teks terlebih dahulu untuk diprediksi.")
        
        # Render footer
        self._render_footer()


def render_sentiment_prediction() -> None:
    """
    Fungsi wrapper untuk menjalankan antarmuka prediksi sentimen.
    
    Fungsi ini mempertahankan kompatibilitas dengan kode existing
    yang memanggil render_sentiment_prediction() secara langsung.
    """
    interface = SentimentPredictionInterface()
    interface.render()


# Entry point untuk testing langsung
if __name__ == "__main__":
    render_sentiment_prediction()
