import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io

# Configuración de IA
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-flash-latest')

st.set_page_config(page_title="Asistente de Fluidez IA", layout="wide")
st.title("🗣️ Asistente de Fluidez mediante Inteligencia Artificial")

# --- INTERFAZ EN COLUMNAS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Práctica de Voz")
    audio_grabado = mic_recorder(
        start_prompt="Empezar a hablar 🎙️",
        stop_prompt="Terminar y Analizar ⏹️",
        key='grabador'
    )

with col2:
    st.subheader("💡 Recomendaciones de Fluidez")
    with st.expander("Ver técnicas para la tartamudez", expanded=True):
        st.markdown("""
        * **Inicio suave:** Deja salir un hilo de aire antes de la primera palabra.
        * **Contacto ligero:** No presiones fuerte los labios en sonidos como /p/, /b/ o /m/.
        * **Pausas tácticas:** Haz pausas breves entre frases para reducir la velocidad.
        * **Cancelación:** Si te bloqueas, detente, relaja la tensión y repite con suavidad.
        """)

# --- PROCESAMIENTO ---
if audio_grabado:
    st.audio(audio_grabado['bytes'])

    if st.button("Analizar el audio grabado"):
        with st.spinner("La IA está analizando tu grabación..."):
            try:
                contenido = [
                    "Analiza este audio. Transcribe el texto y, como experto en logopedia, identifica si hay bloqueos, repeticiones o prolongaciones. Ofrece feedback breve y constructivo.",
                    {
                        "mime_type": "audio/wav",
                        "data": audio_grabado['bytes']
                    }
                ]

                response = model.generate_content(contenido)
                resultado_texto = response.text

                st.subheader("Resultado del análisis:")
                st.write(resultado_texto)

                # --- FUNCIÓN DE LECTURA (TTS) ---
                st.divider()
                st.subheader("🔊 Escuchar análisis")
                tts = gTTS(text=resultado_texto, lang='es')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp, format='audio/mp3')

            except Exception as e:
                st.error(f"Error al procesar: {e}")
