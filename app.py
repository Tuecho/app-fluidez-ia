import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder

# Configuramos con uno de los modelos que tu lista confirmó que están vivos
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-flash-latest') 

st.title("🗣️ Asistente de Fluidez con Gemini 2.0")

audio_grabado = mic_recorder(
    start_prompt="Empezar a hablar 🎙️",
    stop_prompt="Terminar y Analizar ⏹️",
    key='grabador'
)

if audio_grabado:
    st.audio(audio_grabado['bytes'])
    
    if st.button("Analizar con IA"):
        with st.spinner("La IA está escuchando tu patrón de habla..."):
            try:
                # Preparamos el contenido
                contenido = [
                    "Analiza este audio. Transcribe el texto y, como experto en logopedia, identifica si hay bloqueos, repeticiones o prolongaciones. Ofrece feedback constructivo para mejorar la fluidez.",
                    {
                        "mime_type": "audio/wav",
                        "data": audio_grabado['bytes']
                    }
                ]
                
                response = model.generate_content(contenido)
                
                st.subheader("Resultado del análisis:")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Error al procesar: {e}")
