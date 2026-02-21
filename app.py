import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io
import librosa
import numpy as np

# Configuración de IA
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-flash-latest')

st.set_page_config(page_title="Asistente de Fluidez IA", layout="wide")
st.title("🗣️ Asistente de Fluidez mediante Inteligencia Artificial")

# --- INTERFAZ EN COLUMNAS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Práctica de Voz")
    
    # --- INFORMACIÓN DEL USUARIO ---
    st.write("**Información del usuario:**")
    col_genero, col_edad = st.columns([1, 1])
    
    with col_genero:
        genero = st.radio(
            "Género:",
            ["Niño", "Niña"],
            horizontal=True,
            key="genero"
        )
    
    with col_edad:
        edad = st.number_input(
            "Edad:",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="edad"
        )
    
    st.divider()
    
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
                # Calcular duración del audio
                audio_array, sr = librosa.load(io.BytesIO(audio_grabado['bytes']), sr=None)
                duracion_segundos = librosa.get_duration(y=audio_array, sr=sr)
                duracion_minutos = duracion_segundos / 60
                
                # Crear prompt personalizado con información del usuario
                prompt_personalizado = f"""Analiza este audio de un/a {genero.lower()} de {edad} años. 
La duración del audio es de {duracion_segundos:.1f} segundos.

Por favor, proporciona un análisis ESTRUCTURADO con las siguientes secciones:

**MÉTRICAS:**
- Palabras pronunciadas: [número estimado]
- Velocidad de habla (palabras por minuto): [estimado, SIN la palabra "palabras por minuto", solo el número]
- Porcentaje de fluidez: [0-100, SIN el símbolo %, solo el número]
- Problemas detectados: [bloqueos, repeticiones, prolongaciones]

**TRANSCRIPCIÓN:**
[Transcribe el texto completo]

**ANÁLISIS DETALLADO:**
Como experto en logopedia especializado en desarrollo del habla infantil, identifica:
- Bloqueos, repeticiones o prolongaciones específicas
- Palabras o sonidos problemáticos
- Características de desarrollo para esta edad esperadas

**RECOMENDACIONES:**
Ofrece feedback constructivo y recomendaciones personalizadas adaptadas a la edad y características de desarrollo."""
                
                contenido = [
                    prompt_personalizado,
                    {
                        "mime_type": "audio/wav",
                        "data": audio_grabado['bytes']
                    }
                ]

                response = model.generate_content(contenido)
                resultado_texto = response.text

                # Extraer métricas del texto de respuesta
                velocidad_ppm = "N/A"
                porcentaje_fluidez = "N/A"
                
                try:
                    # Buscar velocidad de habla
                    import re
                    lineas = resultado_texto.split('\n')
                    for linea in lineas:
                        if 'Velocidad de habla' in linea or 'velocidad de habla' in linea:
                            # Extraer número de la línea
                            numeros = re.findall(r'\d+', linea)
                            if numeros:
                                velocidad_ppm = numeros[0]
                        if 'Porcentaje de fluidez' in linea or 'porcentaje de fluidez' in linea:
                            # Extraer número de la línea
                            numeros = re.findall(r'\d+', linea)
                            if numeros:
                                porcentaje_fluidez = numeros[0]
                except:
                    pass

                # --- MOSTRAR MÉTRICAS EN TARJETAS ---
                st.subheader("📊 Métricas de Análisis")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("⏱️ Duración", f"{duracion_segundos:.1f}s", delta="segundos")
                
                with col_m2:
                    st.metric("⏱️ Duración", f"{duracion_minutos:.2f}m", delta="minutos")
                
                with col_m3:
                    st.metric("📊 Velocidad", f"{velocidad_ppm} ppm", delta="palabras/min")
                
                with col_m4:
                    st.metric("✨ Fluidez", f"{porcentaje_fluidez}%", delta="porcentaje")

                st.divider()
                
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
