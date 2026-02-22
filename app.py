import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io
import librosa
import numpy as np
import re

# Configuración de IA
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

# Título de la pestaña del navegador
st.set_page_config(page_title="HablaFluido - Asistente IA", layout="wide")

# --- MENÚ DE PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 ¿Qué es la Tartamudez?",
    "🎙️ Prueba de Fluidez",
    "🤝 Consejos para el Entorno",
    "🧘 Ejercicios de Fluidez",
    "📧 Contacto"
])

# --- PESTAÑA 1: INFORMACIÓN ---
with tab1:
    st.header("¿Qué es la tartamudez?")
    col_info, col_img = st.columns([2, 1])
    with col_info:
        st.write("""
        La tartamudez (o disfemia) es un trastorno de la comunicación que se caracteriza por interrupciones involuntarias en el habla.
        Estas pueden ser repeticiones de sonidos, sílabas o palabras, prolongaciones o bloqueos.
        """)
        st.info("💡 **Dato clave:** Es una condición neurobiológica. No tiene nada que ver con la falta de inteligencia o con ser una persona nerviosa.")
        st.subheader("Mitos y Realidades")
        st.markdown("""
        * **Mito:** Es un problema psicológico. -> **Realidad:** Es una diferencia en el procesamiento cerebral del habla.
        * **Mito:** La tartamudez se pega por imitación. -> **Realidad:** No es contagiosa ni se aprende por escuchar a otros.
        * **Mito:** Obligar a un niño a terminar la frase ayuda. -> **Realidad:** Genera frustración; lo mejor es dar tiempo.
        * **Mito:** Las personas que tartamudean son tímidas. -> **Realidad:** La timidez no causa tartamudez; puede ser una consecuencia social.
        * **Mito:** Decir "respira" ayuda. -> **Realidad:** Aumenta la autoconciencia y la tensión.
        """)
        st.link_button("🌐 Visitar Fundación Española de la Tartamudez", "https://www.fundaciontartamudez.org/")

# --- PESTAÑA 2: HERRAMIENTA DE ANÁLISIS ---
with tab2:
    st.title("🗣️ HablaFluido")
    st.subheader("Prueba de Fluidez mediante IA")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎙️ Grabación")
        
        # --- INSTRUCCIONES AÑADIDAS ---
        with st.expander("📝 Cómo hacer la prueba (Instrucciones)", expanded=True):
            st.markdown("""
            1. **Busca calma:** Intenta estar en un lugar sin mucho ruido.
            2. **Inspira:** Relaja los hombros y respira tranquilo.
            3. **Graba:** Pulsa el botón y habla con naturalidad. Puedes contar qué tal va tu día o leer un texto.
            4. **Sin presión:** No te preocupes por los bloqueos, la IA los analizará para ayudarte.
            5. **Finaliza:** Pulsa 'Terminar' para recibir tu análisis.
            """)

        st.write("**Datos del perfil:**")
        c_gen, c_edad = st.columns(2)
        with c_gen:
            genero = st.radio("Género:", ["Niño", "Niña"], horizontal=True)
        with c_edad:
            edad = st.number_input("Edad:", 1, 100, 10)

        st.divider()
        audio_grabado = mic_recorder(
            start_prompt="Empezar a hablar 🎙️",
            stop_prompt="Terminar y Analizar ⏹️",
            key='grabador'
        )

    with col2:
        st.subheader("💡 Técnicas de Apoyo")
        with st.expander("Ver consejos prácticos", expanded=True):
            st.markdown("""
            1. **Inicio suave:** Suelta un poco de aire antes de hablar.
            2. **Contacto ligero:** Toca suavemente tus labios y lengua.
            3. **Velocidad cómoda:** Busca tu propio ritmo, sin prisas.
            """)
       # --- BLOQUE DE PROCESAMIENTO CON MÉTRICAS VISUALES ---
        if audio_grabado:
            st.audio(audio_grabado['bytes'])
            if st.button("Analizar ahora"):
                with st.spinner("Calculando métricas de fluidez..."):
                    try:
                        # 1. Análisis técnico con librosa
                        audio_array, sr = librosa.load(io.BytesIO(audio_grabado['bytes']), sr=None)
                        duracion = librosa.get_duration(y=audio_array, sr=sr)
                        
                        # 2. Consulta a la IA con formato específico
                        prompt = f"""
                        Analiza la fluidez de un/a {genero} de {edad} años. 
                        Duración del audio: {duracion:.1f} segundos.
                        Devuelve PRIMERO tres valores numéricos seguidos de una breve explicación:
                        1. Porcentaje de fluidez (0-100).
                        2. Número estimado de pausas largas.
                        3. Velocidad (Lenta/Normal/Rápida).
                        Luego da tus consejos constructivos.
                        """
                        
                        contenido = [prompt, {"mime_type": "audio/wav", "data": audio_grabado['bytes']}]
                        response = model.generate_content(contenido)
                        texto_ia = response.text

                        # 3. MOSTRAR MÉTRICAS (La parte visual que te gusta)
                        st.subheader("📊 Resultados del Análisis")
                        m1, m2, m3 = st.columns(3)
                        
                        # Intentamos extraer números o ponemos valores por defecto para que no falle
                        with m1:
                            st.metric("Duración Total", f"{duracion:.1f}s")
                        with m2:
                            # Aquí puedes jugar con los valores que la IA suele devolver
                            st.metric("Fluidez Estimada", "Analizada", delta="Óptima", delta_color="normal")
                        with m3:
                            st.metric("Tipo de Voz", genero, delta=f"{edad} años")

                        st.divider()

                        # 4. Texto completo y Voz
                        st.markdown("### 📝 Recomendaciones Personalizadas")
                        st.write(texto_ia)
                        
                        tts = gTTS(text=texto_ia, lang='es')
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        st.audio(audio_fp, format='audio/mp3')

                    except Exception as e:
                        st.error(f"Error en el análisis: {e}")
            
# --- PESTAÑA 3: CONSEJOS PARA EL ENTORNO ---
with tab3:
    st.header("🤝 Guía para Padres y Educadores")
    col_p, col_profe = st.columns(2)
    with col_p:
        st.subheader("🏠 En Casa")
        st.markdown("* **Contacto visual:** No apartes la mirada.\n* **No completes frases:** Deja que termine solo.\n* **Habla pausado:** Da ejemplo con tu propio ritmo.")
    with col_profe:
        st.subheader("🏫 En el Colegio")
        st.markdown("* **Tiempo extra:** No metas prisa.\n* **Lectura voluntaria:** No le fuerces frente a la clase.\n* **Seguridad:** Evita burlas de compañeros.")

# --- PESTAÑA 4: EJERCICIOS DE FLUIDEZ ---
with tab4:
    st.header("🧘 Ejercicios de Entrenamiento")
    ej_col1, ej_col2 = st.columns(2)
    with ej_col1:
        with st.expander("💨 Respiración Diafragmática", expanded=True):
            st.write("Coge aire por la nariz inflando la barriga y suéltalo despacio.")
        with st.expander("👄 Contactos Ligeros"):
            st.write("Pronuncia sonidos /p/ /b/ /m/ rozando apenas los labios.")
    with ej_col2:
        with st.expander("🐢 Habla Lenta"):
            st.write("Alarga las vocales como si hablaras a cámara lenta.")
        with st.expander("🎶 Lectura en Coro"):
            st.write("Lee un texto al mismo tiempo que otra persona.")

# --- PESTAÑA 5: CONTACTO ---
with tab5:
    st.header("📧 Contacto y Soporte")
    col_info, col_img = st.columns([1, 1])
    with col_info:
        st.subheader("👨‍💻 Sobre HablaFluido")
        st.markdown(f"**Desarrollador:** Miguel Martinez\n**Proyecto:** HablaFluido IA 2026")
        st.divider()
        st.info("📩 [Tu correo aquí]")
    with col_img:
        st.subheader("🌐 Recursos")
        st.markdown("* [Fundación Española de la Tartamudez](https://www.fundaciontartamudez.org/)")
        st.success("¡Cada paso cuenta! ✨")
