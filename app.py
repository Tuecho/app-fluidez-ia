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
# --- BLOQUE DE PROCESAMIENTO CON MÉTRICAS TÉCNICAS ---
        if audio_grabado:
            st.audio(audio_grabado['bytes'])
            if st.button("Analizar ahora"):
                with st.spinner("Calculando PPM y Fluidez..."):
                    try:
                        # 1. Análisis de audio
                        audio_array, sr = librosa.load(io.BytesIO(audio_grabado['bytes']), sr=None)
                        duracion = librosa.get_duration(y=audio_array, sr=sr)
                        
                        # 2. Prompt específico para obtener datos numéricos
                        prompt = f"""
                        Analiza la fluidez de un/a {genero} de {edad} años. 
                        Duración: {duracion:.1f}s.
                        
                        IMPORTANTE: Calcula y devuelve los siguientes datos exactos al inicio de tu respuesta con este formato:
                        - PALABRAS_MINUTO: [valor]
                        - PORCENTAJE_FLUIDEZ: [valor]%
                        - BLOQUEOS_DETECTADOS: [valor]
                        
                        Después, proporciona un análisis constructivo y consejos.
                        """
                        
                        contenido = [prompt, {"mime_type": "audio/wav", "data": audio_grabado['bytes']}]
                        response = model.generate_content(contenido)
                        texto_ia = response.text

                        # 3. Extracción de datos con Regex (para las métricas)
                        def extraer_valor(patron, texto):
                            match = re.search(patron, texto)
                            return match.group(1) if match else "--"

                        ppm = extraer_valor(r"PALABRAS_MINUTO:\s*(\d+)", texto_ia)
                        fluidez = extraer_valor(r"PORCENTAJE_FLUIDEZ:\s*(\d+)", texto_ia)
                        bloqueos = extraer_valor(r"BLOQUEOS_DETECTADOS:\s*(\d+)", texto_ia)

                        # 4. MOSTRAR MÉTRICAS VISUALES
                        st.subheader("📊 Resultados Técnicos")
                        m1, m2, m3, m4 = st.columns(4)
                        
                        with m1:
                            st.metric("Velocidad (PPM)", f"{ppm}", delta="Palabras/Min")
                        with m2:
                            st.metric("Nivel de Fluidez", f"{fluidez}%", delta="Estimado")
                        with m3:
                            st.metric("Bloqueos", f"{bloqueos}", delta="Detectados", delta_color="inverse")
                        with m4:
                            st.metric("Tiempo", f"{duracion:.1f}s")

                        st.divider()

                        # 5. Texto completo y Voz
                        st.markdown("### 📝 Análisis Detallado")
                        # Limpiamos el texto para no mostrar los códigos técnicos al usuario
                        texto_limpio = re.sub(r"-(.*):.*", "", texto_ia).strip()
                        st.write(texto_limpio)
                        
                        tts = gTTS(text=texto_limpio, lang='es')
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        st.audio(audio_fp, format='audio/mp3')

                    except Exception as e:
                        st.error(f"Error en el análisis técnico: {e}")            
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
