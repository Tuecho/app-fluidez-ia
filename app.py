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
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Asistente de Fluidez IA", layout="wide")

# --- MENÚ DE PESTAÑAS (Orden cambiado a petición de Miguel) ---

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 ¿Qué es la Tartamudez?", 
    "🎙️ Examen de Fluidez", 
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
        * **Mito:** La tartamudez se pega por imitación. -> **Realidad:** La tartamudez no es contagiosa ni se aprende por escuchar a otros tartamudear. Es una condición con base genética y neurológica clara.
        * **Mito:** Obligar a un niño a terminar la frase le ayuda a aprender. -> **Realidad:** Completar las frases por ellos suele generar frustación e impaciencia. Lo más útil es darle tiempo y demostrarle que lo que dice es más importante que cómo lo dice.
        * **Mito:** Las personas que tartamudean son tímidas o inseguras. ->**Realidad:** La timidez no causa tartamudez. Lo que ocurre es que, debido a las dificultades de la fluidez, algunas personas pueden volverse más reservadas en situaciones sociales para evitar el bloqueo.
        * **Mito:** Decir "respira" ayuda. -> **Realidad:** Aumenta la autoconciencia y puede generar más tensión.
        * **Mito:** Desaparece sola siempre. -> **Realidad:** Muchos niños la superan, pero la intervención temprana es fundamental.
        """)
        
        st.link_button("🌐 Visitar Fundación Española de la Tartamudez", "https://www.fundaciontartamudez.org/")

# --- PESTAÑA 2: HERRAMIENTA DE ANÁLISIS ---
with tab2:
    st.title("Examen de Fluidez mediante IA")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎙️ Grabación")
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
            2. **Contacto ligero:** Toca suavemente tus labios y lengua al hablar.
            3. **Velocidad cómoda:** No hay prisa, busca tu propio ritmo.
            """)


# --- PESTAÑA 3: CONSEJOS PARA EL ENTORNO ---
with tab3:
    st.header("🤝 Guía para Padres y Educadores")
    st.write("El apoyo del entorno es la herramienta más potente para mejorar la confianza de quien tartamudea.")
    
    col_p, col_profe = st.columns(2)
    
    with col_p:
        st.subheader("🏠 En Casa")
        st.markdown("""
        * **Mantén el contacto visual:** No apartes la mirada cuando aparezca un bloqueo; demuestra que estás escuchando con calma.
        * **No completes sus frases:** Deja que la persona termine por sí misma, aunque sepas qué palabra sigue.
        * **Reduce la velocidad general:** Habla tú más despacio en lugar de pedirle a él/ella que lo haga. El ejemplo es mejor que la orden.
        * **Valida el mensaje:** Responde a lo que ha dicho, no a cómo lo ha dicho.
        """)
    
    with col_profe:
        st.subheader("🏫 En el Colegio")
        st.markdown("""
        * **Tiempo extra:** Permite que el alumno se tome su tiempo para responder sin presión de cronómetro.
        * **Lectura en voz alta:** No le fuerces a leer frente a toda la clase si no se siente cómodo; busca alternativas privadas.
        * **Tolerancia cero al acoso:** Asegúrate de que el aula sea un lugar seguro donde nadie se burle de las pausas.
        * **Turnos claros:** Gestiona los turnos de palabra para que no tenga que "luchar" por ser escuchado.
        """)

# --- PESTAÑA 4: EJERCICIOS DE FLUIDEZ ---
with tab4:
    st.header("🧘 Ejercicios de Entrenamiento")
    st.write("Estos ejercicios están diseñados para relajar los órganos del habla y mejorar la coordinación aire-voz.")
    
    ej_col1, ej_col2 = st.columns(2)
    
    with ej_col1:
        with st.expander("💨 Respiración Diafragmática", expanded=True):
            st.write("""
            1. Pon una mano en tu pecho y otra en tu barriga.
            2. Coge aire por la nariz intentando que solo se mueva la mano de la barriga.
            3. Suéltalo muy despacio por la boca. 
            *Objetivo: Evitar la respiración clavicular (de pecho) que genera tensión.*
            """)
        
        with st.expander("👄 Contactos Ligeros"):
            st.write("""
            Practica decir palabras que empiecen por /p/, /b/, /m/ o /t/ de forma muy suave. 
            Imagina que tus labios apenas se rozan, como si fueran plumas.
            *Ejemplo: Di 'barco' rozando los labios lo mínimo posible.*
            """)

    with ej_col2:
        with st.expander("🐢 Habla Lenta y Silabeada"):
            st.write("""
            Lee una frase exagerando las vocales y alargando los sonidos, como si hablaras a cámara lenta.
            *Ejemplo: 'Hooo-laaaa, ¿có-mooo es-tááás?'*
            """)
            
        with st.expander("🎶 Lectura en Coro"):
            st.write("""
            Lee un texto al mismo tiempo que otra persona (o siguiendo un audio). 
            Se ha demostrado que leer al unísono reduce casi por completo los bloqueos.
            """)

# --- PESTAÑA 5: CONTACTO ---
with tab5:
    st.header("📧 Contacto y Soporte")
    st.write("¿Tienes sugerencias o necesitas ayuda con esta aplicación?")
    
    info_col, form_col = st.columns([1, 1])
    
    with info_col:
        st.info(f"""
        **Desarrollador:** Miguel Martinez
        **Proyecto:** Asistente de Fluidez IA 2026
        **Tecnología:** Streamlit + Gemini 1.5 Flash
        
        Esta herramienta ha sido creada para ayudar a personas con tartamudez a practicar en un entorno seguro y privado.
        """)
        st.write("---")
        st.markdown("### 🌐 Enlaces de interés")
        st.write("- [Fundación Española de la Tartamudez](https://www.fundaciontartamudez.org/)")
        st.write("- [Asociación Internacional de Tartamudez (ISA)](https://www.isastutter.org/)")

    with form_col:
        st.subheader("¡Tu opinión cuenta!")
        email = st.text_input("Tu correo electrónico")
        mensaje = st.text_area("Cuéntame tu experiencia o sugerencias")
        if st.button("Enviar mensaje"):
            if email and mensaje:
                st.success(f"¡Gracias Miguel! He recibido tu mensaje (Simulación). En una versión real, esto se enviaría a tu email.")
            else:
                st.warning("Por favor, rellena ambos campos.")




    # PROCESAMIENTO
    if audio_grabado:
        st.audio(audio_grabado['bytes'])

        if st.button("Analizar ahora"):
            with st.spinner("Analizando..."):
                try:
                    # Calculamos duración con librosa
                    audio_array, sr = librosa.load(io.BytesIO(audio_grabado['bytes']), sr=None)
                    duracion = librosa.get_duration(y=audio_array, sr=sr)
                    
                    prompt = f"Analiza la fluidez de un/a {genero} de {edad} años. Duración: {duracion:.1f}s. Sé constructivo."
                    
                    contenido = [prompt, {"mime_type": "audio/wav", "data": audio_grabado['bytes']}]
                    response = model.generate_content(contenido)
                    
                    st.success("¡Análisis completado!")
                    st.write(response.text)

                    # Voz de la IA
                    tts = gTTS(text=response.text, lang='es')
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format='audio/mp3')

                except Exception as e:
                    st.error(f"Error: {e}")
