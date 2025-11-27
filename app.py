import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción de Ventas AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .big-font { font-size:50px !important; font-weight: bold; color: #4CAF50; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO PRINCIPAL ---
st.title("📊 Predicción de Ventas con IA")
st.markdown("Optimiza tu presupuesto de marketing usando **Machine Learning** (Random Forest).")
st.markdown("---")

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        # CORRECCIÓN: Usamos el nombre exacto de tu archivo en el repo
        return joblib.load('modelo_publicidad_rf.joblib')
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Error Crítico: No se encontró el archivo `modelo_publicidad_rf.joblib`. Por favor verifica que esté subido en el repositorio.")
    st.stop()

# --- SIDEBAR (Entradas) ---
with st.sidebar:
    st.header("🎛️ Panel de Control")
    st.write("Ajusta tu inversión en publicidad (x $1000):")
    
    # Sliders para hacer la interacción más dinámica
    tv = st.slider("📺 TV", 0.0, 300.0, 150.0)
    radio = st.slider("📻 Radio", 0.0, 50.0, 20.0)
    diario = st.slider("📰 Diario", 0.0, 100.0, 10.0)
    
    st.markdown("---")
    if st.button("🔄 Resetear Valores"):
        st.rerun()

# --- LÓGICA DE PREDICCIÓN ---
# Crear el DataFrame con los nombres de columnas exactos que usó el modelo al entrenarse
input_data = pd.DataFrame([[tv, radio, diario]], columns=['TV', 'Radio', 'Diario'])
prediccion = model.predict(input_data)[0]

# --- DASHBOARD PRINCIPAL (Columnas) ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("🎯 Resultados")
    # Muestra el número grande
    st.markdown(f'<p class="big-font">{prediccion:.2f} k</p>', unsafe_allow_html=True)
    st.caption("Unidades de venta estimadas")
    
    # Semáforo de rendimiento con mensajes condicionales
    if prediccion > 20:
        st.success("🌟 **¡Excelente Proyección!** La estrategia parece muy efectiva.")
    elif prediccion > 12:
        st.info("✅ **Buen Rendimiento.** Estás en el camino correcto.")
    else:
        st.warning("⚠️ **Rendimiento Bajo.** Considera aumentar la inversión en Radio o TV.")
    
    st.divider()
    st.metric(label="Inversión Total", value=f"${tv + radio + diario:,.2f}")

with col2:
    st.subheader("💡 Distribución del Presupuesto")
    
    # Preparamos los datos para el gráfico
    datos_grafico = pd.DataFrame({
        'Canal': ['TV', 'Radio', 'Diario'],
        'Inversión': [tv, radio, diario],
        'Color': ['#636EFA', '#EF553B', '#00CC96'] # Colores modernos de Plotly
    })

    # GRÁFICO DE DONA INTERACTIVO (Más moderno que las barras simples)
    fig = px.pie(
        datos_grafico, 
        values='Inversión', 
        names='Canal', 
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN DE DETALLES ---
with st.expander("📄 Ver Ficha Técnica de la Predicción"):
    st.table(input_data)
    st.write(f"**Modelo utilizado:** Random Forest Regressor")