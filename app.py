import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Sales Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (Para darle el toque moderno) ---
st.markdown("""
    <style>
    .big-font { font-size:50px !important; font-weight: bold; color: #4CAF50; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO PRINCIPAL ---
st.title("📊 Predicción de Ventas con IA")
st.markdown("Optimiza tu presupuesto de marketing usando **Machine Learning**.")
st.markdown("---")

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('modelo_ventas.joblib')
    except:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Error: No se encontró el archivo 'modelo_ventas.joblib'. Por favor súbelo a tu repositorio.")
    st.stop()

# --- SIDEBAR (Entradas) ---
with st.sidebar:
    st.header("🎛️ Panel de Control")
    st.write("Ajusta tu inversión en publicidad (x $1000):")
    
    tv = st.slider("📺 TV", 0, 300, 150)
    radio = st.slider("📻 Radio", 0, 50, 20)
    diario = st.slider("📰 Diario", 0, 100, 10)
    
    st.markdown("---")
    st.caption("Desarrollado por JEAS-14")

# --- LÓGICA DE PREDICCIÓN ---
input_data = pd.DataFrame([[tv, radio, diario]], columns=['TV', 'Radio', 'Diario'])
prediccion = model.predict(input_data)[0]

# --- DASHBOARD PRINCIPAL (Columnas) ---
col1, col2 = st.columns([1, 2])

with col1:
    # 1. Métrica Principal (El número grande)
    st.subheader("🎯 Ventas Estimadas")
    st.markdown(f'<p class="big-font">{prediccion:.2f} k</p>', unsafe_allow_html=True)
    
    # Semáforo de rendimiento
    if prediccion > 20:
        st.success("¡Excelente Proyección! 🚀")
    elif prediccion > 10:
        st.warning("Rendimiento Moderado 😐")
    else:
        st.error("Rendimiento Bajo 🔻")
    
    st.write(f"Inversión Total: **${tv + radio + diario}**")

with col2:
    # 2. Gráficos Modernos con Plotly
    st.subheader("💡 Análisis de Inversión")
    
    # Preparamos los datos para el gráfico
    datos_grafico = pd.DataFrame({
        'Medio': ['TV', 'Radio', 'Diario'],
        'Inversión': [tv, radio, diario],
        'Color': ['#1f77b4', '#ff7f0e', '#2ca02c'] # Colores personalizados
    })

    # CREAR GRÁFICO DE BARRAS DINÁMICO
    fig = px.bar(
        datos_grafico, 
        x='Medio', 
        y='Inversión', 
        color='Medio',
        text='Inversión',
        title="Distribución del Presupuesto",
        color_discrete_sequence=px.colors.qualitative.Pastel, # Paleta de colores moderna
        template="plotly_white"
    )
    
    fig.update_layout(showlegend=False) # Ocultar leyenda redundante
    st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN INFERIOR (Detalle) ---
with st.expander("Ver desglose detallado del presupuesto"):
    # Gráfico de Dona (Pie Chart)
    fig_pie = px.pie(
        datos_grafico, 
        values='Inversión', 
        names='Medio', 
        title='Porcentaje de Inversión por Canal',
        hole=0.4, # Hace que sea una dona
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_pie, use_container_width=True)