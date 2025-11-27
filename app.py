import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
modelo = joblib.load('modelo_publicidad_rf.joblib')

# Título
st.title("Predicción de Ventas: Marketing Mix 📊")
st.markdown("Introduce tu presupuesto de publicidad para estimar las ventas.")

# Sidebar para inputs
st.sidebar.header("Presupuesto de Publicidad")
tv = st.sidebar.number_input("TV", min_value=0.0, value=100.0)
radio = st.sidebar.number_input("Radio", min_value=0.0, value=20.0)
diario = st.sidebar.number_input("Diario", min_value=0.0, value=10.0)

# Botón de predicción
if st.button("Predecir Ventas"):
    # Crear dataframe con los datos de entrada
    input_data = pd.DataFrame([[tv, radio, diario]], columns=['TV', 'Radio', 'Diario'])
    
    # Predecir
    prediccion = modelo.predict(input_data)[0]
    
    # Mostrar resultado
    st.success(f"📈 Las ventas estimadas son: {prediccion:.2f}")
    
    # Mostrar métricas adicionales visuales
    st.bar_chart(input_data.T)
