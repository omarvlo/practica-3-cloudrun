import streamlit as st
import pandas as pd
import numpy as np
import io
import itertools
import time
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# ==============================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==============================
st.set_page_config(page_title="Aprendizaje en línea con River", page_icon="")
st.title("Aprendizaje en línea con River (Streaming realista desde Cloud Storage)")

st.markdown("""
Este panel demuestra cómo un modelo de **aprendizaje incremental** puede entrenarse y actualizarse 
a partir de un dataset grande alojado en **Google Cloud Storage (GCS)**.  
Cada archivo CSV del bucket se procesa como un *fragmento temporal* del flujo de datos.
""")

# =======================================
# INICIALIZACIÓN DEL MODELO Y LAS MÉTRICAS
# =======================================
if "model" not in st.session_state:
    st.session_state.model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    st.session_state.metric = metrics.R2()
    st.session_state.history = []  # Guarda evolución del R²

model = st.session_state.model
r2 = st.session_state.metric

# =============================
# CONFIGURACIÓN DE PARÁMETROS
# =============================
bucket_name = st.text_input("Nombre del bucket de GCS:", "bucket_131025")
prefix = st.text_input("Carpeta/prefijo dentro del bucket:", "tlc_yellow_trips_2022/")
limite = st.number_input("Número de registros por archivo a procesar:", value=1000, step=100)
mostrar_grafico = st.checkbox("Mostrar gráfico de evolución del R²", value=True)

# ========================================
# FUNCIÓN PRINCIPAL: LECTURA Y STREAMING
# ========================================
def stream_from_bucket(bucket_name, prefix, limite=1000):
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    total = len(blobs)

    st.write(f"Se encontraron **{total} archivos** en `{prefix}`.")
    progreso = st.progress(0)

    for idx, blob in enumerate(blobs, start=1):
        st.markdown(f"### Procesando archivo {idx} de {total}: `{blob.name.split('/')[-1]}`")
        try:
            # Descarga CSV directamente como bytes
            content = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(content))

            # Procesa un subconjunto de filas
            for _, row in itertools.islice(df.iterrows(), 0, limite):
                try:
                    x = {
                        "dist": float(row["trip_distance"]),
                        "pass": float(row["passenger_count"]),
                        "hour": float(row.get("pickup_hour", 0))
                    }
                    y = float(row["fare_amount"])
                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    r2.update(y, y_pred)
                except Exception:
                    continue

            # Avance de progreso
            progreso.progress(idx / total)
            yield blob.name, r2.get()

        except Exception as e:
            st.error(f"Error procesando `{blob.name}`: {e}")
            continue

        time.sleep(0.2)  # Solo para ver el avance más claramente

    progreso.empty()

# =============================
# BOTÓN PARA ACTUALIZAR EL MODELO
# =============================
if st.button("Actualizar modelo con datos del bucket"):
    st.info("Procesando archivos desde el bucket... esto puede tardar un poco ⏳")

    nombres, valores = [], []

    for fname, score in stream_from_bucket(bucket_name, prefix, limite):
        nombres.append(fname.split("/")[-1])
        valores.append(score)
        st.session_state.history.append(score)
        st.write(f"`{fname}` — R² acumulado: **{score:.3f}**")

    st.success("Entrenamiento incremental completado.")

    # Visualización final
    if mostrar_grafico and valores:
        st.line_chart(
            pd.DataFrame({"R²": valores}, index=np.arange(1, len(valores) + 1)),
            height=300,
            use_container_width=True
        )

# =============================
# MOSTRAR ESTADO ACTUAL
# =============================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{r2.get():.3f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history, height=200, use_container_width=True)

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)")
