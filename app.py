import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea con River", page_icon="")
st.title("Aprendizaje en línea con River (Streaming realista desde Cloud Storage)")

st.markdown("""
Este panel demuestra cómo un modelo de **aprendizaje incremental** puede entrenarse y actualizarse 
a partir de un dataset grande alojado en **Google Cloud Storage (GCS)**.  
Cada archivo CSV del bucket se procesa como un *fragmento temporal* del flujo de datos.
""")

# =========================================================
# FUNCIONES AUXILIARES PARA GUARDAR Y CARGAR EL MODELO
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    """Guarda el modelo en formato pickle dentro del bucket de GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{destination_blob}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    """Carga el modelo desde GCS si existe, de lo contrario devuelve None."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        else:
            st.info("No se encontró un modelo previo, se iniciará uno nuevo.")
            return None
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None

# =========================================================
# CONFIGURACIÓN DE PARÁMETROS
# =========================================================
bucket_name = st.text_input("Nombre del bucket de GCS:", "bucket_131025")
prefix = st.text_input("Carpeta/prefijo dentro del bucket:", "tlc_yellow_trips_2022/")
limite = st.number_input("Número de registros por archivo a procesar:", value=1000, step=100)
mostrar_grafico = st.checkbox("Mostrar gráfico de evolución del R²", value=True)

st.markdown("""
Haz clic en **“Procesar siguiente archivo”** para actualizar el modelo con el siguiente fragmento 
del dataset almacenado en tu bucket de GCS.
""")

# =========================================================
# INICIALIZACIÓN DEL MODELO Y LAS MÉTRICAS
# =========================================================
MODEL_PATH = "models/model_incremental.pkl"

if "model" not in st.session_state:
    model = load_model_from_gcs(bucket_name, MODEL_PATH)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []
    st.session_state.current_idx = 0

model = st.session_state.model
r2 = st.session_state.metric

# =========================================================
# PROCESAMIENTO INCREMENTAL UNO A UNO (versión optimizada)
# =========================================================
def process_next_blob(bucket_name, prefix, limite=5000, chunksize=500):
    """Procesa el siguiente archivo del bucket según el índice en sesión."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = sorted(list(bucket.list_blobs(prefix=prefix)), key=lambda b: b.name)

    idx = st.session_state.current_idx
    if idx >= len(blobs):
        st.success("Todos los archivos ya fueron procesados.")
        return None, None

    blob = blobs[idx]
    expected_cols = {"trip_distance", "passenger_count", "fare_amount"}
    st.info(f"Procesando archivo {idx + 1}/{len(blobs)}: `{blob.name}`")

    content = blob.download_as_bytes()
    buffer = io.BytesIO(content)
    count = 0

    try:
        for chunk in pd.read_csv(buffer, chunksize=chunksize):
            if not expected_cols.issubset(set(chunk.columns)):
                st.warning(f"Saltando `{blob.name}` (faltan columnas requeridas)")
                break

            # --- Conversión numérica robusta ---
            for col in ["trip_distance", "passenger_count", "fare_amount"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            # --- Feature adicional: hora del día ---
            if "pickup_datetime" in chunk.columns:
                chunk["pickup_hour"] = pd.to_datetime(
                    chunk["pickup_datetime"], errors="coerce"
                ).dt.hour.fillna(0)
            else:
                chunk["pickup_hour"] = 0

            # --- Feature adicional: indicador de peaje ---
            if "tolls_amount" in chunk.columns:
                chunk["tolls_flag"] = (chunk["tolls_amount"] > 0).astype(int)
            else:
                chunk["tolls_flag"] = 0

            # --- Limpieza y filtros ---
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["trip_distance", "passenger_count", "fare_amount"]
            )
            chunk = chunk[
                (chunk["fare_amount"] > 0)
                & (chunk["trip_distance"] > 0)
                & (chunk["passenger_count"] > 0)
            ]

            if len(chunk) > 1:
                chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                continue

            # --- Entrenamiento incremental ---
            for _, row in chunk.iterrows():
                if count >= limite:
                    break
                try:
                    x = {
                        "dist": float(row["trip_distance"]),
                        "pass": float(row["passenger_count"]),
                        "hour": float(row.get("pickup_hour", 0)),
                        "tolls": float(row.get("tolls_flag", 0)),
                    }
                    y = float(row["fare_amount"])
                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    r2.update(y, y_pred)
                    count += 1
                except Exception:
                    continue

            if count >= limite:
                break

    except Exception as e:
        st.warning(f"Error al procesar `{blob.name}`: {e}")
        return None, None

    score = r2.get()
    st.success(f"R² acumulado tras `{blob.name}`: {score:.3f}")

    st.session_state.history.append(score)
    st.session_state.current_idx += 1  # avanzar al siguiente archivo

    save_model_to_gcs(model, bucket_name, MODEL_PATH)
    return blob.name, score

# =========================================================
# BOTÓN PRINCIPAL (un archivo por clic)
# =========================================================
if st.button("Procesar siguiente archivo"):
    fname, score = process_next_blob(bucket_name, prefix, limite)
    if fname:
        st.write(f"Modelo actualizado con `{fname}` → R² actual: **{score:.3f}**")

    if mostrar_grafico and st.session_state.history:
        st.line_chart(st.session_state.history, height=300, use_container_width=True)

st.markdown(f"**Archivo actual:** {st.session_state.get('current_idx', 0) + 1}")

# =========================================================
# ESTADO ACTUAL DEL MODELO
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{r2.get():.3f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history, height=200, use_container_width=True)

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)")
