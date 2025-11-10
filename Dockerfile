# Imagen base oficial de Python
FROM python:3.13.5-slim

# Mostrar logs en tiempo real
ENV PYTHONUNBUFFERED 1

# Directorio de trabajo
WORKDIR /usr/src/app

# Copiar dependencias e instalarlas
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . ./

# Exponer puerto estándar de Cloud Run
ENV PORT 8080

# Ejecutar Streamlit en el puerto y host esperados por Cloud Run
CMD exec streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

