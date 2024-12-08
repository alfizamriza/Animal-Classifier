# Gunakan base image Python
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Port yang digunakan Streamlit
EXPOSE 8501

# Jalankan aplikasi Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]