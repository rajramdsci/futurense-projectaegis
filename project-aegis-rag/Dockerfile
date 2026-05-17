# Use an official, lightweight Python image
FROM python:3.13.7-slim

# Set environment variables to optimize Python behavior inside the container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for certain PDF/RAG or ML packages)
#

# Copy dependency list first to utilize Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application folders
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY src/ ./src/

# Expose the standard Streamlit port
EXPOSE 8501

# Run the Streamlit application using healthcheck best practices
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py"]
