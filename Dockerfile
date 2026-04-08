FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
# Use Streamlit as primary demo UI; API runs separately if needed
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]