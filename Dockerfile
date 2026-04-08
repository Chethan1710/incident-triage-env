FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
<<<<<<< HEAD
# Use Streamlit as primary demo UI; API runs separately if needed
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
=======
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
>>>>>>> e12b981e38929ad56abec1a80f58e6bac9cc38aa
