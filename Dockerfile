# Base Image
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000

#run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


#build the Docker container:
#docker build -t defectnet-api .
#docker run -p 8000:8000 defectnet-api