$env:MILVUS_HOST = "localhost"
$env:MILVUS_PORT = "19530"
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "123456"
$env:MYSQL_DB = "legal_db"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:PYTHONUNBUFFERED = "1"

Set-Location "e:\agent\legal-chatbot"
& "E:\agent\legal-chatbot\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
