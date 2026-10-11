1. Start backend
cd C:\Users\USER\Desktop\chatbot-main\backend
..\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000

2. Start frontend
Open a second PowerShell:
cd C:\Users\USER\Desktop\chatbot-main\frontend
python -m http.server 8080


3. Then open:
http://localhost:8080
Admin dashboard: http://localhost:8080/admin.html
