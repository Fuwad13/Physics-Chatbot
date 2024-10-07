@echo off
echo Navigating to the project directory...

:: Change to the directory where your project is located
cd /d "E:\Physics-Chatbot"

echo Starting processes...

:: Start Ollama Server and capture its PID
start "Ollama Server" cmd /c "ollama serve"
timeout /t 2 > nul  
for /f "tokens=2" %%i in ('tasklist ^| findstr "ollama"') do set ollama_pid=%%i
echo Ollama PID: %ollama_pid%

:: Activate venv and start Chroma Server
call "E:\Physics-Chatbot\venv\Scripts\activate.bat"
cd /d "E:\Physics-Chatbot\backend\LLM"
start "Chroma Server" cmd /c chroma run --path "chroma server"
timeout /t 2 > nul  
for /f "tokens=2" %%i in ('tasklist ^| findstr "chroma"') do set chroma_pid=%%i
echo Chroma PID: %chroma_pid%

:: Activate venv and start Streamlit
call "E:\Physics-Chatbot\venv\Scripts\activate.bat"
cd /d "E:\Physics-Chatbot\frontend"
start "Streamlit" cmd /c "streamlit run chat_ui.py"
timeout /t 2 > nul  
for /f "tokens=2" %%i in ('tasklist ^| findstr "python"') do set streamlit_pid=%%i
echo Streamlit PID: %streamlit_pid%

:: Activate venv and start FastAPI
call "E:\Physics-Chatbot\venv\Scripts\activate.bat"
cd /d "E:\Physics-Chatbot\backend\LLM\llm-api"
start "FastAPI" cmd /c "uvicorn main:app --port 8080 --reload"
timeout /t 2 > nul
for /f "tokens=2" %%i in ('tasklist ^| findstr "uvicorn"') do set fastapi_pid=%%i
echo FastAPI PID: %fastapi_pid%

:: Wait for the user to press Enter before terminating
echo Processes are running. Press Enter to stop all.
pause

:: Terminate the processes using their PIDs
echo Terminating processes...
taskkill /PID %ollama_pid% /F
taskkill /PID %chroma_pid% /F
taskkill /PID %streamlit_pid% /F
taskkill /PID %fastapi_pid% /F

echo All processes terminated.
pause
