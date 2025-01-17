import os
import subprocess
import signal
import json
import time
from config import config

  # Modify the process creation to include real-time output
def start_process(command, name, cwd=None):
    process = subprocess.Popen(
        command, 
        cwd=cwd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    def print_output(stream, prefix):
        for line in stream:
            print(f"[{name}] {line.strip()}")
        
    # Create threads to monitor output
    from threading import Thread
    Thread(target=print_output, args=(process.stdout, name), daemon=True).start()
    Thread(target=print_output, args=(process.stderr, name), daemon=True).start()

    return process

# Main function
def main():

    # List to hold process objects
    processes = []

    try:
        # Start Ollama Server
        print("Starting Ollama Server...")
        ollama_process = start_process(config.ollama_cmd, "Ollama")
        processes.append(ollama_process)

        # Start Chroma Server 
        print("Starting Chroma Server...")
        chroma_dir = os.path.join(os.getcwd(), config.chromadb_dir)
        args = ["chroma", "run", "--path", chroma_dir, "--port", "8000"]
        chroma_process = start_process(args, "Chroma", chroma_dir)
        processes.append(chroma_process)

        # Start Streamlit
        print("Starting Streamlit...")
        streamlit_file = os.path.join(os.getcwd(), config.streamlit_file)
        args = ["streamlit", "run", streamlit_file]
        streamlit_process = start_process(args, "Streamlit")
        processes.append(streamlit_process)

        # Start FastAPI
        print("Starting FastAPI...")
        fastapi_dir = os.path.join(os.getcwd(), config.fastapi_dir)
        args = ["uvicorn", "main:app", "--port", str(config.fastapi_port), "--reload"]
        fastapi_process = start_process(args, "FastAPI", fastapi_dir)
        processes.append(fastapi_process)
        
        # print("All processes started successfully.")
        # print("Press Ctrl+C to terminate all processes.")

        # Keep the script running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTerminating processes...")
        for process in processes:
            if process.poll() is None:  # Check if the process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.kill(process.pid, signal.SIGKILL)

        print("All processes terminated.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        for process in processes:
            if process.poll() is None:  # Check if the process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.kill(process.pid, signal.SIGKILL)

if __name__ == "__main__":
    main()
