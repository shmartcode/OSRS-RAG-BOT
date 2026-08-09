import os
import shutil
import subprocess
import sys
import time


def main():
    print("⚔️  Starting OSRS RAG Bot Stack...\n")

    # 1. Ensure Docker CLI is available and the background service is running
    if not shutil.which("docker"):
        print("❌ Error: Docker is not installed or not in system PATH.")
        sys.exit(1)

    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("❌ Error: Docker Desktop is not running. Please open Docker Desktop and try again.")
        sys.exit(1)

    # 2. Spin up containers (supports both 'docker compose' V2 and 'docker-compose' V1)
    print("🚀 Launching containers...")
    cmd = ["docker", "compose"] if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0 else ["docker-compose"]

    if subprocess.run([*cmd, "up", "-d", "--build"]).returncode != 0:
        print("❌ Error: Failed to start Docker containers.")
        sys.exit(1)

    # 3. Give services a moment to initialize, then attach to the interactive app
    time.sleep(2)
    print("\n Attaching to interactive session...\n")
    os.system("docker exec -it osrs_rag_app python app.py")


if __name__ == "__main__":
    main()
