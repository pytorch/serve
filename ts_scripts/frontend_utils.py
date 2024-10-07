import os
import sys
import subprocess

def test_frontend():
    print("## Started frontend build and tests")
    frontend_gradlew_path = os.path.join("frontend", "gradlew")
    frontend_gradlew_cmd = [frontend_gradlew_path, "-p", "frontend", "clean", "build"]
    print(f"## In directory: {os.getcwd()} | Executing command: {frontend_gradlew_cmd}")

    try:
        subprocess.run(frontend_gradlew_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        sys.exit("## Frontend Gradle Tests Failed !")
