import os
import sys


def test_frontend():
    print("## Started frontend build and tests")
    frontend_gradlew_path = os.path.join("frontend", "gradlew")
    frontend_gradlew_cmd = f"{frontend_gradlew_path} -p frontend clean build"
    print(f"## In directory: {os.getcwd()} | Executing command: {frontend_gradlew_cmd}")
    frontend_gradlew_exit_code = os.system(frontend_gradlew_cmd)

    if frontend_gradlew_exit_code != 0:
        sys.exit("## Frontend Gradle Tests Failed !")
