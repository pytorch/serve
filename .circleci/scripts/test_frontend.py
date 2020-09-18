import os
import sys

FRONTEND_GRADLEW_PATH = os.path.join("frontend", "gradlew")
FRONTEND_GRADLEW_CMD = f"{FRONTEND_GRADLEW_PATH} -p frontend clean build"
FRONTEND_GRADLEW_EXIT_CODE = os.system(FRONTEND_GRADLEW_CMD)

if FRONTEND_GRADLEW_EXIT_CODE != 0 :
    sys.exit("Frontend Gradle Tests Failed")