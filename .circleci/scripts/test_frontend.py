import os
import sys

EXIT_CODE = os.system("frontend/gradlew -p frontend clean build")

if EXIT_CODE != 0 :
    sys.exit("Frontend Gradle Tests Failed")