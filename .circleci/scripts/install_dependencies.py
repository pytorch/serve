import os

os.system("pip install -r requirements/common.txt")
os.system("pip install -r requirements/cpu_win.txt")
os.system("pip install -r requirements/developer.txt")

os.system("npm install -g newman newman-reporter-html markdown-link-check")