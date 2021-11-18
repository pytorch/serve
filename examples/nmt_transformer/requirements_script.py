import sys

fileName = sys.argv[1]

file1 = open("requirements.txt", "w")

python_dep = [
    "fastBPE\n",
    "regex\n",
    "requests\n",
    "sacremoses\n",
    "subword_nmt\n",
    fileName + "\n",
]

for dep in python_dep:
    file1.writelines(dep)

file1.close()
