import sys
from io import TextIOWrapper
from typing import List

fileName: str = sys.argv[1]

file1: TextIOWrapper = open('requirements.txt','w')

python_dep: List[str] = ["fastBPE\n", "regex\n", "requests\n", "sacremoses\n", "subword_nmt\n", fileName+"\n"]

for dep in python_dep:
    file1.writelines(dep)

file1.close() 
