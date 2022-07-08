import wget
from zipfile import ZipFile
url='https://git.io/JuMq0'
filename = wget.download(url)
#unzip -qq flower_model_bit_0.96875.zip
print(filename)

#unzip
with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()