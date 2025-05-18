print("Welcome to Liver LLM")
pdf = 'papers/A model to predict survival in patients with end-stage liver disease_10.1053jhep.2001.22172.pdf'
file_name = pdf.split('/')[1].rsplit('.', 1)[0]

print(file_name)