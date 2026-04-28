import fitz  # PyMuPDF

# Caminho do PDF
pdf_path = "./documents/ementas do i bloco engenharia da computação.pdf"

# Abrir o PDF
doc = fitz.open(pdf_path)

texto_completo = ""

# Ler todas as páginas
for pagina in doc:
    texto_completo += pagina.get_text()

# Salvar em arquivo TXT
with open("texto_extraido.txt", "w", encoding="utf-8") as f:
    f.write(texto_completo)

print("Texto extraído com sucesso!")
print("Arquivo salvo como: texto_extraido.txt")
