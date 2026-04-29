
import fitz 

def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    
    for page in doc:
        text += page.get_text()
    
    return text
text = load_pdf("data/Engneering_standards/IEC 62443-2-1.pdf")


