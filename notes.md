# RAG with PDF

There are several Python libraries that can extract **tables and images** from PDFs. Here are the best ones:

---

### **1. PDFMiner.six**  
✅ **Best for:** Extracting text and metadata, but weak at tables/images.  
🔹 **Tables:** No structured extraction, just raw text.  
🔹 **Images:** Yes, but requires extra processing.  

```python
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTImage, LTFigure

for page_layout in extract_pages("sample.pdf"):
    for element in page_layout:
        if isinstance(element, LTImage) or isinstance(element, LTFigure):
            print("Found an image!")
```

---

### **2. PyMuPDF (fitz)**  
✅ **Best for:** Extracting **both tables and images** efficiently.  
🔹 **Tables:** Extracts as raw text but needs post-processing.  
🔹 **Images:** Yes, extracts with metadata.  

```python
import fitz  # PyMuPDF

doc = fitz.open("sample.pdf")

# Extract images
for page in doc:
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        with open(f"image_{xref}.png", "wb") as f:
            f.write(image_bytes)
```

---

### **3. Camelot** (for Tables)  
✅ **Best for:** Extracting **structured tables** (PDFs with lines/grid).  
🔹 **Tables:** Extracts tables cleanly, **but only for PDFs with visible lines**.  
🔹 **Images:** ❌ No image extraction.  

```python
import camelot

tables = camelot.read_pdf("sample.pdf", pages="1")
print(tables[0].df)  # Convert to pandas DataFrame
```

---

### **4. PDFPlumber**  
✅ **Best for:** Extracting **text, tables, and images** easily.  
🔹 **Tables:** Works even if tables have no grid.  
🔹 **Images:** Yes, extracts image objects.  

```python
import pdfplumber

with pdfplumber.open("sample.pdf") as pdf:
    for page in pdf.pages:
        # Extract Tables
        tables = page.extract_table()
        print(tables)  # Returns structured table as list of lists

        # Extract Images
        for img in page.images:
            print("Found image:", img)
```

---

### **5. Tabula (Java-based, for Tables)**  
✅ **Best for:** Extracting **tables from PDFs without grid lines**.  
🔹 **Tables:** Works even when grids are missing.  
🔹 **Images:** ❌ No image extraction.  

```python
import pandas as pd
from tabula import read_pdf

df = read_pdf("sample.pdf", pages="all")[0]  # Extract first table
print(df)
```

---

### **Best Choice for Your Needs**  
📌 **For tables:**  
- **Camelot** (if PDFs have table grids)  
- **Tabula** (if PDFs have no table grids)  
- **PDFPlumber** (best all-rounder for tables)

📌 **For images:**  
- **PyMuPDF (fitz)** (Best for extracting images)  
- **PDFPlumber** (Good for structured extraction)

## PDF Parsing Benchmarks

1. [benchmarks](https://github.com/py-pdf/benchmarks?tab=readme-ov-file)
2. [Best PDF Parser for RAG Apps: A Comprehensive Guide](https://nanonets.com/blog/best-pdf-parser-for-rag-apps-a-comprehensive-guide/)
2. [Nomic Embeddings](http://medium.com/@guptak650/nomic-embeddings-a-cheaper-and-better-way-to-create-embeddings-6590868b438f)