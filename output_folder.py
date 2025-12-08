import os
from langchain_community.document_loaders import PyPDFLoader

#  Folder where all your PDF files are stored
PDF_FOLDER = "archive/supreme_court_judgements/1950"   

# 📄 Output file path
OUTPUT_FILE = "merged_output1.txt"

def load_and_merge_pdfs(pdf_folder):
    all_text = ""

    # Loop through all PDF files
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Loading: {file_path}")
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Combine all pages
            for d in docs:
                all_text += d.page_content + "\n\n"

    return all_text


# ▶ Run merging
merged_text = load_and_merge_pdfs(PDF_FOLDER)

# 💾 Save to a single file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(merged_text)

print(f"\nMerged PDF text saved to: {OUTPUT_FILE}")