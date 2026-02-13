
import zipfile
import re
import os

docx_path = r"s:\Projects\object-detection-project\IEEE.docx"

try:
    if not os.path.exists(docx_path):
        print(f"Error: File not found at {docx_path}")
        exit(1)

    print(f"Analyzing {docx_path}...")

    with zipfile.ZipFile(docx_path) as z:
        # Read the main document XML
        xml_content = z.read('word/document.xml').decode('utf-8')
        
        # Search for column definitions in section properties
        # The tag is usually <w:cols ... w:num="2" ... />
        
        # Find all w:cols tags and their attributes
        cols_tags = re.findall(r'<w:cols\s+([^>]*)>', xml_content)
        cols_full_tags = re.findall(r'(<w:cols\s+[^>]*>)', xml_content)
        
        has_two_columns = False
        
        print(f"Found {len(cols_tags)} column definitions.")
        
        for i, (attrs, full_tag) in enumerate(zip(cols_tags, cols_full_tags)):
            print(f"Definition {i+1}: {full_tag}")
            
            # Check for number of columns
            num_match = re.search(r'w:num="(\d+)"', attrs)
            if num_match:
                num_cols = int(num_match.group(1))
                if num_cols == 2:
                    has_two_columns = True
            else:
                # If w:num is missing, it defaults to 1 usually, unless there are other indicators
                pass

        if has_two_columns:
            print("\n✅ VERDICT: YES. The document allows for 2-column formatting in at least one section (likely the main body).")
        else:
            print("\n❌ VERDICT: NO. No explicit 2-column definition found. It appears to be a 1-column layout.")

except Exception as e:
    print(f"Failed to read .docx file: {e}")
