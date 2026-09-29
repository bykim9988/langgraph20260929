# pip install pymupdf

import pymupdf  # PyMuPDF

def extract_clean_visible_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    clean_text_by_page = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        visible_page_strings = []
        
        # Gather geometry of images and vector paths (Scenarios 1 & 5)
        # We find their bounding boxes to check if text is hidden underneath them
        image_rects = [page.get_image_bbox(img) for img in page.get_images(full=True)]
        draw_rects = [draw["rect"] for draw in page.get_drawings() if "rect" in draw]
        obscuring_rects = image_rects + draw_rects

        # Extract precise text layout metadata
        text_page = page.get_text("dict")
        
        for block in text_page.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                # Check for Scenario 2: Text Rendering Mode 3 (Invisible Text)
                # wmode/dir flags usually map rendering types; PyMuPDF drops completely empty text runs automatically,
                # but we can secure the structural text content inside spans below.
                
                for span in line.get("spans", []):
                    text = span["text"]
                    if not text.strip():
                        continue
                        
                    font_size = span["size"]
                    font_color = span["color"]
                    text_bbox = pymupdf.Rect(span["bbox"])
                    
                    # ----------------------------------------------------
                    # FILTER 1 [Scenario 4]: Extremely small font sizes (0 - 1pt)
                    # ----------------------------------------------------
                    if font_size <= 1.0:
                        continue
                        
                    # ----------------------------------------------------
                    # FILTER 2 [Scenario 3]: Text matches background (White text color)
                    # sRGB value for pure white is 16777215
                    # ----------------------------------------------------
                    if font_color == 16777215:
                        continue
                        
                    # ----------------------------------------------------
                    # FILTER 3 [Scenario 1 & 5]: Hidden under shapes or OCR image layers
                    # If text box resides inside a larger solid vector path or image
                    # ----------------------------------------------------
                    is_obscured = False
                    for obs_rect in obscuring_rects:
                        # If the block is trapped inside an image or shape and isn't the shape itself
                        if obs_rect.contains(text_bbox) and obs_rect.width > text_bbox.width:
                            is_obscured = True
                            break
                    if is_obscured:
                        continue

                    # If all filters pass, it is clean visible text
                    visible_page_strings.append(text)
                    
            # Join fragments with spaces to maintain readable words
            visible_page_strings.append("\n")

        # Combine text for the current page
        clean_text_by_page[page_num + 1] = " ".join(visible_page_strings).strip()

    doc.close()
    return clean_text_by_page

# --- Execution Example ---
pdf_file = "../../data/scan_image.pdf"  # Replace with your actual PDF path
extracted_data = extract_clean_visible_text(pdf_file)

# Print clean output text per page
for page, content in extracted_data.items():
    print(f"=== Page {page} ===")
    print(content)
