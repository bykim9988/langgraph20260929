from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import white, black, gray

def create_test_hidden_pdf(filename="test_hidden_text.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)

    # ----------------------------------------------------
    # CASE A: Normal visible text (Should be extracted)
    # ----------------------------------------------------
    c.setFillColor(black)
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, "SUCCESS: This is the only normal visible text.")

    # ----------------------------------------------------
    # CASE B [Scenario 3]: White text on a white canvas (Should be skipped)
    # ----------------------------------------------------
    c.setFillColor(white)
    c.drawString(50, 650, "HIDDEN TEXT (Scenario 3: White text on white background)")

    # ----------------------------------------------------
    # CASE C [Scenario 4]: Microscopic 0.5pt text (Should be skipped)
    # ----------------------------------------------------
    c.setFillColor(black)
    c.setFont("Helvetica", 0.5)
    c.drawString(50, 600, "HIDDEN TEXT (Scenario 4: Microscopic font size 0.5pt)")

    # ----------------------------------------------------
    # CASE D [Scenario 5]: Text covered by a vector shape (Should be skipped)
    # ----------------------------------------------------
    c.setFont("Helvetica", 12)
    c.setFillColor(black)
    c.drawString(50, 550, "HIDDEN TEXT (Scenario 5: Text hidden behind a gray box)")
    
    # Draw an overlapping solid gray rectangle over the text above to mask it visually
    c.setFillColor(gray)
    c.rect(45, 540, 450, 25, fill=True, stroke=False)

    c.save()
    print(f"🎉 Test file successfully generated: '{filename}'")

# Execute generation
create_test_hidden_pdf()
