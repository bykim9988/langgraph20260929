# 설치
#   pip install pymupdf pytesseract pillow
#   Tesseract 엔진 + 한국어 데이터:
#     Ubuntu/Debian : sudo apt install tesseract-ocr tesseract-ocr-kor
#     macOS         : brew install tesseract tesseract-lang
#     Windows       : UB-Mannheim 설치본에서 Korean 선택 후 pytesseract.pytesseract.tesseract_cmd 지정
"""
PDF에서 '눈에 보이는' 텍스트만 추출한다.

(A) 디지털 텍스트 레이어  → 숨겨진 텍스트를 원인별로 분류해 제거
  [1] OCR 투명 레이어      : 이미지와 겹치는 Tr 3 텍스트        → 제거 (이미지는 (B)에서 직접 OCR)
  [2] Text Rendering Mode 3 : 채우기·외곽선 없음                 → 제거 + 화면 출력
  [3] 배경색과 동일 / opacity 0                                  → 제거 + 화면 출력
  [4] 극소 폰트 (<= 1pt)                                         → 제거 + 화면 출력
  [5] 나중에 그려진 불투명 도형·이미지에 가려짐                  → 제거 

(B) 스캔 이미지 → Tesseract OCR
  - 페이지에서 텍스트만 제거한 사본을 렌더링해 이미지 영역을 OCR한다.
    → 이미지 위에 겹쳐 있는 벡터 텍스트가 중복 인식되지 않고,
      이미지를 가리는 도형(흰 박스 등)은 그대로 반영된다(눈에 보이는 부분만 OCR).
  - OCR 결과는 이미지 위치(y좌표) 기준으로 디지털 텍스트와 읽기 순서대로 합친다.
"""
import re

import pymupdf
import pytesseract
from PIL import Image

# ---------------- 설정값: 숨김 텍스트 ----------------
MIN_FONT_SIZE = 1.0        # [4] 이 크기(pt) 이하면 극소 폰트
COLOR_TOL = 0.03           # [3] 글자색-배경색 채널별 최대 차이(0~1)가 이 값 이하면 '같은 색'
OCR_OVERLAP = 0.5          # [1] Tr 3 텍스트가 이미지와 50% 이상 겹치면 OCR 레이어로 간주
COVER_RATIO = 0.9          # [5] 텍스트 bbox의 90% 이상이 덮이면 '가림'
PAGE_BG = (1.0, 1.0, 1.0)  # 도형이 없을 때 가정하는 페이지 배경색(흰색)

# ---------------- 설정값: OCR ----------------
OCR_LANG = "kor+eng"
OCR_DPI = 300              # 렌더링 해상도 (테스트 결과 300dpi가 가장 정확)
OCR_PSM = 6                # 6 = 균일한 텍스트 블록 가정 (표 형태 문서에 유리)
OCR_BG_CLIP = 170          # 이 밝기(0~255) 초과 픽셀은 흰색으로 → 하이라이트·연한 배경 제거
OCR_MIN_IMAGE_PT = 20      # 가로·세로가 이 크기(pt) 미만인 이미지(아이콘 등)는 OCR 생략

LABELS = {
    1: "[1] OCR 투명 레이어 (스캔 이미지 위 Tr 3)",
    2: "[2] Text Rendering Mode 3 (Invisible)",
    3: "[3] 배경색과 동일 / Opacity 0",
    4: "[4] 극소 폰트 (<= %.1fpt)" % MIN_FONT_SIZE,
    5: "[5] 도형·이미지에 의한 가림 (Occlusion)",
}
REPORT = (2, 3, 4)         # 화면에 따로 출력할 시나리오


# =====================================================================
# (A) 디지털 텍스트 레이어: 숨겨진 텍스트 분류
# =====================================================================
def _covered(inner: pymupdf.Rect, outer: pymupdf.Rect) -> bool:
    """inner 면적의 COVER_RATIO 이상이 outer에 포함되는가."""
    if inner.is_empty:
        return outer.contains(inner.tl)
    return (inner & outer).get_area() >= COVER_RATIO * inner.get_area()


def _color_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _to_rgb(c):
    """PyMuPDF 색 튜플(Gray/RGB/CMYK)을 RGB(0~1)로 변환."""
    if c is None:
        return None
    c = tuple(c)
    if len(c) == 1:
        return (c[0],) * 3
    if len(c) == 4:
        k = c[3]
        return tuple((1 - x) * (1 - k) for x in c[:3])
    return c[:3]


def classify_page(page):
    """
    반환: (hidden_keys, findings)
      hidden_keys: 숨김으로 판정된 글자의 (문자, origin_x, origin_y) 집합
      findings   : [(시나리오 번호, 텍스트, 상세정보), ...]
    """
    bboxlog = page.get_bboxlog()                     # index == seqno (그리기 순서)
    fills = {d["seqno"]: d for d in page.get_drawings() if d.get("fill") is not None}

    # 불투명하게 칠하는 객체: (seqno, rect, 채우기색 or None, 종류)
    painters = []
    for seq, (kind, rect) in enumerate(bboxlog):
        r = pymupdf.Rect(rect)
        if kind == "fill-image":
            painters.append((seq, r, None, "image"))
        elif kind == "fill-path" and seq in fills:
            d = fills[seq]
            if (d.get("fill_opacity") or 1.0) >= 0.99:   # 반투명 도형은 가리지 않음
                painters.append((seq, r, _to_rgb(d["fill"]), "path"))
    images = [p[1] for p in painters if p[3] == "image"]

    hidden, findings = set(), []
    for span in page.get_texttrace():
        chars = span["chars"]
        text = "".join(chr(c[0]) for c in chars)
        if not text.strip():
            continue
        seq = span["seqno"]
        bbox = pymupdf.Rect(span["bbox"])
        size = span["size"]
        color = _to_rgb(span["color"]) or (0, 0, 0)
        opacity = span.get("opacity", 1.0)

        below = [p for p in painters if p[0] < seq and _covered(bbox, p[1])]
        above = [p for p in painters if p[0] > seq and _covered(bbox, p[1])]

        scenario, detail = None, ""
        if span["type"] == 3:                                   # Tr 3
            if any((bbox & r).get_area() >= OCR_OVERLAP * bbox.get_area() for r in images):
                scenario, detail = 1, "Tr 3 + 이미지와 겹침"
            else:
                scenario, detail = 2, "render mode 3"
        elif opacity <= 0.01:
            scenario, detail = 3, f"opacity={opacity:.2f}"
        elif size <= MIN_FONT_SIZE or bbox.height <= MIN_FONT_SIZE:
            scenario, detail = 4, f"size={size:.2f}pt"
        else:
            bg = below[-1][2] if below else PAGE_BG             # 바로 아래 배경색
            if bg is not None and _color_diff(color, bg) <= COLOR_TOL:
                hexc = "#%02x%02x%02x" % tuple(round(x * 255) for x in color)
                hexb = "#%02x%02x%02x" % tuple(round(x * 255) for x in bg)
                scenario, detail = 3, f"text {hexc} on bg {hexb}"
            elif above:
                scenario, detail = 5, f"covered by {above[-1][3]} (seqno {above[-1][0]})"

        if scenario:
            findings.append((scenario, text, detail))
            hidden.update((chr(c[0]), round(c[2][0], 2), round(c[2][1], 2)) for c in chars)

    return hidden, findings


def extract_visible_text_lines(page, hidden):
    """숨김 글자를 뺀 디지털 텍스트를 줄 단위로 반환: [(Rect, text), ...]"""
    out = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            kept = [
                ch for span in line["spans"] for ch in span["chars"]
                if (ch["c"], round(ch["origin"][0], 2), round(ch["origin"][1], 2)) not in hidden
            ]
            s = "".join(ch["c"] for ch in kept).strip()
            if s:
                r = pymupdf.Rect(kept[0]["bbox"])
                for ch in kept[1:]:
                    r |= ch["bbox"]
                out.append((r, s))
    return out


# =====================================================================
# (B) 이미지 OCR
# =====================================================================
def preprocess_for_ocr(img: Image.Image, bg_clip: int = OCR_BG_CLIP) -> Image.Image:
    """그레이스케일 변환 후, 밝은 배경(하이라이트·연한 셀 배경)을 흰색으로 날린다."""
    g = img.convert("L")
    if bg_clip:
        g = g.point(lambda v: 255 if v > bg_clip else v)
    return g


def clean_ocr_text(text: str) -> str:
    """OCR 원문 후처리: 아이콘·점에서 생긴 기호 토큰 제거, 넓은 공백(표의 열 간격)은 탭으로."""
    out = []
    for line in text.splitlines():
        tokens = re.split(r"( {3,}| )", line)             # 구분자 유지
        kept = []
        for tok in tokens:
            if tok.strip() and not re.search(r"[\w~()\[\]%]", tok):
                continue                                   # ©, ', ", ・ 같은 단독 기호
            kept.append(tok)
        line = re.sub(r" {3,}", "\t", "".join(kept))
        line = re.sub(r" +", " ", line)
        line = re.sub(r"\s*\t\s*", "\t", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def ocr_image(img: Image.Image, lang: str = OCR_LANG, psm: int = OCR_PSM,
              preprocess: bool = True, postprocess: bool = True) -> str:
    """
    PIL 이미지 한 장을 OCR해서 텍스트를 반환한다. (PDF와 무관하게 단독 사용 가능)

    ※ image_to_data(단어 좌표)로 줄을 재조립하면 한글이 음절 단위('모 델 의')로 쪼개지므로,
      띄어쓰기 처리가 더 정확한 image_to_string을 사용한다.
    """
    if preprocess:
        img = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(
        img, lang=lang,
        config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1",
    )
    return clean_ocr_text(text) if postprocess else text


def extract_text_from_image_file(image_path, **kwargs) -> str:
    """이미지 파일(png/jpg 등)에서 바로 텍스트를 뽑는 편의 함수."""
    with Image.open(image_path) as img:
        return ocr_image(img, **kwargs)


def _text_free_copy(doc, pno):
    """해당 페이지만 복사한 뒤 텍스트를 모두 지운 사본(이미지·도형은 유지)."""
    tmp = pymupdf.open()
    tmp.insert_pdf(doc, from_page=pno, to_page=pno)
    pg = tmp[0]
    pg.add_redact_annot(pg.rect, fill=False)            # fill=False: 흰 박스를 칠하지 않음
    pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                        text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    return tmp


def ocr_page_images(page, dpi: int = OCR_DPI, **ocr_kwargs):
    """
    페이지의 이미지 영역을 OCR한다. 이미지 하나 = 텍스트 블록 하나.
    반환: [(이미지 Rect 페이지좌표, text), ...]
    """
    rects = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if r.width >= OCR_MIN_IMAGE_PT and r.height >= OCR_MIN_IMAGE_PT:
            rects.append(r)
    if not rects:
        return []

    tmp = _text_free_copy(page.parent, page.number)
    clean_page = tmp[0]
    out = []
    for r in rects:
        pix = clean_page.get_pixmap(dpi=dpi, clip=r, colorspace=pymupdf.csRGB, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = ocr_image(img, **ocr_kwargs)
        if text:
            out.append((r, text))
    tmp.close()
    return out


# =====================================================================
# 통합
# =====================================================================
def extract_clean_visible_text(pdf_path, use_ocr=True, report=True, mark_ocr=False):
    """
    반환: (clean_text_by_page, hidden_by_page)
      clean_text_by_page[p] : 보이는 디지털 텍스트 + 이미지 OCR 텍스트 (읽기 순서)
      hidden_by_page[p]     : [(시나리오, 텍스트, 상세), ...]
    """
    doc = pymupdf.open(pdf_path)
    clean_text_by_page, hidden_by_page = {}, {}

    for page in doc:
        pno = page.number + 1
        hidden, findings = classify_page(page)
        hidden_by_page[pno] = findings

        items = [(r, t) for r, t in extract_visible_text_lines(page, hidden)]
        if use_ocr:
            for r, t in ocr_page_images(page):
                if mark_ocr:
                    t = f"----- [OCR] image @ {tuple(round(v) for v in r)} -----\n{t}\n----- [/OCR] -----"
                items.append((r, t))

        # 읽기 순서: 위→아래(3pt 단위로 같은 줄 취급), 왼→오른
        items.sort(key=lambda it: (round(it[0].y0 / 3), it[0].x0))
        clean_text_by_page[pno] = "\n".join(t for _, t in items)

        if report:
            _print_hidden(pno, findings)

    doc.close()
    return clean_text_by_page, hidden_by_page


def _print_hidden(pno, findings):
    targets = [f for f in findings if f[0] in REPORT]
    if not targets:
        print(f"\n##### Page {pno}: 숨겨진 텍스트([2][3][4]) 없음 #####")
        return
    print(f"\n##### Page {pno}: 숨겨진 텍스트 탐지 #####")
    for sc in REPORT:
        items = [f for f in targets if f[0] == sc]
        if items:
            print(f"--- {LABELS[sc]} : {len(items)}건 ---")
            for _, text, detail in items:
                print(f"  • {text!r}   ({detail})")


# --- Execution Example ---
if __name__ == "__main__":
    import sys
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/scan_image.pdf"
    clean, hidden = extract_clean_visible_text(pdf_file, mark_ocr=True)

    for page, content in clean.items():
        print(f"\n=== Page {page} : Clean Text ===")
        print(content)
        removed = [f for f in hidden[page] if f[0] not in REPORT]
        if removed:
            print("(그 외 제거: " + ", ".join(f"{LABELS[s][:3]} {t!r}" for s, t, _ in removed) + ")")
