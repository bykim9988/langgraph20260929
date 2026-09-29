# pip install pymupdf   (>= 1.23 권장: get_texttrace / get_bboxlog 사용)
"""
PDF에서 '눈에 보이는' 텍스트만 추출하고, 숨겨진 텍스트는 원인별로 분류한다.

  [1] OCR 투명 레이어      : 이미지 위/아래에 놓인 Tr 3(invisible) 텍스트   → 제거
  [2] Text Rendering Mode 3 : 채우기·외곽선 없음                            → 제거 + 화면 출력
  [3] 배경색과 동일 / opacity 0                                             → 제거 + 화면 출력
  [4] 극소 폰트 (<= 1pt)                                                    → 제거 + 화면 출력
  [5] 나중에 그려진 불투명 도형·이미지에 가려짐                             → 제거

핵심: get_texttrace()는 span별 렌더링 모드(type), opacity, 색, 크기, 그리기 순서(seqno)를
제공하고, get_bboxlog()는 페이지의 모든 그리기 연산을 seqno 순서대로 제공한다.
→ "텍스트보다 나중에 그려진 것만 텍스트를 가린다"는 z-order를 정확히 판정할 수 있다.
"""
import pymupdf

# ---------------- 설정값 ----------------
MIN_FONT_SIZE = 1.0      # [4] 이 크기(pt) 이하면 극소 폰트
COLOR_TOL = 0.03         # [3] 글자색-배경색 채널별 최대 차이(0~1)가 이 값 이하면 '같은 색'
OCR_OVERLAP = 0.5        # [1] Tr 3 텍스트가 이미지와 50% 이상 겹치면 OCR 레이어로 간주
COVER_RATIO = 0.9        # [5] 텍스트 bbox의 90% 이상이 덮이면 '겹침/가림'
PAGE_BG = (1.0, 1.0, 1.0)  # 도형이 없을 때 가정하는 페이지 배경색(흰색)

LABELS = {
    1: "[1] OCR 투명 레이어 (스캔 이미지 위 Tr 3)",
    2: "[2] Text Rendering Mode 3 (Invisible)",
    3: "[3] 배경색과 동일 / Opacity 0",
    4: "[4] 극소 폰트 (<= %.1fpt)" % MIN_FONT_SIZE,
    5: "[5] 도형·이미지에 의한 가림 (Occlusion)",
}
REPORT = (2, 3, 4)       # 화면에 따로 출력할 시나리오


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
    bboxlog = page.get_bboxlog()                     # index == seqno
    fills = {d["seqno"]: d for d in page.get_drawings() if d.get("fill") is not None}

    # 불투명하게 '덮는' 객체들: (seqno, rect, 배경색 or None)
    painters = []
    for seq, (kind, rect) in enumerate(bboxlog):
        r = pymupdf.Rect(rect)
        if kind == "fill-image":
            painters.append((seq, r, None, "image"))
        elif kind == "fill-path" and seq in fills:
            d = fills[seq]
            if (d.get("fill_opacity") or 1.0) >= 0.99:   # 반투명 도형은 가리지 않음
                painters.append((seq, r, _to_rgb(d["fill"]), "path"))

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
            # 스캔 OCR 레이어: 이미지와 절반 이상 겹치면 [1]
            imgs = [p[1] for p in painters if p[3] == "image"]
            if any((bbox & r).get_area() >= OCR_OVERLAP * bbox.get_area() for r in imgs):
                scenario, detail = 1, "Tr 3 + 이미지와 겹침"
            else:
                scenario, detail = 2, "render mode 3"
        elif opacity <= 0.01:                                   # 투명도 0
            scenario, detail = 3, f"opacity={opacity:.2f}"
        elif size <= MIN_FONT_SIZE or bbox.height <= MIN_FONT_SIZE:
            scenario, detail = 4, f"size={size:.2f}pt"
        else:
            # 텍스트 바로 아래(가장 나중에 그려진) 배경의 색
            bg = below[-1][2] if below else PAGE_BG
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


def extract_clean_visible_text(pdf_path, report=True):
    doc = pymupdf.open(pdf_path)
    clean_text_by_page, hidden_by_page = {}, {}

    for page in doc:
        pno = page.number + 1
        hidden, findings = classify_page(page)
        hidden_by_page[pno] = findings

        # rawdict의 레이아웃(읽기 순서)을 유지하면서 숨긴 글자만 제외
        lines_out = []
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                s = "".join(
                    ch["c"]
                    for span in line["spans"]
                    for ch in span["chars"]
                    if (ch["c"], round(ch["origin"][0], 2), round(ch["origin"][1], 2)) not in hidden
                )
                if s.strip():
                    lines_out.append(s.strip())
        clean_text_by_page[pno] = "\n".join(lines_out)

        if report:
            _print_hidden(pno, findings)

    doc.close()
    return clean_text_by_page, hidden_by_page


def _print_hidden(pno, findings):
    targets = [f for f in findings if f[0] in REPORT]
    if not targets:
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
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/test_hidden.pdf"
    clean, hidden = extract_clean_visible_text(pdf_file)

    for page, content in clean.items():
        print(f"\n=== Page {page} : Clean Text ===")
        print(content)
        removed = [f for f in hidden[page] if f[0] not in REPORT]
        if removed:
            print(f"(그 외 제거: " + ", ".join(f"{LABELS[s][:3]} {t!r}" for s, t, _ in removed) + ")")
