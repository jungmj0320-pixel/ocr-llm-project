import streamlit as st
from PIL import Image
import numpy as np
import pytesseract
import cv2
import requests
import re

# ----------------------------------------------------------
# 기본 페이지 설정 + 전체 폰트/색상 스타일
# ----------------------------------------------------------
st.set_page_config(page_title="사진 속 가격을 한 번에 한국 원화(KRW)로!", layout="wide")

page_css = """
<style>
/* 전체 배경 기본 톤 */
[data-testid="stAppViewContainer"] {
    background-color: #E8F6FF;
}

/* 제목 계열 색상 & 폰트 */
h1, h2, h3, h4, h5, h6 {
    color: #1E2A3A;
    font-family: 'Pretendard', sans-serif;
}

/* 일반 텍스트 색상 & 폰트 */
p, span, label {
    color: #2B3A4B !important;
    font-family: 'Pretendard', sans-serif;
}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# ----------------------------------------------------------
# 세션 상태 초기화
# ----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "upload"

if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = None


# ==========================================================
# OCR 쪽 공통 함수 (EasyOCR → Tesseract 로 교체)
# ==========================================================

def parse_price(text: str):
    """
    문자열에서 가격처럼 보이는 숫자만 추출하는 함수
    - 콤마/점 정리 후, 정규식으로 숫자 패턴 찾기
    """
    clean = text.replace(",", ".").strip()
    m = re.search(r"\d+\.\d+|\d+", clean)
    if not m:
        return None, False
    s = m.group()
    has_dot = "." in s
    return float(s), has_dot


def run_tesseract_boxes(image_np):
    """
    Tesseract로 이미지에서 각 단어별 박스 + 텍스트 + 신뢰도 추출
    EasyOCR의 reader.readtext(...)를 대체하는 역할
    """
    # Tesseract는 RGB 이미지도 잘 읽음 (PIL -> np.array가 이미 RGB)
    data = pytesseract.image_to_data(
        image_np,
        lang="eng",
        output_type=pytesseract.Output.DICT
    )

    results = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i]
        if not text or text.strip() == "":
            continue

        # 신뢰도(conf)가 -1 이면 무시
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1.0
        if conf < 0:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        # EasyOCR 형식과 비슷하게 bbox 4점 구성
        bbox = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        ]
        results.append((bbox, text, conf))

    return results


def biggest_price_from_ocr(image_np):
    """
    (기존 EasyOCR 버전 유지)
    - run_tesseract_boxes() 로 읽은 결과를 기반으로
      '가격처럼 보이는 숫자'를 우선순위에 따라 하나 선택
    """
    try:
        results = run_tesseract_boxes(image_np)
    except Exception:
        return None, None, None

    candidates = []
    for (bbox, text, conf) in results:
        price, has_dot = parse_price(text)
        if price is None:
            continue

        ys = [p[1] for p in bbox]
        height = max(ys) - min(ys)

        # 소수점 포함 여부 + 글자 크기(height)로 우선순위 부여
        priority = (1 if has_dot else 0, height)

        candidates.append((priority, height, price, text))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, h, price, text = candidates[0]
    return price, text, h


# ----------------------------------------------------------
# 흰 박스(가격표) 탐지 → 그 안에서 가격 인식
# ----------------------------------------------------------
def find_white_boxes(image_np):
    """
    이미지에서 흰색 박스(가격표처럼 생긴 부분)를 찾아내는 함수
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 너무 작으면 무시
        if w < 80 or h < 30:
            continue
        # 세로로 너무 긴 박스도 제외
        if w < h * 1.2:
            continue

        boxes.append((x, y, w, h))

    return boxes


def detect_price_from_white_boxes(image_np):
    """
    흰 박스를 먼저 찾고, 각 박스 안에서 OCR을 돌려
    그 중에서 가장 '가격스러운 값' 하나 선택
    """
    boxes = find_white_boxes(image_np)
    if not boxes:
        return None, None

    candidates = []
    for (x, y, w, h) in boxes:
        roi = image_np[y:y + h, x:x + w]
        price, text, height = biggest_price_from_ocr(roi)
        if price is not None:
            candidates.append((height, price, text))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, price, text = candidates[0]
    return price, text


# ==========================================================
# 1 페이지: 업로드 화면
# ==========================================================
def page_upload():
    # ===== 첫 화면 배경 CSS =====
    background_image = "https://i.imgur.com/lrfh4Me.png"

    st.markdown(
        f"""
        <style>
        /* 전체 페이지에 배경 이미지 적용 */
        [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(to bottom,
                    rgba(255,255,255,0.7),
                    rgba(255,255,255,0)),
                url("{background_image}") center/cover no-repeat;
        }}

        /* 헤더 투명 처리 */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        /* 업로드 박스 반투명 흰색 */
        .uploadedFile {{
            background-color: rgba(255,255,255,0.8);
        }}

        /* 전체 컨테이너 반투명 박스 + 둥근 모서리 */
        .block-container {{
            background-color: rgba(255,255,255,0.60);
            padding: 2rem;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------
    # 제목 / 안내 문구
    # ---------------------------
    st.markdown(
        """
        <h1 style='font-size:48px; text-align:center; font-weight:700;'>
        💸 사진 속 가격을 한 번에 한국 원화(KRW)로! 💸
        </h1>
        <p style='text-align:center;'>
        이미지를 업로드해 가격을 확인하세요!
        </p>
        """,
        unsafe_allow_html=True,
    )

    # 파일 업로드
    uploaded = st.file_uploader("이미지 선택", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state["uploaded"] = uploaded
        st.session_state["page"] = "result"
        st.rerun()


# ==========================================================
# 2 페이지: 결과 화면
# ==========================================================
def page_result():
    uploaded = st.session_state["uploaded"]
    pil_image = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil_image)

    # 1) 흰 박스에서 먼저 탐색
    price, line = detect_price_from_white_boxes(image_np)
    used_white = True

    # 2) 흰 박스에서 못 찾으면 전체 이미지에서 탐색
    if price is None:
        used_white = False
        price, line, _ = biggest_price_from_ocr(image_np)

    if price is None:
        st.error("❌ 이미지에서 가격을 찾지 못했습니다.")
        return

    col1, col2 = st.columns([1, 1])

    # -----------------------------
    # 왼쪽: 원본 이미지
    # -----------------------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(pil_image, caption="업로드한 이미지", width=350)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # 오른쪽: 인식된 가격 정보
    # -----------------------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if used_white:
            st.info("☑ 흰색 가격표 영역에서 가격을 감지했습니다.")
        else:
            st.warning("⚠ 흰색 박스를 찾지 못해 전체 이미지에서 가격을 감지했습니다.")

        st.markdown(f"📄 **감지된 텍스트:** {line}")
        st.markdown(f"🔍 **감지된 가격:** {price}")

        # 사용자 검증/수정
        confirm = st.radio("이 가격이 맞나요?", ["네, 맞아요", "아니요, 직접 수정할게요"])
        if confirm == "아니요, 직접 수정할게요":
            price = st.number_input("가격을 직접 입력하세요.", min_value=0.0,
                                    step=0.01, value=float(price))

        # 통화 선택
        currency = st.selectbox("통화 단위를 선택하세요.",
                                ["USD", "CAD", "AUD", "EUR", "JPY", "KRW"],
                                index=0)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # 환율 API 호출 + 원화 계산
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    api_key = "cur_live_bo8IxSQX1WDR4CDzN8cXfMgKFJZmaliymksH2Fuh"
    url = (
        f"https://api.currencyapi.com/v3/latest"
        f"?apikey={api_key}&currencies=KRW&base_currency={currency}"
    )

    data = requests.get(url).json()
    krw_rate = data["data"]["KRW"]["value"]

    st.markdown(f"📈 **실시간 환율:** 1 {currency} = {krw_rate} KRW")

    krw_price = round(price * krw_rate, 2)
    st.success(f"🇰🇷 한국 가격: **{krw_price:,.0f} 원**")

    st.markdown("</div>", unsafe_allow_html=True)

    # 다시 하기 버튼
    if st.button("🔄 다른 이미지로 다시 계산하기"):
        st.session_state["page"] = "upload"
        st.session_state["uploaded"] = None
        st.rerun()


# ==========================================================
# 메인: 페이지 전환
# ==========================================================
if st.session_state["page"] == "upload":
    page_upload()
else:
    page_result()
