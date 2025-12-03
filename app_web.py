import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2
import requests
import re

# ----------------------------------------------------------
# 페이지 설정 + 공통 디자인 CSS
# ----------------------------------------------------------
st.set_page_config(page_title="이미지 가격 → 원화 변환기", layout="wide")

page_css = """
<style>
/* 전체 배경색 */
[data-testid="stAppViewContainer"] {
    background-color: #E8F6FF;
}

/* 제목 & 텍스트 통일 */
h1, h2, h3, h4, h5, h6 {
    color: #1E2A3A;
    font-family: 'Pretendard', sans-serif;
}
p, span, label {
    color: #2B3A4B !important;
    font-family: 'Pretendard', sans-serif;
}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)


# ----------------------------------------------------------
# 세션 초기화
# ----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "upload"

if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = None


# ----------------------------------------------------------
# OCR 준비
# ----------------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)


# ----------------------------------------------------------
# 콤마 → 점 자동 변환 + 숫자만 추출
# ----------------------------------------------------------
def parse_price(text: str):
    clean = text.replace(",", ".").strip()
    m = re.search(r"\d+\.\d+|\d+", clean)
    if not m:
        return None, False
    s = m.group()
    has_dot = "." in s
    return float(s), has_dot


# ----------------------------------------------------------
# EasyOCR: 가장 큰 글씨(가장 큰 숫자) 가격 찾기
# ----------------------------------------------------------
def biggest_price_from_ocr(image_np):
    try:
        results = reader.readtext(image_np, detail=1)
    except:
        return None, None, None

    candidates = []

    for (bbox, text, conf) in results:
        price, has_dot = parse_price(text)
        if price is None:
            continue

        ys = [p[1] for p in bbox]
        height = max(ys) - min(ys)

        priority = (1 if has_dot else 0, height)
        candidates.append((priority, height, price, text))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, h, price, text = candidates[0]
    return price, text, h


# ----------------------------------------------------------
# 흰 가격표 탐지
# ----------------------------------------------------------
def find_white_boxes(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 80 or h < 30:
            continue
        if w < h * 1.2:
            continue
        boxes.append((x, y, w, h))

    return boxes


# ----------------------------------------------------------
# 흰 박스 내부에서 가격 찾기
# ----------------------------------------------------------
def detect_price_from_white_boxes(image_np):
    boxes = find_white_boxes(image_np)
    if not boxes:
        return None, None

    candidates = []
    for (x, y, w, h) in boxes:
        roi = image_np[y:y+h, x:x+w]
        price, text, height = biggest_price_from_ocr(roi)
        if price is not None:
            candidates.append((height, price, text))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, price, text = candidates[0]
    return price, text


# ----------------------------------------------------------
# 1️⃣ 업로드 페이지
# ----------------------------------------------------------
def page_upload():
    # ===== 첫 화면 배경 이미지 CSS 추가 =====
    background_image = "https://i.imgur.com/lrfh4Me.png"

    st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("{background_image}") center/cover no-repeat;
    }}
        <style>
        /* 전체 페이지 배경 + 이미지 설정 */
        [data-testid="stAppViewContainer"] {{
            background: url("{background_image}") center/cover no-repeat;
        }}

        /* 헤더 부분 투명 처리 */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        /* 업로드 박스 배경 살짝 흰색 투명 */
        .uploadedFile {{
            background-color: rgba(255,255,255,0.8);
        }}

        /* 내용을 읽기 쉽게 전체 블록에 반투명 흰 박스 */
        .block-container {{
            background-color: rgba(255,255,255,0.60);
            padding: 2rem;
            border-radius: 15px;
        }}
        </style>
    """, unsafe_allow_html=True)

    # ---------------------------
    # 여기는 기존 페이지 제목/설명
    # ---------------------------
    st.markdown("""
        <h1 style='font-size:48px; text-align:center; font-weight:700;'>
            💸사진 속 가격을 한번에💸<br>한국 원화(KRW)로!
        </h1>
        <p style='text-align:center;'>이미지를 업로드해 가격을 확인하세요!</p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("이미지 선택", type=["png", "jpg", "jpeg"])

    if uploaded:
        st.session_state["uploaded"] = uploaded
        st.session_state["page"] = "result"
        st.rerun()


# ----------------------------------------------------------
# 2️⃣ 결과 페이지
# ----------------------------------------------------------
def page_result():

    uploaded = st.session_state["uploaded"]
    pil_image = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil_image)

    price, line = detect_price_from_white_boxes(image_np)
    used_white = True

    if price is None:
        used_white = False
        price, line, _ = biggest_price_from_ocr(image_np)

    if price is None:
        st.error("❌ 가격을 찾지 못했습니다.")
        return

    # 좌/우 레이아웃
    col1, col2 = st.columns([1, 1])

    # -----------------------------
    # 왼쪽 : 이미지 카드
    # -----------------------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(pil_image, caption="업로드된 이미지", width=350)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # 오른쪽 : 분석 카드
    # -----------------------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if used_white:
            st.info("☑️ 흰 가격표에서 가격을 찾았습니다!")
        else:
            st.warning("⚠ 전체 이미지에서 가장 큰 숫자를 사용했어요.")

        st.markdown(f"📄 **인식된 문장:** {line}")
        st.markdown(f"🔍 **감지된 금액:** {price}")

        confirm = st.radio("금액이 맞나요?", ["네, 맞아요", "아니요, 직접 입력할게요"])

        if confirm == "아니요, 직접 입력할게요":
            price = st.number_input("금액 직접 입력", min_value=0.0, step=0.01, value=float(price))

        currency = st.selectbox("통화 선택", ["USD", "CAD", "AUD", "EUR", "JPY", "KRW"], index=0)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # 아래 카드 : 환율 + 한국돈
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    api_key = "cur_live_bo8IxSQX1WDR4CDzN8cXfMgKFJZmaliymksH2Fuh"
    url = f"https://api.currencyapi.com/v3/latest?apikey={api_key}&currencies=KRW&base_currency={currency}"
    data = requests.get(url).json()
    krw_rate = data["data"]["KRW"]["value"]

    st.markdown(f"📈 **현재 환율:** 1 {currency} = {krw_rate} KRW")

    krw_price = round(price * krw_rate, 2)
    st.success(f"🇰🇷 한국 돈으로: **{krw_price:,} 원**")

    st.markdown("</div>", unsafe_allow_html=True)

    # 새 분석 버튼
    if st.button("🔄 새 이미지 분석하기"):
        st.session_state["page"] = "upload"
        st.session_state["uploaded"] = None
        st.rerun()


# ----------------------------------------------------------
# 페이지 이동
# ----------------------------------------------------------
if st.session_state["page"] == "upload":
    page_upload()
else:
    page_result()
