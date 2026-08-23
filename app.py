"""
그림책 사진관 — 사진을 다양한 손그림 화풍으로 바꾸는 도구 (오프라인 필터 전용)

바깥으로 아무것도 전송하지 않습니다. 업로드한 사진은 앱이 도는 동안 메모리에만 있습니다.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

st.set_page_config(page_title="그림책 사진관", page_icon="🌾", layout="wide")

# ----------------------------------------------------------------------------- 설정값

SIZE_PRESETS = {
    "원본 크기 유지": None,
    "웹·블로그용 (긴 변 1024px)": 1024,
    "책 본문 삽입용 (긴 변 1600px)": 1600,
    "인쇄 고화질 (긴 변 2400px)": 2400,
    "직접 입력": "custom",
}

# 기존 그림책 느낌 프리셋: smooth, levels, line, saturation, warmth, bloom
LOOKS = {
    "그림책 느낌": (3, 64, 0.55, 1.15, 0.35, 0.15),
    "은은하게 (원본 살림)": (2, 64, 0.30, 1.08, 0.20, 0.10),
    "또렷한 셀 애니메이션": (4, 40, 0.80, 1.25, 0.30, 0.10),
    "포근한 수채": (4, 64, 0.25, 1.00, 0.60, 0.35),
    "직접 조절": None,
}

FILTER_TYPES = [
    "📖 그림책 (세밀 조절)", 
    "✏️ 흑백 연필 스케치", 
    "🖍️ 만화책 (카툰)", 
    "🖌️ 수채화 (자동)"
]

PREVIEW_EDGE = 720  # 미리보기는 작게 돌려서 슬라이더 반응을 빠르게 유지


@dataclass
class Result:
    name: str
    before: Image.Image
    after: Image.Image


# ----------------------------------------------------------------------------- 이미지 처리 (공통)

def resize_long_edge(img: Image.Image, long_edge: int | None) -> Image.Image:
    """긴 변을 기준으로 비율을 유지하며 크기를 맞춥니다."""
    if not long_edge:
        return img
    w, h = img.size
    if max(w, h) == long_edge:
        return img
    s = long_edge / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


# ----------------------------------------------------------------------------- 필터 함수들

# 1. 기존 그림책 필터 관련
def _ink_lines(gray: np.ndarray, sigma=1.1, k=1.7, p=22.0, eps=0.09, phi=14.0) -> np.ndarray:
    """가우시안 차이로 잉크선처럼 얇고 매끈한 윤곽을 뽑습니다 (0=검정, 1=흰색)."""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    d = ((1 + p) * g1 - p * g2) / 255.0
    return np.clip(np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps))), 0, 1)

def painterly(img: Image.Image, smooth: int, levels: int, line: float, saturation: float, warmth: float, bloom: float) -> Image.Image:
    """색을 평평하게 정리하고 윤곽선과 부드러운 빛을 얹습니다."""
    rgb = np.array(img.convert("RGB"))

    flat = cv2.pyrMeanShiftFiltering(rgb, sp=max(4, smooth * 5), sr=max(10, smooth * 14), maxLevel=1)
    flat = cv2.bilateralFilter(flat, 9, 45, 9)

    if levels < 56:
        q = Image.fromarray(flat).quantize(colors=int(levels), method=Image.MEDIANCUT, dither=Image.Dither.NONE)
        flat = cv2.bilateralFilter(np.array(q.convert("RGB")), 7, 30, 7)

    if line > 0:
        gray = cv2.bilateralFilter(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), 7, 40, 7)
        edge = 1.0 - (1.0 - _ink_lines(gray)) * line
        flat = (flat.astype(np.float32) * edge[..., None]).clip(0, 255).astype(np.uint8)

    out = Image.fromarray(flat)

    if saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(saturation)
    if warmth:
        a = np.array(out).astype(np.float32)
        a[..., 0] = (a[..., 0] + 14 * warmth).clip(0, 255)   # R
        a[..., 2] = (a[..., 2] - 10 * warmth).clip(0, 255)   # B
        out = Image.fromarray(a.astype(np.uint8))

    if bloom:
        b = np.array(out).astype(np.float32) / 255.0
        g = np.array(out.filter(ImageFilter.GaussianBlur(12))).astype(np.float32) / 255.0
        s = 1 - (1 - b) * (1 - g)
        out = Image.fromarray(((b * (1 - bloom) + s * bloom) * 255).clip(0, 255).astype(np.uint8))

    return out


# 2. 흑백 연필 스케치 필터
def filter_sketch(img: Image.Image, blur_amount: int) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(gray)
    # 홀수 값 유지
    k_size = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
    blur = cv2.GaussianBlur(inv, (k_size, k_size), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return Image.fromarray(cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB))


# 3. 카툰 (만화책) 필터
def filter_cartoon(img: Image.Image, edge_thickness: int, color_reduction: int) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    k_size = edge_thickness if edge_thickness % 2 == 1 else edge_thickness + 1
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k_size, 9)

    h, w = rgb.shape[:2]
    # 처리 속도를 위해 이미지를 줄여서 색상을 단순화
    small = cv2.resize(rgb, (w//2, h//2))
    for _ in range(2):
        small = cv2.bilateralFilter(small, 9, color_reduction, color_reduction)
    color = cv2.resize(small, (w, h))

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)


# 4. 수채화 필터
def filter_watercolor(img: Image.Image, brush_size: int, detail: float) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    # BGR 포맷이어야 색이 정확하게 나오므로 변환 후 적용
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    res_bgr = cv2.stylization(bgr, sigma_s=brush_size, sigma_r=detail)
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(res_rgb)


# --- 공통 필터 적용 라우터
def apply_filter(img: Image.Image, filter_type: str, params: tuple) -> Image.Image:
    if filter_type == FILTER_TYPES[0]: # 그림책
        return painterly(img, *params)
    elif filter_type == FILTER_TYPES[1]: # 스케치
        return filter_sketch(img, *params)
    elif filter_type == FILTER_TYPES[2]: # 카툰
        return filter_cartoon(img, *params)
    elif filter_type == FILTER_TYPES[3]: # 수채화
        return filter_watercolor(img, *params)
    return img


@st.cache_data(show_spinner=False, max_entries=8)
def preview(png: bytes, filter_type: str, params: tuple) -> Image.Image:
    """같은 사진·같은 설정이면 다시 계산하지 않습니다."""
    return apply_filter(Image.open(io.BytesIO(png)), filter_type, params)


def to_bytes(img: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
    else:
        img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def build_zip(results: list[Result], fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    ext = "jpg" if fmt == "JPEG" else "png"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in results:
            z.writestr(f"{r.name.rsplit('.', 1)[0]}_그림.{ext}", to_bytes(r.after, fmt, quality))
    return buf.getvalue()


# ----------------------------------------------------------------------------- 사이드바

st.sidebar.title("🌾 그림책 사진관")
st.sidebar.caption("원하는 화풍을 선택하고 책에 넣을 사진을 만드세요")

st.sidebar.markdown("**🎨 필터 선택**")
filter_type = st.sidebar.selectbox("화풍 모드", FILTER_TYPES, label_visibility="collapsed")
st.sidebar.divider()

st.sidebar.markdown(f"**⚙️ {filter_type.split(' ')[1]} 설정**")
if filter_type == FILTER_TYPES[0]:  # 그림책
    look = st.sidebar.selectbox("프리셋 고르기", list(LOOKS))
    base = LOOKS[look] or LOOKS["그림책 느낌"]
    k = look
    smooth = st.sidebar.slider("색면 정리", 1, 5, base[0], key=f"{k}-sm")
    levels = st.sidebar.slider("색 단계", 8, 64, base[1], key=f"{k}-lv")
    line = st.sidebar.slider("윤곽선", 0.0, 1.0, base[2], 0.05, key=f"{k}-ln")
    saturation = st.sidebar.slider("채도", 0.6, 1.8, base[3], 0.02, key=f"{k}-sa")
    warmth = st.sidebar.slider("따뜻함", -1.0, 1.5, base[4], 0.05, key=f"{k}-wa")
    bloom = st.sidebar.slider("빛 번짐", 0.0, 0.6, base[5], 0.02, key=f"{k}-bl")
    params = (smooth, levels, line, saturation, warmth, bloom)

elif filter_type == FILTER_TYPES[1]:  # 스케치
    blur_amount = st.sidebar.slider("선명도 (낮을수록 선명)", 5, 51, 21, step=2, help="값이 클수록 연필 선이 부드러워지고 퍼집니다.")
    params = (blur_amount, )

elif filter_type == FILTER_TYPES[2]:  # 카툰
    edge_thickness = st.sidebar.slider("검은 윤곽선 굵기", 3, 15, 9, step=2, help="홀수만 가능하며 클수록 선이 두꺼워집니다.")
    color_reduction = st.sidebar.slider("색상 단순화(수채화 느낌 부여)", 10, 150, 70, step=10, help="값이 클수록 색이 뭉개져 만화 느낌이 강해집니다.")
    params = (edge_thickness, color_reduction)

elif filter_type == FILTER_TYPES[3]:  # 수채화
    brush_size = st.sidebar.slider("붓터치 크기", 10, 100, 60, step=5, help="수채화 물감이 번지는 정도를 설정합니다.")
    detail = st.sidebar.slider("디테일(질감)", 0.1, 1.0, 0.5, step=0.1, help="종이와 물감의 질감을 살립니다.")
    params = (brush_size, detail)


st.sidebar.divider()
st.sidebar.markdown("**💾 저장 크기**")
size_name = st.sidebar.selectbox("크기", list(SIZE_PRESETS), index=2)
choice = SIZE_PRESETS[size_name]
if choice == "custom":
    long_edge = st.sidebar.number_input("긴 변 (px)", 256, 6000, 1600, 64)
else:
    long_edge = choice
    if long_edge:
        st.sidebar.caption(f"긴 변 {long_edge}px, 비율 유지")

fmt = st.sidebar.radio("파일 형식", ["JPEG", "PNG"], horizontal=True)
quality = st.sidebar.slider("JPEG 품질", 70, 100, 95) if fmt == "JPEG" else 95

# ----------------------------------------------------------------------------- 본문

st.title("사진을 그림으로")
st.write("다양한 화풍으로 사진을 변경해보세요. 설정값은 미리보기에 즉시 반영됩니다.")

files = st.file_uploader(
    "사진 올리기",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
)

if "results" not in st.session_state:
    st.session_state.results = []

# --- 미리보기 ---------------------------------------------------------------
if files:
    st.subheader("미리보기")
    st.caption(f"{files[0].name} · 설정을 바꾸면 바로 다시 그립니다")
    try:
        first = Image.open(files[0]).convert("RGB")
        small = resize_long_edge(first, PREVIEW_EDGE)
        buf = io.BytesIO()
        small.save(buf, "PNG")
        with st.spinner("그림을 변환하는 중입니다..."):
            shown = preview(buf.getvalue(), filter_type, params)
        a, b = st.columns(2)
        a.image(small, caption="원본")
        b.image(shown, caption="변환")
    except Exception as e:
        st.warning(f"미리보기를 만들지 못했습니다 — {e}")

st.divider()

col_run, col_clear = st.columns([1, 5])
run = col_run.button(
    f"{len(files)}장 모두 변환" if files else "변환하기",
    type="primary", disabled=not files,
)
if col_clear.button("결과 지우기", disabled=not st.session_state.results):
    st.session_state.results = []
    st.rerun()

if run:
    st.session_state.results = []
    bar = st.progress(0.0, text="시작하는 중")
    for i, f in enumerate(files, start=1):
        bar.progress((i - 1) / len(files), text=f"{f.name} ({i}/{len(files)})")
        try:
            src = Image.open(f).convert("RGB")
            work = resize_long_edge(src, min(1400, long_edge or 1400))
            out = resize_long_edge(apply_filter(work, filter_type, params), long_edge)
        except Exception as e:
            st.warning(f"{f.name} 변환 실패 — {e}")
            continue
        st.session_state.results.append(Result(f.name, src, out))
    bar.progress(1.0, text=f"{len(st.session_state.results)}장 완료")

# --- 결과 -------------------------------------------------------------------
results = st.session_state.results

if results:
    st.download_button(
        f"전체 {len(results)}장 ZIP으로 받기",
        data=build_zip(results, fmt, quality),
        file_name=f"그림_{datetime.now():%Y%m%d_%H%M}.zip",
        mime="application/zip",
        type="primary",
    )

    ext = "jpg" if fmt == "JPEG" else "png"
    for idx, r in enumerate(results):
        st.subheader(r.name)
        a, b = st.columns(2)
        a.image(r.before, caption=f"원본 · {r.before.width}×{r.before.height}")
        b.image(r.after, caption=f"변환 · {r.after.width}×{r.after.height}")
        st.download_button(
            "이 장 받기",
            data=to_bytes(r.after, fmt, quality),
            file_name=f"{r.name.rsplit('.', 1)[0]}_그림.{ext}",
            mime=f"image/{'jpeg' if fmt == 'JPEG' else 'png'}",
            key=f"dl_{idx}",
        )
elif not files:
    st.info("사진을 올리면 여기에 결과가 나타납니다. 여러 장을 한 번에 올려도 됩니다.")
