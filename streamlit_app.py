"""CXR Insight research dashboard (v5): enlarged results, collapsible Grad-CAM.
Save as streamlit-web/app/streamlit_app.py in the project root.
Requires cxr_preprocessing.py and test_hr/cxr_inference_uignore_weighted_predicted_gradcam.py at the project root.
"""
from __future__ import annotations

from io import BytesIO
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
from matplotlib import colormaps
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from cxr_preprocessing import MEAN, STD, preprocess_image
from test_hr.cxr_inference_uignore_weighted_predicted_gradcam import (
    CXRInference,
    MODEL_PATH,
    THRESHOLDS,
)
st.set_page_config(page_title="CXR Insight | X-ray Analysis", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --navy: #142d49;
  --blue: #205c98;
  --muted: #657990;
  --line: #dce6f0;
  --shadow: 0 8px 26px rgba(18, 48, 79, .045);
}
.stApp {background: #f5f8fc;}
[data-testid="stSidebar"] {background: #142d49;}
[data-testid="stSidebar"] * {color: #eef5ff !important;}
.block-container {max-width: 1600px; padding-top: 1.35rem; padding-bottom: 3rem;}
h1,h2,h3 {color: #142d49; letter-spacing: -.025em;}
[data-testid="stFileUploader"] {background: white; border: 1px solid #e3eaf2; border-radius: 14px; padding: .6rem;}
[data-testid="stButton"] button[kind="primary"] {background: linear-gradient(100deg,#144c88,#2772c6); border:0; border-radius:11px; font-weight:750; padding:.65rem 1rem;}
[data-testid="stDownloadButton"] button {border-radius:10px; border-color:#c9d7e7;}
[data-testid="stExpander"] {border: 1px solid var(--line); background: #fff; border-radius: 16px; box-shadow: var(--shadow); overflow:hidden;}
[data-testid="stExpander"] summary {color:var(--navy); font-weight:750;}
.cxr-kicker {color:#4476a5;font-weight:800;font-size:.77rem;letter-spacing:.11em;text-transform:uppercase;margin-bottom:.25rem;}
.cxr-sub {color:#62768d;font-size:.93rem;margin-top:-.5rem;margin-bottom:1.1rem;}
.cxr-section {font-size:1.22rem;font-weight:800;color:#142d49;margin:1rem 0 .25rem;}
.cxr-section-sub {color:#62768d;font-size:.84rem;margin-bottom:.95rem;line-height:1.5;}
.cxr-foot {color:#63748a;font-size:.82rem;line-height:1.6;}
.cxr-mode {display:inline-block;padding:.24rem .6rem;border-radius:99px;background:#e5eef9;color:#204f80;font-size:.73rem;font-weight:750;}
.cxr-results-grid {display:grid;grid-template-columns: repeat(auto-fit,minmax(min(100%,230px),1fr));gap:12px;align-items:stretch;}
.cxr-card {
  box-sizing:border-box;
  min-height: 370px;
  padding: 1.25rem 1.13rem;
  border: 1px solid #dce6f0;
  border-radius: 17px;
  background: #fff;
  box-shadow: var(--shadow);
  display:flex;
  flex-direction:column;
  gap:.9rem;
}
.cxr-card--negative {
    border-top: 4px solid #2563EB;
}
.cxr-card--positive {border-top: 4px solid #c94a54;}
.cxr-card--uncertain {border-top: 4px solid #ad792c;}
.cxr-card-head {display:flex;justify-content:space-between;align-items:flex-start;gap:.35rem;}
.cxr-card-title {font-size:1.17rem;font-weight:850;color:#183450;line-height:1.28;}
.cxr-card-sub {font-size:.76rem;color:#748398;margin-top:.22rem;}
.cxr-status {padding:.32rem .6rem;border-radius:999px;font-size:.73rem;font-weight:750;white-space:nowrap;}
.cxr-card--negative .cxr-status {
    color: #1D4ED8;
    background: #DBEAFE;
}
.cxr-card--positive .cxr-status {color:#b53846;background:#ffedf0;}
.cxr-card--uncertain .cxr-status {color:#99691f;background:#fff4df;}
.cxr-score-row {display:flex;align-items:baseline;justify-content:space-between;gap:.45rem;margin-top:.3rem;}
.cxr-score-name {font-size:.85rem;color:#657990;}
.cxr-score {font-size:1.88rem;letter-spacing:-.045em;font-weight:850;}
.cxr-card--negative .cxr-score {
    color: #2563EB;
}
.cxr-card--positive .cxr-score {color:#c43c47;}
.cxr-card--uncertain .cxr-score {color:#ad792c;}
.cxr-track {height:10px;background:#e8eef5;border-radius:100px;overflow:hidden;margin-top:-.35rem;}
.cxr-track-fill {
    height: 100%;
    border-radius: 100px;
    background: #2563EB;
}
.cxr-card--positive .cxr-track-fill {background:#c94a54;}
.cxr-card--uncertain .cxr-track-fill {background:#ad792c;}
.cxr-interpret {padding:.72rem .78rem;min-height:3.65rem;border-radius:12px;background:#f4f8fd;font-size:.8rem;color:#49647d;line-height:1.5;}
.cxr-breakdown {padding-top:.75rem;margin-top:auto;border-top:1px solid #e9eff6;}
.cxr-breakdown-title {color:#566d84;font-size:.72rem;font-weight:700;margin-bottom:.54rem;}
.cxr-mini-row {display:grid;grid-template-columns: 66px 46px minmax(35px,1fr);align-items:center;gap:.35rem;font-size:.71rem;margin:.42rem 0;color:#49647d;}
.cxr-mini-row b {font-weight:750;font-variant-numeric:tabular-nums;text-align:right;}
.cxr-mini-track {height:6px;background:#e8eef5;border-radius:99px;overflow:hidden;}
.cxr-mini-fill {height:100%;border-radius:99px;background:#4e81b2;}
.cxr-wait {min-height:335px;background:#fff;border:1px dashed #cfdaea;border-radius:16px;padding:2.5rem 1.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;box-shadow:var(--shadow);}
.cxr-wait-title {font-weight:800;font-size:1.2rem;color:#173b60;margin:.8rem 0 .35rem;}
.cxr-wait-sub {max-width:330px;color:#657990;font-size:.85rem;line-height:1.7;}
.cxr-model-row {color:#5c7086;font-size:.78rem; margin:.55rem 0 .3rem;}
.cxr-notice {padding:.78rem 1rem;border-radius:12px;background:#eaf1f9;color:#455f79;font-size:.8rem;line-height:1.6;margin-top:.75rem;}
.cxr-report-intro {color:#5d7389;font-size:.83rem;line-height:1.6;margin:.25rem 0 .75rem;}
@media (max-width:1050px) {.cxr-card {padding:1rem .8rem;min-height:345px;} .cxr-card-title{font-size:1.02rem;} .cxr-score{font-size:1.55rem;} .cxr-mini-row{grid-template-columns:48px 39px 1fr;gap:.2rem;font-size:.63rem;}}
</style>
""", unsafe_allow_html=True)

FINDINGS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
KO = {"Cardiomegaly":"심비대", "Edema":"폐부종", "Pleural Effusion":"흉수"}
CLASS_KO = {"Negative":"음성", "Positive":"양성"}
CLASS_COLORS = {"Negative":"#19866b", "Positive":"#c14746"}
MAX_UPLOAD_MB = 20

@st.cache_resource(show_spinner="DenseNet121 U-Ignore 모델 로딩 중...")
def get_predictor(path: str) -> CXRInference:
    return CXRInference(model_path=path, device="cpu")


def overlay_image(image: Image.Image, cam: np.ndarray, image_size: int, strength: float = 0.45) -> Image.Image:
    tensor = preprocess_image(image, image_size=image_size)
    pixels = tensor.permute(1,2,0).cpu().numpy()
    pixels = np.clip(pixels * np.asarray(STD) + np.asarray(MEAN), 0, 1)
    cam = np.asarray(cam, dtype=np.float32)
    if cam.shape != pixels.shape[:2] or not np.isfinite(cam).all():
        raise ValueError("Grad-CAM의 크기 또는 값이 올바르지 않습니다.")
    cam = np.clip(cam,0,1)
    heatmap = colormaps["jet"](cam)[..., :3]
    alpha = (strength * cam)[...,None]
    overlay = np.clip(pixels * (1-alpha) + heatmap * alpha,0,1)
    return Image.fromarray(np.rint(overlay*255).astype(np.uint8))


def processed_image(image: Image.Image, image_size: int) -> Image.Image:
    tensor = preprocess_image(image, image_size=image_size)
    px = tensor.permute(1,2,0).cpu().numpy()
    px = np.clip(px * np.asarray(STD) + np.asarray(MEAN),0,1)
    return Image.fromarray(np.rint(px*255).astype(np.uint8))


def png_bytes(image: Image.Image) -> bytes:
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def interactive_viewer(image: Image.Image, key: str, height: int = 475) -> None:
    """Interactive pan/zoom via Plotly, no remote viewer or patient-image upload."""
    rgb = np.asarray(image.convert("RGB"))
    width, ih = image.size
    fig = go.Figure(go.Image(z=rgb))
    fig.update_xaxes(visible=False, range=[-0.5, width - 0.5], constrain="domain")
    fig.update_yaxes(visible=False, range=[ih - 0.5, -0.5], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=height,
        dragmode="pan", paper_bgcolor="#0b1622", plot_bgcolor="#0b1622",
    )
    st.plotly_chart(
        fig, use_container_width=True, key=key,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawrect"],
            "toImageButtonOptions": {"format": "png", "filename": "cxr_viewer"},
        },
    )



def korean_pdf_font() -> str:
    regular_name = "CXR_Pretendard"
    bold_name = "CXR_Pretendard_Bold"

    font_dir = PROJECT_ROOT / "streamlit-web" / "app" / "assets"

    if regular_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(
                regular_name,
                str(font_dir / "Pretendard-Regular.ttf")
            )
        )

    if bold_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(
                bold_name,
                str(font_dir / "Pretendard-Bold.ttf")
            )
        )

    return regular_name



def pdf_wrapped_lines(text: str, font: str, size: int, max_width: float) -> list[str]:
    """Wrap Korean text using PDF font metrics, including no-space long strings."""
    result, line = [], ""
    for ch in text:
        trial = line + ch
        if line and pdfmetrics.stringWidth(trial, font, size) > max_width:
            result.append(line)
            line = ch
        else:
            line = trial
    if line:
        result.append(line)
    return result


def create_pdf_report(analysis: dict) -> bytes:
    """한글 의료영상 AI 연구용 결과지. 분석 결과는 현재 세션에서만 사용합니다."""
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(595.28, 841.89))
    pdf.setTitle("CXR Insight 흉부 X-ray AI 분석 결과지")
    pdf.setAuthor("CXR Insight")
    font = korean_pdf_font()
    bold_font = "CXR_Pretendard_Bold"
    navy = colors.HexColor("#142D49")
    blue = colors.HexColor("#2563A6")
    muted = colors.HexColor("#62758A")
    pale = colors.HexColor("#F3F7FC")
    border = colors.HexColor("#DDE7F1")
    positive = colors.HexColor("#B94449")
    negative = colors.HexColor("#2563EB")
    
    def text(x, y, value, size=9, color=navy, bold=False):
        pdf.setFillColor(color)
        pdf.setFont(bold_font if bold else font, size)
        pdf.drawString(x, y, str(value))


    def centered(x, y, value, size=9, color=navy):
        pdf.setFillColor(color)
        pdf.setFont(font, size)
        pdf.drawCentredString(x, y, str(value))

    def pill(x, y, label, color, width=52):
        pdf.setFillColor(color)
        pdf.roundRect(x, y-5, width, 19, 9, fill=1, stroke=0)
        centered(x+width/2, y+1, label, 9, colors.white)

    # 헤더 / 비진단용 구분
    pdf.setFillColor(navy)
    pdf.rect(0, 746, 595.28, 95.89, fill=1, stroke=0)
    text(36, 806, "CXR Insight", 20, colors.white, bold=True)
    text(37, 782, "흉부 X-ray AI 분석 결과지", 13, colors.white, bold=True)
    pdf.setFillColor(colors.HexColor("#345371"))
    pdf.roundRect(426, 784, 132, 25, 10, fill=1, stroke=0)
    centered(492, 792, "연구·교육용", 10, colors.white)
    text(37, 762, "의료 진단용이 아닌 연구용 프로토타입 분석 결과", 9, colors.HexColor("#D5E3F1"))

    # 검사 및 모델 정보
    pdf.setFillColor(pale)
    pdf.roundRect(36, 670, 523, 63, 9, fill=1, stroke=0)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    text(49, 710, "분석 일시", 9, muted)
    text(49, 689, generated, 10)
    text(244, 710, "분석 모델", 9, muted)
    text(244, 689, "DenseNet121 · U-Ignore", 10)
    text(415, 710, "모델 입력", 9, muted)
    text(415, 689, f"{analysis['image_size']} × {analysis['image_size']}", 10)

    # 예측 표
    text(37, 644, "01  AI 소견 예측 결과", 13, bold=True)
    pdf.setFillColor(navy)
    pdf.roundRect(36, 602, 523, 26, 5, fill=1, stroke=0)
    # 표 열 너비: 165 + 110 + 124 + 124 = 523
    widths = [165, 110, 124, 124]
    x0 = 36
    labels = ["소견", "예측 결과", "음성 확률", "양성 확률"]
    offset=0
    for label, w in zip(labels,widths):
        centered(x0+offset+w/2, 611, label, 9, colors.white)
        offset += w
    for idx, item in enumerate(analysis["findings"]):
        row_top = 602 - idx*37
        row_bottom = row_top - 37
        pdf.setFillColor(colors.white if idx%2 else pale)
        pdf.rect(36, row_bottom, 523, 37, fill=1, stroke=0)
        name = KO.get(item["finding"], item["finding"])
        pred = item["prediction"]
        probs = item["probabilities"]
        text(47, row_bottom+13, name, 10)
        c = {"Positive":positive,"Negative":negative}.get(pred,muted)
        pill(201, row_bottom+12, CLASS_KO.get(pred,pred), c, 52)
        for x, val in zip((373, 497), (probs["negative"], probs["positive"])):
            centered(x,row_bottom+13,f"{val:.1%}",10)
    threshold_text = " / ".join(
        f"{name} {threshold:.2f}"
        for name, threshold in zip(
            ["심비대", "폐부종", "흉수"],
            THRESHOLDS
        )
    )

    text(
        39, 476,
        f"양성 판정 임계값: {threshold_text}",
        8, muted
    )

    # 모델 입력 영상 및 Grad-CAM 4개
    text(37, 445, "02  X-ray 영상 및 소견별 Grad-CAM", 13, bold=True)
    text(38, 427, "Grad-CAM은 각 소견의 최종 예측 클래스 점수를 기준으로 생성합니다.", 8, muted)
    images = [processed_image(analysis["image"], analysis["image_size"])]
    images.extend(overlay_image(analysis["image"],analysis["cams"][n], analysis["image_size"],.45) for n in FINDINGS)
    captions = ["모델 입력 X-ray"] + [
        f"{KO[item['finding']]} ({CLASS_KO[item['prediction']]})"
        for item in analysis["findings"]
    ]
    for i,(im,label) in enumerate(zip(images,captions)):
        x=36+i*132
        pdf.setFillColor(pale)
        pdf.roundRect(x, 275, 127, 139, 7, fill=1, stroke=0)
        pdf.drawImage(ImageReader(im),x+5,284,width=117,height=117,mask='auto')
        centered(x+63.5,266,label,9)

    # 안내문 카드
    pdf.setFillColor(pale)
    pdf.roundRect(36, 96, 523, 151, 9, fill=1, stroke=0)
    text(48, 224, "03  AI 분석 결과 해석 안내", 12, bold=True)
    
    notes = [
        "본 보고서는 연구·교육용이며 의료 진단을 대체하지 않습니다.",
        "Grad-CAM: 붉은색은 예측 클래스(음성·양성) 점수에 기여한 영역입니다.",
        "붉은색이 양성이나 병변을 뜻하지 않으며, 병변 위치를 확정하지 않습니다.",
        "검은 패딩은 전처리 결과이며, 소견별 히트맵 색상은 직접 비교할 수 없습니다.",
        "출력 확률은 실제 질환 확률이 아니며, 모델의 추가 검증이 필요합니다.",
    ]

    cursor = 203

    for note in notes:
        lines = pdf_wrapped_lines(note, font, 8.5, 493)
        for line in lines:
            text(48, cursor, line, 8.5, muted)
            cursor -= 12
        cursor -= 3

    if cursor < 108:
        raise ValueError("PDF 안내 문구가 표시 영역을 벗어납니다.")


    pdf.setStrokeColor(border)
    pdf.line(36,79,559,79)
    text(37,61,"CXR Insight  |  연구용 AI 분석 결과",8,muted)
    pdf.setFillColor(blue)
    pdf.setFont(font,8)
    pdf.drawRightString(558,61,"1 / 1",)
    pdf.save()
    return buf.getvalue()


def result_card(item: dict) -> str:
    """Return a two-class U-Ignore result card."""
    name = item["finding"]
    pred = item["prediction"]
    probs = item["probabilities"]
    probability = max(0.0, min(1.0, float(probs[pred.lower()])))
    pct = probability * 100
    tone = pred.lower()
    interpretation = {
        "Negative": f"모델은 이 영상의 {KO[name]} 소견을 음성 클래스로 분류했습니다.",
        "Positive": f"모델은 이 영상의 {KO[name]} 소견을 양성 클래스로 분류했습니다.",
    }[pred]
    rows = []
    for cls, bar_color in (("negative", "#2563EB"), ("positive", "#cf5760")):
        val = max(0.0, min(1.0, float(probs[cls]))) * 100
        rows.append(
            f'<div class="cxr-mini-row"><span>{CLASS_KO[cls.title()]}</span>'
            f'<b>{val:.1f}%</b><div class="cxr-mini-track">'
            f'<div class="cxr-mini-fill" style="width:{val:.1f}%;background:{bar_color}"></div>'
            f'</div></div>'
        )
    return f"""
    <div class="cxr-card cxr-card--{tone}">
      <div class="cxr-card-head">
        <div><div class="cxr-card-title">{KO[name]}</div><div class="cxr-card-sub">{name}</div></div>
        <span class="cxr-status">{CLASS_KO[pred]}</span>
      </div>
      <div class="cxr-score-row"><span class="cxr-score-name">예측 클래스 확률</span><strong class="cxr-score">{pct:.1f}%</strong></div>
      <div class="cxr-track"><div class="cxr-track-fill" style="width:{pct:.1f}%"></div></div>
      <div class="cxr-interpret">{interpretation}</div>
      <div class="cxr-breakdown"><div class="cxr-breakdown-title">클래스별 출력 확률</div>{''.join(rows)}</div>
    </div>"""



with st.sidebar:
    st.markdown(
        """
        <div style="
            font-size: 30px;
            font-weight: 850;
            line-height: 1.3;
            white-space: nowrap;
            margin-bottom: 8px;
        ">
            🩻 CXR Insight
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Chest X-ray AI Analysis")
    st.divider()

    st.markdown("**ANALYSIS**")
    st.write("01　X-ray 업로드")
    st.write("02　AI 소견 분석")
    st.write("03　Grad-CAM 비교")

    st.divider()

    st.markdown("**MODEL**")
    st.write("DenseNet121 · U-Ignore")
    st.caption("Cardiomegaly / Edema / Pleural Effusion")

    st.divider()

    st.markdown(
        """
        <div style="font-size: 13px; line-height: 1.8; color: #cbd5e1;">
            연구·교육용 프로토타입.<br>
            의료진의 진단을 대체하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown('<div class="cxr-kicker">Medical imaging · AI interpretation</div>', unsafe_allow_html=True)
st.title("Chest X-ray Analysis")
st.markdown('<div class="cxr-sub">흉부 X-ray의 심비대 · 폐부종 · 흉수 소견을 AI로 분석합니다.</div>', unsafe_allow_html=True)
st.info("연구·교육용 서비스입니다. 예측 클래스 확률은 실제 질병 확률 또는 임상적 확진을 의미하지 않습니다.")

left, right = st.columns([0.95, 2.05], gap="large")
with left:
    st.markdown('<div class="cxr-section">01　X-ray 이미지 업로드</div>',unsafe_allow_html=True)
    uploaded = st.file_uploader("흉부 X-ray 이미지 (JPG/PNG · 최대 20MB)",type=["jpg","jpeg","png"])
    image = None
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB*1024*1024:
            st.error(f"파일 크기는 {MAX_UPLOAD_MB}MB 이하여야 합니다.")
        else:
            try:
                image = Image.open(BytesIO(uploaded.getvalue()))
                image.load()
                image = image.convert("RGB")
                view_source = st.radio("영상 표시", ["원본", "모델 입력 영상"], horizontal=True, key="viewer_source")
                if view_source == "모델 입력 영상":
                    view_img = processed_image(image, 224)
                else:
                    view_img = image
                interactive_viewer(view_img, key="main_xray_viewer", height=450)
                st.caption(f"원본 크기: {image.width} × {image.height} px · 마우스 휠 확대/축소 · 드래그 이동 · 더블클릭 초기화")
            except (UnidentifiedImageError,OSError,ValueError) as exc:
                st.error(f"이미지 파일을 읽을 수 없습니다: {exc}")
    run = st.button("AI 분석 시작",type="primary",use_container_width=True,disabled=image is None)

# Clear stale results before any new analysis, including upload removal.
file_key = (uploaded.name,hashlib.sha256(uploaded.getvalue()).hexdigest()) if uploaded is not None else None
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state.pop("analysis",None)

if run and image is not None:
    try:
        if not Path(MODEL_PATH).is_file():
            raise FileNotFoundError(f"모델 체크포인트가 존재하지 않습니다: {MODEL_PATH}")
        predictor = get_predictor(str(MODEL_PATH))
        start = time.perf_counter()
        with st.spinner("AI 분석 및 Grad-CAM 생성 중..."):
            output = predictor.predict(image,include_gradcam=True)
            findings = output["findings"]
            if len(findings)!=3 or set(output["gradcam"])!=set(FINDINGS):
                raise ValueError("예측 또는 Grad-CAM 결과 구성이 예상과 다릅니다.")
            if [f["finding"] for f in findings]!=FINDINGS:
                raise ValueError("소견 출력 순서가 예상과 다릅니다.")
            for item in findings:
                if item["prediction"] not in CLASS_KO or set(item["probabilities"]) != {"negative", "positive"}:
                    raise ValueError("U-Ignore 예측은 음성/양성 두 클래스여야 합니다.")
            st.session_state["analysis"] = {
                "findings":findings,
                "cams":{k:np.asarray(output["gradcam"][k]).copy() for k in FINDINGS},
                "image":image.copy(),
                "image_size":predictor.image_size,
                "elapsed":time.perf_counter()-start,
                "file_key":file_key,
            }
    except Exception as exc:
        st.session_state.pop("analysis",None)
        st.error(f"분석 실패 ({type(exc).__name__}): {exc}")

saved = st.session_state.get("analysis")
if saved is not None and saved["file_key"] != file_key:
    saved = None

with right:
    st.markdown('<div class="cxr-section">02　AI 소견 분석 결과</div>', unsafe_allow_html=True)
    st.markdown('<div class="cxr-section-sub">세 소견에 대한 예측 클래스와 각 클래스의 출력 확률을 표시합니다.</div>', unsafe_allow_html=True)
    if saved is None:
        st.markdown(
            '<div class="cxr-wait"><div style="font-size:2rem">🩻</div>'
            '<div class="cxr-wait-title">분석 대기 중</div>'
            '<div class="cxr-wait-sub">왼쪽에서 X-ray 이미지를 업로드한 뒤 <b>AI 분석 시작</b>을 누르면 세 소견의 예측 결과를 이곳에서 확인할 수 있습니다.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        card_html = "".join(result_card(item) for item in saved["findings"])
        st.markdown(
            f'<div class="cxr-results-grid">{card_html}</div>',
            unsafe_allow_html=True,
        )
        
        st.markdown(
            f'<div class="cxr-model-row">DenseNet121 · U-Ignore　|　분석 소요 시간: {saved["elapsed"]:.1f}초</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="cxr-notice"><b>결과 해석 안내</b> — 표시된 수치는 모델의 클래스 출력 확률이며 실제 질병의 존재 확률 또는 진단 결과가 아닙니다.</div>',
            unsafe_allow_html=True,
        )

    # No Grad-CAM image, thumbnail, or slider is visible until the user expands this section.
    st.markdown('<div style="height:13px"></div>', unsafe_allow_html=True)
    with st.expander("03　소견별 Grad-CAM 시각화 · 클릭해서 보기", expanded=False):
        st.caption(
            "심비대·폐부종·흉수 각각의 **예측 클래스 기준 Grad-CAM**을 보여줍니다. "
            "활성화가 높은 영역은 병변의 확정 위치가 아닙니다."
        )
        if saved is None:
            st.info("X-ray 분석 후 Grad-CAM 시각화를 확인할 수 있습니다.")
        else:
            strength = st.slider(
                "히트맵 오버레이 강도", 0, 80, 45, 5,
                help="이미지 표시 강도만 변경하며 모델 예측은 바뀌지 않습니다.",
            ) / 100
            cam_cols = st.columns(3, gap="small")
            for col, item in zip(cam_cols, saved["findings"]):
                finding = item["finding"]
                with col:
                    st.markdown(f"**{KO[finding]}**")
                    overlay = overlay_image(saved["image"], saved["cams"][finding], saved["image_size"], strength)
                    st.image(overlay, use_container_width=True)
                    pred = item["prediction"]
                    st.caption(f"예측: {CLASS_KO[pred]} · {item['probabilities'][pred.lower()]:.1%}")
                    st.download_button(
                        "히트맵 저장", png_bytes(overlay),
                        file_name=f"gradcam_{finding.replace(' ', '_')}.png",
                        mime="image/png", key=f"download_{finding}",
                        use_container_width=True,
                    )
            st.caption("소견별 히트맵은 독립적으로 정규화되어 색상 강도를 직접 비교할 수 없습니다. 패딩 영역의 활성화에도 유의하세요.")

    # PDF remains available independent of whether the Grad-CAM panel is expanded.
    if saved is not None:
        st.markdown('<div class="cxr-section">04　분석 보고서 다운로드</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="cxr-report-intro">AI 예측 결과와 소견별 Grad-CAM을 포함한 연구용 PDF 보고서를 저장할 수 있습니다.</div>',
            unsafe_allow_html=True,
        )
        try:
            report = create_pdf_report(saved)
            st.download_button(
                "PDF 분석 보고서 다운로드", report,
                file_name="cxr_insight_analysis_report.pdf", mime="application/pdf",
                key="download_pdf_report", use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF 보고서 생성 실패 ({type(exc).__name__}): {exc}")
