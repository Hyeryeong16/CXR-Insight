"""CXR Insight viewer + PDF report (v4).
Save as app/streamlit_app.py; run from project root with streamlit.
Requires cxr_preprocessing.py and test_hr/cxr_inference.py in the project root.
"""
from __future__ import annotations

from io import BytesIO
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from test_hr.cxr_inference import CXRInference, MODEL_PATH

st.set_page_config(page_title="CXR Insight | X-ray Analysis", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --navy:#142d49; --blue:#2563a6; --muted:#667b91; --line:#dce6f0; }
.stApp { background: #f5f8fc; }
[data-testid="stSidebar"] { background: #142d49; }
[data-testid="stSidebar"] * { color: #eef5ff !important; }
.block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 3rem; }
h1,h2,h3 { color:#142d49; letter-spacing:-.02em; }
[data-testid="stFileUploader"] { background:white; border-radius:14px; padding:.5rem; }
[data-testid="stButton"] button[kind="primary"] { background:#205c98; border:0; border-radius:10px; font-weight:650; }
[data-testid="stDownloadButton"] button { border-radius:10px; }
.cxr-kicker {color:#4476a5;font-weight:800;font-size:.77rem;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.3rem;}
.cxr-sub {color:#62768d;font-size:.94rem;margin-top:-.55rem;margin-bottom:1rem;}
.cxr-card {padding:1.08rem 1.05rem;border:1px solid #dce6f0;border-radius:15px;background:#fff;box-shadow:0 2px 9px rgba(18,48,79,.035);height:100%;}
.cxr-card-title {font-size:.93rem;font-weight:800;color:#183450;}
.cxr-card-sub {font-size:.73rem;color:#748398;margin-top:.06rem;}
.cxr-pred {font-size:1.3rem;font-weight:800;margin:.55rem 0 .1rem;}
.cxr-prob {color:#64748b;font-size:.77rem;margin-bottom:.5rem;}
.cxr-progress {height:8px;background:#eaf0f6;border-radius:99px;overflow:hidden;}
.cxr-progress-fill {height:100%;border-radius:99px;}
.cxr-foot {color:#63748a;font-size:.81rem;line-height:1.55;}
.cxr-section {font-size:1.16rem;font-weight:800;color:#142d49;margin:1.3rem 0 .55rem;}
.cxr-rule {border:0;border-top:1px solid #dce6f0;margin:1.1rem 0;}
.cxr-mode {display:inline-block;padding:.24rem .6rem;border-radius:99px;background:#e5eef9;color:#204f80;font-size:.73rem;font-weight:700;}
</style>
""", unsafe_allow_html=True)

FINDINGS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
KO = {"Cardiomegaly":"심비대", "Edema":"폐부종", "Pleural Effusion":"흉수"}
CLASS_KO = {"Negative":"음성", "Positive":"양성", "Uncertain":"불확실"}
CLASS_COLORS = {"Negative":"#19866b", "Positive":"#c14746", "Uncertain":"#aa771f"}
MAX_UPLOAD_MB = 20

@st.cache_resource(show_spinner="DenseNet121 모델 로딩 중...")
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
    """Register a Korean-capable PDF font; no font file is distributed with the app."""
    font_name = "CXRKorean"
    if font_name in pdfmetrics.getRegisteredFontNames():
        return font_name
    candidates = [
        PROJECT_ROOT / "app" / "assets" / "NanumGothic.ttf",  # optional local installation
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            pdfmetrics.registerFont(TTFont(font_name, str(path)))
            return font_name
    # Built-in Korean CID font: does not require bundling external font files.
    pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
    return "HYGothic-Medium"


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
    navy = colors.HexColor("#142D49")
    blue = colors.HexColor("#2563A6")
    muted = colors.HexColor("#62758A")
    pale = colors.HexColor("#F3F7FC")
    border = colors.HexColor("#DDE7F1")
    positive = colors.HexColor("#B94449")
    negative = colors.HexColor("#16836A")
    uncertain = colors.HexColor("#A77621")

    def text(x, y, value, size=9, color=navy):
        pdf.setFillColor(color)
        pdf.setFont(font, size)
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
    text(36, 806, "CXR Insight", 20, colors.white)
    text(37, 782, "흉부 X-ray AI 분석 결과지", 13, colors.white)
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
    text(244, 689, "DenseNet121 · Pilot", 10)
    text(415, 710, "모델 입력", 9, muted)
    text(415, 689, f"{analysis['image_size']} × {analysis['image_size']}", 10)

    # 예측 표
    text(37, 644, "01  AI 소견 예측 결과", 13)
    pdf.setFillColor(navy)
    pdf.roundRect(36, 602, 523, 26, 5, fill=1, stroke=0)
    # 표 열 너비: 138 + 85 + 100 + 100 + 100 = 523
    widths = [138, 85, 100, 100, 100]
    x0 = 36
    labels = ["소견", "예측 결과", "음성 확률", "양성 확률", "불확실 확률"]
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
        c = {"Positive":positive,"Negative":negative,"Uncertain":uncertain}.get(pred,muted)
        pill(179, row_bottom+12, CLASS_KO.get(pred,pred), c, 52)
        for x, val in zip((36+138+85+50,36+138+85+150,36+138+85+250),
                          (probs["negative"],probs["positive"],probs["uncertain"])):
            centered(x,row_bottom+13,f"{val:.1%}",10)
    text(39, 476, "표의 수치는 모델 출력 확률이며 실제 질병의 존재 확률이 아닙니다.", 8, muted)

    # 모델 입력 영상 및 Grad-CAM 4개
    text(37, 445, "02  X-ray 영상 및 소견별 Grad-CAM", 13)
    text(38, 427, "Grad-CAM은 각 소견의 양성(Positive) 클래스 점수를 기준으로 생성합니다.", 8, muted)
    images = [processed_image(analysis["image"], analysis["image_size"])]
    images.extend(overlay_image(analysis["image"],analysis["cams"][n], analysis["image_size"],.45) for n in FINDINGS)
    captions = ["모델 입력 X-ray", "심비대", "폐부종", "흉수"]
    for i,(im,label) in enumerate(zip(images,captions)):
        x=36+i*132
        pdf.setFillColor(pale)
        pdf.roundRect(x, 275, 127, 139, 7, fill=1, stroke=0)
        pdf.drawImage(ImageReader(im),x+5,284,width=117,height=117,mask='auto')
        centered(x+63.5,266,label,9)

    # 안내문 카드
    pdf.setFillColor(pale)
    pdf.roundRect(36, 96, 523, 151, 9, fill=1, stroke=0)
    text(48, 224, "03  AI 분석 결과 해석 안내", 12)
    notes = [
        "본 보고서는 연구·교육용 AI 분석 결과이며 의료 진단을 대체하지 않습니다.",
        "Grad-CAM: 붉은색은 양성 클래스 점수에 상대적으로 기여한 영역이며 병변의 확정 위치가 아닙니다. 검은 패딩은 인위적으로 추가된 영역입니다.",
        "예측 결과: 모델 출력 확률은 실제 질환 확률이 아니며, '불확실'은 학습된 라벨 클래스입니다.",
        "모델 한계: 현재 Pilot 모델의 성능과 일반화 가능성은 추가 검증이 필요합니다.",
        "히트맵은 소견별로 독립 정규화되므로 서로 색상 강도를 직접 비교할 수 없습니다.",
    ]
    cursor=203
    for note in notes:
        lines=pdf_wrapped_lines(note,font,8,493)
        for line in lines:
            text(48,cursor,line,8,muted)
            cursor-=12
        cursor-=4
    if cursor<100:
        raise ValueError("PDF 안내 문구가 표시 영역을 벗어납니다.")

    pdf.setStrokeColor(border)
    pdf.line(36,79,559,79)
    text(37,61,"CXR Insight  |  연구용 AI 분석 결과",8,muted)
    pdf.setFillColor(blue)
    pdf.setFont(font,8)
    pdf.drawRightString(558,61,"1 / 1",)
    pdf.save()
    return buf.getvalue()


def result_card(item: dict) -> None:
    name = item["finding"]
    pred = item["prediction"]
    prob = float(item["probabilities"][pred.lower()])
    color = CLASS_COLORS[pred]
    pct = max(0,min(100,prob*100))
    st.markdown(f"""<div class="cxr-card">
       <div class="cxr-card-title">{KO[name]}</div>
       <div class="cxr-card-sub">{name}</div>
       <div class="cxr-pred" style="color:{color}">{CLASS_KO[pred]} <span style="font-size:.72rem;font-weight:650">{pred}</span></div>
       <div class="cxr-prob">예측 클래스 확률 &nbsp; <b>{pct:.1f}%</b></div>
       <div class="cxr-progress"><div class="cxr-progress-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
    </div>""",unsafe_allow_html=True)
    with st.expander("3개 클래스 확률 보기"):
        for cls in ("negative", "positive", "uncertain"):
            st.caption(f"{CLASS_KO[cls.title()]}  {item['probabilities'][cls]:.1%}")
            st.progress(float(item["probabilities"][cls]))


with st.sidebar:
    st.markdown("## 🩻 CXR Insight")
    st.caption("Chest X-ray AI Analysis")
    st.divider()
    st.markdown("**ANALYSIS**")
    st.write("01　X-ray 업로드")
    st.write("02　AI 소견 분석")
    st.write("03　Grad-CAM 비교")
    st.divider()
    st.markdown("**MODEL**")
    st.write("DenseNet121 · U-MultiClass")
    st.caption("Cardiomegaly / Edema / Pleural Effusion")
    st.caption("Viewer: Plotly · Reports: PDF")
    st.markdown("<span class='cxr-mode'>Pilot model</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("연구·교육용 프로토타입. 의료진의 진단을 대체하지 않습니다.")

st.markdown('<div class="cxr-kicker">Medical imaging · AI interpretation</div>', unsafe_allow_html=True)
st.title("Chest X-ray Analysis")
st.markdown('<div class="cxr-sub">흉부 X-ray의 세 가지 소견을 분류하고, Positive 클래스 Grad-CAM을 비교합니다.</div>',unsafe_allow_html=True)
st.info("연구·교육용 서비스입니다. 예측 클래스 확률은 실제 질병 확률 또는 임상적 확진을 의미하지 않습니다.")

left, right = st.columns([1, 1.65], gap="large")
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
                interactive_viewer(view_img, key="main_xray_viewer")
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
    st.markdown('<div class="cxr-section">02　AI 소견 분석 결과</div>',unsafe_allow_html=True)
    if saved is None:
        st.markdown('<div class="cxr-card"><div class="cxr-card-title">분석 대기 중</div><p class="cxr-foot">왼쪽에서 이미지를 올리고 AI 분석 시작을 누르면 세 소견의 결과가 표시됩니다.</p></div>',unsafe_allow_html=True)
    else:
        cols = st.columns(3,gap="small")
        for col, item in zip(cols,saved["findings"]):
            with col:
                result_card(item)
        st.caption(f"분석 소요 시간(모델 추론 + Grad-CAM): {saved['elapsed']:.1f}초 · Pilot 모델")
        st.markdown('<div class="cxr-foot">Uncertain은 모델이 학습한 불확실 라벨 클래스이며, 임상적 불확실성이나 진단 결론을 나타내지 않습니다.</div>',unsafe_allow_html=True)


    # 03 is placed directly below 02 in the right-hand column.
    st.markdown('<div class="cxr-section">03　소견별 Grad-CAM 시각화</div>', unsafe_allow_html=True)
    st.caption("같은 X-ray에 대한 **Positive 클래스 Grad-CAM**입니다. 빨간색은 상대적 활성도가 높은 영역이지 병변 확정 표시가 아닙니다.")
    if saved is None:
        st.info("분석 후 심비대·폐부종·흉수의 Grad-CAM이 이곳에 표시됩니다.")
    else:
        strength = st.slider(
            "히트맵 오버레이 강도", 0, 80, 45, 5,
            help="시각화 강도만 변경하며 모델 예측은 변하지 않습니다."
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
        st.caption("소견별 히트맵은 독립적으로 정규화되어 색상 강도를 서로 직접 비교할 수 없습니다. 패딩 영역의 활성화에도 유의하세요.")


# Report exports are tied to the current input, never to a previous uploaded image.
if saved is not None:
    st.divider()
    st.markdown('<div class="cxr-section">04　분석 보고서 다운로드</div>', unsafe_allow_html=True)
    st.caption("실제 분석 결과 및 Positive 클래스 Grad-CAM을 포함한 익명 연구용 PDF입니다. 예측값은 확진 결과가 아닙니다.")
    try:
        report = create_pdf_report(saved)
        st.download_button(
            "PDF 분석 보고서 다운로드", report,
            file_name="cxr_insight_analysis_report.pdf", mime="application/pdf",
            key="download_pdf_report", use_container_width=False,
        )
    except Exception as exc:
        st.error(f"PDF 보고서 생성 실패 ({type(exc).__name__}): {exc}")
