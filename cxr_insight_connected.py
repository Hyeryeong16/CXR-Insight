"""CXR Insight: Streamlit interface using test_hr/cxr_inference.py.
Run from the project root: python -m streamlit run cxr_insight_connected.py
"""
from io import BytesIO
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import colormaps
from PIL import Image, UnidentifiedImageError

from cxr_preprocessing import preprocess_image, MEAN, STD
from test_hr.cxr_inference import CXRInference, MODEL_PATH

st.set_page_config(page_title="CXR Insight", page_icon="🩻", layout="wide")
st.markdown("""
<style>
.stApp {background: #f5f8fc;}
[data-testid="stSidebar"] {background: #102a46;}
[data-testid="stSidebar"] * {color: #eef5ff !important;}
.block-container {max-width: 1530px; padding-top: 1.5rem;}
h1, h2, h3 {color: #102a46;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #dfe8f4; border-radius: 12px; padding: 12px;}
</style>
""", unsafe_allow_html=True)

KOREAN = {"Cardiomegaly": "심비대", "Edema": "폐부종", "Pleural Effusion": "흉수"}
CLASS_KO = {"Negative": "음성", "Positive": "양성", "Uncertain": "불확실"}

@st.cache_resource(show_spinner="DenseNet121 모델을 불러오는 중입니다...")
def get_predictor(model_path: str):
    # Use existing trained model & inference code, without modifying either.
    return CXRInference(model_path=model_path, device="cpu")


def make_overlay(image: Image.Image, cam: np.ndarray, image_size: int) -> Image.Image:
    """Overlay a Grad-CAM array on the *same padded/resized image* seen by the model."""
    tensor = preprocess_image(image, image_size=image_size)
    pixels = tensor.permute(1, 2, 0).cpu().numpy()
    pixels = np.clip(pixels * np.asarray(STD) + np.asarray(MEAN), 0, 1)
    cam = np.asarray(cam, dtype=np.float32)
    if cam.shape != pixels.shape[:2]:
        raise ValueError(f"Grad-CAM 크기 불일치: {cam.shape} / {pixels.shape[:2]}")
    if not np.isfinite(cam).all():
        raise ValueError("Grad-CAM에 유효하지 않은 값이 있습니다.")
    cam = np.clip(cam, 0, 1)
    heatmap = colormaps["jet"](cam)[..., :3]
    alpha = (0.4 * cam)[..., None]
    overlay = np.clip(pixels * (1 - alpha) + heatmap * alpha, 0, 1)
    return Image.fromarray((overlay * 255).round().astype(np.uint8))


def png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


with st.sidebar:
    st.title("🩻 CXR Insight")
    st.caption("AI Chest X-ray Analysis")
    st.divider()
    st.markdown("**분석 메뉴**")
    st.markdown("01 · X-ray 업로드")
    st.markdown("02 · AI 소견 분석")
    st.markdown("03 · 소견별 Grad-CAM")
    st.divider()
    st.markdown("**Model Information**")
    st.write("DenseNet121 / U-MultiClass")
    st.write("Cardiomegaly · Edema · Pleural Effusion")
    st.caption("연구·교육용 프로토타입 · 의료 진단용 아님")

st.title("Chest X-ray Analysis")
st.caption("흉부 X-ray 이미지 한 장에 대한 세 소견의 예측과 Positive 클래스 Grad-CAM을 함께 보여줍니다.")
st.info("연구·교육용 시스템입니다. 결과는 의료 진단을 대체하지 않으며, 모델 성능은 별도 검증이 필요합니다.")

left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("01  X-ray 이미지 업로드")
    uploaded = st.file_uploader("JPG 또는 PNG 이미지", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            image.load()
            image = image.convert("RGB")
            st.image(image, caption="업로드한 원본 X-ray", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"이미지 읽기 실패: {exc}")
    analyze = st.button("AI 분석 시작", type="primary", disabled=image is None, use_container_width=True)

# Avoid showing an old patient's result after upload changes.
current_key = (uploaded.name, hashlib.sha256(uploaded.getvalue()).hexdigest()) if uploaded is not None else None
if st.session_state.get("last_file_key") != current_key:
    st.session_state["last_file_key"] = current_key
    st.session_state.pop("analysis_result", None)

if analyze and image is not None:
    try:
        if not Path(MODEL_PATH).is_file():
            raise FileNotFoundError(f"설정된 모델이 서버에 없습니다: {MODEL_PATH}")
        predictor = get_predictor(str(MODEL_PATH))
        with st.spinner("모델 추론 및 Grad-CAM 3개 생성 중..."):
            result = predictor.predict(image, include_gradcam=True)
        if len(result.get("findings", [])) != 3 or len(result.get("gradcam", {})) != 3:
            raise RuntimeError("예측 또는 Grad-CAM 결과가 3개가 아닙니다.")
        overlays = {
            name: make_overlay(image, result["gradcam"][name], predictor.image_size)
            for name in KOREAN
        }
        st.session_state["analysis_result"] = (result["findings"], overlays, image.copy())
    except Exception as exc:
        st.session_state.pop("analysis_result", None)
        st.error(f"분석 실패: {type(exc).__name__}: {exc}")

with right:
    st.subheader("02  AI 분석 결과 (U-MultiClass)")
    saved = st.session_state.get("analysis_result")
    if saved is None:
        st.info("왼쪽에서 X-ray를 업로드하고 'AI 분석 시작'을 누르면 실제 모델 결과가 표시됩니다.")
    else:
        findings, overlays, analyzed_image = saved
        cols = st.columns(3)
        for col, finding in zip(cols, findings):
            with col:
                name = finding["finding"]
                prediction = finding["prediction"]
                probabilities = finding["probabilities"]
                st.markdown(f"**{KOREAN[name]}**  ")
                st.caption(name)
                st.metric("예측", CLASS_KO[prediction])
                st.metric("예측 클래스 확률", f"{probabilities[prediction.lower()]:.1%}")
                with st.expander("클래스별 확률"):
                    for cls in ("negative", "positive", "uncertain"):
                        st.write(f"{cls.title()}: {probabilities[cls]:.1%}")

st.divider()
st.subheader("03  소견별 예측 및 Grad-CAM")
st.caption("세 히트맵은 동일한 X-ray에 대한 **Positive 클래스** Grad-CAM입니다. 색상이 붉을수록 해당 클래스 점수에 대한 상대적 기여가 큰 영역입니다.")
saved = st.session_state.get("analysis_result")
if saved is None:
    st.info("분석이 완료되면 원본 X-ray + 심비대·폐부종·흉수 Grad-CAM 네 이미지가 나란히 표시됩니다.")
else:
    findings, overlays, analyzed_image = saved
    columns = st.columns(4, gap="small")
    with columns[0]:
        st.markdown("**원본 X-ray**")
        st.image(analyzed_image, use_container_width=True)
    for col, item in zip(columns[1:], findings):
        finding = item["finding"]
        with col:
            st.markdown(f"**{KOREAN[finding]}**")
            st.image(overlays[finding], use_container_width=True)
            cls = item["prediction"]
            confidence = item["probabilities"][cls.lower()]
            st.caption(f"{CLASS_KO[cls]} · {confidence:.1%}")
            st.download_button(
                "Grad-CAM 저장", png_bytes(overlays[finding]),
                file_name=f"gradcam_{finding.replace(' ', '_')}.png",
                mime="image/png", key=f"download_{finding}", use_container_width=True,
            )
    st.caption("Grad-CAM은 의학적 병변 위치를 확정하는 지도가 아닙니다. 모델의 Positive 클래스 활성화 근거를 참고용으로 보여줍니다.")
