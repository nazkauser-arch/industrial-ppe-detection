from pathlib import Path
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

API_URL = "http://127.0.0.1:8000/predict"

def load_css(file_name):
    css_path = Path(__file__).parent / file_name
    with open(css_path, encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )

def draw_detections(image, detections):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        bbox = detection["bbox"]
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        label = f'{detection["class"]} {detection["confidence"]:.2f}'

        is_violation = detection["class"].startswith("no_")
        color = "red" if is_violation else "lime"

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=3
        )

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]

        label_y = max(0, y1 - text_height - 6)

        draw.text(
            (x1, label_y),
            label,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="black"
        )

    return image

st.set_page_config(
    page_title="Industrial PPE Detection",
    page_icon="🦺",
    layout="wide"
)

import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def load_css(file_name):
    css_path = Path(__file__).parent / file_name
    with open(css_path, encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )

st.set_page_config(
    page_title="Industrial PPE Detection",
    page_icon="🦺",
    layout="wide"
)

load_css("style.css")

st.title("Industrial PPE Detection System")
st.write("Upload an image to detect personal protective equipment and safety violations.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", width=350)
    
    if st.button("Detect PPE", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    data={"conf": confidence},
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()
                annotated_image = draw_detections(
                    image,
                    result["detections"]
                )

                with col2:
                    st.image(
                        annotated_image,
                        caption="Annotated Image",
                        width=350
                    )

                st.subheader("Detection Results")
                st.metric("Total Detections", result["total_detections"])

                st.write(
                    f"Confidence Threshold: "
                    f"{result['confidence_threshold']}"
                )

                if result["detections"]:
                    st.dataframe(
                        [
                            {
                                "Class": detection["class"],
                                "Confidence": detection["confidence"],
                                "Bounding Box": detection["bbox"]
                            }
                            for detection in result["detections"]
                        ],
                        use_container_width=True
                    )
                else:
                    st.info("No PPE objects detected in this image.")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure FastAPI is running.")

            except requests.exceptions.Timeout:
                st.error("The prediction timed out. Please try again.")

            except requests.exceptions.HTTPError:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
