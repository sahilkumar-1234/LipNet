#Import dependencies 
import streamlit as st 
import numpy as np 
import torch
import os 
import imageio
from data_preprocessing import load_video, decode, load_annot
from data_preprocessing import ctc_decode
from model import LipNet
from torchnlp.encoders import LabelEncoder


device = torch.device("cpu")
media_path = os.path.join(os.path.dirname(__file__), "../media")
model_path = os.path.join(os.path.dirname(__file__), "../models")
data_path = os.path.join(os.path.dirname(__file__), "../data")

#Layout config
st.set_page_config(layout="wide")
st.title("Lip Reader App")

# Session states
if "loss" not in st.session_state:
    st.session_state["loss"] = False
if "split" not in st.session_state:
    st.session_state["split"] = "training_files"
if "model" not in st.session_state:
    st.session_state["selected_model"] = "LIPNET_100_EPOCHS.pt"
if "raw_button" not in st.session_state:
    st.session_state["raw_button"] = False
if "decoded_button" not in st.session_state:
    st.session_state["decoded_button"] = False

# --- SIDEBAR ---
with st.sidebar:
    st.image(os.path.join(media_path, "computerVision.jpg"))
    
    st.info("""This application is an implemention of the LipNet paper. 
            The goal is to create a computer vision model (LipNet) 
            that can read lips through videos. Hope you enjoy it!!""")
    
    def click_loss_button():
        st.session_state.loss = True
    def unclick_loss_button():
        st.session_state.loss = False
    
    col00, col01 = st.columns(2)
    with col00:
        st.button("Show loss graph", on_click=click_loss_button)
    with col01:
        st.button("Hide loss graph", on_click=unclick_loss_button)

    if st.session_state.loss: 
        st.image(os.path.join(media_path, "LossGraph.png"))


# --- VIDEO SELECTION / UPLOAD SECTION ---
st.header("📹 Choose or Upload Your Video")

upload_mode = st.radio(
    "Select Input Mode:",
    ["Use dataset video", "Upload your own video", "Record from camera"],
    horizontal=True
)

uploaded_video_path = None

if upload_mode == "Use dataset video":
    files = os.listdir(os.path.join(data_path, "s1"))
    split = ["training files", "validation files"]
    st.session_state["split"] = st.selectbox("Choose between training and validation split", split)
    n = len(files)
    train_files = files[: int(0.9 * n)]
    val_files = files[int(0.9 * n):]
    options = {"training files": train_files, "validation files": val_files}
    selected_option = st.selectbox(
        f"Choose a video from {st.session_state.split}",
        options[st.session_state["split"]]
    )
    if selected_option:
        uploaded_video_path = os.path.join(data_path, "s1", selected_option)

elif upload_mode == "Upload your own video":
    uploaded_file = st.file_uploader("Upload your own video (MP4 format)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        uploaded_video_path = "uploaded_video.mp4"
        st.success("✅ Your video was uploaded successfully!")

elif upload_mode == "Record from camera":
    recorded_file = st.camera_input("Record a short video sample")
    if recorded_file is not None:
        with open("recorded_video.mp4", "wb") as f:
            f.write(recorded_file.read())
        uploaded_video_path = "recorded_video.mp4"
        st.success("🎥 Your recording is ready!")


# --- CONTINUE ONLY IF WE HAVE A VALID VIDEO ---
if uploaded_video_path:
    file_path = uploaded_video_path
    st.info(f"Using video: **{os.path.basename(file_path)}**")

    # Convert and render the video
# Convert and render the video safely (absolute path)
    converted_path = os.path.join(os.path.dirname(__file__), "selected_video.mp4")
    
    # Run ffmpeg to convert the file to standard MP4
    os.system(f"ffmpeg -i \"{file_path}\" -vcodec libx264 \"{converted_path}\" -y")
    
    # Check and display
    if os.path.exists(converted_path):
        with open(converted_path, "rb") as video:
            video_bytes = video.read()
        st.video(video_bytes)
    else:
        st.error("❌ Could not find or generate selected_video.mp4 — please check ffmpeg or your file path.")


    # If it's a dataset video, try loading true labels
    filename = os.path.splitext(os.path.basename(file_path))[0]
    labels_path = os.path.join(data_path, "align", f"{filename}.align")
    if os.path.exists(labels_path):
        labels = load_annot(labels_path, True)
        delim = ""
        st.text(f"The true labels are : {delim.join(decode(labels))}")

    # --- MODEL SECTION ---
    st.info("LIPNET_100_EPOCHS performs the best on test data")
    models = os.listdir(model_path)
    st.session_state.selected_model = st.selectbox("Choose a model", models)

    st.info("This is what the ML model sees")
    video_tensor = load_video(file_path, from_path=True)

    # Convert to GIF
    frames_np = video_tensor.numpy()
    frames_scaled = ((frames_np - frames_np.min()) / (frames_np.max() - frames_np.min()) * 255).astype(np.uint8)
    frames_list = [frame for frame in frames_scaled]
    imageio.mimsave("animation.gif", frames_list, fps=10)
    st.image(os.path.join(os.path.dirname(__file__), "animation.gif"), width=400)

    # Preprocess frames
    frames = video_tensor.float().unsqueeze(0).permute(1, 2, 3, 0).unsqueeze(0)

    # Load model 
    st.info("This is the output of the model")
    model = torch.load(os.path.join(model_path, st.session_state.selected_model),
                       weights_only=False, map_location=device).to(device)
    y_hat = model(frames)
    y_pred = torch.argmax(y_hat, axis=2).squeeze(dim=0)

    # Raw output buttons
    def click_raw_button():
        st.session_state.raw_button = True
    def unclick_raw_button():
        st.session_state.raw_button = False

    col3, col4 = st.columns(2)
    with col3:
        st.button("Show raw model outputs", on_click=click_raw_button)
    with col4:
        st.button("Hide raw model outputs", on_click=unclick_raw_button)

    if st.session_state.raw_button:
        st.text(y_pred)

    # Decode output
    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    encoder = LabelEncoder(vocab, reserved_labels=['...'], unknown_index=-1)

    st.info("This is the decoded output")

    y_pred_ctc = ctc_decode(y_pred)
    delim = ""
    seq = delim.join(decode(y_pred_ctc))

    col5, col6 = st.columns(2)
    def click_decoded_button():
        st.session_state.decoded_button = True
    def unclick_decoded_button():
        st.session_state.decoded_button = False

    with col5:
        st.button("Show decoded model output", on_click=click_decoded_button)
    with col6:
        st.button("Hide decoded model output", on_click=unclick_decoded_button)

    if st.session_state.decoded_button:
        st.text(seq)
