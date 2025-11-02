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
    val_files = files[int(0.9 * n) :]
    options = {"training files": train_files, "validation files": val_files}
    selected_option = st.selectbox(
        f"Choose a video from {st.session_state.split}",
        options[st.session_state.split]
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

# --- Continue only if we have a valid video ---
if uploaded_video_path:
    file_path = uploaded_video_path
    st.info(f"Using video: **{os.path.basename(file_path)}**")

    # Convert and render the video
    os.system(f"ffmpeg -i {file_path} -vcodec libx264 selected_video.mp4 -y")
    video = open("selected_video.mp4", "rb")
    video_bytes = video.read()
    st.video(video_bytes)

    # If it's a dataset video, try loading true labels
    filename = os.path.splitext(os.path.basename(file_path))[0]
    labels_path = os.path.join(data_path, "align", f"{filename}.align")
    if os.path.exists(labels_path):
        labels = load_annot(labels_path, True)
        delim = ""
        st.text(f"The true labels are : {delim.join(decode(labels))}")
