import os
import cv2
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

import torch
# import faiss
import open_clip


@dataclass
class WindowItem:
    t_start: float
    t_end: float
    thumb_path: str  # representative frame path (for UI)


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def get_video_info(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, total_frames


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def build_windows(
    video_path: str,
    out_dir: str,
    window_sec: float = 3.0,
    stride_sec: float = 1.0,
    frames_per_window: int = 4,
    max_windows: int = 900,
) -> Tuple[List[WindowItem], List[List[str]]]:
    """
    Returns:
      - windows: list of WindowItem with timestamps + representative thumbnail
      - window_frame_paths: list parallel to windows, each containing frame paths for that window
    """
    os.makedirs(out_dir, exist_ok=True)
    fps, total_frames = get_video_info(video_path)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    windows: List[WindowItem] = []
    window_frame_paths: List[List[str]] = []

    w_frames = max(1, int(round(window_sec * fps)))
    s_frames = max(1, int(round(stride_sec * fps)))

    # frame positions to sample inside each window (evenly spaced)
    sample_positions = np.linspace(0, max(0, w_frames - 1), frames_per_window).astype(int).tolist()

    start_frame = 0
    w_idx = 0
    while start_frame < total_frames and w_idx < max_windows:
        end_frame = min(total_frames - 1, start_frame + w_frames - 1)
        t_start = start_frame / fps
        t_end = end_frame / fps

        paths = []
        # save sampled frames
        for j, off in enumerate(sample_positions):
            f_idx = min(end_frame, start_frame + off)
            img = read_frame_at(cap, f_idx)
            if img is None:
                continue
            p = os.path.join(out_dir, f"w{w_idx:05d}_f{j}_t{t_start:.2f}.jpg")
            img.save(p, quality=90)
            paths.append(p)

        if len(paths) > 0:
            thumb_path = paths[len(paths) // 2]  # representative frame
            windows.append(WindowItem(t_start=t_start, t_end=t_end, thumb_path=thumb_path))
            window_frame_paths.append(paths)
            w_idx += 1

        start_frame += s_frames

    cap.release()
    return windows, window_frame_paths


@st.cache_resource
def load_clip_model():
    # Good speed/quality tradeoff
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device


def embed_image_paths(model, preprocess, device, paths: List[str], batch_size: int = 32) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
            imgs_t = torch.stack(imgs, dim=0).to(device)
            emb = model.encode_image(imgs_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.detach().cpu().numpy())
    return np.vstack(feats).astype("float32")


def embed_text(model, tokenizer, device, text: str) -> np.ndarray:
    with torch.no_grad():
        tokens = tokenizer([text]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy().astype("float32")


# def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
#     dim = vecs.shape[1]
#     idx = faiss.IndexFlatIP(dim)  # cosine similarity if vectors normalized
#     idx.add(vecs)
#     return idx


def format_time(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


st.set_page_config(page_title="Video Scene Jump (Text Search)", layout="wide")
st.title("🎬 Jump to a video moment by typing what you want to see")

with st.sidebar:
    st.header("Index settings")
    window_sec = st.slider("Window length (seconds)", 1.0, 6.0, 3.0, 0.5)
    stride_sec = st.slider("Stride (seconds)", 0.5, 3.0, 1.0, 0.5)
    frames_per_window = st.slider("Frames per window", 1, 8, 4, 1)
    max_windows = st.slider("Max windows (cap)", 100, 2000, 900, 50)
    top_k = st.slider("Top-K results", 1, 10, 5, 1)

uploaded = st.file_uploader("Upload a ~10 min video", type=["mp4", "mov", "mkv", "webm", "avi"])

if uploaded is None:
    st.info("Upload a video to begin.")
    st.stop()

video_path = save_uploaded_file(uploaded)
st.video(video_path)

model, preprocess, tokenizer, device = load_clip_model()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.button("🔎 Build / Rebuild index"):
    with st.spinner("Building windows + extracting frames..."):
        tmp_dir = tempfile.mkdtemp(prefix="vidwin_")
        windows, window_frame_paths = build_windows(
            video_path,
            tmp_dir,
            window_sec=window_sec,
            stride_sec=stride_sec,
            frames_per_window=frames_per_window,
            max_windows=max_windows,
        )

    st.write(f"Created **{len(windows)}** windows.")

    # embed all frames then average per window
    with st.spinner("Embedding frames (OpenCLIP)..."):
        # flatten frame paths
        flat_paths = [p for paths in window_frame_paths for p in paths]
        frame_vecs = embed_image_paths(model, preprocess, device, flat_paths)

    # map back and average per window
    with st.spinner("Aggregating to window embeddings..."):
        window_vecs = []
        cursor = 0
        for paths in window_frame_paths:
            n = len(paths)
            v = frame_vecs[cursor:cursor+n].mean(axis=0)
            # normalize again
            v = v / (np.linalg.norm(v) + 1e-8)
            window_vecs.append(v.astype("float32"))
            cursor += n
        window_vecs = np.vstack(window_vecs).astype("float32")

    # index = build_faiss_index(window_vecs)

    st.session_state.windows = windows
    st.session_state.window_vecs = window_vecs
    # st.session_state.index = index
    st.session_state.indexed = True
    st.success("Index ready. Try a query!")

if not st.session_state.indexed:
    st.warning("Click **Build / Rebuild index** to enable search.")
    st.stop()

query = st.text_input("Describe what you want to jump to (e.g., 'two people kiss', 'a car chase', 'someone crying')")

if query:
    qvec = embed_text(model, tokenizer, device, query)
    # D, I = st.session_state.index.search(qvec, top_k)

# window_vecs shape: (N, D)
# qvec shape: (1, D)
    window_vecs = st.session_state.window_vecs
    query_vec = qvec[0]

    # cosine similarity (vectors already normalized)
    scores = window_vecs @ query_vec

    # get top-k highest scores
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_idx:
        w = st.session_state.windows[int(idx)]
        results.append((float(scores[idx]), w))

    st.subheader("Matches")
    cols = st.columns(len(results))

    for col, (score, w) in zip(cols, results):
        with col:
            st.image(w.thumb_path, caption=f"{format_time(w.t_start)}–{format_time(w.t_end)} • score {score:.3f}", use_container_width=True)
            if st.button(f"Jump to {format_time(w.t_start)}", key=f"jump_{w.thumb_path}"):
                st.session_state.jump_to = int(w.t_start)

if "jump_to" in st.session_state:
    st.divider()
    st.subheader(f"▶️ Playing from {format_time(st.session_state.jump_to)}")
    st.video(video_path, start_time=st.session_state.jump_to)
