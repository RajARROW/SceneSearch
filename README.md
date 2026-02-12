# 🎬 Ctrl+F for Video (Semantic Video Search)

Search inside a video using natural language.

**Upload a video → Type what you want to see → Instantly jump to the most relevant moment.**

This project uses **OpenCLIP** to embed both video frames and text into the same semantic space, enabling text-to-video moment retrieval.

---

## 🚀 Example Queries

```text
"a car chase"
"two people hugging"
"someone crying"
```

The system ranks video moments by semantic similarity and lets you jump directly to matching timestamps.

---

## 🧠 How It Works

### 1️⃣ Windowing
The video is split into short overlapping time windows (e.g., 3 seconds).

### 2️⃣ Frame Sampling
A few representative frames are extracted from each window.

### 3️⃣ Embedding (OpenCLIP)
- Frames → Image embeddings  
- Query text → Text embedding  

Both live in the same vector space.

### 4️⃣ Aggregation
Frame embeddings are averaged to create a single embedding per window.

### 5️⃣ Similarity Search
Cosine similarity is computed between the text embedding and all window embeddings.

### 6️⃣ Jump to Timestamp
Top matches are displayed with thumbnails and playable timestamps.

---

## 🛠 Tech Stack

- Python
- Streamlit
- OpenCV
- PyTorch
- OpenCLIP
- NumPy

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## ⚠️ Current Limitations (v1)

- Visual-only (no audio understanding yet)
- Struggles with subtle actions or complex scenes
- No narrative or contextual understanding
- Brute-force similarity search (no FAISS / ANN yet)
- Window averaging may blur very fast transitions

---

## 📈 Roadmap

- [ ] Better scene segmentation
- [ ] Smarter frame sampling
- [ ] Audio transcription integration
- [ ] Multi-modal fusion (audio + vision)
- [ ] Approximate nearest neighbor search (FAISS)
- [ ] Performance optimization for longer videos

---

## 💡 Why This Project?

We can search text instantly.  
We can search images.  
But searching inside videos is still hard.

This project is an early step toward making video content as searchable as text.

---

## 🤝 Contributions & Feedback

Feedback, ideas, and contributions are welcome.

If you find this interesting, feel free to open an issue or submit a PR.

---

## 📜 License

MIT License
