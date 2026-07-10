import streamlit as st
from PIL import Image
import numpy as np
import torch
import time
from src.config import TRAIN_INDEX, TRAIN_PATHS
from src.model import FeatureExtractor
from src.transforms import img_transform
from src.utils import load_index
from src.search import Image_Search

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Visual Product Search",page_icon="🔍",layout="wide",initial_sidebar_state="expanded")
@st.cache_resource
def load_backend():
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)
    index = load_index(TRAIN_INDEX)
    image_paths = np.load(TRAIN_PATHS,allow_pickle=True)
    engine = Image_Search(model=model,index=index,img_path=image_paths,transform=img_transform,device=device,)
    return engine, device
engine, device = load_backend()
st.session_state["device"] = device.type

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown(
"""<style>
.main {
    background-color: #0E1117;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.card{
    background:#262730;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 0px 8px rgba(0,0,0,0.3);
}

.result-card{
    background:#1E1E1E;
    border-radius:12px;
    padding:10px;
    text-align:center;
    border:1px solid #333;
}

.result-card:hover{
    border:1px solid #4F8BF9;
}

h1,h2,h3{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Header
# --------------------------------------------------------

col1,col2 = st.columns([8,2])
with col1:
    st.title("🔍 Visual Product Search Engine")
with col2:
    if st.session_state.get("device","CPU")=="cuda":
        st.success("🟢 GPU")
    else:
        st.info("⚪ CPU")
st.divider()

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------

with st.sidebar:
    st.header("⚙ Search Settings")
    st.markdown("---")
    st.write("### Model")
    st.info("ResNet50")
    top_k = st.slider("Top K Results",1,20,5)
    st.write("### Similarity")
    st.success("Cosine Similarity")
    uploaded_file = st.file_uploader("Upload Image",type=["jpg","jpeg","png"])
    search = st.button("🔍 Search Similar Images",use_container_width=True)

# --------------------------------------------------------
# Query Image
# --------------------------------------------------------

left,right = st.columns([2,3])
with left:
    st.subheader("Query Image")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image,use_container_width=False)
    else:
        st.info("Upload an image to begin.")

# --------------------------------------------------------
# Statistics
# --------------------------------------------------------

with right:
    st.subheader("Search Statistics")
    c1,c2,c3,c4 = st.columns(4)
    search_time = st.session_state.get("search_time", 0)
    c1.metric("Search Time",f"{search_time:.2f} ms")
    c2.metric("Model","ResNet50")
    c3.metric("Embedding","2048")
    c4.metric("Results",top_k+1)

# --------------------------------------------------------
# Search Results
# --------------------------------------------------------

st.divider()
st.subheader("Similar Products")
if search and uploaded_file is not None:
    with st.spinner("Searching..."):
        start = time.perf_counter()
        results = engine.search(image=image,top_k=top_k+1)[1:]
        end = time.perf_counter()
        search_time = (end - start) * 1000
        st.session_state["search_time"] = search_time
    cols = st.columns(top_k+1)
    for i, (col, result) in enumerate(zip(cols, results)):
        with col:
            st.markdown("<div class='result-card'>",unsafe_allow_html=True)
            st.image(result["image_path"],use_container_width=True)
            score = float(result["score"])
            progress = max(0.0, min(score, 1.0))
            st.progress(progress)
            st.write(f"Similarity : {score:.2%}")
            st.markdown("</div>",unsafe_allow_html=True)

# --------------------------------------------------------
# Footer
# --------------------------------------------------------
st.divider()
st.caption(
"""
Model : ResNet50 (ImageNet) |
Similarity Search : FAISS IndexHNSWFFlat |
Dataset : Stanford Online Products
"""
)