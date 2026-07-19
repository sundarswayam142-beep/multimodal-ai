import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import ollama
from datetime import date
import os as _os
from collections import Counter
import io

st.set_page_config(
    page_title="AI Quality Assurance",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data tracking profiles inside session memory arrays
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []
if "auth_status" not in st.session_state:
    st.session_state.auth_status = {"signed_in": False, "provider": None, "user": None}
if "converted_neu_buffer" not in st.session_state:
    st.session_state.converted_neu_buffer = None

# Mandatory curriculum baseline limits
CONF_THRESHOLD = 0.15          
CONF_THRESHOLD_TILED = 0.35    
NEU_SIZE = 200
NEU_SIZE_TOLERANCE = 40  

def load_label_font(size=16):
    """Loads a bold, high-impact font for box labels, falling back gracefully
    if no truetype font is available on the system."""
    candidates = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()

def is_already_neu_format(pil_image):
    """Detects whether an uploaded image matches native NEU-DET training profiles."""
    w, h = pil_image.size
    is_square_ish = abs(w - h) <= max(w, h) * 0.1
    is_neu_sized = abs(w - NEU_SIZE) <= NEU_SIZE_TOLERANCE and abs(h - NEU_SIZE) <= NEU_SIZE_TOLERANCE

    rgb = pil_image.convert("RGB")
    r, g, b = rgb.split()
    r_px, g_px, b_px = list(r.getdata()), list(g.getdata()), list(b.getdata())
    sample = range(0, len(r_px), max(1, len(r_px) // 500))

    total_spread = 0
    for i in sample:
        total_spread += max(r_px[i], g_px[i], b_px[i]) - min(r_px[i], g_px[i], b_px[i])

    avg_channel_spread = total_spread / len(list(sample))
    is_grayscale_ish = avg_channel_spread < 15

    return is_square_ish and is_neu_sized and is_grayscale_ish

def normalize_contrast(pil_tile):
    """Grayscale + histogram equalization toward NEU-DET style with unsharp pass."""
    gray = pil_tile.convert("L")
    equalized = ImageOps.equalize(gray)
    sharpened = equalized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    return sharpened.convert("RGB")

def deduplicate_boxes(boxes, iou_threshold=0.25, proximity_ratio=0.5):
    """Weighted Box Fusion cluster aggregation across overlapping layout tiles."""
    if not boxes:
        return []

    def iou(a, b):
        ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
        ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def contains_or_intense_overlap(a, b):
        ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
        ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        if area_b == 0:
            return True
        return (inter / area_b) > 0.65

    def centers_close(a, b):
        acx, acy = (a["x1"] + a["x2"]) / 2.0, (a["y1"] + a["y2"]) / 2.0
        bcx, bcy = (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0
        dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
        avg_size = ((a["x2"] - a["x1"]) + (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) + (b["y2"] - b["y1"])) / 4.0
        return dist < (avg_size * proximity_ratio)

    def belongs_together(a, b):
        if a["cls_name"] != b["cls_name"]:
            return False
        return (iou(a, b) >= iou_threshold or centers_close(a, b) or contains_or_intense_overlap(a, b))

    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if belongs_together(boxes[i], boxes[j]):
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(boxes[i])

    fused = []
    for cluster in clusters.values():
        total_weight = sum(b["conf"] for b in cluster)
        if total_weight == 0:
            total_weight = len(cluster)
        fx1 = sum(b["x1"] * b["conf"] for b in cluster) / total_weight
        fy1 = sum(b["y1"] * b["conf"] for b in cluster) / total_weight
        fx2 = sum(b["x2"] * b["conf"] for b in cluster) / total_weight
        fy2 = sum(b["y2"] * b["conf"] for b in cluster) / total_weight
        max_conf = max(b["conf"] for b in cluster)
        agreement_bonus = min(5.0 * (len(cluster) - 1), 15.0)
        fused_conf = min(max_conf + agreement_bonus, 99.0)
        fused.append({
            "cls_name": cluster[0]["cls_name"],
            "conf": fused_conf,
            "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
        })

    return fused

def get_tiles(pil_image, tile_size=200, overlap=0.3):
    """Splits full image into overlapping tiles ensuring edge snapping grid coverage."""
    img = pil_image.convert("RGB")
    w, h = img.size
    stride = max(1, int(tile_size * (1 - overlap)))
    tiles = []

    y_positions = list(range(0, max(h - tile_size, 0) + 1, stride))
    if not y_positions or y_positions[-1] + tile_size < h:
        y_positions.append(max(h - tile_size, 0))

    x_positions = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not x_positions or x_positions[-1] + tile_size < w:
        x_positions.append(max(w - tile_size, 0))

    for y_start in y_positions:
        for x_start in x_positions:
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            tile = img.crop((x_start, y_start, x_end, y_end))
            if tile.size != (tile_size, tile_size):
                tile = tile.resize((tile_size, tile_size))

            tile = normalize_contrast(tile)
            tiles.append((tile, x_start, y_start))

    return tiles, (w, h)

SEVERITY_COLORS = {
    "low": ("#0F6E56", "#E1F5EE"),
    "medium": ("#854F0B", "#FAEEDA"),
    "high": ("#A32D2D", "#FCEBEB"),
}

# ---- High-End Industrial Styling Moving Dashboard CSS ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --steel-900: #12151A;
        --steel-800: #1A1F26;
        --steel-700: #242B34;
        --steel-line: #2E3742;
        --amber: #F2A93B;
        --amber-dim: rgba(242,169,59,0.14);
        --blue-accent: #6E93B0;
        --text-hi: #F2F1EC;
        --text-mid: #A9AEB6;
        --text-low: #767C86;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .main > div { padding-top: 1.5rem; max-width: 1100px; }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scanSweep {
        0% { transform: translateX(-10%); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateX(110%); opacity: 0; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.8); }
    }
    @keyframes gridMove {
        0% { background-position: 0 0; }
        100% { background-position: 34px 34px; }
    }

    /* Target Streamlit containers to make them look like sleek dashboard widgets */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--steel-800);
        border: 1px solid var(--steel-line) !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #3A4450 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .hero {
        background:
            repeating-linear-gradient(90deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 34px),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 34px),
            linear-gradient(150deg, var(--steel-900) 0%, var(--steel-800) 55%, var(--steel-900) 100%);
        background-size: 34px 34px, 34px 34px, 100% 100%;
        animation: gridMove 8s linear infinite, fadeSlideUp 0.6s ease-out;
        border-radius: 8px; padding: 48px 44px 40px; margin-bottom: 2rem; position: relative; overflow: hidden;
        border: 1px solid var(--steel-line); border-left: 4px solid var(--amber);
    }
    .hero::after {
        content: ""; position: absolute; top: 0; left: 0; height: 100%; width: 18%;
        background: linear-gradient(90deg, transparent, rgba(242,169,59,0.08), transparent);
        animation: scanSweep 4s ease-in-out infinite; pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500;
        letter-spacing: 0.06em; text-transform: uppercase; color: var(--amber); background: var(--amber-dim); border: 1px solid rgba(242,169,59,0.3);
        padding: 4px 12px; border-radius: 20px; margin-bottom: 20px;
    }
    .hero-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); animation: pulse 1.8s ease-in-out infinite; }
    .hero h1 {
        font-family: 'Big Shoulders Display', sans-serif !important; font-weight: 800 !important; font-size: 46px !important; line-height: 1.05 !important; margin: 0 0 12px !important;
        color: var(--text-hi) !important; letter-spacing: -0.01em !important; text-transform: uppercase;
    }
    .hero h1 .accent { color: var(--amber); }
    .hero p.hero-sub { color: var(--text-mid); font-size: 14.5px; margin: 0; max-width: 460px; line-height: 1.65; }

    .section-label { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-hi); margin: 0 0 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid var(--steel-line); padding-bottom: 8px;}
    .defect-pill {
        display: inline-flex; align-items: center; gap: 6px; background: var(--steel-800); color: var(--amber); border: 1px solid var(--steel-line); border-left: 2px solid var(--amber);
        padding: 6px 14px; border-radius: 4px; font-size: 13px; margin: 0 8px 8px 0; transition: all 0.2s ease; cursor: default; font-family: 'JetBrains Mono', monospace; animation: fadeSlideUp 0.4s ease-out backwards;
    }
    .defect-pill:hover { background: var(--steel-700); transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }

    .report-card { background: var(--steel-800); border: 1px solid var(--steel-line); border-radius: 8px; padding: 24px 28px; margin-top: 12px; transition: border-color 0.3s ease; animation: fadeSlideUp 0.5s ease-out; }
    .report-card:hover { border-color: #3A4450; }
    .report-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
    .report-card-header h3 { margin: 0; font-size: 16px; font-family: 'JetBrains Mono', monospace; color: var(--text-hi); text-transform: uppercase; letter-spacing: 0.03em; }
    .report-date { font-size: 13px; color: var(--text-low); margin: 0 0 14px; font-family: 'JetBrains Mono', monospace; }
    .severity-badge { margin-left: auto; font-size: 11.5px; padding: 4px 12px; border-radius: 4px; font-weight: 600; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.04em; }

    .stButton>button {
        background-color: var(--amber); color: var(--steel-900); border-radius: 6px; padding: 0.65rem 1.5rem; font-weight: 700; border: none;
        width: 100%; transition: all 0.18s ease; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; font-size: 13px; letter-spacing: 0.04em;
    }
    .stButton>button:hover { background-color: #FFC266; transform: translateY(-1px); box-shadow: 0 8px 20px rgba(242,169,59,0.3); }
    
    .panel-title {
        font-family: 'JetBrains Mono', monospace; color: var(--text-hi); font-size: 13px; 
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; border-bottom: 1px solid var(--steel-line); padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar Container ----
with st.sidebar:
    st.header("About this platform")
    st.write("""
    Welcome to the **Automated Industrial Quality Assurance** system!
    
    We built this tool to make manufacturing inspections faster, safer, and entirely objective. Instead of relying on manual spot-checks, this application acts as a tireless digital assistant.
    
    Simply upload a photo of a steel surface. The integrated AI instantly scans the material for common defects like scratches, crazing, or scale. It then hands the data over to a language model that writes a clear, professional assessment. 
    
    Whether you are managing a factory floor or handling supply chain verification, this helps you catch and document anomalies before they turn into costly liabilities.
    """)
    st.divider()
    st.caption("⚙️ **Engine Config**")
    st.caption(f"• Standard Base: {CONF_THRESHOLD}")
    st.caption(f"• Tiled Array: {CONF_THRESHOLD_TILED}")
    st.divider()
    st.caption("SoC Multi-Modal Industrial Vision QA")

# ---- Hero Panel (Rendered First for Hierarchy) ----
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow"><span class="dot"></span>YOLOv8 + Llama 3.2</span>
    <h1>Industrial quality<br><span class="accent">assurance</span>, automated.</h1>
    <p class="hero-sub">Upload a steel surface image or run conversion profiles natively using the horizontal dashboard interface below.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# HORIZONTAL INTEGRATED FRONTFACE DASHBOARD ROW
# ==========================================
st.markdown('<p class="section-label">🛠️ INTEGRATED CONTROL DASHBOARD TERMINAL</p>', unsafe_allow_html=True)
dash_col1, dash_col2, dash_col3 = st.columns([1, 1, 1.5], gap="medium")

# Panel Segment 1: User Profile Configuration
with dash_col1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">👤 Profile Identity</div>', unsafe_allow_html=True)
        if not st.session_state.auth_status["signed_in"]:
            auth_choice = st.selectbox("Method", ["Select Identity...", "Google Account", "GitHub Developer", "Guest Entry"], label_visibility="collapsed")
            if auth_choice in ["Google Account", "GitHub Developer"]:
                em = st.text_input("Email", placeholder="user@domain.com", label_visibility="collapsed")
                nm = st.text_input("Name", placeholder="Inspector Name", label_visibility="collapsed")
                if st.button("Authenticate Identity", use_container_width=True):
                    if em and nm:
                        st.session_state.auth_status = {"signed_in": True, "provider": auth_choice, "user": nm}
                        st.rerun()
            elif auth_choice == "Guest Entry":
                if st.button("Enter Portal As Guest", use_container_width=True):
                    st.session_state.auth_status = {"signed_in": True, "provider": "Guest Terminal", "user": "Guest Inspector"}
                    st.rerun()
        else:
            st.success(f"**Active:** {st.session_state.auth_status['user']}")
            st.caption(f"Connected via: {st.session_state.auth_status['provider']}")
            if st.button("Log Out & Clear Session", use_container_width=True):
                st.session_state.auth_status = {"signed_in": False, "provider": None, "user": None}
                st.session_state.converted_neu_buffer = None
                st.rerun()

# Panel Segment 2: NEU Image Preprocessing & Exporter
with dash_col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">🔄 Format Preprocessor</div>', unsafe_allow_html=True)
        st.caption("Upload a raw image to standardize it into a clean 200x200 YOLO training format.")
        c_file = st.file_uploader("Upload Raw Target", type=["jpg", "jpeg", "png"], key="dashboard_conv", label_visibility="collapsed")
        
        if c_file:
            raw_img = Image.open(c_file)
            proc_img = ImageOps.equalize(raw_img.convert("L")).resize((200, 200))
            
            buf = io.BytesIO()
            proc_img.save(buf, format="JPEG")
            st.session_state.converted_neu_buffer = buf.getvalue()
            
            st.success("✓ Image standard converted!")
            st.download_button(
                label="💾 Download YOLO Format",
                data=st.session_state.converted_neu_buffer,
                file_name="converted_yolo_target.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        elif st.session_state.converted_neu_buffer:
            st.info("A converted image is currently loaded in memory pipeline.")
            if st.button("Clear Converter Memory", use_container_width=True):
                st.session_state.converted_neu_buffer = None
                st.rerun()

# Panel Segment 3: Chronological Tracking History Logs
with dash_col3:
    history_container = st.container(border=True) # DEFERRED RENDER CONTAINER to prevent empty states

st.markdown("<br>", unsafe_allow_html=True)

# ---- Main Upload and Inference Section ----
st.markdown('<p class="section-label">📤 PRIMARY UPLOAD & INFERENCE ENGINE</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

ENSEMBLE_WEIGHTS = ["best.pt"]
USE_TTA = False

@st.cache_resource
def load_models():
    models = []
    for path in ENSEMBLE_WEIGHTS:
        try:
            models.append(YOLO(path))
        except Exception:
            continue
    return models if models else [YOLO("best.pt")]

models = load_models()

# Priority routing: Use converted image if available, otherwise use uploaded file
source_image = None
using_converted_flag = False

if st.session_state.converted_neu_buffer is not None:
    source_image = Image.open(io.BytesIO(st.session_state.converted_neu_buffer))
    using_converted_flag = True
    st.info("✅ Injecting your recently standardized image from the Format Preprocessor above directly into the pipeline.")
elif uploaded_file is not None:
    source_image = Image.open(uploaded_file)

if source_image is not None:
    image = source_image
    
    # Friendly UI warning if the user uploads an unoptimized image without converting
    if not using_converted_flag:
        already_neu_check = is_already_neu_format(image)
        if not already_neu_check:
            st.warning("⚠️ **Notice:** This uploaded image is not in the optimal NEU training format (200x200 grayscale). The AI will attempt to tile it, but for the most reliable results, we recommend passing it through the **Format Preprocessor** (Dashboard ☝️) first!")

    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-label">Original Target Layout</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    run = st.button("Run automated inspection pass", use_container_width=True)

    if run:
        with st.spinner("Executing structural validation sweeps..."):
            already_neu = is_already_neu_format(image)
            raw_boxes = []
            full_img = image.convert("RGB").copy()
            draw = ImageDraw.Draw(full_img)

            if already_neu:
                for m in models:
                    results = m(image, conf=CONF_THRESHOLD, augment=USE_TTA, imgsz=640)
                    for box in results[0].boxes:
                        cls_name = m.names[int(box.cls)]
                        conf = float(box.conf) * 100
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        raw_boxes.append({"cls_name": cls_name, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            else:
                tiles, (full_w, full_h) = get_tiles(image, tile_size=200, overlap=0.3)
                for tile_img, x_off, y_off in tiles:
                    for m in models:
                        tile_results = m(tile_img, conf=CONF_THRESHOLD_TILED, augment=USE_TTA)
                        for box in tile_results[0].boxes:
                            cls_name = m.names[int(box.cls)]
                            conf = float(box.conf) * 100
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            raw_boxes.append({
                                "cls_name": cls_name, "conf": conf,
                                "x1": x1 + x_off, "y1": y1 + y_off,
                                "x2": x2 + x_off, "y2": y2 + y_off,
                            })

            deduped = deduplicate_boxes(raw_boxes)
            defects = [(b["cls_name"], b["conf"]) for b in deduped]

            for b in deduped:
                draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(216, 90, 48), width=3)

            label_font = load_label_font(size=max(9, int(min(full_img.size) * 0.013)))
            canvas_w, canvas_h = full_img.size

            AMBER = (242, 169, 59)
            used_label_rects = []
            label_specs = []
            for b in deduped:
                label = f"{b['cls_name'].replace('_', ' ').title()} {b['conf']:.0f}%".upper()
                text_bbox = draw.textbbox((0, 0), label, font=label_font)
                label_w = (text_bbox[2] - text_bbox[0]) + 10
                label_h = (text_bbox[3] - text_bbox[1]) + 7

                lx = min(b["x1"], canvas_w - label_w - 4)
                lx = max(lx, 4)
                ly = b["y1"] - label_h
                if ly < 4:
                    ly = min(b["y1"] + 4, canvas_h - label_h - 4)
                ly = max(ly, 4)

                loop_count = 0
                while any(
                    not (lx + label_w < ux or lx > ux + uw or ly + label_h < uy or ly > uy + uh)
                    for (ux, uy, uw, uh) in used_label_rects
                ) and loop_count < 8:
                    ly += label_h + 2
                    loop_count += 1
                    if ly + label_h > canvas_h:
                        ly = max(b["y1"] - label_h, 4)
                        break

                used_label_rects.append((lx, ly, label_w, label_h))
                label_specs.append((label, lx, ly, label_w, label_h))

            glow_layer = Image.new("RGBA", full_img.size, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            for label, lx, ly, label_w, label_h in label_specs:
                glow_draw.rounded_rectangle([lx, ly, lx + label_w, ly + label_h], radius=4, fill=(*AMBER, 110))
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))

            full_img_rgba = full_img.convert("RGBA")
            full_img_rgba = Image.alpha_composite(full_img_rgba, glow_layer)
            annotated = full_img_rgba.convert("RGB")
            sharp_draw = ImageDraw.Draw(annotated)

            for label, lx, ly, label_w, label_h in label_specs:
                sharp_draw.rounded_rectangle([lx, ly, lx + label_w, ly + label_h], radius=4, fill=(15, 17, 21), outline=AMBER, width=1)
                sharp_draw.text((lx + 6, ly + 4), label, fill=(255, 231, 181), font=label_font)

        with col2:
            st.markdown('<p class="section-label">Automated Vision Plot Overlay</p>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)
            path_label = "Already NEU-format — direct inference" if already_neu else "Tiled inference (image normalized)"
            st.caption(f"Pipeline Execution path: {path_label}")

        st.divider()

        if defects:
            st.markdown('<p class="section-label">Detections Metrics Fleet</p>', unsafe_allow_html=True)
            pills = "".join(f'<span class="defect-pill">{name.replace("_", " ").title()} — {conf:.0f}%</span>' for name, conf in defects)
            st.markdown(pills, unsafe_allow_html=True)

            with st.spinner("Consulting local synthesis records via Llama..."):
                total_count = len(defects)
                defect_counts = Counter(name for name, _ in defects)
                dominant_defect_raw, dominant_num = defect_counts.most_common(1)[0]
                dominant_defect_title = dominant_defect_raw.replace('_', ' ').title()
                exact_percentage = int((dominant_num / total_count) * 100)

                highest_conf = max(conf for _, conf in defects)
                if total_count >= 10 or highest_conf >= 85:
                    severity = "high"
                elif total_count >= 3 or highest_conf >= 35:
                    severity = "medium"
                else:
                    severity = "low"

                defect_list = [f"• {name.replace('_', ' ').title()} ({conf:.0f}%)" for name, conf in defects]
                defect_lines = "\n".join(defect_list)
                inspection_date = date.today().strftime('%d %B %Y')

                prompt = f"""You are a professional industrial quality control inspector. Fill in the EXACT template below. Do not rename, reorder, add, or remove any section headers. Do not add a title like "Structural Summary" or any other heading — output must start directly with "Inspection Report".

Inspection Report
Inspection Date: {inspection_date}
Detected Defects:
{defect_lines}
Summary:
[Write 2-3 sentences. State that {dominant_defect_title} is the primary issue, accounting for {exact_percentage}% of detected defects. Note the impact on surface quality/uniformity.]
Severity:
{severity.title()}
Recommended Action:
• Inspect the affected region manually
• Remove or repair the damaged section if required
• Monitor the production line for recurring defects
• Perform quality verification before shipment

Only output the filled template above, starting with "Inspection Report". No preamble, no extra headers, no commentary."""

                response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
                report_text = response['message']['content']

            text_color, bg_color = SEVERITY_COLORS[severity]

            st.session_state.inspection_history.append({
                "date": inspection_date,
                "defects_summary": f"{total_count} anomalies ({dominant_defect_title} primary)",
                "severity": severity.title(),
                "full_text": report_text
            })

            st.markdown(f"""
            <div class="report-card">
                <div class="report-card-header">
                    <h3>📋 Custom compiled maintenance overview</h3>
                    <span class="severity-badge" style="color:{text_color}; background:{bg_color};">
                        {severity.title()} severity
                    </span>
                </div>
                <p class="report-date">{inspection_date}</p>
                <div style="white-space: pre-wrap; font-size: 14px; line-height: 1.6;">{report_text}</div>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                "Download report as text",
                data=report_text,
                file_name=f"inspection_report_{date.today()}.txt",
                use_container_width=True
            )
        else:
            st.success("✅ No defects detected — surface matrix passes manufacturing norms.")
            st.session_state.inspection_history.append({
                "date": date.today().strftime('%d %B %Y'),
                "defects_summary": "0 anomalies detected",
                "severity": "Passed",
                "full_text": "Surface matrix passes manufacturing norms. No defects detected."
            })

# ==========================================
# RENDER DEFERRED HISTORY DASHBOARD LAST
# ==========================================
# This ensures the history panel at the top of the app is filled
# *after* the YOLO / LLaMA model has finished pushing data to the array!
with history_container:
    st.markdown('<div class="panel-title">📜 Historical Inspection Tracking Logs</div>', unsafe_allow_html=True)
    if st.session_state.inspection_history:
        # Show the two most recent runs prominently
        for idx, hist in enumerate(reversed(st.session_state.inspection_history)):
            if idx < 2:  
                st.write(f"⏱️ **Run #{len(st.session_state.inspection_history) - idx}** | `{hist['severity']}` | {hist['defects_summary']}")
        # Hide older runs inside an expander
        if len(st.session_state.inspection_history) > 2:
            with st.expander("View older operational logs"):
                for idx_old, hist_old in enumerate(reversed(st.session_state.inspection_history)):
                    if idx_old >= 2:
                        st.caption(f"**Run #{len(st.session_state.inspection_history) - idx_old}** [{hist_old['date']}]: {hist_old['defects_summary']}")
    else:
        st.info("No operational records compiled yet. Logs will populate here automatically after your first inspection.")