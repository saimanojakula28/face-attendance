import streamlit as st
import cv2
import pandas as pd
from pathlib import Path
from datetime import date
import time

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from db import init_db, add_user, get_users, mark_attendance, get_attendance
from vision import ensure_dirs, capture_images, train_encodings, load_encodings, recognize_from_frame

st.set_page_config(page_title="Face Attendance", page_icon="✅", layout="wide")

# ----------------- HOLO CYBER GRID / NEON THEME + HUD FRAME + FONT HIGHLIGHTS -----------------
def inject_holo_cyber_theme():
    st.markdown(
        """
        <style>
        /* ===== GLOBAL FONT ===== */
        * {
          font-family: "Segoe UI", "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ====== Background: deep space + animated grid ====== */
        .stApp{
            background:
                radial-gradient(900px circle at 15% 15%, rgba(0,255,255,0.14), transparent 45%),
                radial-gradient(900px circle at 85% 25%, rgba(255,0,255,0.12), transparent 48%),
                radial-gradient(800px circle at 30% 85%, rgba(0,255,140,0.10), transparent 50%),
                linear-gradient(135deg, #04040c 0%, #050019 45%, #001019 100%);
            background-attachment: fixed;
            color: #EAF6FF;
        }

        /* Hologram glow fog */
        .stApp:before{
            content:"";
            position: fixed;
            inset: 0;
            pointer-events:none;
            background: conic-gradient(
                from 180deg at 50% 50%,
                rgba(0,255,255,0.10),
                rgba(255,0,255,0.08),
                rgba(0,255,140,0.06),
                rgba(0,140,255,0.08),
                rgba(0,255,255,0.10)
            );
            filter: blur(42px);
            opacity: 0.65;
            animation: holoGlow 12s ease-in-out infinite alternate;
            z-index: 0;
        }
        @keyframes holoGlow{
            0% { transform: translate3d(-2%, -1%, 0) scale(1.02) rotate(0.5deg); }
            100%{ transform: translate3d( 2%,  1%, 0) scale(1.06) rotate(-0.5deg); }
        }

        /* Animated neon grid */
        .stApp:after{
            content:"";
            position: fixed;
            inset: 0;
            pointer-events:none;
            background-image:
              linear-gradient(rgba(0,255,255,0.10) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,0,255,0.08) 1px, transparent 1px);
            background-size: 70px 70px, 70px 70px;
            background-position: 0 0, 0 0;
            opacity: 0.18;
            transform: perspective(800px) rotateX(56deg) translateY(120px);
            transform-origin: center;
            animation: gridMove 8s linear infinite;
            z-index: 0;
        }
        @keyframes gridMove{
            0% { background-position: 0 0, 0 0; }
            100% { background-position: 0 140px, 140px 0; }
        }

        /* Keep content above overlays */
        .block-container{
            position: relative;
            z-index: 1;
            padding-top: 2.0rem;
        }

        /* ===== FONT HIGHLIGHTING ===== */
        h1{
            font-size: 2.6rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.6px;
            background: linear-gradient(90deg, #00fff0, #ff3dfc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow:
              0 0 18px rgba(0,255,255,0.35),
              0 0 30px rgba(255,0,255,0.25);
        }
        h2{
            font-weight: 800 !important;
            letter-spacing: 0.3px;
            color: #eaffff;
            text-shadow: 0 0 10px rgba(0,255,255,0.25);
        }
        h3{
            font-weight: 700 !important;
            color: #dff9ff;
            text-shadow: 0 0 8px rgba(255,0,255,0.18);
        }
        label, .stTextInput label, .stSlider label {
          font-weight: 600 !important;
          color: rgba(234,246,255,0.85) !important;
          letter-spacing: 0.2px;
        }
        input::placeholder, textarea::placeholder {
          color: rgba(180,220,255,0.55) !important;
          font-style: italic;
        }

        .accent {
          color: #00fff0;
          font-weight: 700;
          text-shadow: 0 0 10px rgba(0,255,255,0.35);
        }
        .warn {
          color: #ffb347;
          font-weight: 700;
          text-shadow: 0 0 10px rgba(255,180,71,0.35);
        }
        .success {
          color: #2cff9a;
          font-weight: 700;
          text-shadow: 0 0 10px rgba(44,255,154,0.35);
        }
        .small-text {
          font-size: 0.8rem;
          color: rgba(234,246,255,0.65);
          letter-spacing: 0.15px;
        }

        /* KPI number highlight (you were using this class but it was missing) */
        .kpi-number {
          font-size: 2.2rem;
          font-weight: 900;
          background: linear-gradient(90deg, #00ffd5, #ff5cf0);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          text-shadow:
            0 0 14px rgba(0,255,255,0.35),
            0 0 24px rgba(255,0,255,0.25);
        }

        /* ====== Glass panels ====== */
        [data-testid="stVerticalBlockBorderWrapper"],
        .stTabs [data-baseweb="tab-list"],
        [data-testid="stExpander"]{
            background: rgba(6, 10, 22, 0.50) !important;
            border: 1px solid rgba(0, 255, 255, 0.22) !important;
            border-radius: 20px !important;
            box-shadow:
              0 0 0 1px rgba(255,0,255,0.10) inset,
              0 18px 60px rgba(0,0,0,0.45);
            backdrop-filter: blur(14px);
        }

        /* ====== Tabs (Neon pill tabs) ====== */
        .stTabs [data-baseweb="tab-list"]{
          padding: 10px 10px !important;
          gap: 8px !important;
        }
        .stTabs [data-baseweb="tab"]{
          border-radius: 999px !important;
          background: rgba(0,0,0,0.18) !important;
          border: 1px solid rgba(0,255,255,0.18) !important;
          padding: 10px 14px !important;
          color: rgba(234,246,255,0.82) !important;
          font-weight: 800 !important;
          letter-spacing: 0.2px;
        }
        .stTabs [aria-selected="true"]{
          background: linear-gradient(90deg, rgba(0,255,255,0.14), rgba(255,0,255,0.14)) !important;
          border: 1px solid rgba(255,0,255,0.35) !important;
          box-shadow: 0 0 18px rgba(0,255,255,0.22), 0 0 26px rgba(255,0,255,0.14);
          color: #ffffff !important;
          text-shadow: 0 0 12px rgba(0,255,255,0.35);
        }

        /* ====== Buttons ====== */
        .stButton > button{
            background: rgba(0,0,0,0.25) !important;
            border: 1px solid rgba(0,255,255,0.45) !important;
            color: #EAF6FF !important;
            border-radius: 16px !important;
            padding: 0.6rem 1.05rem !important;
            box-shadow:
              0 0 0 1px rgba(255,0,255,0.18) inset,
              0 0 18px rgba(0,255,255,0.20),
              0 0 26px rgba(255,0,255,0.10);
            transition: transform 120ms ease, box-shadow 200ms ease, border-color 200ms ease;
        }
        .stButton > button:hover{
            transform: translateY(-1px) scale(1.02);
            border-color: rgba(255,0,255,0.55) !important;
            box-shadow:
              0 0 0 1px rgba(0,255,255,0.22) inset,
              0 0 26px rgba(0,255,255,0.30),
              0 0 34px rgba(255,0,255,0.18);
        }

        /* ====== Inputs ====== */
        input, textarea{
            background: rgba(2, 6, 16, 0.55) !important;
            color: #EAF6FF !important;
            border: 1px solid rgba(0,255,255,0.22) !important;
            border-radius: 14px !important;
        }

        /* ====== Dataframe ====== */
        [data-testid="stDataFrame"]{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(0,255,255,0.18);
            box-shadow: 0 14px 44px rgba(0,0,0,0.45);
        }
        [data-testid="stDataFrame"] * {
          color: rgba(234,246,255,0.92) !important;
          font-weight: 500;
        }

        /* ====== Sidebar ====== */
        [data-testid="stSidebar"]{
            background: rgba(2, 6, 14, 0.68) !important;
            border-right: 1px solid rgba(0,255,255,0.16);
            backdrop-filter: blur(14px);
        }

        /* ====== HUD FRAME for images ====== */
        @keyframes hudPulse {
            0%   { box-shadow: 0 0 0 1px rgba(0,255,255,0.18) inset, 0 0 22px rgba(0,255,255,0.18), 0 0 28px rgba(255,0,255,0.10); }
            100% { box-shadow: 0 0 0 1px rgba(255,0,255,0.18) inset, 0 0 28px rgba(0,255,255,0.26), 0 0 36px rgba(255,0,255,0.16); }
        }
        [data-testid="stImage"]{
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(0,255,255,0.22);
            background: rgba(0,0,0,0.25);
            animation: hudPulse 2.6s ease-in-out infinite alternate;
        }
        [data-testid="stImage"] img{
            border-radius: 20px !important;
        }
        [data-testid="stImage"]::before{
            content:"";
            position:absolute;
            inset: 10px;
            pointer-events:none;
            background:
              linear-gradient(rgba(0,255,255,0.75), rgba(0,255,255,0.0)) left top/2px 26px no-repeat,
              linear-gradient(90deg, rgba(0,255,255,0.75), rgba(0,255,255,0.0)) left top/26px 2px no-repeat,
              linear-gradient(rgba(255,0,255,0.60), rgba(255,0,255,0.0)) right bottom/2px 26px no-repeat,
              linear-gradient(270deg, rgba(255,0,255,0.60), rgba(255,0,255,0.0)) right bottom/26px 2px no-repeat;
            filter: drop-shadow(0 0 6px rgba(0,255,255,0.35));
        }
        [data-testid="stImage"]::after{
            content:"";
            position:absolute;
            inset: 0;
            pointer-events:none;
            background: linear-gradient(
                120deg,
                rgba(0,255,255,0.08),
                rgba(255,0,255,0.06),
                rgba(0,0,0,0)
            );
            mix-blend-mode: screen;
            opacity: 0.6;
        }

        /* Clean UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_holo_cyber_theme()
# -------------------------------------------------------------------------

ensure_dirs()
init_db()

# Hero header
st.markdown("""
<div style="
    padding:18px 22px;
    border-radius:22px;
    border:1px solid rgba(0,255,255,0.22);
    background: rgba(6,10,22,0.55);
    box-shadow: 0 0 0 1px rgba(255,0,255,0.10) inset, 0 18px 60px rgba(0,0,0,0.45);
    backdrop-filter: blur(14px);
">
  <div style="font-size:34px; font-weight:900; letter-spacing:0.4px;">
    ✅ <span class="accent">Face Attendance</span>
  </div>
  <div class="small-text" style="margin-top:6px;">
    Real-time recognition • Neon HUD UI • SQLite + CSV Reports
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

tabs = st.tabs(["👤 Register", "🧠 Train", "📸 Mark Attendance", "📊 Reports"])

# ---------- Register ----------
with tabs[0]:
    st.subheader("Register a new user")
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.text_input("User ID (unique)", placeholder="e.g., 1001 or FAU001")
        name = st.text_input("Name", placeholder="e.g., Sai Manoj")
        num_images = st.slider("Number of images to capture", 10, 60, 30, 5)
        cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)

        if st.button("Save User in Database"):
            if not user_id or not name:
                st.error("Please enter both User ID and Name.")
            else:
                add_user(user_id, name)
                st.success(f"Saved user: {user_id} - {name}")

    with col2:
        st.markdown("<span class='accent'>Capture face images (webcam)</span>", unsafe_allow_html=True)
        if st.button("Start Capturing Images"):
            if not user_id:
                st.error("Enter User ID first.")
            else:
                st.info("Capturing... Look at the camera. Keep face centered. Good lighting helps.")
                try:
                    saved, last_rgb = capture_images(
                        user_id=user_id.strip(),
                        num_images=int(num_images),
                        cam_index=int(cam_index)
                    )
                    st.success(f"Captured and saved {saved} images for {user_id}.")
                    if last_rgb is not None:
                        st.image(last_rgb, caption="Last captured frame (preview)", channels="RGB")
                except Exception as e:
                    st.error(f"Capture failed: {e}")

    st.divider()
    st.subheader("Registered Users")
    rows = get_users()
    if rows:
        df = pd.DataFrame(rows, columns=["user_id", "name", "created_at"])
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No users registered yet.")

# ---------- Train ----------
with tabs[1]:
    st.subheader("Train face encodings")
    st.markdown(
        "<span class='small-text'>This will scan <b>data/images/&lt;user_id&gt;</b> and create <b>data/encodings/encodings.pkl</b>.</span>",
        unsafe_allow_html=True
    )

    if st.button("Train Now"):
        with st.spinner("Training encodings..."):
            stats = train_encodings()
        st.success("Training complete!")
        st.json(stats)

    st.markdown(
        "<span class='warn'>Tip:</span> <span class='small-text'>After registering new users, train again so they can be recognized.</span>",
        unsafe_allow_html=True
    )

# ---------- Mark Attendance (WEBRTC / BROWSER CAMERA) ----------
with tabs[2]:
    st.subheader("Live webcam recognition → mark attendance (Browser Camera)")
    known = load_encodings()

    tol = st.slider("Recognition strictness (lower = stricter)", 0.30, 0.60, 0.45, 0.01)
    run = st.toggle("Start Camera")

    if known is None:
        st.warning("No encodings found. Please go to 🧠 Train tab and train first.")
        st.stop()

    status_placeholder = st.empty()

    # Map user_id -> name
    user_map = {uid: nm for uid, nm, _ in get_users()}

    if "last_mark_time" not in st.session_state:
        st.session_state.last_mark_time = 0.0
    if "last_mark_label" not in st.session_state:
        st.session_state.last_mark_label = None
    if "last_marked" not in st.session_state:
        st.session_state.last_marked = None

    RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    # NOTE: Removed av.VideoFrame type hints to avoid NameError issues on some setups
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = recognize_from_frame(img, known, tolerance=float(tol))

        now = time.time()
        for label, (top, right, bottom, left) in results:
            cv2.rectangle(
                img,
                (left, top),
                (right, bottom),
                (0, 255, 0) if label != "Unknown" else (0, 0, 255),
                2
            )

            display = label
            if label != "Unknown":
                nm = user_map.get(label, "Unknown Name")
                display = f"{label} - {nm}"

                if now - st.session_state.last_mark_time > 1.5:
                    marked = mark_attendance(label, nm)
                    st.session_state.last_mark_time = now
                    st.session_state.last_mark_label = display
                    st.session_state.last_marked = bool(marked)

            cv2.putText(
                img,
                display,
                (left, max(20, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    if run:
        webrtc_streamer(
            key="attendance",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

        label = st.session_state.get("last_mark_label")
        marked = st.session_state.get("last_marked")
        if label and marked is not None:
            if marked:
                status_placeholder.success(f"✅ Marked attendance: {label} ({date.today().isoformat()})")
            else:
                status_placeholder.warning(f"Already marked today: {label}")
    else:
        st.info("Toggle **Start Camera** to begin browser webcam streaming.")

# ---------- Reports ----------
with tabs[3]:
    st.subheader("Attendance Reports")
    date_pick = st.date_input("Filter by date", value=date.today())
    date_str = date_pick.isoformat()

    today_rows = get_attendance(date_filter=date_str)
    present_count = len(today_rows)
    total_users = len(get_users())

    st.markdown(f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin: 8px 0 16px 0;">
      <div style="flex:1; min-width:220px; padding:14px 16px; border-radius:18px;
                  background:rgba(6,10,22,0.50); border:1px solid rgba(0,255,255,0.20);
                  box-shadow:0 0 0 1px rgba(255,0,255,0.10) inset, 0 14px 44px rgba(0,0,0,0.40);
                  backdrop-filter: blur(14px);">
        <div class="small-text">Selected Date</div>
        <div style="font-size:20px; font-weight:800;">{date_str}</div>
      </div>

      <div style="flex:1; min-width:220px; padding:14px 16px; border-radius:18px;
                  background:rgba(6,10,22,0.50); border:1px solid rgba(0,255,255,0.20);
                  box-shadow:0 0 0 1px rgba(255,0,255,0.10) inset, 0 14px 44px rgba(0,0,0,0.40);
                  backdrop-filter: blur(14px);">
        <div class="small-text">Present</div>
        <div class="kpi-number">{present_count}</div>
      </div>

      <div style="flex:1; min-width:220px; padding:14px 16px; border-radius:18px;
                  background:rgba(6,10,22,0.50); border:1px solid rgba(0,255,255,0.20);
                  box-shadow:0 0 0 1px rgba(255,0,255,0.10) inset, 0 14px 44px rgba(0,0,0,0.40);
                  backdrop-filter: blur(14px);">
        <div class="small-text">Total Users</div>
        <div class="kpi-number">{total_users}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if today_rows:
        df = pd.DataFrame(today_rows, columns=["user_id", "name", "date", "time"])
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Export CSV")
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        out_file = reports_dir / f"attendance_{date_str}.csv"
        df.to_csv(out_file, index=False)

        with open(out_file, "rb") as f:
            st.download_button(
                label=f"Download {out_file.name}",
                data=f,
                file_name=out_file.name,
                mime="text/csv"
            )
        st.markdown(
            f"<span class='success'>Saved locally:</span> <span class='small-text'>{out_file}</span>",
            unsafe_allow_html=True
        )
    else:
        st.write(f"No attendance records for {date_str}.")
