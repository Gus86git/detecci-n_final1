import streamlit as st
import numpy as np
from PIL import Image
import io
from datetime import datetime
import time
from io import BytesIO
import os
import pandas as pd
import tempfile

# --- Defensive cv2 import ---
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# --- Plotly ---
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- YOLO ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeBuild AI – Detección de EPP",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    font-size: 2.8rem; color: #60A5FA; text-align: center;
    margin-bottom: 0.2rem; font-weight: 700; letter-spacing: -1px;
}
.sub-header { text-align:center; color:#94A3B8; margin-bottom:1.5rem; font-size:1.1rem; }
.card {
    background: #1E293B; padding: 1.4rem; border-radius: 12px;
    border: 1px solid #334155; margin: 0.6rem 0; color: #F1F5F9;
}
.alert-high {
    background: #1E293B; padding:1.4rem; border-radius:12px;
    border-left: 6px solid #EF4444; margin:1rem 0; color:#F1F5F9;
}
.alert-medium {
    background: #1E293B; padding:1.4rem; border-radius:12px;
    border-left: 6px solid #F59E0B; margin:1rem 0; color:#F1F5F9;
}
.alert-ok {
    background: #1E293B; padding:1.4rem; border-radius:12px;
    border-left: 6px solid #10B981; margin:1rem 0; color:#F1F5F9;
}
.alert-title { font-size:1.3rem; font-weight:700; margin-bottom:0.7rem; }
.alert-action {
    background:#0F172A; padding:0.8rem 1rem; border-radius:8px;
    margin-top:0.8rem; font-size:0.95rem;
}
.badge {
    display:inline-block; padding:0.3rem 0.8rem; border-radius:20px;
    font-size:0.82rem; font-weight:600; margin:0.3rem 0.2rem;
    background:#334155; color:#CBD5E1;
}
.stButton>button {
    width:100%; background:linear-gradient(135deg,#2563EB,#1D4ED8);
    color:white; font-weight:700; border:none; padding:0.75rem 1rem;
    border-radius:10px; transition:all 0.25s; font-size:1rem;
}
.stButton>button:hover { background:linear-gradient(135deg,#1D4ED8,#1E40AF); transform:translateY(-1px); }
.error-box {
    background:#450A0A; border:1px solid #EF4444; padding:1.2rem;
    border-radius:10px; color:#FCA5A5; margin:1rem 0;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── GUARD – show error if critical dependencies missing ───────────────────────
if not CV2_AVAILABLE:
    st.markdown("""
    <div class="error-box">
    <strong>❌ OpenCV no está disponible</strong><br>
    Asegúrate de que <code>packages.txt</code> existe con <code>libgl1-mesa-glx</code> y
    que <code>requirements.txt</code> contiene <code>opencv-python-headless</code>.<br>
    <em>Si estás en Streamlit Cloud, verifica los logs de construcción.</em>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not YOLO_AVAILABLE:
    st.error("❌ Ultralytics YOLO no está instalado. Revisa requirements.txt.")
    st.stop()

# ── EXPERT SYSTEM ─────────────────────────────────────────────────────────────
class SafetyExpertSystem:
    def __init__(self):
        self.rules = [
            ('height_critical',  lambda s: s['persons_high_risk']>0 and s['helmets_high_risk']==0,
             "CRÍTICO: Personal en altura sin ningún casco", "ALTA",
             "🚫 SUSPENDER trabajos en altura – Implementar andamios y redes de seguridad", 1),
            ('height_partial',   lambda s: s['persons_high_risk']>0 and s['helmets_high_risk']<s['persons_high_risk'],
             "ALTO RIESGO: Personal elevado sin protección completa", "ALTA",
             "📏 DELIMITAR área de riesgo – Proveer EPP inmediatamente", 2),
            ('no_ppe_complete',  lambda s: s['persons']>0 and s['full_ppe']==0,
             "PROTECCIÓN INCOMPLETA: Ningún trabajador con EPP completo", "ALTA",
             "🛑 DETENER actividades – Verificar dotación de EPP completo", 3),
            ('no_helmet_all',    lambda s: s['persons']>0 and s['helmets']==0,
             "CRÍTICO: Ningún trabajador usa casco de seguridad", "ALTA",
             "DETENER actividades y notificar al supervisor de seguridad", 4),
            ('no_helmet_some',   lambda s: s['persons']>0 and s['helmets']<s['persons'],
             "ALTA: Trabajadores detectados sin casco de seguridad", "ALTA",
             "Aislar el área y proveer EPP inmediatamente", 5),
            ('no_vest_all',      lambda s: s['persons']>0 and s['vests']==0,
             "MEDIA: Ningún trabajador usa chaleco reflectante", "MEDIA",
             "Notificar al supervisor y proveer chalecos de seguridad", 6),
            ('no_vest_some',     lambda s: s['persons']>0 and s['vests']<s['persons'],
             "MEDIA: Trabajadores detectados sin chaleco reflectante", "MEDIA",
             "Recordar uso obligatorio de chaleco en reunión de seguridad", 7),
            ('all_ok',           lambda s: s['persons']>0 and s['helmets']>=s['persons'] and s['vests']>=s['persons'],
             "OK: Todo el personal cuenta con EPP completo", "OK",
             "Continuar monitoreo y mantener los estándares de seguridad", 8),
            ('no_persons',       lambda s: s['persons']==0,
             "OK: No se detectaron trabajadores en el área analizada", "OK",
             "Continuar con el monitoreo rutinario del área", 9),
        ]

    def analyze(self, detections, conf_threshold=0.4, image_size=None):
        PERSON_CLS = {'person','worker'}
        HELMET_CLS = {'helmet','hardhat','hard-hat'}
        VEST_CLS   = {'safety_vest','vest','safety-vest'}
        persons = sum(1 for d in detections if d['class'] in PERSON_CLS and d['confidence']>=conf_threshold)
        helmets = sum(1 for d in detections if d['class'] in HELMET_CLS and d['confidence']>=conf_threshold)
        vests   = sum(1 for d in detections if d['class'] in VEST_CLS   and d['confidence']>=conf_threshold)
        ctx     = self._context(detections, image_size, conf_threshold)
        stats   = {'persons':persons,'helmets':helmets,'vests':vests,
                   'total_detections':len(detections),**ctx}
        for name,cond,msg,level,action,_ in sorted(self.rules, key=lambda x:x[5]):
            if cond(stats):
                return {'level':level,'message':msg,'action':action,'stats':stats,
                        'compliance':self._compliance(stats),'rule':name}
        return {'level':'OK','message':'Condiciones normales de seguridad','action':'Continuar monitoreo',
                'stats':stats,'compliance':100.0,'rule':'default'}

    def _context(self, detections, image_size, conf_threshold):
        PERSON_CLS={'person','worker'}; HELMET_CLS={'helmet','hardhat','hard-hat'}; VEST_CLS={'safety_vest','vest','safety-vest'}
        if image_size is None:
            return {'persons_high_risk':0,'helmets_high_risk':0,'full_ppe':0}
        h,_ = image_size
        persons = [d for d in detections if d['class'] in PERSON_CLS and d['confidence']>=conf_threshold]
        phr=0; hhr=0; full=0
        for p in persons:
            yc = (p['bbox'][1]+p['bbox'][3])/2
            has_h = any(self._overlap_v(p['bbox'],d['bbox']) for d in detections if d['class'] in HELMET_CLS)
            has_v = any(self._overlap_v(p['bbox'],d['bbox']) for d in detections if d['class'] in VEST_CLS)
            if yc < h*0.4:
                phr+=1
                if has_h: hhr+=1
            if has_h and has_v: full+=1
        return {'persons_high_risk':phr,'helmets_high_risk':hhr,'full_ppe':full}

    def _overlap_v(self, b1, b2, t=0.3):
        ov = min(b1[3],b2[3])-max(b1[1],b2[1])
        return ov > (b1[3]-b1[1])*t

    def _compliance(self, s):
        if s['persons']==0: return 100.0
        hc = (s['helmets']/s['persons'])*100
        vc = (s['vests']/s['persons'])*100
        pen = 0
        if s['persons_high_risk']>0:
            pen = (1 - s['helmets_high_risk']/s['persons_high_risk'])*20
        return max(0, round(hc*0.6 + vc*0.4 - pen, 1))

# ── DETECTION HELPERS ─────────────────────────────────────────────────────────
HELMET_COLORS = [
    ([20,100,100],[30,255,255]),  # yellow
    ([5,100,100],[15,255,255]),   # orange
    ([0,100,100],[10,255,255]),   # red1
    ([160,100,100],[180,255,255]),# red2
    ([100,100,100],[130,255,255]),# blue
    ([0,0,200],[180,30,255]),     # white
]
VEST_COLORS = [
    ([20,100,150],[35,255,255]),  # yellow-fl
    ([5,150,150],[20,255,255]),   # orange-fl
    ([35,100,100],[85,255,255]),  # lime
]

def _color_pct(region, ranges):
    if region is None or region.size==0: return 0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo,hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return np.count_nonzero(mask)/(region.shape[0]*region.shape[1])

def enhance_detections(img_bgr, detections):
    enhanced = list(detections)
    persons = [d for d in detections if d['class']=='person']
    for p in persons:
        x1,y1,x2,y2 = p['bbox']; ph=y2-y1
        head = img_bgr[y1:int(y1+ph*0.25), x1:x2]
        torso= img_bgr[int(y1+ph*0.25):int(y1+ph*0.75), x1:x2]
        if _color_pct(head, HELMET_COLORS)>0.15:
            enhanced.append({'class':'helmet','confidence':0.70,'inferred':True,
                             'bbox':[x1,y1,x2,int(y1+ph*0.25)],
                             'area':(x2-x1)*int(ph*0.25)})
        if _color_pct(torso, VEST_COLORS)>0.20:
            enhanced.append({'class':'safety_vest','confidence':0.65,'inferred':True,
                             'bbox':[x1,int(y1+ph*0.25),x2,int(y1+ph*0.75)],
                             'area':(x2-x1)*int(ph*0.5)})
    return enhanced

@st.cache_resource(show_spinner=False)
def load_model():
    for path in ['models/best.pt','yolov8n.pt']:
        try:
            m = YOLO(path)
            return m, path
        except Exception:
            continue
    return None, None

def run_detection(image_pil, model, conf):
    img = np.array(image_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, conf=conf, iou=0.45, imgsz=640,
                            augment=False, verbose=False, max_det=300)
    dets=[]
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            dets.append({'class': model.names[int(box.cls[0])].lower(),
                         'confidence': float(box.conf[0]),
                         'bbox':[int(x1),int(y1),int(x2),int(y2)],
                         'area':(x2-x1)*(y2-y1), 'inferred':False})
    dets = enhance_detections(img_bgr, dets)
    return dets

COLORS = {
    'person':(0,120,255),'worker':(0,120,255),
    'helmet':(50,205,50),'hardhat':(50,205,50),'hard-hat':(50,205,50),
    'safety_vest':(0,215,255),'vest':(0,215,255),'safety-vest':(0,215,255),
}

def draw_boxes(image_pil, detections, conf):
    img = np.array(image_pil)
    for d in detections:
        if d['confidence']<conf: continue
        x1,y1,x2,y2 = d['bbox']
        col = COLORS.get(d['class'],(200,200,0))
        thick = 2 if d.get('inferred') else 3
        cv2.rectangle(img,(x1,y1),(x2,y2),col,thick)
        label = f"{d['class']}: {d['confidence']:.0%}"
        if d.get('inferred'): label += " (IA)"
        tw,th = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0]
        cv2.rectangle(img,(x1,y1-th-8),(x1+tw+4,y1),col,-1)
        cv2.putText(img,label,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    return Image.fromarray(img)

def process_video(video_bytes, model, conf, max_frames=60):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes); tmp = f.name
    cap = cv2.VideoCapture(tmp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    step  = max(1, total//(max_frames))
    frames_dets=[]; frame_idx=0; processed=0
    while cap.isOpened() and processed<max_frames:
        ret,frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            dets = run_detection(pil, model, conf)
            frames_dets.append(dets)
            processed+=1
        frame_idx+=1
    cap.release()
    os.unlink(tmp)
    return frames_dets, fps

# ── EXPORT ────────────────────────────────────────────────────────────────────
def build_export_df():
    rows=[]
    for i,r in enumerate(st.session_state.history):
        rows.append({
            'ID': i+1, 'Fecha': r['ts'].strftime('%Y-%m-%d'),
            'Hora': r['ts'].strftime('%H:%M:%S'), 'Archivo': r['file'],
            'Alerta': r['level'], 'Cumplimiento': f"{r['compliance']:.1f}%",
            'Personas': r['stats']['persons'], 'Cascos': r['stats']['helmets'],
            'Chalecos': r['stats']['vests'], 'EPP_Completo': r['stats']['full_ppe'],
            'Riesgo_Altura': r['stats']['persons_high_risk'],
            'Total_Det': r['stats']['total_detections'],
        })
    return pd.DataFrame(rows)

def export_excel():
    df = build_export_df()
    buf=BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
        df.to_excel(w,sheet_name='Análisis',index=False)
        ws=w.sheets['Análisis']
        for ci,col in enumerate(df.columns):
            ws.set_column(ci,ci,max(df[col].astype(str).map(len).max(),len(col))+2)
    buf.seek(0); return buf

# ── PLOTLY CHART ──────────────────────────────────────────────────────────────
def compliance_chart():
    if not PLOTLY_AVAILABLE or not st.session_state.history: return
    h = st.session_state.history
    labels=[f"#{i+1} {r['file'][:12]}" for i,r in enumerate(h)]
    vals  =[r['compliance'] for r in h]
    colors=["#10B981" if v>=80 else "#F59E0B" if v>=50 else "#EF4444" for v in vals]
    fig = go.Figure(go.Bar(x=labels,y=vals,marker_color=colors,
                           text=[f"{v:.0f}%" for v in vals],textposition='auto'))
    fig.update_layout(
        title="Cumplimiento EPP por Análisis",
        yaxis=dict(title="Cumplimiento (%)",range=[0,105]),
        paper_bgcolor='#1E293B', plot_bgcolor='#1E293B',
        font=dict(color='#F1F5F9'), margin=dict(l=20,r=20,t=40,b=60),
        height=280,
    )
    fig.add_hline(y=80,line_dash='dash',line_color='#10B981',annotation_text='Meta 80%')
    st.plotly_chart(fig, use_container_width=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state: st.session_state.history=[]
expert = SafetyExpertSystem()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🦺 SafeBuild AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema Inteligente de Detección de EPP • Cascos y Chalecos con YOLO</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    conf = st.slider("Confianza mínima", 0.10, 0.95, 0.40, 0.05,
                     help="Umbral de confianza para aceptar una detección")
    show_boxes = st.checkbox("Mostrar bounding boxes", True)
    st.markdown("---")

    # Model status
    with st.spinner("Cargando modelo YOLO…"):
        model, model_path = load_model()
    if model:
        tag = "personalizado ✅" if 'best.pt' in str(model_path) else "YOLOv8n base ℹ️"
        st.success(f"🤖 Modelo {tag}")
        st.caption(f"Clases detectables: {len(model.names)}")
    else:
        st.error("❌ No se pudo cargar el modelo YOLO")

    st.markdown("---")
    st.markdown("**🎨 Detección por color (IA)**")
    st.info("El sistema infiere EPP por análisis de color cuando YOLO no lo detecta directamente.")

    st.markdown("---")
    st.markdown("**📖 Cómo usar**")
    st.markdown("""
1. Sube una imagen 📸 o video 🎬
2. Ajusta la confianza si es necesario
3. Haz clic en **Analizar**
4. Revisa alertas y métricas
5. Exporta los resultados 📤
""")
    st.markdown("---")
    st.caption("SafeBuild AI v3.0 • YOLOv8 + Sistema Experto")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_hist, tab_export = st.tabs(
    ["📸 Imagen", "🎬 Video", "📋 Historial", "📤 Exportar"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – IMAGEN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_img:
    col_up, col_res = st.columns([3,2])
    with col_up:
        uploaded = st.file_uploader("Selecciona una imagen de la obra:",
                                    type=['jpg','jpeg','png','bmp','webp'],
                                    help="Máx. 200 MB")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption=f"📷 {uploaded.name}", use_container_width=True)
            if st.button("🔍 Analizar con YOLO", use_container_width=True):
                if not model:
                    st.error("❌ Modelo no disponible")
                else:
                    with st.spinner("Analizando imagen…"):
                        prog = st.progress(0)
                        dets = run_detection(image, model, conf)
                        prog.progress(70)
                        arr  = np.array(image)
                        res  = expert.analyze(dets, conf, arr.shape[:2])
                        prog.progress(100); time.sleep(0.1); prog.empty()

                    st.session_state.history.append({
                        'ts': datetime.now(), 'file': uploaded.name,
                        'level': res['level'], 'stats': res['stats'],
                        'compliance': res['compliance'], 'rule': res['rule'],
                    })

                    # Annotated image
                    if show_boxes and dets:
                        ann = draw_boxes(image, dets, conf)
                        with col_res:
                            st.image(ann, caption="🎯 Detecciones", use_container_width=True)

                    # Metrics row
                    st.markdown("---")
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("📦 Total", res['stats']['total_detections'])
                    m2.metric("👥 Personas", res['stats']['persons'])
                    m3.metric("🪖 Cascos", res['stats']['helmets'])
                    m4.metric("🦺 Chalecos", res['stats']['vests'])
                    m5.metric("📈 Cumplimiento", f"{res['compliance']:.0f}%")

                    if res['stats']['persons_high_risk']>0:
                        st.warning(f"⚠️ {res['stats']['persons_high_risk']} persona(s) en zona de altura detectadas")

                    # Alert panel
                    lvl = res['level']
                    cls = 'alert-high' if lvl=='ALTA' else 'alert-medium' if lvl=='MEDIA' else 'alert-ok'
                    icon= '🚨' if lvl=='ALTA' else '⚠️' if lvl=='MEDIA' else '✅'
                    st.markdown(f"""
                    <div class="{cls}">
                      <div class="alert-title">{icon} {res['message']}</div>
                      <div class="alert-action">📋 Acción: {res['action']}</div>
                      <div style="margin-top:0.8rem">
                        <span class="badge">Nivel: {lvl}</span>
                        <span class="badge">EPP completo: {res['stats']['full_ppe']} persona(s)</span>
                        <span class="badge">Cumplimiento: {res['compliance']:.1f}%</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # Detection details
                    with st.expander("📋 Detalle de detecciones"):
                        filtered=[d for d in dets if d['confidence']>=conf]
                        if filtered:
                            det_df=pd.DataFrame([{
                                'Clase':d['class'],'Confianza':f"{d['confidence']:.0%}",
                                'Bbox':str(d['bbox']),'Inferido':d.get('inferred',False),
                            } for d in filtered])
                            st.dataframe(det_df, use_container_width=True)
                        else:
                            st.info("No hay detecciones sobre el umbral seleccionado")
        else:
            st.info("👆 Sube una imagen para comenzar el análisis")
            st.markdown("""
            <div class="card">
            <strong>📸 Recomendaciones para mejores resultados:</strong><br><br>
            • Imágenes con buena iluminación y contraste<br>
            • Trabajadores visibles (no ocluidos)<br>
            • Resolución mínima recomendada: 640×480 px<br>
            • Distancia razonable (cascos y chalecos visibles)
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VIDEO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown("### 🎬 Análisis de Video")
    st.info("El sistema analiza hasta 60 fotogramas distribuidos uniformemente en el video y muestra resultados agregados.")
    vid_file = st.file_uploader("Selecciona un video:", type=['mp4','avi','mov','mkv'])
    max_fr = st.slider("Máximo de fotogramas a analizar", 10, 60, 30)

    if vid_file:
        st.success(f"✅ Video cargado: {vid_file.name} ({vid_file.size/(1024*1024):.1f} MB)")
        if st.button("▶️ Analizar Video", use_container_width=True):
            if not model:
                st.error("❌ Modelo no disponible")
            else:
                with st.spinner(f"Procesando video… (hasta {max_fr} fotogramas)"):
                    pb = st.progress(0)
                    frames_dets, fps = process_video(vid_file.read(), model, conf, max_fr)
                    pb.progress(100); time.sleep(0.1); pb.empty()

                if not frames_dets:
                    st.error("No se pudieron extraer fotogramas del video")
                else:
                    st.success(f"✅ Procesados {len(frames_dets)} fotogramas")
                    # Aggregate stats
                    all_p = [sum(1 for d in fd if d['class']=='person' and d['confidence']>=conf) for fd in frames_dets]
                    all_h = [sum(1 for d in fd if d['class'] in {'helmet','hardhat','hard-hat'} and d['confidence']>=conf) for fd in frames_dets]
                    all_v = [sum(1 for d in fd if d['class'] in {'safety_vest','vest','safety-vest'} and d['confidence']>=conf) for fd in frames_dets]
                    comps = []
                    for p,h,v in zip(all_p,all_h,all_v):
                        hc=(h/p*100) if p>0 else 100
                        vc=(v/p*100) if p>0 else 100
                        comps.append(max(0,hc*0.6+vc*0.4))

                    mc1,mc2,mc3,mc4 = st.columns(4)
                    mc1.metric("🎞️ Fotogramas", len(frames_dets))
                    mc2.metric("👥 Max Personas/frame", max(all_p) if all_p else 0)
                    mc3.metric("🪖 Avg Cascos/frame", f"{sum(all_h)/len(all_h):.1f}" if all_h else "0")
                    mc4.metric("📈 Cumplimiento promedio", f"{sum(comps)/len(comps):.1f}%" if comps else "N/A")

                    if PLOTLY_AVAILABLE:
                        fr_labels=[f"F{i+1}" for i in range(len(comps))]
                        cfig=go.Figure()
                        cfig.add_trace(go.Scatter(x=fr_labels,y=comps,mode='lines+markers',
                                                  line=dict(color='#60A5FA',width=2),
                                                  marker=dict(size=5),name='Cumplimiento'))
                        cfig.add_trace(go.Bar(x=fr_labels,y=all_p,name='Personas',
                                              marker_color='rgba(239,68,68,0.4)',yaxis='y2'))
                        cfig.update_layout(
                            title="Cumplimiento EPP y Personas por fotograma",
                            yaxis=dict(title="Cumplimiento (%)",range=[0,105]),
                            yaxis2=dict(title="Personas",overlaying='y',side='right'),
                            paper_bgcolor='#1E293B',plot_bgcolor='#1E293B',
                            font=dict(color='#F1F5F9'),height=300,
                            margin=dict(l=20,r=20,t=50,b=40),
                            legend=dict(orientation='h',yanchor='bottom',y=1.02)
                        )
                        st.plotly_chart(cfig, use_container_width=True)

                    # Add to history
                    avg_comp = sum(comps)/len(comps) if comps else 0
                    avg_p = sum(all_p)/len(all_p) if all_p else 0
                    avg_h = sum(all_h)/len(all_h) if all_h else 0
                    avg_v = sum(all_v)/len(all_v) if all_v else 0
                    st.session_state.history.append({
                        'ts': datetime.now(), 'file': f"[VIDEO] {vid_file.name}",
                        'level': 'OK' if avg_comp>=80 else 'MEDIA' if avg_comp>=50 else 'ALTA',
                        'stats': {'persons':round(avg_p),'helmets':round(avg_h),'vests':round(avg_v),
                                  'total_detections':sum(len(fd) for fd in frames_dets),
                                  'full_ppe':0,'persons_high_risk':0},
                        'compliance': round(avg_comp,1), 'rule': 'video_aggregate',
                    })
    else:
        st.info("👆 Sube un video para comenzar el análisis")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HISTORIAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("### 📋 Historial de Análisis")
    if st.session_state.history:
        compliance_chart()
        st.markdown("---")
        for i, r in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history)-i
            icon = '🚨' if r['level']=='ALTA' else '⚠️' if r['level']=='MEDIA' else '✅'
            cls  = 'alert-high' if r['level']=='ALTA' else 'alert-medium' if r['level']=='MEDIA' else 'alert-ok'
            st.markdown(f"""
            <div class="{cls}" style="padding:1rem">
              <strong>{icon} Análisis #{idx}</strong> — {r['file']}<br>
              🕐 {r['ts'].strftime('%Y-%m-%d %H:%M:%S')} &nbsp;&nbsp;
              👥 {r['stats']['persons']} personas &nbsp;&nbsp;
              🪖 {r['stats']['helmets']} cascos &nbsp;&nbsp;
              🦺 {r['stats']['vests']} chalecos &nbsp;&nbsp;
              📈 {r['compliance']:.1f}% cumplimiento
            </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🗑️ Limpiar historial"):
            st.session_state.history=[]; st.rerun()
    else:
        st.info("📝 Aún no hay análisis realizados. Ve a la pestaña 📸 Imagen o 🎬 Video.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – EXPORTAR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("### 📤 Exportar Resultados")
    if not st.session_state.history:
        st.warning("📭 No hay datos para exportar. Realiza al menos un análisis.")
    else:
        df = build_export_df()
        ca, cb, cc = st.columns(3)

        with ca:
            st.markdown("#### 📊 Excel (.xlsx)")
            st.markdown("Incluye hoja de análisis detallado con todas las métricas")
            xl = export_excel()
            st.download_button("📥 Descargar Excel",
                data=xl,
                file_name=f"safebuild_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

        with cb:
            st.markdown("#### 📄 CSV (.csv)")
            st.markdown("Formato universal compatible con cualquier software")
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV",
                data=csv_bytes,
                file_name=f"safebuild_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True)

        with cc:
            st.markdown("#### 📋 Resumen")
            total = len(st.session_state.history)
            altas = sum(1 for r in st.session_state.history if r['level']=='ALTA')
            oks   = sum(1 for r in st.session_state.history if r['level']=='OK')
            avg_c = sum(r['compliance'] for r in st.session_state.history)/total
            st.markdown(f"""
            <div class="card">
            📊 <strong>Total análisis:</strong> {total}<br>
            🚨 <strong>Alertas críticas:</strong> {altas}<br>
            ✅ <strong>Condiciones OK:</strong> {oks}<br>
            📈 <strong>Cumplimiento promedio:</strong> {avg_c:.1f}%
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 👀 Vista previa")
        st.dataframe(df, use_container_width=True, hide_index=True)

# ── GLOBAL STATS BAR ──────────────────────────────────────────────────────────
st.markdown("---")
g1,g2,g3,g4 = st.columns(4)
total = len(st.session_state.history)
g1.metric("🔍 Total Análisis", total)
g2.metric("🚨 Alertas Generadas", sum(1 for r in st.session_state.history if r['level'] in ['ALTA','MEDIA']))
g3.metric("👥 Promedio Personas", f"{np.mean([r['stats']['persons'] for r in st.session_state.history]):.1f}" if total else "0")
g4.metric("📊 Cumplimiento Promedio", f"{np.mean([r['compliance'] for r in st.session_state.history]):.1f}%" if total else "0%")

st.markdown("""
<div style="text-align:center;color:#475569;padding:2rem 0 1rem;font-size:0.9rem">
    <strong>SafeBuild AI v3.0</strong> — Sistema de Detección de EPP con YOLO<br>
    🤖 Powered by YOLOv8 + Sistema Experto + Análisis de Color IA
</div>
""", unsafe_allow_html=True)
