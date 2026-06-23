import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="ORBITAL CONJUNCTION COMMAND",
                   layout="wide", page_icon="🛰️")

# ── Custom fonts + SpaceX-style theming ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Rajdhani', sans-serif; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 2px; text-transform: uppercase; }
.stApp { background: linear-gradient(180deg, #05070d 0%, #0a0f1c 100%); }
[data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; color: #5ad1ff; }
[data-testid="stMetricLabel"] { font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; text-transform: uppercase; color: #8ba3c7; }
.status-line { font-family: 'Rajdhani', sans-serif; font-size: 15px; letter-spacing: 1px;
               color: #5ad1ff; border-left: 3px solid #5ad1ff; padding-left: 12px; margin: 8px 0; }
.status-green { color: #3fe07a; border-left-color: #3fe07a; }
.status-amber { color: #ffb84d; border-left-color: #ffb84d; }
.status-red   { color: #ff5d5d; border-left-color: #ff5d5d; }
.mono { font-family: 'Rajdhani', monospace; letter-spacing: 1px; color: #8ba3c7; }
</style>
""", unsafe_allow_html=True)

PC_ACTION = 1e-4
PC_FLOOR  = 1e-15
RISK_COLORS = {"RED": "#ff5d5d", "YELLOW": "#ffb84d", "GREEN": "#3fe07a"}

def safe_load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

@st.cache_data
def load_positions():
    try: return np.load("data/positions.npy")
    except Exception: return None

@st.cache_data
def load_catalog(): return safe_load_json("data/satellites.json")

operational = safe_load_json("data/operational_results.json")
debris      = safe_load_json("data/debris_results.json")
positions   = load_positions()

def classify(pc, dist, action):
    if pc >= action: return "RED"
    if pc >= action/10 or dist < 5.0: return "YELLOW"
    return "GREEN"

def status(msg, cls=""):
    st.markdown(f'<div class="status-line {cls}">▸ {msg}</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#ffffff'>◢ ORBITAL CONJUNCTION COMMAND ◣</h1>",
            unsafe_allow_html=True)
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"<div class='mono'>MISSION CLOCK: {now} &nbsp;|&nbsp; "
            f"DATA SOURCE: CELESTRAK LIVE CATALOG &nbsp;|&nbsp; "
            f"STATUS: <span style='color:#3fe07a'>OPERATIONAL</span></div>",
            unsafe_allow_html=True)
status("All systems nominal. Tracking real orbital objects in real coordinate space.", "status-green")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h3 style='color:#5ad1ff'>⚙ FLIGHT CONTROLS</h3>", unsafe_allow_html=True)
action_thr = st.sidebar.select_slider("ACTION THRESHOLD (Pc)",
    options=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3], value=1e-4,
    format_func=lambda x: f"{x:.0e}")
max_dist = st.sidebar.slider("MAX MISS DISTANCE (KM)", 1, 25, 25)
st.sidebar.markdown("<div class='mono'>Adjust thresholds to re-triage the catalog "
                    "in real time. Lower thresholds = stricter screening.</div>",
                    unsafe_allow_html=True)

# ════ OVERVIEW ════════════════════════════════════════════════════════════════
st.header("◢ Mission Overview")
if operational:
    df = pd.DataFrame(operational)
    df = df[df["dist_refined"] <= max_dist].copy()
    df["risk_live"] = df.apply(lambda r: classify(r["pc"], r["dist_refined"], action_thr), axis=1)
    reds = int((df["risk_live"]=="RED").sum())
    yellows = int((df["risk_live"]=="YELLOW").sum())
    greens = int((df["risk_live"]=="GREEN").sum())

    if reds == 0:
        status(f"No critical conjunctions. {len(df)} objects tracked, catalog clear of action-level risk.", "status-green")
    else:
        status(f"⚠ {reds} CRITICAL conjunction(s) flagged — maneuver assessment advised.", "status-red")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("OBJECTS TRACKED", len(df))
    c2.metric("🔴 CRITICAL", reds)
    c3.metric("🟡 WATCH", yellows)
    c4.metric("🟢 NOMINAL", greens)

    left, right = st.columns([3,2])
    with left:
        status("Rendering orbital shell — drag to rotate the constellation.")
        if positions is not None:
            sub = positions[:1500]
            u,v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]; R=6371
            fig = go.Figure()
            fig.add_surface(x=R*np.cos(u)*np.sin(v), y=R*np.sin(u)*np.sin(v),
                            z=R*np.cos(v), colorscale="Blues", showscale=False, opacity=0.45)
            fig.add_scatter3d(x=sub[:,0],y=sub[:,1],z=sub[:,2],mode="markers",
                              marker=dict(size=1.6,color="#5ad1ff",opacity=0.7),name="Tracked")
            fig.update_layout(scene=dict(aspectmode="data"),height=460,
                              margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
    with right:
        status("Live risk triage by classification.")
        fig = go.Figure(go.Bar(x=["CRITICAL","WATCH","NOMINAL"],y=[reds,yellows,greens],
              marker_color=["#ff5d5d","#ffb84d","#3fe07a"]))
        fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
              font_color="#8ba3c7",font_family="Rajdhani",yaxis_title="PAIRS")
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"<div class='mono'>CLOSEST APPROACH: <span style='color:#5ad1ff'>"
                    f"{df['dist_refined'].min():.2f} KM</span> &nbsp; | &nbsp; "
                    f"PEAK Pc: <span style='color:#5ad1ff'>{df['pc'].max():.2e}</span></div>",
                    unsafe_allow_html=True)
else:
    st.info("Run phase13_operational.py to populate the overview.")
st.divider()

# ════ CONJUNCTIONS ════════════════════════════════════════════════════════════
st.header("◢ Conjunction Catalog")
if operational:
    status("Every flagged pair, ranked by collision probability. Risk recomputes with the slider.")
    view = df.sort_values("pc", ascending=False)
    disp = view[["a","b","dist_refined","tca_refined_h","pc","risk_live"]].copy()
    disp.columns=["OBJECT A","OBJECT B","MISS (KM)","TCA (H)","Pc","RISK"]
    disp["Pc"]=disp["Pc"].apply(lambda x:f"{x:.2e}")
    disp["MISS (KM)"]=disp["MISS (KM)"].round(2); disp["TCA (H)"]=disp["TCA (H)"].round(1)
    st.dataframe(disp.style.map(lambda v:f"color:{RISK_COLORS.get(v,'white')};font-weight:bold",
                 subset=["RISK"]), height=340, width="stretch", hide_index=True)

    status("KEY INSIGHT: the riskiest pair is not always the closest — geometry and uncertainty matter.", "status-amber")
    plot_df=df.copy(); plot_df["pc_disp"]=plot_df["pc"].clip(lower=PC_FLOOR)
    fig=go.Figure()
    for risk in ["GREEN","YELLOW","RED"]:
        sel=plot_df[plot_df["risk_live"]==risk]
        if len(sel): fig.add_scatter(x=sel["dist_refined"],y=sel["pc_disp"],mode="markers",
                     name=risk,marker=dict(color=RISK_COLORS[risk],size=10))
    fig.add_hline(y=action_thr,line_dash="dash",line_color="#ff5d5d",
                  annotation_text=f"ACTION THRESHOLD {action_thr:.0e}")
    fig.update_layout(yaxis_type="log",yaxis=dict(range=[np.log10(PC_FLOOR),0]),
        xaxis_title="MISS DISTANCE (KM)",yaxis_title="COLLISION PROBABILITY",height=400,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8ba3c7",
        font_family="Rajdhani")
    st.plotly_chart(fig, width="stretch")
st.divider()

# ════ LIVE SCREEN ═════════════════════════════════════════════════════════════
st.header("◢ Live Asset Screening")
status("Select a protected asset and screen it against the catalog with live SGP4 propagation.")
catalog = load_catalog()
if catalog:
    names_all=[s["name"] for s in catalog[:1500]]
    didx=names_all.index("ISS (ZARYA)") if "ISS (ZARYA)" in names_all else 0
    chosen=st.selectbox("PROTECTED ASSET", names_all, index=didx)
    screen_dist=st.slider("SCREENING RADIUS (KM)", 10, 200, 100)
    if st.button("◤ INITIATE LIVE SCREEN ◥"):
        with st.spinner(f"Propagating {chosen} against catalog over 12h..."):
            subset=catalog[:1500]; recs=[]; names=[]
            for s in subset:
                try: recs.append(Satrec.twoline2rv(s["line1"],s["line2"])); names.append(s["name"])
                except Exception: pass
            tgt=names.index(chosen); start=datetime.now(timezone.utc); n_steps=int(12*60/5)+1
            closest={}
            for ti in range(n_steps):
                t=start+timedelta(minutes=ti*5)
                jd,fr=jday(t.year,t.month,t.day,t.hour,t.minute,t.second)
                et,rt,vt=recs[tgt].sgp4(jd,fr)
                if et!=0 or any(np.isnan(rt)): continue
                rt=np.array(rt)
                for j,rec in enumerate(recs):
                    if j==tgt: continue
                    e,r,v=rec.sgp4(jd,fr)
                    if e!=0 or any(np.isnan(r)): continue
                    d=np.linalg.norm(rt-np.array(r))
                    if d<screen_dist and (j not in closest or d<closest[j][0]):
                        closest[j]=(d,ti*5/60.0)
            threats=sorted([(names[j],d,t) for j,(d,t) in closest.items() if d>=0.5],key=lambda x:x[1])
        if threats:
            status(f"SCREEN COMPLETE — {len(threats)} object(s) within {screen_dist} km of {chosen}.", "status-amber")
            tdf=pd.DataFrame(threats,columns=["OBJECT","MISS (KM)","TCA (H)"])
            tdf["MISS (KM)"]=tdf["MISS (KM)"].round(2); tdf["TCA (H)"]=tdf["TCA (H)"].round(1)
            st.dataframe(tdf,height=300,width="stretch",hide_index=True)
        else:
            status(f"SCREEN COMPLETE — airspace clear. No objects within {screen_dist} km of {chosen}.", "status-green")
st.divider()

# ════ DEBRIS ══════════════════════════════════════════════════════════════════
st.header("◢ Debris Threat Field")
status("Active satellites screened against real debris from historic fragmentation events.")
if debris:
    c1,c2,c3=st.columns(3)
    c1.metric("DEBRIS FRAGMENTS", sum(debris["counts"].values()))
    c2.metric("ACTIVE↔DEBRIS", debris["active_debris_pairs"])
    c3.metric("DEBRIS↔DEBRIS", debris["debris_debris_pairs"])
    status("These fragments cannot maneuver. When an active satellite nears one, only the satellite can act.", "status-red")

    act=np.array(debris["active"]); deb=np.array(debris["debris"])
    fig=go.Figure(); u,v=np.mgrid[0:2*np.pi:30j,0:np.pi:20j]; R=6371
    fig.add_surface(x=R*np.cos(u)*np.sin(v),y=R*np.sin(u)*np.sin(v),z=R*np.cos(v),
                    colorscale="Greys",showscale=False,opacity=0.3)
    if len(act): fig.add_scatter3d(x=act[:,0],y=act[:,1],z=act[:,2],mode="markers",
                 marker=dict(size=1.6,color="#5ad1ff",opacity=0.55),name="ACTIVE")
    if len(deb): fig.add_scatter3d(x=deb[:,0],y=deb[:,1],z=deb[:,2],mode="markers",
                 marker=dict(size=1.6,color="#ff5d5d",opacity=0.55),name="DEBRIS")
    fig.update_layout(scene=dict(aspectmode="data"),height=560,margin=dict(l=0,r=0,t=0,b=0),
                      paper_bgcolor="rgba(0,0,0,0)",legend=dict(font=dict(color="#8ba3c7")))
    st.plotly_chart(fig, width="stretch")

    if debris.get("threats"):
        status("Closest documented active-vs-debris approaches in the current window.")
        tdf=pd.DataFrame(debris["threats"]); tdf.columns=["ACTIVE SATELLITE","DEBRIS FRAGMENT","MISS (KM)"]
        st.dataframe(tdf,height=280,width="stretch",hide_index=True)
else:
    st.info("Run phase16_debris.py to populate the debris section.")

st.divider()
st.markdown("<div class='mono'>COLLISION PROBABILITIES USE MODELED COVARIANCES BY ORBIT REGIME "
            "(TLEs CARRY NO UNCERTAINTY DATA) AND THE 2D SHORT-ENCOUNTER (FOSTER) METHOD. "
            "HEAVY SCREENS PRECOMPUTED · SINGLE-ASSET SCREEN RUNS LIVE.<br>"
            "M.Sc. HPC PROJECT — ADITYA RAJENDRA SONAWANE · TH DEGGENDORF</div>",
            unsafe_allow_html=True)

# To run my dashboard - Download the requirements from requirements.txt and run this command on terminal- streamlit run dashboard.py
