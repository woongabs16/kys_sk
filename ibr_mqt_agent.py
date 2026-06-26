"""
IBR Model Quality Test (MQT) AI Agent
=====================================
인버터(IBR) 모델 품질테스트 자동화를 위한 AI 에이전트.
실행: streamlit run ibr_mqt_agent.py
"""
import os
import io
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak,
    Image as RLImage, ListFlowable, ListItem
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------------
# API 키
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        secret_key = st.secrets["OPENAI_API_KEY"]
        if secret_key:
            return secret_key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

def get_client():
    key = get_api_key()
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)

# ----------------------------------------------------------------------------
# 이미지 URL
# ----------------------------------------------------------------------------
GRID_IMG_URL = "https://raw.githubusercontent.com/woongabs16/kys_sk/main/grid_interconnection.png"
MODEL_IMG_URL = "https://raw.githubusercontent.com/woongabs16/kys_sk/main/pv_plant_model.png"

# ----------------------------------------------------------------------------
# Plant Information
# ----------------------------------------------------------------------------
PLANT_INFO = {
    "Plant capacity": "180 MW PV Plant",
    "POI": "345 kV (Grid 999 INF)",
    "Substation Transformer": "345 kV / 34.5 kV",
    "Collector Bus": "34.5 kV",
    "Equivalent Collector System": "R = 0.002 pu, X = 0.008 pu, B = 0.03 pu",
    "POC": "34.5 kV",
    "Pad-mounted Transformer": "34.5 kV / 0.69 kV",
    "IBR": "0.69 kV (IBR 91003 GEN)",
    "Dynamic models": "REPCAU1 (Plant) -> REECAU1 (Electrical) -> REGCAU1 (Generator)",
}

# ----------------------------------------------------------------------------
# Test Cases
# ----------------------------------------------------------------------------
PLANT_MW = 180.0
DT = 0.01
T_END = 10.0
FAULT_START = 2.0
F_NOM = 60.0
SETTLE_WIN = 1.5

TEST_CASES = {
    "LVRT": [
        {"id": "LVRT_01", "level": 0.10, "dur": 0.15, "recover": True,
         "desc": "V=0.10pu, 0.15s @POI (IEEE2800 5.3.2)"},
        {"id": "LVRT_02", "level": 0.20, "dur": 0.32, "recover": False,
         "desc": "V=0.20pu, 0.32s SLG fault - 회복 지연/감쇠 부족 (Fail 사례)"},
    ],
    "HVRT": [
        {"id": "HVRT_01", "level": 1.15, "dur": 0.50, "recover": True,
         "desc": "V=1.15pu, 0.50s swell (IEEE2800 5.3.5)"},
        {"id": "HVRT_02", "level": 1.18, "dur": 0.20, "recover": True,
         "desc": "V=1.18pu, 0.20s swell"},
    ],
    "VSTEP": [
        {"id": "VSTEP_01",
         "steps": [(2.0, 3.5, 1.05), (5.5, 7.0, 0.95)],
         "desc": "±5% voltage step change"},
        {"id": "VSTEP_02",
         "steps": [(2.0, 3.5, 1.10), (5.5, 7.0, 0.90)],
         "desc": "±10% voltage step change"},
    ],
}

TEST_LABEL = {
    "LVRT": "Low Voltage Ride-Through Test",
    "HVRT": "High Voltage Ride-Through Test",
    "VSTEP": "Voltage Step Change Test",
}

def _make_timeseries(case):
    t = np.arange(0, T_END, DT)
    rng = np.random.default_rng(abs(hash(case["id"])) % (2**32))

    if "steps" in case:
        v_cmd = np.ones_like(t)
        for (s, e, lvl) in case["steps"]:
            v_cmd[(t >= s) & (t < e)] = lvl
        v = np.empty_like(t)
        v[0] = v_cmd[0]
        alpha = DT / (DT + 0.03)
        for k in range(1, len(t)):
            v[k] = v[k-1] + alpha * (v_cmd[k] - v[k-1])
        p = np.full_like(t, PLANT_MW) - PLANT_MW * 0.02 * (v - 1.0)
        q = -PLANT_MW * 0.8 * (v - 1.0)
    else:
        level, dur = case["level"], case["dur"]
        v = np.ones_like(t)
        fault_mask = (t >= FAULT_START) & (t < FAULT_START + dur)
        v[fault_mask] = level
        post_mask = t >= FAULT_START + dur
        v_at_clear = v[fault_mask][-1] if fault_mask.any() else 1.0

        if case.get("recover", True):
            # HVRT 개선: 빠른 회복 + 낮은 오버슈트
            tau, target = 0.12, 1.0
            v[post_mask] = target + (v_at_clear - target) * np.exp(-(t[post_mask]-(FAULT_START+dur))/tau)
        else:
            tau, target = 0.9, 1.0
            decay = np.exp(-(t[post_mask]-(FAULT_START+dur))/tau)
            osc = 0.12 * np.sin(2*np.pi*1.2*(t[post_mask]-(FAULT_START+dur)))
            v[post_mask] = target - (target - v_at_clear)*decay + osc*decay

        p = PLANT_MW * np.clip(v, 0, 1.2)
        if not case.get("recover", True):
            p[post_mask] *= 0.55
        q = np.zeros_like(t)
        q[fault_mask] = PLANT_MW * 0.5 * (1.0 - v[fault_mask])

    v = v + rng.normal(0, 0.002, size=v.shape)
    p = p + rng.normal(0, 0.3, size=p.shape)
    q = q + rng.normal(0, 0.3, size=q.shape)
    freq = np.full_like(t, F_NOM) + rng.normal(0, 0.005, size=t.shape)

    return pd.DataFrame({
        "Time (s)": t, "Voltage (pu)": v, "Frequency (Hz)": freq,
        "Active Power (MW)": p, "Reactive Power (MVar)": q,
    })

def generate_all_cases(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for kind, cases in TEST_CASES.items():
        rows = []
        for case in cases:
            df = _make_timeseries(case)
            csv_path = out_dir / f"{case['id']}.csv"
            df.to_csv(csv_path, index=False)
            rows.append((case, df, csv_path))
        generated[kind] = rows
    return generated

# ----------------------------------------------------------------------------
# Judgment
# ----------------------------------------------------------------------------
def quick_metrics(df):
    pre = df[df["Time (s)"] < FAULT_START]
    settle = df[df["Time (s)"] > (T_END - SETTLE_WIN)]
    p_pre = pre["Active Power (MW)"].mean()
    p_post = settle["Active Power (MW)"].mean()
    v_overshoot = df["Voltage (pu)"].max()
    v_settle_band = settle["Voltage (pu)"].std()
    recovery_ratio = (p_post / p_pre) if p_pre else 0.0
    return {
        "pre_fault_P_MW": round(p_pre, 2),
        "post_fault_P_MW": round(p_post, 2),
        "max_V_pu": round(v_overshoot, 4),
        "post_V_std": round(v_settle_band, 4),
        "P_recovery_ratio": round(recovery_ratio, 4),
    }

DEFAULT_PARAM_TABLE = [
    {"name": "Tp", "model": "REGCAU1", "desc": "Voltage filter time constant", "current": "0.5", "recommended": "0.02"},
    {"name": "Kvp", "model": "REECAU1", "desc": "Voltage proportional gain", "current": "0.1", "recommended": "0.9"},
    {"name": "Kvi", "model": "REECAU1", "desc": "Voltage integral gain", "current": "0.1", "recommended": "0.4"},
]

def _rule_based_judgment(metrics):
    recov = metrics["P_recovery_ratio"]
    overshoot = metrics["max_V_pu"]
    vstd = metrics["post_V_std"]
    fail_reasons = []
    if recov < 0.95:
        fail_reasons.append(f"유효전력 회복률 {recov:.0%} < 95%")
    if overshoot > 1.20:
        fail_reasons.append(f"전압 오버슈트 {overshoot:.3f}pu > 1.20pu")
    if vstd > 0.025:
        fail_reasons.append(f"전압 정착 불안정 (std {vstd:.3f}pu)")
    if fail_reasons:
        return {"verdict": "Fail", "reason": "; ".join(fail_reasons),
                "param_table": DEFAULT_PARAM_TABLE, "rationale": ["Control gains are too low and filter time constant is too large."]}
    return {"verdict": "Pass", "reason": "All criteria satisfied per IEEE 2800.2",
            "param_table": [], "rationale": []}

def judge_with_openai(case, kind, metrics):
    client = get_client()
    fallback = _rule_based_judgment(metrics)
    if client is None:
        fallback["source"] = "rule-based"
        return fallback
    # ... (기존 OpenAI 판정 로직 유지)
    prompt = f"""..."""  # 기존 prompt 유지 (간략화)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}], temperature=0.1)
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        data.setdefault("param_table", [])
        data.setdefault("rationale", [])
        data["source"] = "OpenAI (gpt-4o-mini)"
        return data
    except Exception:
        fallback["source"] = "rule-based (fallback)"
        return fallback

# ----------------------------------------------------------------------------
# PDF Report
# ----------------------------------------------------------------------------
ORANGE = colors.HexColor("#E8721C")
ORANGE_DEEP = colors.HexColor("#C9551A")
CREAM = colors.HexColor("#FFF1E2")
RED = colors.HexColor("#c0392b")

def _plot_case(df, case_id):
    fig, ax1 = plt.subplots(figsize=(6, 2.8))
    ax1.plot(df["Time (s)"], df["Voltage (pu)"], color="#E8721C", lw=1.8, label="V (pu)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (pu)", color="#E8721C")
    ax1.axhline(1.0, color="grey", ls="--", lw=0.7)
    ax2 = ax1.twinx()
    ax2.plot(df["Time (s)"], df["Active Power (MW)"], color="#8a5a2b", lw=1.4, label="P (MW)")
    ax2.set_ylabel("P (MW)", color="#8a5a2b")
    ax1.set_title(case_id, fontsize=11, weight="bold")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(results, out_path):
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], alignment=TA_CENTER, textColor=ORANGE_DEEP, fontSize=20)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=ORANGE_DEEP, spaceAfter=12)
    body = styles["BodyText"]

    elems = [Paragraph("IBR Model Quality Test Report", title),
             Paragraph(f"180 MW PV Plant | IEEE 2800.2 Conformity | {datetime.now():%Y-%m-%d}", 
                       ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.grey)),
             Spacer(1, 20)]

    n_fail = sum(1 for r in results if r["judgment"]["verdict"] == "Fail")
    elems.append(Paragraph(f"Summary — Total {len(results)} cases, {len(results)-n_fail} Pass / {n_fail} Fail", h2))

    # ... (기존 테이블 및 상세 페이지 로직 유지, 필요 시 더 정리)
    # (전체 build_pdf 함수는 이전 버전과 동일하게 유지하되, Fail reason이 깨끗하게 나오도록)

    doc.build(elems)
    return out_path

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="IBR MQT AI Agent", page_icon="🟠", layout="wide")

WORK_DIR = Path("./mqt_workspace")
CSV_DIR = WORK_DIR / "csv"
PDF_PATH = WORK_DIR / "MQT_Report.pdf"

# (CSS는 이전과 동일)

def sidebar():
    st.sidebar.markdown("### 🟠 MQT AI Agent")
    st.sidebar.caption("IBR Model Quality Test")
    if get_api_key():
        st.sidebar.success("MQT 준비완료")
        st.sidebar.caption("개발자 OPEN AI API KEY로 동작 (사용자 입력 불필요)")
    else:
        st.sidebar.warning("키 미설정 → 룰베이스 판정")
    st.sidebar.markdown("---")
    return st.sidebar.radio("MENU", ["Model Quality Test", "Power System Chatbot"])

def render_plant_model():
    st.markdown("<div class='section-title'>Applied Model · 180 MW PV Plant</div><div class='hr'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.markdown("<div class='diagram'><div class='cap'>Grid Interconnection Diagram</div></div>", unsafe_allow_html=True)
        st.image(GRID_IMG_URL, use_container_width=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='diagram'><div class='cap'>PV Plant Model (WECC Generic REGC / REEC / REPC)</div></div>", unsafe_allow_html=True)
        st.image(MODEL_IMG_URL, width=380)  # 크기 축소
    with c2:
        st.markdown("<div class='card'><div class='cap' style='color:#c9551a;font-weight:800;margin-bottom:8px'>Plant Model Specifications</div>"
                    + "".join(f"<div class='kv'><b>{k}</b> : {v}</div>" for k, v in PLANT_INFO.items())
                    + "</div>", unsafe_allow_html=True)

def page_run():
    st.markdown("<div class='hero'><h1>IBR Model Quality Test · AI Agent</h1><p>180 MW PV Plant IEEE 2800.2 Compliance Test</p></div>", unsafe_allow_html=True)
    render_plant_model()

    st.markdown("<div class='section-title' style='margin-top:18px'>품질테스트 실행</div><div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("""
    **Performed Tests**  
    • Voltage Ride-Through Test (LVRT & HVRT)  
    • Voltage Step Change Test
    """)

    c1, _, _ = st.columns([1, 1, 1])
    with c1:
        run = st.button("▶ 시뮬레이션 & AI 판정 실행", use_container_width=True)

    if run:
        with st.spinner("IEEE 2800.2 테스트 판정 중..."):
            generated = generate_all_cases(CSV_DIR)
            results = []
            progress = st.progress(0.0)
            total = sum(len(v) for v in generated.values())
            done = 0
            for kind, items in generated.items():
                for case, df, _ in items:
                    metrics = quick_metrics(df)
                    judgment = judge_with_openai(case, kind, metrics)
                    results.append({"kind": kind, "case": case, "df": df, "metrics": metrics, "judgment": judgment})
                    done += 1
                    progress.progress(done / total)
            st.session_state["results"] = results
        st.toast("판정 완료!", icon="✅")

    # ... (나머지 UI 로직은 이전 코드와 동일)

    if st.button("📄 PDF 보고서 생성"):
        with st.spinner("PDF 생성 중..."):
            build_pdf(st.session_state["results"], PDF_PATH)
        with open(PDF_PATH, "rb") as f:
            st.download_button("⬇ 보고서 다운로드", f.read(), file_name="IBR_MQT_Report.pdf", mime="application/pdf")

# Chatbot 함수는 기존과 동일

choice = sidebar()
if choice == "Model Quality Test":
    page_run()
else:
    page_chatbot()
