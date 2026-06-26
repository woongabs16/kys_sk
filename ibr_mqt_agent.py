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
# 테스트 케이스 및 로직 (이전 개선 유지)
# ----------------------------------------------------------------------------
PLANT_MW = 180.0
DT = 0.01
T_END = 10.0
FAULT_START = 2.0
F_NOM = 60.0
SETTLE_WIN = 1.5

TEST_CASES = {
    "LVRT": [
        {"id": "LVRT_01", "level": 0.10, "dur": 0.15, "recover": True, "desc": "V=0.10pu, 0.15s @POI (IEEE2800 5.3.2)"},
        {"id": "LVRT_02", "level": 0.20, "dur": 0.32, "recover": False, "desc": "V=0.20pu, 0.32s SLG fault - 회복 지연/감쇠 부족 (Fail 사례)"},
    ],
    "HVRT": [
        {"id": "HVRT_01", "level": 1.15, "dur": 0.50, "recover": True, "desc": "V=1.15pu, 0.50s swell (IEEE2800 5.3.5)"},
        {"id": "HVRT_02", "level": 1.18, "dur": 0.20, "recover": True, "desc": "V=1.18pu, 0.20s swell"},
    ],
    "VSTEP": [
        {"id": "VSTEP_01", "steps": [(2.0, 3.5, 1.05), (5.5, 7.0, 0.95)], "desc": "±5% voltage step change"},
        {"id": "VSTEP_02", "steps": [(2.0, 3.5, 1.10), (5.5, 7.0, 0.90)], "desc": "±10% voltage step change"},
    ],
}

TEST_LABEL = {
    "LVRT": "Low Voltage Ride-Through Test",
    "HVRT": "High Voltage Ride-Through Test",
    "VSTEP": "Voltage Step Change Test",
}

# _make_timeseries, generate_all_cases, quick_metrics, judgment 함수들은 이전 버전 유지 (HVRT Pass 개선 포함)
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
            tau = 0.12 if "HVRT" in case["id"] else 0.15
            target = 1.0
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

DEFAULT_RATIONALE = [
    "Control gains are too low and filter time constant is too large.",
    "High Tp introduces excessive delay in voltage measurement.",
    "Raising Kvp/Kvi improves response and damping per IEEE 2800.2."
]

def _rule_based_judgment(metrics):
    recov = metrics["P_recovery_ratio"]
    overshoot = metrics["max_V_pu"]
    vstd = metrics["post_V_std"]
    fail_reasons = []
    if recov < 0.95:
        fail_reasons.append(f"유효전력 회복률 {recov:.0%} < 95% (출력 미복귀)")
    if overshoot > 1.20:
        fail_reasons.append(f"전압 오버슈트 {overshoot:.3f}pu > 1.20pu")
    if vstd > 0.025:
        fail_reasons.append(f"고장 후 전압 정착 불안정 (std {vstd:.3f}pu)")
    if fail_reasons:
        return {"verdict": "Fail", "reason": "; ".join(fail_reasons), "param_table": DEFAULT_PARAM_TABLE, "rationale": DEFAULT_RATIONALE}
    return {"verdict": "Pass", "reason": "전압 회복 및 출력 복귀 정상", "param_table": [], "rationale": []}

def judge_with_openai(case, kind, metrics):
    client = get_client()
    fallback = _rule_based_judgment(metrics)
    if client is None:
        fallback["source"] = "rule-based"
        return fallback
    # OpenAI 판정 (간략 버전)
    try:
        prompt = f"""..."""  # 실제로는 기존 상세 prompt 사용
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        data.setdefault("param_table", [])
        data.setdefault("rationale", [])
        data["source"] = "OpenAI"
        return data
    except Exception:
        return fallback

# ----------------------------------------------------------------------------
# PDF (기존 유지)
# ----------------------------------------------------------------------------
ORANGE = colors.HexColor("#E8721C")
ORANGE_DEEP = colors.HexColor("#C9551A")
CREAM = colors.HexColor("#FFF1E2")
RED = colors.HexColor("#c0392b")

def _plot_case(df, case_id):
    fig, ax1 = plt.subplots(figsize=(5.6, 2.6))
    ax1.plot(df["Time (s)"], df["Voltage (pu)"], color="#E8721C", lw=1.5)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Voltage (pu)", color="#E8721C")
    ax1.axhline(1.0, color="grey", ls="--")
    ax2 = ax1.twinx()
    ax2.plot(df["Time (s)"], df["Active Power (MW)"], color="#8a5a2b", lw=1.2)
    ax2.set_ylabel("P (MW)", color="#8a5a2b")
    ax1.set_title(case_id, weight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(results, out_path):
    # (기존 build_pdf 함수 전체 유지 - 공간 관계로 생략, 이전 코드와 동일하게 사용)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    # ... (전체 build_pdf 내용은 이전 버전과 동일)
    # 실제 구현 시 이전 응답의 build_pdf 함수를 그대로 복사
    pass  # 실제 코드에서는 전체 build_pdf 함수 넣으세요

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="IBR MQT AI Agent", page_icon="🟠", layout="wide")
WORK_DIR = Path("./mqt_workspace")
CSV_DIR = WORK_DIR / "csv"
PDF_PATH = WORK_DIR / "MQT_Report.pdf"

CUSTOM_CSS = """<style>
/* 이전 CSS 전체 유지 */
:root { --orange:#E8721C; --orange-deep:#C9551A; --cream:#FFF1E2; }
.stApp { background: radial-gradient(1100px 560px at 12% -8%, #fff2e3 0%, #fff8f1 45%, #fdf6ef 100%); }
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #f0e2d3; }
.hero { background: linear-gradient(120deg, #f59b4d 0%, #e8721c 60%, #d9591a 100%); color:#fff; padding:26px 30px; border-radius:16px; margin-bottom:18px; }
.metric-card, .card, .diagram { /* 기존 스타일 유지 */ }
</style>"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def metric_card(col, value, label, cls=""):
    col.markdown(f"<div class='metric-card'><div class='v {cls}'>{value}</div><div class='l'>{label}</div></div>", unsafe_allow_html=True)

def sidebar():
    st.sidebar.markdown("### 🟠 MQT AI Agent")
    st.sidebar.caption("IBR Model Quality Test")
    if get_api_key():
        st.sidebar.success("개발자 Open AI API키로 동작 (사용자 입력 불필요)")
    else:
        st.sidebar.warning("키 미설정 → 룰베이스 판정")
    st.sidebar.markdown("---")
    return st.sidebar.radio("MENU", ["Model Quality Test", "Power System Chatbot"])

def render_plant_model():
    st.markdown("<div class='section-title'>적용 모델 · 180 MW PV Plant</div><div class='hr'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("**Grid interconnection of the plant model**")
        st.image(GRID_IMG_URL, use_container_width=True)
        st.markdown("**PV plant model (WECC Generic REGC / REEC / REPC)**")
        st.image(MODEL_IMG_URL, use_container_width=True)
    with col2:
        st.markdown("<div class='card'><div style='color:#c9551a;font-weight:800;margin-bottom:12px'>Plant Model Specifications</div>" +
                    "".join(f"<div class='kv'><b>{k}</b> : {v}</div>" for k, v in PLANT_INFO.items()) +
                    "</div>", unsafe_allow_html=True)

def page_run():
    st.markdown("<div class='hero'><h1>IBR Model Quality Test · AI Agent</h1><p>180 MW PV Plant 인버터 모델 품질테스트 자동 판정 (IEEE 2800.2)</p></div>", unsafe_allow_html=True)
    render_plant_model()
    st.markdown("<div class='section-title'>품질테스트 실행</div><div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("**Performed Tests**  \n• Voltage Ride-Through Test (LVRT & HVRT)  \n• Voltage Step Change Test")
    # ... (나머지 page_run 로직은 이전과 동일)

def page_chatbot():
    st.markdown(
        "<div class='hero'><h1>⚡ Power System Chatbot 🤖</h1>"
        "<p>🔌 IBR 계통연계 · 📘 IEEE 2800/2800.2 · 🌊 LVRT/HVRT · 🧩 PSS/E 모델 질의응답</p></div>",
        unsafe_allow_html=True)
    client = get_client()
    if client is None:
        st.info("🔑 서비스 키가 설정되지 않아 챗봇을 사용할 수 없습니다.")
        return
    st.markdown("##### 💡 추천 질문")
    st.caption("⚡ IEEE 2800 LVRT 기준은? 🌊 REGCAU1 Tp는 무슨 역할? 🔋 Voltage Step Change Test 절차는?")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "👋 안녕하세요! ⚡ IBR 계통연계 전문 챗봇입니다. 무엇이든 물어보세요. 😊"}]
    for m in st.session_state.chat:
        avatar = "🤖" if m["role"] == "assistant" else "🧑‍🔧"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
    if prompt := st.chat_input("💬 질문을 입력하세요"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(prompt)
        system = "You are a friendly power-system engineer assistant specializing in IBR grid interconnection, IEEE 2800/2800.2..."
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, *st.session_state.chat]
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ 오류: {e}"
        st.session_state.chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)

choice = sidebar()
if choice == "Model Quality Test":
    page_run()
else:
    page_chatbot()
