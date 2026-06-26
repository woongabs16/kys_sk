"""
IBR Model Quality Test (MQT) AI Agent
=====================================
인버터(IBR) 모델 품질테스트 자동화를 위한 AI 에이전트.

흐름:
  1. PSS/E 생략 → 180MW PV Plant의 LVRT / HVRT / Voltage Step Change
     테스트를 각 2케이스씩 모사한 CSV 데이터 생성
  2. 생성된 CSV를 바탕으로 OpenAI가 IEEE 2800.2 기준으로 Pass/Fail 판정
  3. 판정 결과 PDF 보고서 제공 (Fail이면 실패 이유 + 모델 파라미터 수정방안 포함)
  4. 챗봇으로 사용자 질문 응답

실행:  streamlit run ibr_mqt_agent.py

[B방법] 키는 코드/깃에 두지 않고 st.secrets 또는 환경변수에서만 읽는다.
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
# API 키 (B방법: secrets / env 만)
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
# 1) PSS/E 생략 - 시험 케이스용 CSV 시계열 데이터 생성
# ----------------------------------------------------------------------------
PLANT_MW = 180.0
DT = 0.01
T_END = 10.0
FAULT_START = 2.0
F_NOM = 60.0

# 시험 유형별 2케이스
#  - LVRT: 저전압 통과시험 (LVRT_02 = Fail 케이스: 회복 지연/감쇠 부족)
#  - HVRT: 고전압 통과시험 (모두 Pass)
#  - VSTEP: Voltage Step Change Test (±5% step 응답)
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
    "VSTEP": [   # Voltage Step Change Test
        {"id": "VSTEP_01", "level": 1.05, "dur": 2.00, "recover": True,
         "desc": "+5% voltage step change response"},
        {"id": "VSTEP_02", "level": 0.95, "dur": 2.00, "recover": True,
         "desc": "-5% voltage step change response"},
    ],
}

TEST_LABEL = {
    "LVRT": "Low Voltage Ride-Through Test",
    "HVRT": "High Voltage Ride-Through Test",
    "VSTEP": "Voltage Step Change Test",
}


def _make_timeseries(case):
    """단순 응답 모델로 V/P/Q 시계열을 생성 (PSS/E 대체)."""
    t = np.arange(0, T_END, DT)
    v = np.ones_like(t)
    level, dur = case["level"], case["dur"]

    fault_mask = (t >= FAULT_START) & (t < FAULT_START + dur)
    v[fault_mask] = level

    post_mask = t >= FAULT_START + dur
    v_at_clear = v[fault_mask][-1] if fault_mask.any() else 1.0

    if case.get("recover", True):
        tau = 0.15
        target = 1.0
        v[post_mask] = target + (v_at_clear - target) * np.exp(-(t[post_mask] - (FAULT_START + dur)) / tau)
    else:
        # Fail 사례: 느린 회복 + 감쇠 부족(언더댐핑) → 1.0pu 미도달 + 진동
        tau = 0.9
        target = 1.0
        decay = np.exp(-(t[post_mask] - (FAULT_START + dur)) / tau)
        osc = 0.12 * np.sin(2 * np.pi * 1.2 * (t[post_mask] - (FAULT_START + dur)))
        v[post_mask] = target - (target - v_at_clear) * decay + osc * decay

    p = PLANT_MW * np.clip(v, 0, 1.2)
    if not case.get("recover", True):
        p[post_mask] *= 0.55  # 출력 미복귀

    q = np.zeros_like(t)
    q[fault_mask] = PLANT_MW * 0.5 * (1.0 - v[fault_mask])

    rng = np.random.default_rng(abs(hash(case["id"])) % (2**32))
    v += rng.normal(0, 0.002, size=v.shape)
    p += rng.normal(0, 0.3, size=p.shape)
    q += rng.normal(0, 0.3, size=q.shape)
    freq = np.full_like(t, F_NOM) + rng.normal(0, 0.005, size=t.shape)

    return pd.DataFrame({
        "Time (s)": t,
        "Voltage (pu)": v,
        "Frequency (Hz)": freq,
        "Active Power (MW)": p,
        "Reactive Power (MVar)": q,
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
# 2) IEEE 2800.2 기준 Pass/Fail 판정
# ----------------------------------------------------------------------------
def quick_metrics(df):
    pre = df[df["Time (s)"] < FAULT_START]
    post = df[df["Time (s)"] > FAULT_START + 1.0]
    p_pre = pre["Active Power (MW)"].mean()
    p_post = post["Active Power (MW)"].mean()
    v_post = post["Voltage (pu)"].mean()
    v_overshoot = df["Voltage (pu)"].max()
    v_settle_band = post["Voltage (pu)"].std()
    recovery_ratio = (p_post / p_pre) if p_pre else 0.0
    return {
        "pre_fault_P_MW": round(p_pre, 2),
        "post_fault_P_MW": round(p_post, 2),
        "post_fault_V_pu": round(v_post, 4),
        "max_V_pu": round(v_overshoot, 4),
        "post_V_std": round(v_settle_band, 4),
        "P_recovery_ratio": round(recovery_ratio, 4),
    }


# Fail 시 기본 파라미터 추천 표 (사진 예시 기준)
DEFAULT_PARAM_TABLE = [
    {"name": "Tp",  "model": "REGCAU1", "desc": "Voltage filter time constant (voltage measurement filter)",
     "current": "0.5", "recommended": "0.02"},
    {"name": "Tiq", "model": "REGCAU1", "desc": "Current filter time constant (or related delay)",
     "current": "10.0", "recommended": "10.0"},
    {"name": "Kvp", "model": "REECAU1", "desc": "Voltage proportional gain (local V control loop)",
     "current": "0.1", "recommended": "0.9"},
    {"name": "Kvi", "model": "REECAU1", "desc": "Voltage integral gain (local V control loop)",
     "current": "0.1", "recommended": "0.4"},
]

DEFAULT_RATIONALE = [
    "Current values cause excessively slow / mismatched filtering and low gains, "
    "leading to sluggish response or insufficient damping in voltage/reactive power control during transients.",
    "High Tp (0.5 s) introduces excessive delay in voltage measurement, slowing down the control loop response.",
    "Raising Kvp/Kvi accelerates the local voltage control loop so active power recovers to >= 95% "
    "of pre-fault output and post-fault voltage settles near 1.0 pu (WECC Generic Model Validation & IEEE 2800.2).",
]


def _rule_based_judgment(metrics):
    recov = metrics["P_recovery_ratio"]
    overshoot = metrics["max_V_pu"]
    vstd = metrics["post_V_std"]
    fail_reasons = []
    if recov < 0.95:
        fail_reasons.append(f"유효전력 회복률 {recov:.0%} < 95% (출력 미복귀)")
    if overshoot > 1.22:  # 측정 노이즈 여유 0.02pu 포함 (정착 기준 1.20pu)
        fail_reasons.append(f"전압 오버슈트 {overshoot:.3f}pu > 1.20pu")
    if vstd > 0.02:
        fail_reasons.append(f"고장 후 전압 정착 불안정 (std {vstd:.3f}pu, 감쇠 부족)")

    if fail_reasons:
        return {
            "verdict": "Fail",
            "reason": "; ".join(fail_reasons),
            "param_table": DEFAULT_PARAM_TABLE,
            "rationale": DEFAULT_RATIONALE,
        }
    return {"verdict": "Pass", "reason": "전압 회복 및 출력 복귀 정상",
            "param_table": [], "rationale": []}


def judge_with_openai(case, kind, metrics):
    client = get_client()
    fallback = _rule_based_judgment(metrics)

    if client is None:
        fallback["source"] = "rule-based (API 키 없음)"
        return fallback

    prompt = f"""You are a grid-code compliance expert. Judge the following IBR (inverter-based resource)
ride-through test result against IEEE 2800.2 conformity criteria.

Test type: {TEST_LABEL.get(kind, kind)} ({kind})
Test case: {case['id']} - {case['desc']}
Computed metrics: {json.dumps(metrics)}

IEEE 2800.2 / WECC Generic Model expectations:
- Plant must remain connected through the disturbance (no trip).
- Active power must recover to >= 95% of pre-fault value after fault clearance.
- Post-fault voltage should settle near 1.0 pu with adequate damping; transient overshoot <= ~1.2 pu.
- For LVRT, reactive current injection should support voltage during the dip.

If the case FAILS, provide concrete REGCAU1 / REECAU1 parameter recommendations
(e.g. Tp, Tiq, Kvp, Kvi) with current vs recommended values, plus rationale.

Return STRICT JSON only, no markdown:
{{"verdict": "Pass" or "Fail",
  "reason": "one concise sentence (korean ok)",
  "param_table": [{{"name":"Tp","model":"REGCAU1","desc":"...","current":"0.5","recommended":"0.02"}}],
  "rationale": ["reason 1", "reason 2"]}}
For Pass, param_table and rationale must be empty arrays."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if data.get("verdict") == "Fail" and not data.get("param_table"):
            data["param_table"] = DEFAULT_PARAM_TABLE
            data["rationale"] = data.get("rationale") or DEFAULT_RATIONALE
        data.setdefault("param_table", [])
        data.setdefault("rationale", [])
        data["source"] = "OpenAI (gpt-4o-mini)"
        return data
    except Exception:
        fallback["source"] = "rule-based (OpenAI 호출 실패)"
        return fallback


# ----------------------------------------------------------------------------
# 3) PDF 보고서 생성
# ----------------------------------------------------------------------------
NAVY = colors.HexColor("#1f3a5f")
ACCENT = colors.HexColor("#2563eb")
LIGHT = colors.HexColor("#eef2f7")
RED = colors.HexColor("#c0392b")


def _plot_case(df, case_id):
    fig, ax1 = plt.subplots(figsize=(5.6, 2.6))
    ax1.plot(df["Time (s)"], df["Voltage (pu)"], color="#2563eb", lw=1.4, label="V (pu)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (pu)", color="#2563eb")
    ax1.axhline(1.0, color="grey", ls="--", lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df["Time (s)"], df["Active Power (MW)"], color="#c0392b", alpha=0.8, lw=1.2, label="P (MW)")
    ax2.set_ylabel("P (MW)", color="#c0392b")
    ax1.set_title(case_id, fontsize=10, weight="bold")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _mitigation_flowables(j, body, h3):
    """사진 예시 형식: 실패 이유 + 파라미터 추천."""
    elems = [Paragraph("Fail Reasons and Parameter Recommendations", h3)]

    elems.append(Paragraph("<b>1. Parameters to Modify (REGCAU1 &amp; REECAU1 Model)</b>", body))
    items = [ListItem(Paragraph(f"<b>{p['name']}</b> : {p['desc']}", body))
             for p in j["param_table"]]
    elems.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=14))
    elems.append(Spacer(1, 6))

    elems.append(Paragraph("<b>2. Current vs Recommended Values</b>", body))
    rows = [["Parameter", "Model", "Current", "Recommended"]]
    for p in j["param_table"]:
        rows.append([p["name"], p["model"], p["current"], p["recommended"]])
    tbl = Table(rows, colWidths=[1.3*inch, 1.3*inch, 1.3*inch, 1.5*inch])
    tstyle = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    for i, p in enumerate(j["param_table"], start=1):
        if str(p["current"]) != str(p["recommended"]):
            tstyle.append(("TEXTCOLOR", (3, i), (3, i), RED))
            tstyle.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(tstyle))
    elems.append(tbl)
    elems.append(Spacer(1, 6))

    elems.append(Paragraph(
        "<b>3. Why These Changes Are Required "
        "(WECC Generic Model Validation &amp; Stability Improvement)</b>", body))
    r_items = [ListItem(Paragraph(r, body)) for r in j["rationale"]]
    elems.append(ListFlowable(r_items, bulletType="bullet", start="circle", leftIndent=14))
    return elems


def build_pdf(results, out_path):
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], alignment=TA_CENTER,
                           textColor=NAVY, fontSize=18)
    subtitle = ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER,
                              textColor=colors.grey, fontSize=9)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], textColor=ACCENT)
    body = styles["BodyText"]

    elems = [
        Paragraph("IBR Model Quality Test Report", title),
        Paragraph(f"180 MW PV Plant  |  IEEE 2800.2 Conformity  |  {datetime.now():%Y-%m-%d %H:%M}", subtitle),
        Spacer(1, 18),
    ]

    n_fail = sum(1 for r in results if r["judgment"]["verdict"] == "Fail")
    elems.append(Paragraph(
        f"Summary &nbsp;-&nbsp; Total {len(results)} cases, "
        f"<font color='#1f8a3b'>{len(results)-n_fail} Pass</font> / "
        f"<font color='#c0392b'>{n_fail} Fail</font>", h2))
    rows = [["Test", "Case", "Verdict", "P recovery", "Max V (pu)"]]
    for r in results:
        rows.append([r["kind"], r["case"]["id"], r["judgment"]["verdict"],
                     f"{r['metrics']['P_recovery_ratio']:.0%}",
                     f"{r['metrics']['max_V_pu']:.3f}"])
    tbl = Table(rows, colWidths=[0.9*inch, 1.3*inch, 1.0*inch, 1.2*inch, 1.2*inch])
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
             ("ALIGN", (0, 0), (-1, -1), "CENTER"),
             ("FONTSIZE", (0, 0), (-1, -1), 9),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    for i, r in enumerate(results, start=1):
        if r["judgment"]["verdict"] == "Fail":
            style.append(("TEXTCOLOR", (2, i), (2, i), RED))
            style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    elems += [tbl, PageBreak()]

    for r in results:
        j = r["judgment"]
        verdict_color = "#1f8a3b" if j["verdict"] == "Pass" else "#c0392b"
        elems.append(Paragraph(f"{TEST_LABEL.get(r['kind'], r['kind'])} - {r['case']['id']}", h2))
        elems.append(Paragraph(r["case"]["desc"], body))
        elems.append(Paragraph(
            f"<b>Verdict:</b> <font color='{verdict_color}'><b>{j['verdict']}</b></font> "
            f"&nbsp;(<i>{j.get('source','')}</i>)<br/><b>Reason:</b> {j['reason']}", body))
        elems.append(Spacer(1, 8))
        elems.append(RLImage(_plot_case(r["df"], r["case"]["id"]), width=5.1*inch, height=2.4*inch))
        elems.append(Spacer(1, 8))
        if j["verdict"] == "Fail" and j.get("param_table"):
            elems += _mitigation_flowables(j, body, h3)
        elems.append(PageBreak())

    doc.build(elems)
    return out_path


# ----------------------------------------------------------------------------
# Streamlit UI (전문 테마)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="IBR MQT AI Agent", page_icon="⚡", layout="wide")

WORK_DIR = Path("./mqt_workspace")
CSV_DIR = WORK_DIR / "csv"
PDF_PATH = WORK_DIR / "MQT_Report.pdf"

CUSTOM_CSS = """
<style>
:root { --navy:#1f3a5f; --accent:#2563eb; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #eef4ff 0%, #f7f9fc 40%, #f4f6fa 100%);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #16243b 0%, #1f3a5f 100%);
}
section[data-testid="stSidebar"] * { color:#e6edf7 !important; }
.hero {
  background: linear-gradient(120deg, #1f3a5f 0%, #2563eb 100%);
  color:#fff; padding:26px 30px; border-radius:16px; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(31,58,95,.25);
}
.hero h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:-.5px; }
.hero p { margin:6px 0 0; opacity:.9; font-size:14px; }
.badge {
  display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
  padding:3px 10px; border-radius:999px; font-size:12px; margin-right:6px; margin-top:10px;
}
.metric-card {
  background:#fff; border:1px solid #e6ebf3; border-radius:14px; padding:16px 18px;
  box-shadow:0 4px 14px rgba(15,23,42,.05); text-align:center;
}
.metric-card .v { font-size:30px; font-weight:800; color:var(--navy); }
.metric-card .l { font-size:12px; color:#64748b; margin-top:2px; }
.pass { color:#1f8a3b !important; }
.fail { color:#c0392b !important; }
.stButton>button {
  background:linear-gradient(120deg,#2563eb,#1f3a5f); color:#fff; border:0;
  border-radius:10px; padding:10px 18px; font-weight:700;
  box-shadow:0 6px 16px rgba(37,99,235,.30);
}
.stButton>button:hover { filter:brightness(1.07); }
div[data-testid="stExpander"] {
  border:1px solid #e6ebf3; border-radius:12px; background:#fff;
}
.section-title { font-size:18px; font-weight:800; color:var(--navy); margin:6px 0 2px; }
.hr { height:1px; background:linear-gradient(90deg,#2563eb33,transparent); margin:10px 0 18px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(col, value, label, cls=""):
    col.markdown(
        f"<div class='metric-card'><div class='v {cls}'>{value}</div>"
        f"<div class='l'>{label}</div></div>", unsafe_allow_html=True)


def sidebar():
    st.sidebar.markdown("### ⚡ MQT AI Agent")
    st.sidebar.markdown("---")
    if get_api_key():
        st.sidebar.success("서비스 준비 완료")
        st.sidebar.caption("개발자 키로 동작 (사용자 입력 불필요)")
    else:
        st.sidebar.error("키 미설정 → 룰베이스 판정")
        st.sidebar.caption("Secrets에 OPENAI_API_KEY 등록 필요")
    st.sidebar.markdown("---")
    return st.sidebar.radio("MENU", ["대시보드 · 판정", "Power System Chatbot"])


def page_run():
    st.markdown(
        "<div class='hero'><h1>IBR Model Quality Test · AI Agent</h1>"
        "<p>180 MW PV Plant 인버터 모델 품질테스트 자동 판정 (IEEE 2800.2)</p>"
        "<span class='badge'>LVRT</span><span class='badge'>HVRT</span>"
        "<span class='badge'>Voltage Step Change</span>"
        "<span class='badge'>AI Pass/Fail</span><span class='badge'>PDF Report</span></div>",
        unsafe_allow_html=True)

    c1, _, _ = st.columns([1, 1, 1])
    with c1:
        run = st.button("▶  시뮬레이션 & AI 판정 실행", use_container_width=True)

    if run:
        with st.spinner("CSV 생성 및 IEEE 2800.2 판정 중..."):
            generated = generate_all_cases(CSV_DIR)
            results = []
            progress = st.progress(0.0)
            total = sum(len(v) for v in generated.values())
            done = 0
            for kind, items in generated.items():
                for case, df, _ in items:
                    metrics = quick_metrics(df)
                    judgment = judge_with_openai(case, kind, metrics)
                    results.append({"kind": kind, "case": case, "df": df,
                                    "metrics": metrics, "judgment": judgment})
                    done += 1
                    progress.progress(done / total)
            progress.empty()
            st.session_state["results"] = results
        st.toast("판정 완료!", icon="✅")

    results = st.session_state.get("results")
    if not results:
        st.info("‘시뮬레이션 & AI 판정 실행’을 눌러 시작하세요.")
        return

    n_total = len(results)
    n_fail = sum(1 for r in results if r["judgment"]["verdict"] == "Fail")
    n_pass = n_total - n_fail

    st.markdown("<div class='section-title'>Overview</div><div class='hr'></div>",
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, n_total, "Total Cases")
    metric_card(m2, n_pass, "Pass", "pass")
    metric_card(m3, n_fail, "Fail", "fail")
    metric_card(m4, f"{n_pass/n_total*100:.0f}%", "Pass Rate")

    st.markdown("<div class='section-title' style='margin-top:18px'>Results</div>"
                "<div class='hr'></div>", unsafe_allow_html=True)
    summary = pd.DataFrame([{
        "Test": r["kind"], "Case": r["case"]["id"],
        "Verdict": r["judgment"]["verdict"],
        "P recovery": f"{r['metrics']['P_recovery_ratio']:.0%}",
        "Max V (pu)": r["metrics"]["max_V_pu"],
        "Reason": r["judgment"]["reason"],
    } for r in results])

    def color_verdict(val):
        return "color:#c0392b;font-weight:700" if val == "Fail" else "color:#1f8a3b;font-weight:700"
    st.dataframe(summary.style.map(color_verdict, subset=["Verdict"]),
                 use_container_width=True, hide_index=True)

    for r in results:
        v = r["judgment"]["verdict"]
        icon = "🟢" if v == "Pass" else "🔴"
        with st.expander(f"{icon}  {TEST_LABEL.get(r['kind'], r['kind'])} · {r['case']['id']} — {v}"):
            st.caption(r["case"]["desc"])
            st.line_chart(r["df"].set_index("Time (s)")[["Voltage (pu)", "Active Power (MW)"]])
            if v == "Fail":
                st.markdown("**Fail Reasons and Parameter Recommendations**")
                st.error(r["judgment"]["reason"])
                if r["judgment"].get("param_table"):
                    pt = pd.DataFrame(r["judgment"]["param_table"]).rename(
                        columns={"name": "Parameter", "model": "Model", "desc": "Description",
                                 "current": "Current", "recommended": "Recommended"})
                    st.table(pt[["Parameter", "Model", "Current", "Recommended", "Description"]])
                    for rt in r["judgment"].get("rationale", []):
                        st.markdown(f"- {rt}")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    if st.button("📄  PDF 보고서 생성"):
        with st.spinner("PDF 생성 중..."):
            build_pdf(results, PDF_PATH)
        with open(PDF_PATH, "rb") as f:
            st.download_button("⬇  보고서 다운로드", f.read(),
                               file_name="IBR_MQT_Report.pdf", mime="application/pdf")


def page_chatbot():
    st.markdown(
        "<div class='hero'><h1>Power System Chatbot</h1>"
        "<p>IBR 계통연계 · IEEE 2800/2800.2 · LVRT/HVRT · PSS/E 모델 관련 질의응답</p></div>",
        unsafe_allow_html=True)
    client = get_client()
    if client is None:
        st.info("서비스 키가 설정되지 않아 챗봇을 사용할 수 없습니다. (배포자 문의)")
        return

    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("질문을 입력하세요 (예: IEEE 2800 LVRT 기준?)"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        system = ("You are a helpful power-system engineer assistant specializing in "
                  "IBR grid interconnection, IEEE 2800 / 2800.2, LVRT/HVRT, voltage step "
                  "change tests, and PSS/E inverter models (REGCAU1/REECAU1/REPCAU1). "
                  "Answer concisely in Korean.")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, *st.session_state.chat])
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"오류: {e}"
        st.session_state.chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


choice = sidebar()
if choice == "대시보드 · 판정":
    page_run()
else:
    page_chatbot()
