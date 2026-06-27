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
# 180 MW PV Plant 모델 정보
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
# 챗봇 도메인 지식 (정확도 향상용 시스템 프롬프트 + few-shot)
# ----------------------------------------------------------------------------
CHATBOT_SYSTEM = """You are an expert power-system engineer assistant specializing in
inverter-based resource (IBR) grid interconnection. Your domains: IEEE Std 2800-2022 and
IEEE 2800.2 conformity testing, LVRT/HVRT ride-through, voltage/frequency step change tests,
flat start tests, and WECC generic PSS/E models (REGCAU1 / REECAU1 / REPCAU1).

ACCURACY RULES (follow strictly):
- Answer in Korean, concise and technically precise. Add a few relevant emojis to stay friendly.
- Ground every claim in the reference facts below. Do NOT invent numeric thresholds.
- If a user asks for an exact value that depends on the specific .dyr definition or a
  specific table/region of IEEE 2800-2022, state the typical/representative value AND tell the
  user to confirm against the official IEEE 2800-2022 table or the model's .dyr file.
- Distinguish IEEE 2800 (requirements) from IEEE 2800.2 (conformity test/validation procedure).
- If you are unsure, say so plainly rather than guessing.
- Keep parameter-name caveats: the same physical quantity may appear as Tp or Tfltr etc.
  depending on the model variant.

REFERENCE FACTS (authoritative for this assistant):
[IEEE 2800 LVRT]
- Continuous operation band is typically 0.9-1.1 pu (no curtailment).
- Low-voltage ride-through is a stepwise voltage-vs-time envelope: the deeper the dip,
  the shorter the mandatory ride-through time.
- The minimum ride-through duration near the near-zero-voltage region is representatively 0.32 s
  (reflecting transmission primary fault-clearing time).
- During the dip the IBR must inject dynamic reactive current proportional to voltage deviation
  to support voltage recovery; after clearance, active power must recover toward pre-disturbance
  level with transient overshoot kept within about 1.2 pu. Momentary cessation is restricted.

[IEEE 2800 HVRT]
- High-voltage ride-through applies above ~1.1 pu swell; the higher the voltage, the shorter the
  ride-through time. During overvoltage the IBR absorbs reactive power (negative Q).

[REGCAU1 Tp]
- Tp is the terminal-voltage measurement filter time constant (a 1st-order low-pass filter).
- Large Tp delays voltage sensing -> slower/under-damped LVRT recovery. Too small -> noise sensitivity.
- Typical recommended value is around 0.02 s. A value like 0.5 s is a classic cause of sluggish LVRT.
- Naming may differ (Tp vs Tfltr) by model variant; confirm in the .dyr definition.

[Voltage Step Change Test]
- Purpose: verify plant voltage/reactive-power control dynamics (mainly REPC Q-control and REEC).
- Procedure: stabilize at an operating point (e.g., P fixed, V=1.0 pu) -> apply a step UP in
  reference/source voltage (e.g., +2% or +5%) -> hold to reach new steady state -> return to nominal
  -> apply a step DOWN (-2% or -5%) -> return. Record V, P, Q, current throughout.
- Evaluate rise time, settling time, overshoot, steady-state error, and the reactive-power response
  (droop/voltage-control gain, time constant, damping) vs. the reference/measured response.

[REGC / REEC / REPC roles]
- REGC_A (REGCAU1): converter/generator interface (current injection, LVPL, high-voltage reactive
  current management).
- REEC (REECAU1): electrical control (active/reactive current commands, current limit logic,
  Q/V and P control modes).
- REPC (REPCAU1): plant-level control (POI voltage/reactive power, frequency droop, reference commands).

[IEEE 2800 vs 2800.2]
- IEEE 2800: interconnection REQUIREMENTS for IBR.
- IEEE 2800.2: conformity TEST / validation procedures for those requirements (basis of MQT pass/fail).

[Reactive current injection gain]
- During LVRT, reactive current is injected proportional to voltage deviation; the slope K (pu/pu)
  is set by the operator/standard (often >= 2 pu/pu is cited) and implemented via REEC (e.g., Kqv).

[Typical LVRT-fail mitigation]
- Reduce voltage measurement filter Tp (e.g., 0.5 -> 0.02), raise local voltage-loop gains
  Kvp/Kvi (e.g., 0.1 -> 0.9 / 0.1 -> 0.4), and widen Imax/Qmax to speed up recovery and improve damping.

[Flat Start Test]
- Checks that with no disturbance, V/P/Q stay flat (proper initialization/equilibrium);
  small standard deviation = pass.
"""

CHATBOT_FEWSHOT = [
    {"role": "user", "content": "IEEE 2800 LVRT 기준은?"},
    {"role": "assistant", "content":
     "⚡ IEEE 2800-2022는 IBR이 측정점(POM)에서 전압-시간 포락선을 따라 외란을 견디고 "
     "연결을 유지(ride-through)하도록 규정해요.\n\n"
     "• 연속 운전 영역은 보통 0.9~1.1 pu (출력 저감 없이 운전)\n"
     "• 저전압일수록 견뎌야 하는 시간이 짧아지는 계단형 곡선이며, 영점 부근 최소 ride-through "
     "시간은 대표적으로 0.32초로 인용돼요 (송전 1차 보호 고장제거 시간 반영)\n"
     "• 저전압 구간엔 전압편차에 비례하는 동적 무효전류 주입으로 전압 회복 지원, 고장 제거 후엔 "
     "유효전력을 외란 이전 수준으로 회복하되 과도 오버슈트는 약 1.2 pu 이내로 제한 🌊\n\n"
     "정확한 수치 곡선은 적용 표(IEEE 2800-2022 ride-through 절)를 직접 대조하시길 권해요. 📘"},

    {"role": "user", "content": "REGCAU1 Tp는 무슨 역할?"},
    {"role": "assistant", "content":
     "🌊 REGCAU1의 Tp는 단자전압 측정 필터의 시정수예요 (1차 저역통과 필터). 컨버터가 전압을 "
     "인지할 때 노이즈·급변을 걸러줍니다.\n\n"
     "• Tp가 크면 전압 감지가 늦어져 LVRT 회복이 느려지고 감쇠가 부족해질 수 있어요\n"
     "• 너무 작으면 측정 노이즈에 민감\n"
     "• 일반 권장값은 약 0.02초이며, 0.5초처럼 크면 응답이 둔해지는 전형적 원인이에요 ⚙️\n\n"
     "모델 변형에 따라 Tp 또는 Tfltr로 표기가 다를 수 있으니 .dyr 정의서에서 확인하세요. 🔍"},

    {"role": "user", "content": "Voltage Step Change Test 절차는?"},
    {"role": "assistant", "content":
     "🔋 전압 스텝 변화 시험은 플랜트의 전압/무효전력 제어(REPC Q-control, REEC) 동특성을 검증해요.\n\n"
     "1️⃣ 정상 상태로 안정화 (예: P 일정, V=1.0 pu)\n"
     "2️⃣ 기준전압을 계단형으로 상승 (예: +2%, +5%) → 새 정상 상태까지 유지\n"
     "3️⃣ 공칭으로 복귀 → 같은 방식으로 하강 스텝 (−2%, −5%) → 복귀\n"
     "4️⃣ 전 구간 V·P·Q·전류 기록 📈\n\n"
     "평가 항목: 상승시간, 정정시간, 오버슈트, 정상상태 오차, 전압변동에 대한 무효전력 응답"
     "(droop/게인·시정수·감쇠). '공칭→위 스텝→공칭→아래 스텝→공칭' 파형이 전형적 입력이에요. 😊"},
]

# ----------------------------------------------------------------------------
# 테스트 케이스
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
         "desc": "±5% voltage step change (up -> nominal -> down -> nominal)"},
        {"id": "VSTEP_02",
         "steps": [(2.0, 3.5, 1.10), (5.5, 7.0, 0.90)],
         "desc": "±10% voltage step change (up -> nominal -> down -> nominal)"},
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
            tau = 0.12 if "HVRT" in case["id"] else 0.15
            target = 1.0
            v[post_mask] = target + (v_at_clear - target) * np.exp(-(t[post_mask]-(FAULT_START+dur))/tau)
        else:
            # LVRT_02 Fail 유도: 느린 회복 + 진동 + 낮은 출력 회복
            tau, target = 0.95, 1.0
            decay = np.exp(-(t[post_mask]-(FAULT_START+dur))/tau)
            osc = 0.15 * np.sin(2*np.pi*1.1*(t[post_mask]-(FAULT_START+dur)))
            v[post_mask] = target - (target - v_at_clear)*decay*0.7 + osc*decay*0.6

        p = PLANT_MW * np.clip(v, 0, 1.2)
        if not case.get("recover", True):
            p[post_mask] *= 0.52   # 출력 회복률 낮게 유지
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
# Pass/Fail 판정
# ----------------------------------------------------------------------------
def quick_metrics(df):
    pre = df[df["Time (s)"] < FAULT_START]
    settle = df[df["Time (s)"] > (T_END - SETTLE_WIN)]
    p_pre = pre["Active Power (MW)"].mean()
    p_post = settle["Active Power (MW)"].mean()
    v_overshoot = df["Voltage (pu)"].max()
    v_settle_band = settle["Voltage (pu)"].std()
    recovery_ratio = (p_post / p_pre) if p_pre > 0 else 0.0
    return {
        "pre_fault_P_MW": round(p_pre, 2),
        "post_fault_P_MW": round(p_post, 2),
        "max_V_pu": round(v_overshoot, 4),
        "post_V_std": round(v_settle_band, 4),
        "P_recovery_ratio": round(recovery_ratio, 4),
    }

DEFAULT_PARAM_TABLE = [
    {"name": "Tp", "model": "REGCAU1", "desc": "Voltage filter time constant (voltage measurement filter)",
     "current": "0.5", "recommended": "0.02"},
    {"name": "Kvp", "model": "REECAU1", "desc": "Voltage proportional gain (local V control loop)",
     "current": "0.1", "recommended": "0.9"},
    {"name": "Kvi", "model": "REECAU1", "desc": "Voltage integral gain (local V control loop)",
     "current": "0.1", "recommended": "0.4"},
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
        fail_reasons.append(f"Active power recovery ratio {recov:.0%} < 95% (insufficient recovery)")
    if overshoot > 1.20:
        fail_reasons.append(f"Voltage overshoot {overshoot:.3f}pu > 1.20pu")
    if vstd > 0.025:
        fail_reasons.append(f"Post-fault voltage instability (std {vstd:.3f}pu, insufficient damping)")
    if fail_reasons:
        return {"verdict": "Fail", "reason": "; ".join(fail_reasons),
                "param_table": DEFAULT_PARAM_TABLE, "rationale": DEFAULT_RATIONALE}
    return {"verdict": "Pass", "reason": "All criteria satisfied",
            "param_table": [], "rationale": []}

def judge_with_openai(case, kind, metrics):
    client = get_client()
    fallback = _rule_based_judgment(metrics)
    if client is None:
        fallback["source"] = "rule-based (API 키 없음)"
        return fallback

    prompt = f"""You are a grid-code compliance expert. Judge the following IBR
ride-through / step-change test result against IEEE 2800.2 conformity criteria.

Test type: {TEST_LABEL.get(kind, kind)} ({kind})
Test case: {case['id']} - {case['desc']}
Computed metrics: {json.dumps(metrics)}

IEEE 2800.2 / WECC Generic Model expectations:
- Plant remains connected (no trip).
- Active power recovers to >= 95% of pre-fault value after clearance.
- Post-fault voltage settles near 1.0 pu with adequate damping; overshoot <= ~1.2 pu.

If FAIL, give concrete REGCAU1/REECAU1 parameter recommendations (Tp, Kvp, Kvi)
with current vs recommended values, plus rationale.

Return STRICT JSON only:
{{"verdict":"Pass"|"Fail","reason":"...(korean ok)",
  "param_table":[{{"name":"Tp","model":"REGCAU1","desc":"...","current":"0.5","recommended":"0.02"}}],
  "rationale":["...","..."]}}
For Pass, param_table and rationale are empty arrays."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}], temperature=0.1)
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if data.get("verdict") == "Fail" and not data.get("param_table"):
            data["param_table"] = DEFAULT_PARAM_TABLE
            data["rationale"] = DEFAULT_RATIONALE
        data.setdefault("param_table", [])
        data.setdefault("rationale", [])
        data["source"] = "OpenAI (gpt-4o-mini)"
        return data
    except Exception:
        fallback["source"] = "rule-based (OpenAI 호출 실패)"
        return fallback

# ----------------------------------------------------------------------------
# PDF 보고서
# ----------------------------------------------------------------------------
ORANGE = colors.HexColor("#E8721C")
ORANGE_DEEP = colors.HexColor("#C9551A")
CREAM = colors.HexColor("#FFF1E2")
RED = colors.HexColor("#c0392b")

def _plot_case(df, case_id):
    fig, ax1 = plt.subplots(figsize=(5.6, 2.6))
    ax1.plot(df["Time (s)"], df["Voltage (pu)"], color="#E8721C", lw=1.5, label="V (pu)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Voltage (pu)", color="#E8721C")
    ax1.axhline(1.0, color="grey", ls="--", lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df["Time (s)"], df["Active Power (MW)"], color="#8a5a2b", alpha=0.85, lw=1.2, label="P (MW)")
    ax2.set_ylabel("P (MW)", color="#8a5a2b")
    ax1.set_title(case_id, fontsize=10, weight="bold"); ax1.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120); plt.close(fig); buf.seek(0)
    return buf

def _mitigation_flowables(j, body, h3):
    elems = [Paragraph("Fail Reasons and Parameter Recommendations", h3)]
    elems.append(Paragraph("<b>1. Parameters to Modify (REGCAU1 &amp; REECAU1 Model)</b>", body))
    elems.append(ListFlowable(
        [ListItem(Paragraph(f"<b>{p['name']}</b> : {p['desc']}", body)) for p in j["param_table"]],
        bulletType="bullet", start="circle", leftIndent=14))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("<b>2. Current vs Recommended Values</b>", body))
    rows = [["Parameter", "Model", "Current", "Recommended"]]
    for p in j["param_table"]:
        rows.append([p["name"], p["model"], p["current"], p["recommended"]])
    tbl = Table(rows, colWidths=[1.3*inch, 1.3*inch, 1.3*inch, 1.5*inch])
    tstyle = [("BACKGROUND", (0, 0), (-1, 0), ORANGE),
              ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
              ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
              ("ALIGN", (0, 0), (-1, -1), "CENTER"),
              ("FONTSIZE", (0, 0), (-1, -1), 9),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM])]
    for i, p in enumerate(j["param_table"], start=1):
        if str(p["current"]) != str(p["recommended"]):
            tstyle.append(("TEXTCOLOR", (3, i), (3, i), RED))
            tstyle.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(tstyle)); elems.append(tbl); elems.append(Spacer(1, 6))
    elems.append(Paragraph("<b>3. Why These Changes Are Required</b>", body))
    elems.append(ListFlowable([ListItem(Paragraph(r, body)) for r in j["rationale"]],
                              bulletType="bullet", start="circle", leftIndent=14))
    return elems

def build_pdf(results, out_path):
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], alignment=TA_CENTER, textColor=ORANGE_DEEP, fontSize=18)
    subtitle = ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.grey, fontSize=9)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=ORANGE_DEEP)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], textColor=ORANGE)
    body = styles["BodyText"]
    elems = [Paragraph("IBR Model Quality Test Report", title),
             Paragraph(f"180 MW PV Plant | IEEE 2800.2 Conformity | {datetime.now():%Y-%m-%d %H:%M}", subtitle),
             Spacer(1, 18)]
    n_fail = sum(1 for r in results if r["judgment"]["verdict"] == "Fail")
    elems.append(Paragraph(
        f"Summary &nbsp;-&nbsp; Total {len(results)} cases, "
        f"<font color='#1f8a3b'>{len(results)-n_fail} Pass</font> / "
        f"<font color='#c0392b'>{n_fail} Fail</font>", h2))
    rows = [["Test", "Case", "Verdict", "P recovery", "Max V (pu)"]]
    for r in results:
        rows.append([r["kind"], r["case"]["id"], r["judgment"]["verdict"],
                     f"{r['metrics']['P_recovery_ratio']:.0%}", f"{r['metrics']['max_V_pu']:.3f}"])
    tbl = Table(rows, colWidths=[0.9*inch, 1.3*inch, 1.0*inch, 1.2*inch, 1.2*inch])
    style = [("BACKGROUND", (0, 0), (-1, 0), ORANGE),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
             ("ALIGN", (0, 0), (-1, -1), "CENTER"),
             ("FONTSIZE", (0, 0), (-1, -1), 9),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM])]
    for i, r in enumerate(results, start=1):
        if r["judgment"]["verdict"] == "Fail":
            style.append(("TEXTCOLOR", (2, i), (2, i), RED))
            style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style)); elems += [tbl, PageBreak()]
    for r in results:
        j = r["judgment"]
        vc = "#1f8a3b" if j["verdict"] == "Pass" else "#c0392b"
        elems.append(Paragraph(f"{TEST_LABEL.get(r['kind'], r['kind'])} - {r['case']['id']}", h2))
        elems.append(Paragraph(r["case"]["desc"], body))
        elems.append(Paragraph(
            f"<b>Verdict:</b> <font color='{vc}'><b>{j['verdict']}</b></font> "
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
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="IBR MQT AI Agent", page_icon="🟠", layout="wide")
WORK_DIR = Path("./mqt_workspace")
CSV_DIR = WORK_DIR / "csv"
PDF_PATH = WORK_DIR / "MQT_Report.pdf"

CUSTOM_CSS = """
<style>
:root { --orange:#E8721C; --orange-deep:#C9551A; --cream:#FFF1E2; }
.stApp {
  background: radial-gradient(1100px 560px at 12% -8%, #fff2e3 0%, #fff8f1 45%, #fdf6ef 100%);
}
section[data-testid="stSidebar"] {
  background:#ffffff; border-right:1px solid #f0e2d3;
}
section[data-testid="stSidebar"] * { color:#5b3a1c !important; }
section[data-testid="stSidebar"] h3 { color:#c9551a !important; }
.hero {
  background: linear-gradient(120deg, #f59b4d 0%, #e8721c 60%, #d9591a 100%);
  color:#fff; padding:26px 30px; border-radius:16px; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(232,114,28,.28);
}
.hero h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:-.5px; }
.hero p { margin:6px 0 0; opacity:.95; font-size:14px; }
.badge {
  display:inline-block; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.35);
  padding:3px 10px; border-radius:999px; font-size:12px; margin-right:6px; margin-top:10px;
}
.card {
  background:#fff; border:1px solid #f3ddc7; border-radius:14px; padding:16px 18px;
  box-shadow:0 4px 14px rgba(201,85,26,.06);
}
.metric-card {
  background:#fff; border:1px solid #f3ddc7; border-radius:14px; padding:16px 18px;
  box-shadow:0 4px 14px rgba(201,85,26,.06); text-align:center;
}
.metric-card .v { font-size:30px; font-weight:800; color:var(--orange-deep); }
.metric-card .l { font-size:12px; color:#9a6a3a; margin-top:2px; }
.pass { color:#1f8a3b !important; }
.fail { color:#c0392b !important; }
.kv { font-size:13px; color:#5b3a1c; margin:2px 0; }
.kv b { color:#9a4a12; }
.stButton>button {
  background:linear-gradient(120deg,#f59b4d,#e8721c); color:#fff; border:0;
  border-radius:10px; padding:10px 18px; font-weight:700;
  box-shadow:0 6px 16px rgba(232,114,28,.30);
}
.stButton>button:hover { filter:brightness(1.06); }
div[data-testid="stExpander"] { border:1px solid #f3ddc7; border-radius:12px; background:#fff; }
.section-title { font-size:18px; font-weight:800; color:var(--orange-deep); margin:6px 0 2px; }
.hr { height:1px; background:linear-gradient(90deg,#e8721c44,transparent); margin:10px 0 18px; }
.diagram { background:#fff; border:1px solid #f3ddc7; border-radius:14px; padding:14px 16px; }
.diagram .cap { font-weight:800; color:var(--orange-deep); font-size:13px; margin-bottom:8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def metric_card(col, value, label, cls=""):
    col.markdown(f"<div class='metric-card'><div class='v {cls}'>{value}</div>"
                 f"<div class='l'>{label}</div></div>", unsafe_allow_html=True)

def sidebar():
    # MENU 글씨 크기 확대 (20px)
    st.markdown("""
    <style>
    div[role="radiogroup"] label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🟠 MQT AI Agent")

    if get_api_key():
        st.sidebar.success("AI Agent-Based Automated System for Dynamic Model Quality Test of Inverter-Based Resources")
        st.sidebar.caption("개발자 Open AI API키로 동작 (사용자 입력 불필요)")
    else:
        st.sidebar.warning("키 미설정 → 룰베이스 판정")
        st.sidebar.caption("Secrets에 OPENAI_API_KEY 등록 필요")

    st.sidebar.markdown("---")
    menu_choice = st.sidebar.radio("MENU", ["Model Quality Test", "Power System Chatbot"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("<span style='font-size: 13px;'><b>📌 사용 방법</b></span>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <span style='font-size: 13px;'>
    • Model Quality Test: 시뮬레이션&AI 판정 실행 클릭 → PDF 보고서 생성 클릭 → 보고서 다운로드<br>
    • Power System Chatbot: 질문 입력
    </span>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.image("https://raw.githubusercontent.com/woongabs16/kys_sk/main/logo.png", width=60)
    st.sidebar.markdown("<span style='font-size: 12px;'><b>YEONSOO KIM</b></span>", unsafe_allow_html=True)

    return menu_choice

def render_plant_model():
    st.markdown("<div class='section-title'>적용 모델 · 180 MW PV Plant</div>"
                "<div class='hr'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.markdown("<div class='diagram'><div class='cap'>Grid interconnection of the plant model</div></div>",
                    unsafe_allow_html=True)
        st.image(GRID_IMG_URL, width=520)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='diagram'><div class='cap'>PV plant model (WECC Generic REGC / REEC / REPC)</div></div>",
                    unsafe_allow_html=True)
        st.image(MODEL_IMG_URL, width=380)
    with c2:
        st.markdown("<div class='card'><div class='cap' style='color:#c9551a;font-weight:800;margin-bottom:8px'>"
                    "Plant Model Specifications</div>"
                    + "".join(f"<div class='kv'><b>{k}</b> : {v}</div>" for k, v in PLANT_INFO.items())
                    + "</div>", unsafe_allow_html=True)

def page_run():
    st.markdown(
        "<div class='hero'><h1>IBR Model Quality Test AI Agent</h1>"
        "<p>인버터 기반 자원의 동적 모델 품질테스트 자동화를 위한 AI Agent</p>"
        "<span class='badge'>Generative AI</span><span class='badge'>Power System Chatbot</span>"
        "<span class='badge'>AI Pass/Fail</span>"
        "<span class='badge'>PDF Report</span><span class='badge'>IEEE Std 2800.2</span></div>",
        unsafe_allow_html=True)
    render_plant_model()
    st.markdown("<div class='section-title' style='margin-top:18px'>모델품질테스트(MQT) 실행</div>"
                "<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("**Performed Tests**  \n"
                "• Voltage Ride-Through Test (LVRT & HVRT)  \n"
                "• Voltage Step Change Test")
    c1, _, _ = st.columns([1, 1, 1])
    with c1:
        run = st.button("▶ 시뮬레이션 & AI 판정 실행", use_container_width=True)
    if run:
        with st.spinner("IEEE 2800.2 기반 모델품질테스트 판정 중..."):
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
    st.markdown("<div class='section-title'>Overview</div><div class='hr'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, n_total, "Total Cases")
    metric_card(m2, n_pass, "Pass", "pass")
    metric_card(m3, n_fail, "Fail", "fail")
    metric_card(m4, f"{n_pass/n_total*100:.0f}%", "Pass Rate")
    st.markdown("<div class='section-title' style='margin-top:18px'>Results</div>"
                "<div class='hr'></div>", unsafe_allow_html=True)
    summary = pd.DataFrame([{
        "Test": r["kind"], "Case": r["case"]["id"], "Verdict": r["judgment"]["verdict"],
        "P recovery": f"{r['metrics']['P_recovery_ratio']:.0%}",
        "Max V (pu)": r["metrics"]["max_V_pu"], "Reason": r["judgment"]["reason"],
    } for r in results])
    def color_verdict(val):
        return "color:#c0392b;font-weight:700" if val == "Fail" else "color:#1f8a3b;font-weight:700"
    st.dataframe(summary.style.map(color_verdict, subset=["Verdict"]),
                 use_container_width=True, hide_index=True)
    for r in results:
        v = r["judgment"]["verdict"]
        icon = "🟢" if v == "Pass" else "🔴"
        with st.expander(f"{icon} {TEST_LABEL.get(r['kind'], r['kind'])} · {r['case']['id']} — {v}"):
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
    if st.button("📄 PDF 보고서 생성"):
        with st.spinner("PDF 생성 중..."):
            build_pdf(results, PDF_PATH)
        with open(PDF_PATH, "rb") as f:
            st.download_button("⬇ 보고서 다운로드", f.read(),
                               file_name="IBR_MQT_Report.pdf", mime="application/pdf")

def page_chatbot():
    st.markdown(
        "<div class='hero'><h1>⚡ Power System Chatbot 🤖</h1></div>",
        unsafe_allow_html=True)
    client = get_client()
    if client is None:
        st.info("🔑 서비스 키가 설정되지 않아 챗봇을 사용할 수 없습니다. (배포자 문의)")
        return
    st.markdown("##### 💡 질문예시")
    st.caption("⚡ IEEE 2800 LVRT 기준은? 🌊 REGCAU1 Tp는 무슨 역할? "
               "🔋 Voltage Step Change Test 절차는?")
    if "chat" not in st.session_state:
        st.session_state.chat = [{
            "role": "assistant",
            "content": "👋 안녕하세요! ⚡ IBR 계통연계 전문 챗봇입니다. 무엇이든 물어보세요. 😊"
        }]
    for m in st.session_state.chat:
        avatar = "🤖" if m["role"] == "assistant" else "🧑‍🔧"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
    if prompt := st.chat_input("💬 질문을 입력하세요 (예: IEEE 2800 LVRT 기준?)"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(prompt)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,                         # 환각 억제, 일관된 답변
                messages=[
                    {"role": "system", "content": CHATBOT_SYSTEM},
                    *CHATBOT_FEWSHOT,                    # 검증된 예시 답변(few-shot)
                    *st.session_state.chat,              # 실제 대화 기록
                ],
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
