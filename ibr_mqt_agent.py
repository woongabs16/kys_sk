"""
IBR Model Quality Test (MQT) AI Agent
=====================================
인버터(IBR) 모델 품질테스트 자동화를 위한 AI 에이전트.

흐름:
  1. PSS/E 생략 → 180MW PV Plant의 LVRT / HVRT / Voltage Ride-Through
     테스트를 각 2케이스씩 모사한 CSV 데이터 생성
  2. 생성된 CSV를 바탕으로 OpenAI가 IEEE 2800.2 기준으로 Pass/Fail 판정
  3. 판정 결과 PDF 보고서 제공 (Fail이면 모델 수정방안 포함)
  4. 챗봇으로 사용자 질문 응답

실행:  streamlit run ibr_mqt_agent.py

API 키 우선순위:
  1) 사이드바 입력값 (사용자 본인 키 - 권장)
  2) 환경변수 OPENAI_API_KEY (.env)
  3) st.secrets["OPENAI_API_KEY"] (Streamlit Cloud 배포 시)
"""

import os
import io
import json
import textwrap
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
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
)

# matplotlib은 그래프 이미지를 PDF에 넣기 위해 사용
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# OpenAI 클라이언트 (지연 초기화: 키가 있을 때만 생성)
# ----------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 로컬 실행 시 .env 자동 로드 (배포 환경에서는 Secrets 사용)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_api_key() -> str | None:
    """[B방법] 키를 코드에 두지 않고 st.secrets 또는 환경변수에서만 읽는다.
    - Streamlit Cloud 배포: 앱 Settings > Secrets 에 OPENAI_API_KEY 등록
    - 로컬 실행: .env 파일 또는 환경변수 OPENAI_API_KEY
    사용자는 키를 입력하지 않고 그대로 앱을 사용한다(개발자 키 사용)."""
    try:
        secret_key = st.secrets["OPENAI_API_KEY"]
        if secret_key:
            return secret_key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_client() -> "OpenAI | None":
    key = get_api_key()
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


# ----------------------------------------------------------------------------
# 1) PSS/E 생략 - 시험 케이스용 CSV 시계열 데이터 생성
# ----------------------------------------------------------------------------
PLANT_MW = 180.0          # 180MW PV Plant
DT = 0.01                 # 10ms step
T_END = 10.0              # 10s 시뮬레이션
FAULT_START = 2.0         # 고장 시작
F_NOM = 60.0

# 각 시험 유형별 2케이스. (voltage_dip pu, duration s, 회복여부)
TEST_CASES = {
    "LVRT": [
        {"id": "LVRT_01", "dip": 0.10, "dur": 0.15, "recover": True,
         "desc": "V=0.10pu, 0.15s @POI (IEEE2800 5.3.2)"},
        {"id": "LVRT_02", "dip": 0.45, "dur": 1.00, "recover": True,
         "desc": "V=0.45pu, 1.00s 3φ fault (IEEE2800 5.3.4)"},
    ],
    "HVRT": [
        {"id": "HVRT_01", "swell": 1.15, "dur": 0.50, "recover": True,
         "desc": "V=1.15pu, 0.50s swell (IEEE2800 5.3.5)"},
        {"id": "HVRT_02", "swell": 1.20, "dur": 0.20, "recover": False,
         "desc": "V=1.20pu, 0.20s swell - 모델 미회복 사례"},
    ],
    "VRT": [   # Voltage step / ride-through
        {"id": "VRT_01", "step": 1.05, "dur": 2.00, "recover": True,
         "desc": "+5% voltage step response"},
        {"id": "VRT_02", "step": 0.95, "dur": 2.00, "recover": True,
         "desc": "-5% voltage step response"},
    ],
}


def _make_timeseries(case: dict, kind: str) -> pd.DataFrame:
    """단순 1차 응답 모델로 V/P/Q 시계열을 생성한다 (PSS/E 대체용)."""
    t = np.arange(0, T_END, DT)
    v = np.ones_like(t)  # pre-fault 1.0 pu

    if kind == "LVRT":
        level, dur = case["dip"], case["dur"]
    elif kind == "HVRT":
        level, dur = case["swell"], case["dur"]
    else:  # VRT
        level, dur = case["step"], case["dur"]

    fault_mask = (t >= FAULT_START) & (t < FAULT_START + dur)
    v[fault_mask] = level

    # 고장 해소 후 회복 (recover=False면 회복 실패하도록 잔류 오차)
    post_mask = t >= FAULT_START + dur
    tau = 0.15
    target = 1.0 if case.get("recover", True) else level + 0.3 * (1.0 - level)
    v_at_clear = v[fault_mask][-1] if fault_mask.any() else 1.0
    v[post_mask] = target + (v_at_clear - target) * np.exp(-(t[post_mask] - (FAULT_START + dur)) / tau)

    # 유효전력: 전압에 따라 출력 저하, 회복 시 복귀
    p = PLANT_MW * np.clip(v, 0, 1.2)
    if not case.get("recover", True):
        p[post_mask] *= 0.6  # 출력 미복귀

    # 무효전력: 저전압 시 무효전류 주입(+), 과전압 시 흡수(-)
    q = np.zeros_like(t)
    q[fault_mask] = PLANT_MW * 0.5 * (1.0 - v[fault_mask])  # K=2 비례 주입 근사

    # 측정 노이즈
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


def generate_all_cases(out_dir: Path) -> dict:
    """모든 시험 케이스 CSV를 생성하고 {kind: [(case, df, csv_path)]} 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for kind, cases in TEST_CASES.items():
        rows = []
        for case in cases:
            df = _make_timeseries(case, kind)
            csv_path = out_dir / f"{case['id']}.csv"
            df.to_csv(csv_path, index=False)
            rows.append((case, df, csv_path))
        generated[kind] = rows
    return generated


# ----------------------------------------------------------------------------
# 2) IEEE 2800.2 기준 Pass/Fail 판정
# ----------------------------------------------------------------------------
def quick_metrics(df: pd.DataFrame) -> dict:
    """판정에 쓰일 핵심 지표를 계산한다."""
    pre = df[df["Time (s)"] < FAULT_START]
    post = df[df["Time (s)"] > FAULT_START + 1.0]
    p_pre = pre["Active Power (MW)"].mean()
    p_post = post["Active Power (MW)"].mean()
    v_post = post["Voltage (pu)"].mean()
    v_overshoot = df["Voltage (pu)"].max()
    recovery_ratio = (p_post / p_pre) if p_pre else 0.0
    return {
        "pre_fault_P_MW": round(p_pre, 2),
        "post_fault_P_MW": round(p_post, 2),
        "post_fault_V_pu": round(v_post, 4),
        "max_V_pu": round(v_overshoot, 4),
        "P_recovery_ratio": round(recovery_ratio, 4),
    }


def judge_with_openai(case: dict, kind: str, metrics: dict) -> dict:
    """OpenAI로 IEEE 2800.2 기준 판정. 키가 없으면 룰베이스 fallback."""
    client = get_client()

    fallback = _rule_based_judgment(metrics)

    if client is None:
        fallback["source"] = "rule-based (API 키 없음)"
        return fallback

    prompt = f"""You are a grid-code compliance expert. Judge the following IBR (inverter-based resource)
ride-through test result against IEEE 2800.2 conformity criteria.

Test type: {kind}
Test case: {case['id']} - {case['desc']}
Computed metrics: {json.dumps(metrics)}

IEEE 2800.2 key expectations:
- Plant must remain connected through the disturbance (no trip).
- Active power must recover to >= 95% of pre-fault value after fault clearance.
- Post-fault voltage should settle near 1.0 pu; transient overshoot should stay within ~1.2 pu.
- For LVRT, reactive current injection should support voltage during the dip.

Return STRICT JSON only, no markdown:
{{"verdict": "Pass" or "Fail",
  "reason": "one concise sentence",
  "mitigation": "if Fail, concrete inverter model parameter changes (e.g. REGCAU1/REECAU1 gains, Kqv, Imax, Qmax); if Pass, empty string"}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        data["source"] = "OpenAI (gpt-4o-mini)"
        return data
    except Exception as e:
        fallback["source"] = f"rule-based (OpenAI 호출 실패: {e})"
        return fallback


def _rule_based_judgment(metrics: dict) -> dict:
    recov = metrics["P_recovery_ratio"]
    overshoot = metrics["max_V_pu"]
    fail_reasons = []
    if recov < 0.95:
        fail_reasons.append(f"유효전력 회복률 {recov:.0%} < 95%")
    if overshoot > 1.2:
        fail_reasons.append(f"전압 오버슈트 {overshoot:.3f}pu > 1.20pu")
    if fail_reasons:
        return {
            "verdict": "Fail",
            "reason": "; ".join(fail_reasons),
            "mitigation": ("REECAU1의 무효전류 게인(Kqv) 상향(예: 0.9), Imax 1.22pu로 확대, "
                           "Qmax 1.15pu 확대하여 고장 중 무효전류 주입과 전압 회복 속도를 개선."),
        }
    return {"verdict": "Pass", "reason": "전압 회복 및 출력 복귀 정상", "mitigation": ""}


# ----------------------------------------------------------------------------
# 3) PDF 보고서 생성
# ----------------------------------------------------------------------------
def _plot_case(df: pd.DataFrame, case_id: str) -> io.BytesIO:
    fig, ax1 = plt.subplots(figsize=(5.5, 2.6))
    ax1.plot(df["Time (s)"], df["Voltage (pu)"], color="tab:blue", label="V (pu)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (pu)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["Time (s)"], df["Active Power (MW)"], color="tab:red", alpha=0.7, label="P (MW)")
    ax2.set_ylabel("P (MW)", color="tab:red")
    ax1.set_title(case_id)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(results: list, out_path: Path) -> Path:
    """results: [{kind, case, metrics, judgment, df}] → PDF 생성."""
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], alignment=TA_CENTER)
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    elems = [Paragraph("IBR Model Quality Test Report", title),
             Paragraph(f"180 MW PV Plant &nbsp;|&nbsp; IEEE 2800.2 Conformity &nbsp;|&nbsp; "
                       f"{datetime.now():%Y-%m-%d %H:%M}", body),
             Spacer(1, 16)]

    # 요약 테이블
    header = ["Test", "Case", "Verdict", "P recovery", "Max V (pu)"]
    rows = [header]
    for r in results:
        rows.append([r["kind"], r["case"]["id"], r["judgment"]["verdict"],
                     f"{r['metrics']['P_recovery_ratio']:.0%}",
                     f"{r['metrics']['max_V_pu']:.3f}"])
    tbl = Table(rows, colWidths=[0.9*inch, 1.2*inch, 1.0*inch, 1.2*inch, 1.2*inch])
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
             ("ALIGN", (0, 0), (-1, -1), "CENTER"),
             ("FONTSIZE", (0, 0), (-1, -1), 9)]
    # Fail 행 강조
    for i, r in enumerate(results, start=1):
        if r["judgment"]["verdict"] == "Fail":
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fdecea")))
            style.append(("TEXTCOLOR", (2, i), (2, i), colors.red))
    tbl.setStyle(TableStyle(style))
    elems += [Paragraph("Summary", h2), tbl, PageBreak()]

    # 케이스별 상세
    for r in results:
        j = r["judgment"]
        elems.append(Paragraph(f"{r['kind']} - {r['case']['id']}", h2))
        elems.append(Paragraph(r["case"]["desc"], body))
        elems.append(Paragraph(
            f"<b>Verdict:</b> {j['verdict']} &nbsp;&nbsp; "
            f"(<i>{j.get('source','')}</i>)<br/><b>Reason:</b> {j['reason']}", body))
        if j["verdict"] == "Fail" and j.get("mitigation"):
            elems.append(Spacer(1, 4))
            elems.append(Paragraph("<b>Mitigation (모델 수정방안)</b>", body))
            elems.append(Paragraph(j["mitigation"], body))
        elems.append(Spacer(1, 8))
        img = _plot_case(r["df"], r["case"]["id"])
        elems.append(RLImage(img, width=5.0*inch, height=2.4*inch))
        elems.append(PageBreak())

    doc.build(elems)
    return out_path


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="IBR MQT AI Agent", layout="wide")

WORK_DIR = Path("./mqt_workspace")
CSV_DIR = WORK_DIR / "csv"
PDF_PATH = WORK_DIR / "MQT_Report.pdf"


def sidebar_api():
    st.sidebar.header("🔑 API 상태")
    if get_api_key():
        st.sidebar.success("서비스 준비 완료 ✅")
        st.sidebar.caption("개발자 키로 동작 중 (사용자 키 입력 불필요)")
    else:
        st.sidebar.error("키 미설정 → 룰베이스 판정으로 동작")
        st.sidebar.caption("배포자: Secrets에 OPENAI_API_KEY 등록 필요")


def page_run():
    st.title("IBR Model Quality Test · AI Agent")
    st.markdown("180 MW PV Plant · LVRT / HVRT / Voltage Ride-Through (각 2케이스) · IEEE 2800.2")

    if st.button("▶ 시뮬레이션 데이터 생성 & 판정 실행", type="primary"):
        with st.spinner("CSV 생성 중..."):
            generated = generate_all_cases(CSV_DIR)

        results = []
        progress = st.progress(0.0)
        total = sum(len(v) for v in generated.values())
        done = 0
        for kind, items in generated.items():
            for case, df, csv_path in items:
                metrics = quick_metrics(df)
                judgment = judge_with_openai(case, kind, metrics)
                results.append({"kind": kind, "case": case, "df": df,
                                "metrics": metrics, "judgment": judgment})
                done += 1
                progress.progress(done / total)
        progress.empty()
        st.session_state["results"] = results
        st.success("판정 완료!")

    results = st.session_state.get("results")
    if results:
        summary = pd.DataFrame([{
            "Test": r["kind"], "Case": r["case"]["id"],
            "Verdict": r["judgment"]["verdict"],
            "P recovery": f"{r['metrics']['P_recovery_ratio']:.0%}",
            "Max V (pu)": r["metrics"]["max_V_pu"],
            "Reason": r["judgment"]["reason"],
        } for r in results])

        def color_verdict(val):
            return "color: red; font-weight:bold" if val == "Fail" else "color: green"
        st.dataframe(summary.style.map(color_verdict, subset=["Verdict"]),
                     use_container_width=True)

        # 케이스별 그래프
        for r in results:
            with st.expander(f"{r['kind']} · {r['case']['id']} — {r['judgment']['verdict']}"):
                st.line_chart(r["df"].set_index("Time (s)")[["Voltage (pu)", "Active Power (MW)"]])
                if r["judgment"]["verdict"] == "Fail":
                    st.error(f"수정방안: {r['judgment'].get('mitigation','')}")

        # PDF
        if st.button("📄 PDF 보고서 생성"):
            with st.spinner("PDF 생성 중..."):
                build_pdf(results, PDF_PATH)
            with open(PDF_PATH, "rb") as f:
                st.download_button("⬇ 보고서 다운로드", f.read(),
                                   file_name="IBR_MQT_Report.pdf", mime="application/pdf")


def page_chatbot():
    st.title("Power System Chatbot")
    client = get_client()
    if client is None:
        st.info("서비스 키가 설정되지 않아 챗봇을 사용할 수 없습니다. (배포자에게 문의)")
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
                  "IBR grid interconnection, IEEE 2800 / 2800.2, LVRT/HVRT, and PSS/E "
                  "inverter models. Answer concisely in Korean.")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, *st.session_state.chat],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"오류: {e}"
        st.session_state.chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


PAGES = {
    "1) 시뮬레이션 & 판정": page_run,
    "2) 챗봇": page_chatbot,
}

sidebar_api()
st.sidebar.header("메뉴")
choice = st.sidebar.radio("이동", list(PAGES.keys()))
PAGES[choice]()
