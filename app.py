# -*- coding: utf-8 -*-
"""
유사 차트 검색기 — 퀀티와이즈 628종목 3년치 데이터 기반
실행: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="유사 차트 검색기", layout="wide")

DB_PATH = "stock_db.parquet"

# ----------------------------- 데이터 로드 -----------------------------
@st.cache_data(show_spinner="데이터베이스 로딩 중...")
def load_db():
    df = pd.read_parquet(DB_PATH)
    df["date"] = pd.to_datetime(df["date"])
    stocks = {}
    for code, g in df.groupby("code"):
        g = g.sort_values("date").reset_index(drop=True)
        stocks[code] = g
    name_map = df.groupby("code")["name"].first().to_dict()
    return stocks, name_map

def fetch_latest(code: str, days: int = 400):
    """FinanceDataReader로 최신 주가 조회. 실패 시 None."""
    try:
        import FinanceDataReader as fdr
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=int(days * 1.8))
        df = fdr.DataReader(code, start.strftime("%Y-%m-%d"))
        if df is None or len(df) == 0:
            return None
        df = df.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "value"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "value"]].dropna(subset=["close"])
    except Exception:
        return None

# ----------------------------- 유사도 계산 -----------------------------
def normalize(x: np.ndarray) -> np.ndarray:
    """로그가격 → z-score 정규화 (모양만 비교)"""
    lx = np.log(x)
    s = lx.std()
    if s < 1e-9:
        return lx * 0
    return (lx - lx.mean()) / s

@st.cache_data(show_spinner=False)
def build_window_matrix(window: int, stride: int = 3):
    """모든 종목의 슬라이딩 윈도우를 하나의 행렬로 구축 (정규화된 종가)"""
    stocks, _ = load_db()
    mats, meta = [], []
    for code, g in stocks.items():
        c = g["close"].to_numpy(dtype=float)
        n = len(c)
        if n < window + 1:
            continue
        for s in range(0, n - window + 1, stride):
            seg = c[s:s + window]
            if np.any(seg <= 0) or np.any(~np.isfinite(seg)):
                continue
            mats.append(normalize(seg))
            meta.append((code, s))
    M = np.vstack(mats)
    return M, meta

def find_similar(query_close: np.ndarray, window: int, top_k: int,
                 exclude_code: str | None, min_gap: int = 10):
    """query 최근 window일과 가장 비슷한 과거 구간 검색"""
    q = normalize(query_close[-window:])
    M, meta = build_window_matrix(window)

    # 피어슨 상관 (정규화된 벡터이므로 내적/윈도우 길이)
    corr = (M @ q) / window
    # 유클리드 거리 (모양 차이)
    dist = np.sqrt(np.sum((M - q) ** 2, axis=1) / window)
    score = corr - 0.5 * dist  # 상관 높고 거리 짧을수록 점수↑

    order = np.argsort(-score)
    picked, used = [], {}
    for idx in order:
        code, s = meta[idx]
        if exclude_code and code == exclude_code:
            # 같은 종목의 "지금 그 구간"이 그대로 뽑히는 것 방지:
            # 자기 자신 종목은 최근 window일과 겹치는 구간만 제외
            stocks, _ = load_db()
            n = len(stocks[code])
            if s > n - window - min_gap:
                continue
        # 같은 종목 내 겹치는 구간 중복 방지
        key_ok = True
        for ps in used.get(code, []):
            if abs(ps - s) < window // 2:
                key_ok = False
                break
        if not key_ok:
            continue
        used.setdefault(code, []).append(s)
        picked.append((code, s, float(corr[idx]), float(dist[idx])))
        if len(picked) >= top_k:
            break
    return picked

# ----------------------------- 차트 -----------------------------
def candle_fig(df: pd.DataFrame, title: str, highlight: tuple | None = None,
               height: int = 420):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#d33", decreasing_line_color="#36c",
        name="가격"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["value"],
                         marker_color="#999", name="거래대금"), row=2, col=1)
    if highlight:
        x0, x1 = highlight
        fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.15,
                      line_width=0, row=1, col=1)
    fig.update_layout(title=title, height=height, showlegend=False,
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig

# ----------------------------- UI -----------------------------
st.title("📈 유사 차트 검색기")
st.caption("시총 3,000억 이상 628종목 · 2023-07 ~ 2026-07 일봉 데이터베이스 기반")

stocks, name_map = load_db()

with st.sidebar:
    st.header("검색 설정")
    code_in = st.text_input("종목코드 (6자리)", value="005930",
                            help="예: 005930 (삼성전자)")
    window = st.select_slider("비교 구간 (거래일)", options=[20, 40, 60, 90, 120], value=60)
    top_k = st.slider("결과 개수", 3, 10, 5)
    after_days = st.slider("패턴 이후 표시 기간 (거래일)", 10, 60, 20, step=5,
                           help="매칭된 패턴이 끝난 뒤 며칠간의 실제 흐름을 보여줄지")
    use_live = st.checkbox("최신 데이터 온라인 조회 (FinanceDataReader)", value=True,
                           help="끄면 내장 데이터베이스의 최신 시점 기준으로 검색")
    run = st.button("🔍 유사 차트 검색", use_container_width=True, type="primary")

if run:
    code = code_in.strip().replace("A", "").zfill(6)

    # 1) 쿼리 데이터 확보
    qdf, source = None, ""
    if use_live:
        qdf = fetch_latest(code)
        source = "온라인 (FinanceDataReader)"
    if qdf is None or len(qdf) < window:
        if code in stocks:
            qdf = stocks[code]
            source = "내장 데이터베이스"
        else:
            st.error(f"'{code}' 데이터를 가져올 수 없습니다. "
                     "온라인 조회 실패 + 데이터베이스에도 없는 종목입니다.")
            st.stop()

    qname = name_map.get(code, code)
    qclose = qdf["close"].to_numpy(dtype=float)
    if len(qclose) < window:
        st.error(f"데이터가 {len(qclose)}일뿐이라 {window}일 비교가 불가합니다.")
        st.stop()

    # 2) 쿼리 차트
    st.subheader(f"기준: {qname} ({code}) — 최근 {window}거래일")
    st.caption(f"데이터 출처: {source} · 마지막 날짜 {qdf['date'].iloc[-1]:%Y-%m-%d}")
    qtail = qdf.tail(min(len(qdf), window * 2))
    hl = (qdf["date"].iloc[-window], qdf["date"].iloc[-1])
    st.plotly_chart(candle_fig(qtail, f"{qname} (주황 = 비교 구간)", highlight=hl),
                    use_container_width=True)

    # 3) 검색
    with st.spinner("628종목 × 3년 구간에서 유사 패턴 검색 중..."):
        results = find_similar(qclose, window, top_k, exclude_code=code)

    if not results:
        st.warning("유사 구간을 찾지 못했습니다.")
        st.stop()

    st.subheader(f"유사 차트 Top {len(results)}")
    st.caption("주황 구간 = 매칭 패턴 · 이후 봉 = 패턴 뒤 실제 흐름 (참고용)")

    for rank, (mcode, s, corr, dist) in enumerate(results, 1):
        g = stocks[mcode]
        mname = name_map.get(mcode, mcode)
        after = after_days  # 패턴 이후 표시 기간 (사이드바에서 조절)
        e = s + window
        show = g.iloc[max(0, s - 10): min(len(g), e + after)]
        hl = (g["date"].iloc[s], g["date"].iloc[e - 1])

        # 패턴 이후 수익률
        aft_txt = "이후 데이터 없음"
        if e < len(g):
            base = g["close"].iloc[e - 1]
            fut_end = min(len(g) - 1, e - 1 + after)
            ret = (g["close"].iloc[fut_end] / base - 1) * 100
            aft_txt = f"패턴 종료 후 {fut_end - (e - 1)}거래일 수익률: **{ret:+.1f}%**"

        c1, c2 = st.columns([3, 1])
        with c1:
            st.plotly_chart(candle_fig(
                show,
                f"#{rank} {mname} ({mcode}) · {g['date'].iloc[s]:%Y-%m-%d} ~ {g['date'].iloc[e-1]:%Y-%m-%d}",
                highlight=hl, height=380), use_container_width=True)
        with c2:
            st.metric("상관계수", f"{corr:.3f}")
            st.metric("형태 거리", f"{dist:.3f}")
            st.markdown(aft_txt)
        st.divider()
else:
    st.info("왼쪽에서 종목코드를 입력하고 **유사 차트 검색**을 누르세요.")
