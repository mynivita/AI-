# -*- coding: utf-8 -*-
"""
유사 차트 검색기 v2 — 패턴 통계 분석 강화판
실행: streamlit run app.py

v2 추가 기능
1. 수익률 분포 분석: 유사 상위 200개 구간의 이후 수익률 통계 (평균/중앙값/승률/최악)
2. 거래량 패턴 동시 매칭: 가격 모양 + 거래대금 모양을 함께 비교
3. 베이스라인 비교: "아무 시점"의 평균 수익률 대비 이 패턴이 실제로 나은지 표시
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

st.set_page_config(page_title="유사 차트 검색기", layout="wide")

DB_PATH = "stock_db.parquet"

# ----------------------------- 데이터 로드 -----------------------------
@st.cache_data(show_spinner="데이터베이스 로딩 중...")
def load_db():
    df = pd.read_parquet(DB_PATH)
    # 한글 컬럼(퀀티와이즈 원본 형식)도 자동 변환
    df = df.rename(columns={"종가": "close", "시가": "open", "고가": "high",
                            "저가": "low", "거래대금": "value"})
    df["code"] = df["code"].astype(str).str.replace("A", "", regex=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    # 데이터 100일 미만 종목 제외
    ok = df.groupby("code")["close"].transform("size") >= 100
    df = df[ok]
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

# ----------------------------- 유사도 엔진 -----------------------------
def _znorm(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0

@st.cache_data(show_spinner="윈도우 행렬 구축 중 (최초 1회, 약 20초)...")
def build_matrices(window: int, stride: int = 3):
    """전 종목 슬라이딩 윈도우: 가격/거래량 정규화 행렬 + 메타"""
    stocks, _ = load_db()
    Mp, Mv, meta = [], [], []
    for code, g in stocks.items():
        c = g["close"].to_numpy(float)
        v = np.nan_to_num(g["value"].to_numpy(float), nan=0.0)
        if len(c) < window + 1:
            continue
        lc, lv = np.log(c), np.log1p(v)
        for s in range(0, len(c) - window + 1, stride):
            seg = lc[s:s + window]
            if np.any(~np.isfinite(seg)):
                continue
            Mp.append(_znorm(seg))
            Mv.append(_znorm(lv[s:s + window]))
            meta.append((code, s))
    return (np.vstack(Mp).astype(np.float32),
            np.vstack(Mv).astype(np.float32), meta)

@st.cache_data(show_spinner=False)
def build_fwd_returns(window: int, after: int, stride: int = 3):
    """각 윈도우 종료 시점 기준 after 거래일 뒤 수익률 (meta와 정렬 일치)"""
    stocks, _ = load_db()
    _, _, meta = build_matrices(window, stride)
    starts_by, idx_by = defaultdict(list), defaultdict(list)
    for i, (c, s) in enumerate(meta):
        starts_by[c].append(s)
        idx_by[c].append(i)
    fwd = np.full(len(meta), np.nan)
    for c, ss in starts_by.items():
        cl = stocks[c]["close"].to_numpy(float)
        S = np.asarray(ss)
        E = S + window - 1
        ok = E + after < len(cl)
        r = np.full(len(S), np.nan)
        r[ok] = cl[E[ok] + after] / cl[E[ok]] - 1
        fwd[np.asarray(idx_by[c])] = r
    return fwd

def search(query_close, query_value, window, vol_weight,
           exclude_code, n_stats=200, min_gap=10):
    """유사 구간 검색 → (통계용 상위 n_stats개 인덱스 리스트, 지표 배열들)"""
    stocks, _ = load_db()
    Mp, Mv, meta = build_matrices(window)

    qp = _znorm(np.log(query_close[-window:])).astype(np.float32)
    corr_p = (Mp @ qp) / window
    dist_p = np.sqrt(np.sum((Mp - qp) ** 2, axis=1) / window)

    if query_value is not None and vol_weight > 0:
        qv = _znorm(np.log1p(np.nan_to_num(
            query_value[-window:], nan=0.0))).astype(np.float32)
        corr_v = (Mv @ qv) / window
    else:
        corr_v = np.zeros(len(meta), dtype=np.float32)
        vol_weight = 0.0

    score = (1 - vol_weight) * (corr_p - 0.5 * dist_p) + vol_weight * corr_v
    order = np.argsort(-score)

    used, picked = defaultdict(list), []
    for i in order:
        c, s = meta[i]
        if exclude_code and c == exclude_code and \
                s > len(stocks[c]) - window - min_gap:
            continue  # 자기 자신의 "지금 그 구간" 제외
        if any(abs(s - ps) < window // 2 for ps in used[c]):
            continue  # 같은 종목 내 겹치는 구간 중복 제거
        used[c].append(s)
        picked.append(i)
        if len(picked) >= n_stats:
            break
    return picked, meta, corr_p, dist_p, corr_v, score

# ----------------------------- 차트 -----------------------------
def candle_fig(df, title, highlight=None, height=400):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#d33", decreasing_line_color="#36c"),
        row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["value"],
                         marker_color="#999"), row=2, col=1)
    if highlight:
        fig.add_vrect(x0=highlight[0], x1=highlight[1], fillcolor="orange",
                      opacity=0.15, line_width=0, row=1, col=1)
    fig.update_layout(title=title, height=height, showlegend=False,
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig

# ----------------------------- UI -----------------------------
st.title("📈 유사 차트 검색기 v2")
st.caption("618종목 · 2019-07 ~ 2026-07 (7년) · 가격+거래량 패턴 매칭 · 수익률 분포 분석")

stocks, name_map = load_db()

with st.sidebar:
    st.header("검색 설정")
    code_in = st.text_input("종목코드 (6자리)", value="005930")
    window = st.select_slider("비교 구간 (거래일)",
                              options=[20, 40, 60, 90, 120], value=60)
    after_days = st.slider("패턴 이후 분석 기간 (거래일)", 10, 60, 20, step=5)
    vol_weight = st.slider("거래량 패턴 반영 비중", 0.0, 0.5, 0.2, step=0.05,
                           help="0 = 가격 모양만 비교, 높일수록 거래대금 패턴까지 비슷한 구간 우선")
    top_k = st.slider("차트 표시 개수", 3, 10, 5)
    n_stats = st.slider("통계 표본 수 (유사 상위 N개)", 50, 300, 100, step=25,
                        help="수익률 분포 통계에 사용할 유사 구간 개수. "
                             "적을수록 더 닮은 구간만, 많을수록 표본이 커지지만 덜 닮은 구간까지 포함")
    use_live = st.checkbox("최신 데이터 온라인 조회", value=True)
    run = st.button("🔍 유사 차트 검색", use_container_width=True, type="primary")

if run:
    st.session_state["has_run"] = True
if not st.session_state.get("has_run"):
    st.info("왼쪽에서 종목코드를 입력하고 **유사 차트 검색**을 누르세요.")
    st.stop()

code = code_in.strip().replace("A", "").zfill(6)

# 1) 쿼리 데이터
qdf, source = None, ""
if use_live:
    qdf = fetch_latest(code)
    source = "온라인 (FinanceDataReader)"
if qdf is None or len(qdf) < window:
    if code in stocks:
        qdf, source = stocks[code], "내장 데이터베이스"
    else:
        st.error(f"'{code}' 데이터를 가져올 수 없습니다.")
        st.stop()

qname = name_map.get(code, code)
qclose = qdf["close"].to_numpy(float)
qvalue = qdf["value"].to_numpy(float) if "value" in qdf.columns else None
if len(qclose) < window:
    st.error(f"데이터가 {len(qclose)}일뿐이라 {window}일 비교가 불가합니다.")
    st.stop()

# 2) 쿼리 차트
st.subheader(f"기준: {qname} ({code}) — 최근 {window}거래일")
st.caption(f"데이터 출처: {source} · 마지막 날짜 {qdf['date'].iloc[-1]:%Y-%m-%d}")
hl = (qdf["date"].iloc[-window], qdf["date"].iloc[-1])
st.plotly_chart(candle_fig(qdf.tail(window * 2), f"{qname} (주황 = 비교 구간)",
                           highlight=hl), use_container_width=True)

# 3) 검색 + 통계
with st.spinner("618종목 × 7년 (31만 구간) 검색 중..."):
    picked, meta, corr_p, dist_p, corr_v, score_arr = search(
        qclose, qvalue, window, vol_weight, exclude_code=code, n_stats=300)
    fwd = build_fwd_returns(window, after_days)

pat_ret_all = fwd[picked]                      # 유사도 순 최대 300개
pat_ret = pat_ret_all[:n_stats]
pat_ret = pat_ret[np.isfinite(pat_ret)]
base = fwd[np.isfinite(fwd)]

st.subheader(f"📊 패턴 통계 — 유사 상위 {len(picked)}개 구간의 이후 {after_days}거래일")
if len(pat_ret) < 30:
    st.warning("유효 표본이 30개 미만이라 통계 신뢰도가 낮습니다.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("표본 수", f"{len(pat_ret)}개")
c2.metric("평균 수익률", f"{pat_ret.mean()*100:+.2f}%",
          delta=f"{(pat_ret.mean()-base.mean())*100:+.2f}%p vs 전체평균")
c3.metric("중앙값", f"{np.median(pat_ret)*100:+.2f}%")
c4.metric("승률 (>0%)", f"{(pat_ret>0).mean()*100:.0f}%",
          delta=f"{((pat_ret>0).mean()-(base>0).mean())*100:+.0f}%p vs 전체")
c5.metric("최고", f"{pat_ret.max()*100:+.1f}%")
c6.metric("최악", f"{pat_ret.min()*100:+.1f}%")

# 유사도-수익률 꺾은선: X=유사도, Y=이후 수익률, 마커 크기=구간 표본 수
pat_idx = np.array(picked[:n_stats])
pat_scores = score_arr[pat_idx]
pat_fwd_v = fwd[pat_idx]
okm = np.isfinite(pat_fwd_v)
pat_idx, pat_scores, pat_fwd_v = pat_idx[okm], pat_scores[okm], pat_fwd_v[okm]

nb = 8
edges = np.linspace(pat_scores.min(), pat_scores.max() + 1e-9, nb + 1)
bins = []
for b in range(nb):
    m = (pat_scores >= edges[b]) & (pat_scores < edges[b + 1]) \
        if b < nb - 1 else (pat_scores >= edges[b])
    if m.sum() == 0:
        continue
    bins.append({"x": float(pat_scores[m].mean()),
                 "y": float(pat_fwd_v[m].mean() * 100),
                 "n": int(m.sum()),
                 "members": list(pat_idx[np.where(m)[0]])})
bins.sort(key=lambda d: d["x"])
max_n = max(b["n"] for b in bins)
sizes = [10 + 28 * b["n"] / max_n for b in bins]

sfig = go.Figure(go.Scatter(
    x=[b["x"] for b in bins], y=[b["y"] for b in bins],
    mode="lines+markers",
    marker=dict(size=sizes, color="#4a7ebb",
                line=dict(color="#2c5aa0", width=1)),
    line=dict(color="#9bb8dd"),
    customdata=[[b["n"]] for b in bins],
    hovertemplate=("유사도 %{x:.3f}<br>평균 수익률 %{y:+.2f}%"
                   "<br>표본 %{customdata[0]}개<extra></extra>")))
sfig.add_hline(y=float(base.mean()) * 100, line_dash="dash",
               line_color="#2a9d3a", annotation_text="전체 평균(베이스라인)")
sfig.update_layout(height=340, showlegend=False,
                   xaxis_title="유사도 점수 (오른쪽일수록 기준 차트와 더 닮음)",
                   yaxis_title=f"이후 {after_days}거래일 평균 수익률 (%)",
                   margin=dict(l=10, r=10, t=25, b=10))
event = st.plotly_chart(sfig, use_container_width=True,
                        on_select="rerun", key="sim_bins")
st.caption("● 마커 크기 = 해당 유사도 구간의 표본 수 · **마커를 클릭**하면 "
           "그 구간에 속한 실제 차트들이 아래에 표시됩니다")

sel = None
try:
    pts = event.selection.points
    if pts:
        sel = pts[0].get("point_index", pts[0].get("point_number"))
except Exception:
    sel = None

if sel is not None and 0 <= sel < len(bins):
    b = bins[sel]
    st.markdown(f"**🔎 선택 구간 — 유사도 {b['x']:.3f} 부근 · "
                f"표본 {b['n']}개 · 평균 {b['y']:+.2f}%**")
    show_n = min(b["n"], 6)
    if b["n"] > show_n:
        st.caption(f"이 중 유사도 상위 {show_n}개만 표시합니다.")
    bcols = st.columns(2)
    for k, i in enumerate(b["members"][:show_n]):
        mcode, s = meta[i]
        g = stocks[mcode]
        e = s + window
        showdf = g.iloc[max(0, s - 10): min(len(g), e + after_days)]
        hlb = (g["date"].iloc[s], g["date"].iloc[e - 1])
        if e < len(g):
            fe = min(len(g) - 1, e - 1 + after_days)
            ret_txt = f"이후 {(g['close'].iloc[fe]/g['close'].iloc[e-1]-1)*100:+.1f}%"
        else:
            ret_txt = "이후 데이터 없음"
        with bcols[k % 2]:
            st.plotly_chart(candle_fig(
                showdf,
                f"{name_map.get(mcode, mcode)} ({mcode}) · {ret_txt}",
                highlight=hlb, height=300),
                use_container_width=True, key=f"binchart_{sel}_{k}")

better = pat_ret.mean() - base.mean()
win_diff = (pat_ret > 0).mean() - (base > 0).mean()
if abs(better) < 0.005 and abs(win_diff) < 0.03:
    verdict = ("이 패턴의 과거 성과는 **아무 시점이나 고른 것과 사실상 차이가 없습니다.** "
               "모양 자체에 특별한 정보가 없었다는 뜻입니다.")
elif better > 0:
    verdict = (f"이 패턴군은 전체 평균 대비 **+{better*100:.2f}%p** 높은 수익률을 보였습니다. "
               "다만 생존편향(현재 시총 3,000억 이상 종목만 포함)이 있어 실제보다 낙관적일 수 있습니다.")
else:
    verdict = (f"이 패턴군은 전체 평균 대비 **{better*100:.2f}%p** 낮은 수익률을 보였습니다.")
st.markdown(f"> {verdict}")
st.caption("⚠️ 과거 통계일 뿐 미래를 보장하지 않습니다. 생존편향이 내장되어 있어 "
           "전반적으로 실제보다 낙관적으로 보일 수 있습니다.")

# 3-1) 통계 안정성 자동 검증: 표본 수 50~300에서 결과가 출렁이는지
Ns = [50, 100, 150, 200, 250, 300]
rob = []
for n in Ns:
    r = pat_ret_all[:n]
    r = r[np.isfinite(r)]
    if len(r) >= 20:
        rob.append((n, r.mean(), (r > 0).mean()))
if len(rob) >= 3:
    means = np.array([m for _, m, _ in rob])
    wins = np.array([w for _, _, w in rob])
    excess = means - base.mean()
    spread = means.max() - means.min()
    sign_flip = (excess.max() > 0) and (excess.min() < 0)

    if sign_flip:
        s_verdict = ("🔴 **불안정** — 표본 수에 따라 전체 평균보다 좋았다 나빴다가 "
                     "뒤집힙니다. 이 패턴의 통계는 우연의 영향이 커서 신뢰하기 어렵습니다.")
    elif spread < 0.03:
        s_verdict = ("🟢 **안정적** — 표본을 50개로 줄여도 300개로 늘려도 결과가 "
                     "거의 같습니다. 이 패턴군의 통계는 소수 극단값에 좌우되지 않습니다.")
    elif spread < 0.07:
        s_verdict = ("🟡 **보통** — 표본 수에 따라 수치가 다소 움직이지만 방향은 "
                     "일관됩니다. 참고는 하되 수치 자체를 믿지는 마세요.")
    else:
        s_verdict = ("🔴 **불안정** — 표본 수에 따라 평균이 크게 출렁입니다. "
                     "극단적인 사례 몇 개가 통계를 좌우하고 있다는 신호입니다.")

    with st.expander("🧪 통계 안정성 검증 (표본 수 50→300 변화 테스트)", expanded=True):
        st.markdown(s_verdict)
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(
            x=[n for n, _, _ in rob], y=means * 100,
            mode="lines+markers", name="패턴군 평균",
            line=dict(color="#d33")))
        rfig.add_hline(y=float(base.mean()) * 100, line_dash="dash",
                       line_color="#2a9d3a",
                       annotation_text="전체 평균(베이스라인)")
        rfig.update_layout(height=260,
                           xaxis_title="통계 표본 수 (유사 상위 N개)",
                           yaxis_title=f"이후 {after_days}거래일 평균 수익률 (%)",
                           margin=dict(l=10, r=10, t=20, b=10),
                           showlegend=False)
        st.plotly_chart(rfig, use_container_width=True)
        st.caption(f"승률 범위: {wins.min()*100:.0f}% ~ {wins.max()*100:.0f}% · "
                   f"평균 수익률 범위: {means.min()*100:+.2f}% ~ {means.max()*100:+.2f}% "
                   f"(변동폭 {spread*100:.2f}%p)")

st.divider()

# 4) Top K 차트
st.subheader(f"유사 차트 Top {top_k}")
st.caption("주황 구간 = 매칭 패턴 · 이후 봉 = 패턴 뒤 실제 흐름")

for rank, i in enumerate(picked[:top_k], 1):
    mcode, s = meta[i]
    g = stocks[mcode]
    mname = name_map.get(mcode, mcode)
    e = s + window
    show = g.iloc[max(0, s - 10): min(len(g), e + after_days)]
    hl = (g["date"].iloc[s], g["date"].iloc[e - 1])

    aft_txt = "이후 데이터 없음"
    if e < len(g):
        fe = min(len(g) - 1, e - 1 + after_days)
        ret = (g["close"].iloc[fe] / g["close"].iloc[e - 1] - 1) * 100
        aft_txt = f"이후 {fe-(e-1)}거래일: **{ret:+.1f}%**"

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(candle_fig(
            show,
            f"#{rank} {mname} ({mcode}) · "
            f"{g['date'].iloc[s]:%Y-%m-%d} ~ {g['date'].iloc[e-1]:%Y-%m-%d}",
            highlight=hl, height=380), use_container_width=True)
    with col2:
        st.metric("가격 상관", f"{corr_p[i]:.3f}")
        st.metric("형태 거리", f"{dist_p[i]:.3f}")
        if vol_weight > 0:
            st.metric("거래량 상관", f"{corr_v[i]:.3f}")
        st.markdown(aft_txt)
    st.divider()
