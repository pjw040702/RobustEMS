# -*- coding: utf-8 -*-
"""
자과캠 PV·ESS·MG 경제성 모델 — A / C-1 / C-2 시나리오, 25년 현금흐름 NPV
=========================================================================
파라미터 기준: reference/1113_통합 파라미터 원장.pdf (팀 원장).
원장에 값이 없는 항목은 이 파일 주석에 출처(약관·공고·PNNL 등)를 달았다.

시나리오
  기준선  PV 0 + ESS 0 + MG 0 — 실제 미도입. 모든 절감액·NPV 의 기준
  A      시장단가 PV 130만원/kW, 국비 없음, ESS duration 1~5h 비교, 공고 제약 없음
  C-1    공고단가 PV 103.6만원/kW, 국비 없음, ESS 5h + 공고 제약(경부하기 의무충전)
  C-2    C-1 과 같은 조건 + 조건부 국비(적격 사업비 최대 70%, 캠퍼스 분야 130억 가상 한도)
         C-1 과 보조금 외 조건이 같으므로 C-2 − C-1 = 순수 보조금 효과

설계 원칙
  - ESS 출력(kW)은 LP 결정변수, 용량 = duration x 출력. PV 용량은 최대용량의 0~100% 11점 고정
  - 설비비는 LP 목적함수에 '25년 현금흐름의 등가연간비용(EAC)'으로만 들어간다. NPV 는 0년차
    초기투자 + 연차별 현금흐름으로 따로 계산하므로 연간환산 투자비와 초기투자비가 겹치지 않는다
  - PV 열화는 연차별 (1-0.94%)^(y-1). 0.8949(25년 평균계수)는 쓰지 않는다. 설비 최적화에는
    할인가중 대표계수 S_REP 를 쓰고, NPV 는 연차별 운영비를 3점(1년차/대표/25년차) 2차보간한다
  - 기본요금은 약관 제68조 월별 요금적용전력. 정상상태(매년 같은 운전)에서 '직전 12개월 중
    12·1·2·7·8·9월과 당월 최대' = max(당월 피크, 지정월 피크)
  - 열화비 C_DEG 기준 0 — 배터리 교체를 현금흐름에 넣으므로 throughput 비용은 이중계산

실행
  python econ_model.py --validate   실측 부하로 2025.09~2026.08 청구액 재현(원장 13,885,735,690원)
  python econ_model.py --quick      대표 1건(C-1, PV 100%) 시간·메모리 측정
  python econ_model.py --all        본 실행 + 민감도 -> results/econ/*.csv
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "econ")
INPUTS = os.path.join(ROOT, "data", "econ_inputs.npz")      # 재생성 가능한 캐시(.gitignore)

# =====================================================================
# 1. 파라미터 — 1113 통합 파라미터 원장
# =====================================================================
DT_H = 0.25
R_DISC, YEARS = 0.045, 25                                    # 실질할인율 4.5%, 프로젝트 25년
DF = 1.0 / (1.0 + R_DISC) ** np.arange(0, YEARS + 1)         # DF[y], y = 0..25
AF = float(DF[1:].sum())                                     # 연금현가계수 14.828
CRF = 1.0 / AF                                               # 0.06744

# ---- 전기요금: 교육용(을) 고압A 선택Ⅱ ----
CONTRACT_KW = 30_000.0
C_PEAK = 6_980.0                                             # 원/kW·월
RATE = {"summer": {"off": 72.0, "mid": 116.7, "peak": 182.6},          # 원장: 개정 전후 가격 유지
        "spring_fall": {"off": 72.0, "mid": 86.4, "peak": 106.9},
        "winter": {"off": 76.0, "mid": 115.2, "peak": 153.9}}
CLIMATE_WON, FUEL_WON = 9.0, 5.0                             # 원/kWh
VAT, FUND = 0.10, 0.027                                      # 부가세 10%, 전력산업기반기금 2.7%(2025.07.01~)
TAX_F = 1.0 + VAT + FUND                                     # 1.127 — 둘 다 전기요금계에 항목별 부과
REFORM = pd.Timestamp("2026-06-01")                          # 계시별 시간대 개편, 교육용(을) 적용일
S_MONTHS = (12, 1, 2, 7, 8, 9)                               # 기본공급약관 제68조① 참조월
DEMAND_FLOOR = 0.30 * CONTRACT_KW                            # 제68조① '계약전력의 30% 미만이면 30%'
# 시간대·주말·공휴일: 전기공급약관 별표3(교육용전력(을) 포함). 개편: 기후부 2026.3.13, 교육용(을) 2026.6.1

# ---- 탄소 ----
CO2 = 0.4173                 # tCO2eq/MWh 국가 승인 2023 소비단 계수(2025.12 공표) — 감축량 기준
CO2_OLD = 0.45941            # 종전 모델(발표자료 p.53) — 비교
CO2_ONLY = 0.4149            # CO2 단일 가스 — 비교
CO2_PP = 0.4541              # 파워플래너 고지 — 비교
CARBON_PRICE = 25_500.0      # 원/tCO2eq, 2025 대학 매입단가(발표자료 p.30)
SCC = {"SCC 중위(51USD)": 69_349.0, "SCC 고위(76USD)": 103_344.0}   # 사회적 편익 — NPV 에 넣지 않음

# ---- PV ----
PV_MAX_KW = 10_305.4
PV_PRICE = {"A": 1_300_000.0, "C-1": 1_036_000.0, "C-2": 1_036_000.0}
PV_PRICE_A_HIGH = 1_500_000.0                                # A 시장단가 상향 민감도
PV_DIAG = 33_478.0                                           # 구조안전진단 원/kW (1,500만 x 23동 / 10,305.4)
PV_OM = {"low": 15_000.0, "high": 20_000.0, "ext": 27_196.0}  # 원/kW·년
PV_DEG = 0.0094
PV_F = (1.0 - PV_DEG) ** np.arange(0, YEARS)                 # PV_F[y-1] = y년차 출력계수
S_REP = float((PV_F * DF[1:]).sum() / AF)                    # 할인가중 대표 출력계수(설비 최적화용)
S_END = float(PV_F[-1])                                      # 25년차
G_STC, T_STC, NOCT, ETA_SYS, GAMMA = 1000.0, 25.0, 45.0, 0.96, -0.0045
PV_FRACS = [round(i / 10, 1) for i in range(11)]

# ---- BIPV (확장 시나리오, 국비 0) ----
BIPV_RATED_KW = 1_274.525                                    # 설치비는 정격으로
BIPV_MWH = 1_153.0                                           # 연간 발전량(발표자료 p.57)
BIPV_EQ_KW = 902.54                                          # 옥상PV 등가용량 — 발전량 환산에만
BIPV_PRICE = (4_000_000.0, 5_000_000.0, 6_000_000.0)
BIPV_OM = {"15,000": 15_000.0, "27,196": 27_196.0, "27,196x1.5": 27_196.0 * 1.5}

# ---- ESS ----
ESS_PCS = 249_000.0                                          # 원/kW
ESS_CELL = 100_000.0                                         # 원/kWh
ESS_BOS = 300_000.0                                          # 원/kWh (기준: 용량 비례)
ESS_BOS_ALT_KW = ESS_BOS * 5 + ESS_PCS                       # 1,749,000원/kW (대체: BOS 를 출력 비례로)
ESS_P_MAX = 20_000.0                                         # PCS 출력 상한
ESS_REPL = 100_000.0                                         # 배터리 교체비 원/kWh (셀만)
ESS_REPL_YEARS = (10, 20)
ESS_RESID = 0.5                                              # 25년차 잔존 = 20년차 교체분 5/10
ESS_OM = 10_000.0                                            # 원/kWh·년
C_DEG = 0.0                                                  # 기준 0, 비교 10
C_DEG_OLD = 10.0
SELF_DIS_OLD = 0.05 / (30 * 96)                              # 5%/월 — 비교(새 모델 기준 0)
RTE = 0.83
ETA = RTE ** 0.5
SOC_MIN, SOC_MAX, SOC_INIT = 0.20, 0.90, 0.50
N_EFC = 5_000                                                # 사이클 수명 점검용(10년 교체 가정의 유효성)
# PCS: 기준은 교체 없음 — PNNL ESGC 2022 Table 6.5 는 배터리 기술에서 20~25년 동안 DC 저장블록만
# 교체·증설하고 전력설비 교체를 두지 않는다. 민감도는 같은 보고서 4.1.2.1 의 '전력설비 대정비 =
# 전력설비 비용의 30%, 10년마다'(Austin 2021; Vartanian 2021)를 10·20년차에 건다.
PCS_REFURB_FRAC, PCS_REFURB_YEARS = 0.30, (10, 20)

# ---- 보조금 (C-2) ----
SUB_RATE = 0.70
SUB_CAP = 13_000_000_000.0                                   # 캠퍼스 분야 전체 예산 — 개별 보장액 아님(가상 한도)
ESS_ELIG_CAP = 586_000.0                                     # 원/kWh 적격 인정상한
# 공고: '정부지원 규모 = 총사업비의 70% 이내'. 항목별 배분 규칙은 없다. 원장의 'PV 적격분 먼저, ESS 잔액'
# 순서로 한도 하나를 나눠 쓰면 총 국비 = min(0.7 x 적격 총사업비, 한도) 로 순서와 무관하게 같다.
# 코드는 PV -> 플랫폼 -> ESS 순으로 배분하지만 결과는 이 식과 동치다.

# ---- MG 플랫폼 ----
PLATFORM_LEVELS = (0.0, 5e8, 10e8, 20e8)                     # 0 = 진단용(무료라는 뜻 아님)
PLATFORM_BASE = 10e8
# 유지관리비 = 구축비 x SW 비중 x 요율. 원장: SW 개발비 부분의 10~15%(SW사업 대가산정 가이드), 총액 단순곱 금지.
# SW 비중 25%: 한국BEMS협회 — 계측기 등 하드웨어 비용이 소프트웨어의 3~4배(KHARN) -> SW 20~25%의 상단.
# AI MG 플랫폼은 예측·최적화 기능 때문에 BEMS 보다 SW 비중이 클 수 있다 -> 1억원/년 = NPV -14.8억원 환산 병기.
PLATFORM_SW_SHARE = 0.25
PLATFORM_OM_RATE = {"10%": 0.10, "15%": 0.15}

# ---- 공고 제약: 경부하기 대책기간 ESS 충전 의무 ----
# 시간대는 공고 p.2('25 가을 9~16시, '26 봄 10~16시). 날짜는 정부 발표:
#   '25 가을 9/20~11/16 (58일), '26 봄 2/28~6/14 (107일, 기후부 2026.2.26 발표)
MANDATE = [("2025-09-20", "2025-11-16", 9, 16), ("2026-02-28", "2026-06-14", 10, 16)]
# 충전 '의무량'은 공고·정부 발표 어디에도 정량 기준이 없다('충전시간 조정 운영실적 제출'만 요구).
# 그래서 level 0 / 0.5 / 1.0 을 범위로 보고, 기준은 가장 보수적인 1.0(대책일마다 SOC 가용대역 전폭 충전
# = 하루 충전을 대책 시간대로 옮긴다는 해석)으로 둔다. 공고상 의무는 사후관리 5년이나 25년 전체에 건다.
MANDATE_LEVEL = 1.0

# ---- 요금용 공휴일: 공급약관 별표3 3호 '관공서의 공휴일에 관한 규정'의 공휴일(임시공휴일 제외) ----
# data.csv holiday 는 캠퍼스 저활동일로 근로자의날을 포함하므로 요금 계산에서는 뺀다.
NON_OFFICIAL_HOLIDAYS = ("2026-05-01",)

# =====================================================================
# 2. 입력 — 회귀 present 부하(ess_pv_pipeline.ipynb 와 같은 스펙), PV 단위출력, 캘린더
# =====================================================================


def build_inputs():
    import statsmodels.formula.api as smf

    d = pd.read_csv(os.path.join(ROOT, "data", "data_long.csv"), parse_dates=["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d["lnusage"] = np.log(d["usage_kwh"])
    d["temp_lag1"] = d["temp_c"].shift(1)
    d["temp_lag2"] = d["temp_c"].shift(2)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["humidity_bin"] = pd.qcut(d["humidity_pct"], 6, labels=False)
    est = d.dropna(subset=["lnusage", "temp_c", "temp_lag1", "temp_lag2",
                           "humidity_pct", "precip_mm"]).copy()
    formula = ("lnusage ~ cr(temp_c, df=7) + cr(temp_lag1, df=7) + cr(temp_lag2, df=7) "
               "+ C(humidity_bin) + precip_mm + holiday + vacation + exam_period "
               "+ C(day_of_week) + C(month) + C(year)"
               " + heat_wave_advisory + heat_wave_warning + tropical_night + heavy_snow_advisory")
    res = smf.ols(formula, data=est).fit(cov_type="HAC", cov_kwds={"maxlags": 15})
    smear = np.exp(res.resid).mean()

    d15 = pd.read_csv(os.path.join(ROOT, "data", "data.csv"), parse_dates=["date"])
    d15["temp_c"] = d15["temp_c"].interpolate()
    d15["slot"] = d15.groupby("date").cumcount()                       # 0~95, 구간 시작 = slot x 15분
    d15["shape"] = d15["usage_kwh"] / d15.groupby("date")["usage_kwh"].transform("sum")
    cal = d15[["date"]].drop_duplicates().merge(est, on="date", how="left")
    cal["daily_kwh"] = np.exp(res.predict(cal)) * smear
    d15 = d15.merge(cal[["date", "daily_kwh"]], on="date", how="left")
    L = (d15["shape"] * d15["daily_kwh"] * 4).to_numpy()                # kW

    G = (d15["solar_rad_15min_mj_m2"] * 1e6 / 900).to_numpy()
    t_cell = d15["temp_c"].to_numpy() + (NOCT - 20) / 800 * G
    p_unit = ((G / G_STC) * (1 + GAMMA * (t_cell - T_STC)) * ETA_SYS).clip(min=0)

    dates = d15["date"]
    days = sorted(dates.unique())
    win = []
    for d0, d1, h0, h1 in MANDATE:
        for di, dd in enumerate(days):
            if pd.Timestamp(d0) <= dd <= pd.Timestamp(d1):
                win.append((di * 96 + h0 * 4, di * 96 + h1 * 4 - 1))

    t_hol = d15["holiday"].to_numpy().copy()
    t_hol[dates.isin(pd.to_datetime(list(NON_OFFICIAL_HOLIDAYS))).to_numpy()] = 0
    np.savez(INPUTS, L=L, L_meas=d15["usage_kwh"].to_numpy() * 4, p_unit=p_unit,
             day=(dates - pd.Timestamp("1970-01-01")).dt.days.to_numpy(),
             slot=d15["slot"].to_numpy(), month=dates.dt.month.to_numpy(),
             dow=dates.dt.dayofweek.to_numpy(), holiday=t_hol,
             win=np.array(win, dtype=int))
    print(f"입력 생성: {len(L):,}구간, 회귀 present {L.sum() * DT_H / 1e3:,.0f}MWh / "
          f"실측 {d15['usage_kwh'].sum() / 1e3:,.0f}MWh, 의무충전 {len(win)}개 구간")


_INP = None


def inputs():
    global _INP
    if _INP is None:
        if not os.path.exists(INPUTS):
            build_inputs()
        z = np.load(INPUTS)
        _INP = {k: z[k] for k in z.files}
        _INP["date"] = pd.to_datetime(_INP["day"], unit="D")
    return _INP


# =====================================================================
# 3. 요금 — 계시별 시간대, 월별 요금적용전력
# =====================================================================


def season_of(month):
    month = np.asarray(month)
    return np.where(np.isin(month, (6, 7, 8)), "summer",
                    np.where(np.isin(month, (11, 12, 1, 2)), "winter", "spring_fall"))


def bands(inp, grid="split"):
    """0=경부하, 1=중간, 2=최대. 구간 시작시각 기준.

    grid: 'split' = 2026-06-01 전 개편 전 / 이후 개편 후 (원장 기준)
          'new'   = 1년 전체 개편 후 (사업기간 2027~ 의 실제 조건)
          'old'   = 1년 전체 개편 전
    개편 전 (봄·여름·가을 평일): 경부하 22~08, 최대 11~12·13~18, 나머지 중간
    개편 후 (봄·여름·가을 평일): 11~12·13~15 최대->중간, 18~21 중간->최대 => 최대 15~21
    겨울(11~2월): 개편 전후 동일 — 최대 09~12·16~19
    토요일: 최대 -> 중간.  일요일·공휴일: 종일 경부하
    """
    h = inp["slot"] / 4.0
    winter = np.isin(inp["month"], (11, 12, 1, 2))
    if grid == "split":
        new = np.asarray(inp["date"] >= REFORM)
    elif grid == "new":
        new = np.ones(len(h), bool)
    else:
        new = np.zeros(len(h), bool)
    off = (h >= 22) | (h < 8)
    pk_w = ((h >= 9) & (h < 12)) | ((h >= 16) & (h < 19))
    pk_old = ((h >= 11) & (h < 12)) | ((h >= 13) & (h < 18))
    pk_new = (h >= 15) & (h < 21)
    pk = np.where(winter, pk_w, np.where(new, pk_new, pk_old)) & ~off
    b = np.where(off, 0, np.where(pk, 2, 1))
    sat = (inp["dow"] == 5) & (inp["holiday"] == 0)
    b = np.where(sat & (b == 2), 1, b)
    b = np.where(inp["holiday"] == 1, 0, b)              # data.csv holiday = 공휴일·대체공휴일·일요일
    return b


def bands_legacy(inp):
    """종전 코드(ess_pv_pipeline / 조원 재현코드)의 시간대 — 검증 비교용.
    라벨(구간 종료) 시각을 시작시각처럼 썼고, 주말 규칙이 없다."""
    t = ((inp["slot"] + 1) / 4.0) % 24
    winter = np.isin(inp["month"], (11, 12, 1, 2))
    off = (t >= 23) | (t < 9)
    pk_w = ((t >= 10) & (t < 12)) | ((t >= 17) & (t < 20)) | ((t >= 22) & (t < 23))
    pk_s = ((t >= 10) & (t < 12)) | ((t >= 13) & (t < 17))
    pk = np.where(winter, pk_w, pk_s)
    return np.where(off, 0, np.where(pk, 2, 1))


def tariff(inp, grid="split"):
    """(전력량 단가 원/kWh, 요금적용전력 대상 구간 마스크).

    공급약관 별표3 2호: 요금적용전력은 중간·최대부하 시간대의 최대수요전력만 대상으로 한다.
    1호에 따라 공휴일은 최대수요전력도 경부하로 계량되므로 같이 빠진다. 토요일은 사용전력량만
    중간부하로 옮기고 최대수요전력은 그대로 대상이다. 'legacy'(종전 코드 재현)는 전 구간.
    """
    if grid == "legacy":
        b = bands_legacy(inp)
        mask = np.ones(len(b), bool)
    else:
        b = bands(inp, grid)
        mask = b != 0              # 경부하(22~08시)와 공휴일 종일 제외. 토요일 최대->중간은 b=1 이라 포함
    s = season_of(inp["month"])
    tab = {k: np.array([v["off"], v["mid"], v["peak"]]) for k, v in RATE.items()}
    out = np.empty(len(b))
    for k, arr in tab.items():
        m = s == k
        out[m] = arr[b[m]]
    return out, mask


def tou_price(inp, grid="split"):
    return tariff(inp, grid)[0]


def billing_demand(grid_kw, month, demand="monthly", mask=None):
    """월별 요금적용전력(kW) 12개. 'monthly' = 약관 제68조 정상상태, 'x12' = 연중 최대 x 12."""
    if mask is None:
        mask = np.ones(len(grid_kw), bool)
    pk = np.array([grid_kw[(month == m) & mask].max() if np.any((month == m) & mask) else 0.0
                   for m in range(1, 13)])
    if demand == "x12":
        return np.full(12, max(pk.max(), DEMAND_FLOOR)), pk
    ps = max(DEMAND_FLOOR, max(pk[m - 1] for m in S_MONTHS))
    bill = np.array([ps if m in S_MONTHS else max(pk[m - 1], ps) for m in range(1, 13)])
    return bill, pk


def bill_of(grid_kw, price, month, demand="monthly", mask=None):
    kwh = grid_kw.sum() * DT_H
    energy = float((price * grid_kw).sum() * DT_H)
    bd, pk = billing_demand(grid_kw, month, demand, mask)
    demand_c = float(C_PEAK * bd.sum())
    base = energy + (CLIMATE_WON + FUEL_WON) * kwh + demand_c
    return dict(전력량요금=energy, 기후환경요금=CLIMATE_WON * kwh, 연료비조정요금=FUEL_WON * kwh,
                기본요금=demand_c, 부가세=VAT * base, 전력기금=FUND * base, 청구액=base * TAX_F,
                수전kWh=float(kwh), 연간피크kW=float(grid_kw.max()), 적용전력합kW=float(bd.sum()))


# =====================================================================
# 4. 디스패치·설비 LP
# =====================================================================


def ess_cost_terms(hours, ess_cost="base", repl=ESS_REPL, pcs_repl=False):
    """출력 1kW 당 (초기투자, 초기 적격투자, 등가연간 후속비용) — 용량 = hours x 출력."""
    if ess_cost == "base":
        init = ESS_PCS + hours * (ESS_CELL + ESS_BOS)
    else:                                                    # BOS 출력비례 대체식
        init = ESS_BOS_ALT_KW + hours * ESS_CELL
    elig = min(init, ESS_ELIG_CAP * hours)
    later_pv = hours * (repl * sum(DF[y] for y in ESS_REPL_YEARS) - ESS_RESID * repl * DF[YEARS])
    if pcs_repl:
        later_pv += ESS_PCS * PCS_REFURB_FRAC * sum(DF[y] for y in PCS_REFURB_YEARS)
    eac_later = CRF * later_pv + hours * ESS_OM
    return init, elig, eac_later


def subsidy_split(c_pv, platform, plat_eligible=True, sub_cap=SUB_CAP):
    """C-2 국비: PV 적격분 -> 플랫폼 -> ESS 잔액 순서. (PV 보조, 플랫폼 보조, ESS 에 남은 한도)."""
    s_pv = min(SUB_RATE * c_pv * (PV_PRICE["C-2"] + PV_DIAG), sub_cap)
    s_pl = min(SUB_RATE * platform, sub_cap - s_pv) if plat_eligible else 0.0
    return s_pv, s_pl, max(0.0, sub_cap - s_pv - s_pl)


def solve(c_pv, hours, scen, s=S_REP, p_fix=None, basis="total", tou="split", demand="monthly",
          co2=CO2, c_deg=C_DEG, self_dis=0.0, mandate=None, ess_cost="base", repl=ESS_REPL,
          pcs_repl=False, platform=PLATFORM_BASE, plat_eligible=True, sub_cap=SUB_CAP):
    """한 해 운전 + ESS 출력 결정. s = PV 출력계수(열화). p_fix 가 있으면 출력 고정(연차 평가용).

    basis 'self' = 학교 부담(C-2 국비 차감), 'total' = 총사업비.
    mandate = None 이면 의무충전 없음, 숫자면 그 level 로 의무충전.
    """
    inp = inputs()
    L, pu, month = inp["L"], inp["p_unit"], inp["month"]
    T = len(L)
    pv = s * c_pv * pu
    price, dmask = tariff(inp, tou)
    if p_fix is not None and p_fix < 1e-6:                 # ESS 없음 -> 결정할 것이 없다(LP 불필요)
        g = np.maximum(L - pv, 0.0)
        out = bill_of(g, price, month, demand, dmask)
        out.update(P_ess_kW=0.0, ESS_kWh=0.0, Z=0.0, 배출권비용=out["수전kWh"] / 1000 * co2 * CARBON_PRICE,
                   청구액_x12=bill_of(g, price, month, "x12", dmask)["청구액"], PV발전kWh=float(pv.sum() * DT_H),
                   출력제한kWh=float(np.maximum(pv - L, 0).sum() * DT_H), EFC_년=0.0, 동시충방전=0,
                   목적함수=np.nan, LP초=0.0)
        return out
    S = set(S_MONTHS) if demand == "monthly" else set(range(1, 13))
    nonS = [m for m in range(1, 13) if m not in S]

    o_ch, o_dis, o_g, o_cu, o_E = 0, T, 2 * T, 3 * T, 4 * T
    o_Pm = 5 * T + 1
    o_PS = o_Pm + 12
    o_B = o_PS + 1
    o_P = o_B + len(nonS)
    o_Z = o_P + 1
    N = o_Z + 1
    t = np.arange(T)

    init, elig, eac_later = ess_cost_terms(hours, ess_cost, repl, pcs_repl)
    c = np.zeros(N)
    c[o_g:o_g + T] = DT_H * ((price + CLIMATE_WON + FUEL_WON) * TAX_F + co2 / 1000 * CARBON_PRICE)
    c[o_ch:o_ch + T] = DT_H * c_deg
    c[o_dis:o_dis + T] = DT_H * c_deg
    c[o_PS] = C_PEAK * TAX_F * len(S)
    c[o_B:o_B + len(nonS)] = C_PEAK * TAX_F
    if basis == "self":
        c[o_Z] = CRF
        c[o_P] = eac_later
    else:
        c[o_P] = CRF * init + eac_later

    # ---- 등식: 전력수지 / SOC 동역학 / 연간 순환 ----
    keep = 1.0 - self_dis
    r = np.concatenate([t, t, t, t, T + t, T + t, T + t, T + t, [2 * T, 2 * T, 2 * T + 1, 2 * T + 1]])
    cc = np.concatenate([o_g + t, o_cu + t, o_dis + t, o_ch + t,
                         o_E + t + 1, o_E + t, o_ch + t, o_dis + t,
                         [o_E, o_P, o_E + T, o_P]])
    v = np.concatenate([np.ones(T), -np.ones(T), np.ones(T), -np.ones(T),
                        np.ones(T), np.full(T, -keep), np.full(T, -ETA * DT_H), np.full(T, DT_H / ETA),
                        [1.0, -SOC_INIT * hours, 1.0, -SOC_INIT * hours]])
    A_eq = sp.csr_matrix((v, (r, cc)), shape=(2 * T + 2, N))
    b_eq = np.concatenate([L - pv, np.zeros(T), [0.0, 0.0]])

    # ---- 부등식 ----
    R, C, V, b = [], [], [], []
    n = 0

    def blk(rows_local, cols, vals, rhs):
        nonlocal n
        R.append(rows_local + n); C.append(cols); V.append(vals); b.append(rhs)
        n += int(rows_local.max()) + 1

    td = t[dmask]                                                                     # 요금적용전력 대상 구간만
    jd = np.arange(len(td))
    blk(np.concatenate([jd, jd]), np.concatenate([o_g + td, o_Pm + month[td] - 1]),
        np.concatenate([np.ones(len(td)), -np.ones(len(td))]), np.zeros(len(td)))   # P_grid <= 월피크
    blk(np.concatenate([t, t]), np.concatenate([o_ch + t, np.full(T, o_P)]),
        np.concatenate([np.ones(T), -np.ones(T)]), np.zeros(T))                       # 충전 <= 출력
    blk(np.concatenate([t, t]), np.concatenate([o_dis + t, np.full(T, o_P)]),
        np.concatenate([np.ones(T), -np.ones(T)]), np.zeros(T))                       # 방전 <= 출력
    k = np.arange(1, T)
    m_ = np.arange(T - 1)
    blk(np.concatenate([m_, m_]), np.concatenate([o_E + k, np.full(T - 1, o_P)]),
        np.concatenate([np.ones(T - 1), np.full(T - 1, -SOC_MAX * hours)]), np.zeros(T - 1))
    blk(np.concatenate([m_, m_]), np.concatenate([o_E + k, np.full(T - 1, o_P)]),
        np.concatenate([-np.ones(T - 1), np.full(T - 1, SOC_MIN * hours)]), np.zeros(T - 1))
    Sl = sorted(S)
    i = np.arange(len(Sl))
    blk(np.concatenate([i, i]), np.concatenate([o_Pm + np.array(Sl) - 1, np.full(len(Sl), o_PS)]),
        np.concatenate([np.ones(len(Sl)), -np.ones(len(Sl))]), np.zeros(len(Sl)))     # 지정월 피크 <= P_S
    if nonS:
        j = np.arange(len(nonS))
        blk(np.concatenate([j, j]), np.concatenate([o_Pm + np.array(nonS) - 1, o_B + j]),
            np.concatenate([np.ones(len(nonS)), -np.ones(len(nonS))]), np.zeros(len(nonS)))   # 당월 <= B
        blk(np.concatenate([j, j]), np.concatenate([np.full(len(nonS), o_PS), o_B + j]),
            np.concatenate([np.ones(len(nonS)), -np.ones(len(nonS))]), np.zeros(len(nonS)))   # P_S <= B
    if mandate is not None and mandate > 0:
        w = inp["win"]
        q = np.arange(len(w))
        blk(np.concatenate([q, q, q]),
            np.concatenate([o_E + w[:, 0], o_E + w[:, 1] + 1, np.full(len(w), o_P)]),
            np.concatenate([np.ones(len(w)), -np.ones(len(w)),
                            np.full(len(w), mandate * (SOC_MAX - SOC_MIN) * hours)]),
            np.zeros(len(w)))                                                        # 구간 순충전 >= need
    if basis == "self":
        _, _, rem = subsidy_split(c_pv, platform, plat_eligible, sub_cap)
        blk(np.array([0, 0]), np.array([o_P, o_Z]), np.array([init - SUB_RATE * elig, -1.0]), np.array([0.0]))
        blk(np.array([0, 0]), np.array([o_P, o_Z]), np.array([init, -1.0]), np.array([rem]))
    A_ub = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, N))
    b_ub = np.concatenate(b)

    lb = np.zeros(N)
    ub = np.full(N, np.inf)
    ub[o_g:o_g + T] = CONTRACT_KW
    ub[o_cu:o_cu + T] = pv
    if mandate is not None:
        for t0, t1 in inp["win"]:
            ub[o_dis + t0:o_dis + t1 + 1] = 0.0                                       # 대책기간 방전 금지
    lb[o_PS] = DEMAND_FLOOR
    if p_fix is None:
        ub[o_P] = ESS_P_MAX
    else:
        lb[o_P] = ub[o_P] = p_fix
    if basis != "self":
        ub[o_Z] = 0.0

    t0_ = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP 실패({scen}, PV {c_pv:.0f}kW, {hours}h): {res.message}")
    x = res.x
    g, ch, dis = x[o_g:o_g + T], x[o_ch:o_ch + T], x[o_dis:o_dis + T]
    P = max(0.0, float(x[o_P]))
    if P < 1e-3:
        P = 0.0
    kwh_cap = hours * P
    out = bill_of(g, price, month, demand, dmask)
    out_x12 = bill_of(g, price, month, "x12", dmask)
    thr = float((ch + dis).sum() * DT_H)
    out.update(
        P_ess_kW=P, ESS_kWh=kwh_cap, Z=float(x[o_Z]),
        배출권비용=out["수전kWh"] / 1000 * co2 * CARBON_PRICE,
        청구액_x12=out_x12["청구액"],
        PV발전kWh=float(pv.sum() * DT_H), 출력제한kWh=float(x[o_cu:o_cu + T].sum() * DT_H),
        EFC_년=thr / (2 * kwh_cap) if kwh_cap > 1 else 0.0,
        동시충방전=int(np.sum((ch > 1e-3) & (dis > 1e-3))),
        목적함수=float(res.fun), LP초=time.time() - t0_)
    return out


# =====================================================================
# 5. 현금흐름 · NPV
# =====================================================================
_CURVE_KEYS = ("청구액", "수전kWh", "전력량요금", "기본요금")


def curve(points):
    """[(s, out), ...] -> s 의 함수(키별 2차 보간). 점이 1개면 상수."""
    if len(points) == 1:
        o = points[0][1]
        return lambda key, s: o[key]
    xs = np.array([p[0] for p in points])
    coef = {k: np.polyfit(xs, [p[1][k] for p in points], len(points) - 1) for k in _CURVE_KEYS}
    return lambda key, s: float(np.polyval(coef[key], s))


def cashflow(design, f, base, *, pv_price, pv_om, platform, plat_om_rate, subsidy=True,
             plat_eligible=True, sub_cap=SUB_CAP, repl=ESS_REPL, pcs_repl=False, ess_cost="base",
             co2=CO2, carbon_price=CARBON_PRICE):
    """연차별 현금흐름(원) -> NPV 등. subsidy=True 는 C-2 학교 관점(국비 차감)."""
    c_pv, P, h = design["C_PV_kW"], design["P_ess_kW"], design["hours"]
    kwh = h * P
    init, elig, _ = ess_cost_terms(h, ess_cost, repl, pcs_repl)
    capex_ess, capex_pv = P * init, c_pv * (pv_price + PV_DIAG)
    capex0 = capex_ess + capex_pv + platform
    sub = 0.0
    if subsidy:
        s_pv, s_pl, rem = subsidy_split(c_pv, platform, plat_eligible, sub_cap)
        sub = s_pv + s_pl + min(SUB_RATE * P * elig, rem)

    y = np.arange(1, YEARS + 1)
    sy = PV_F
    bill = np.array([f("청구액", s) for s in sy])
    grid_kwh = np.array([f("수전kWh", s) for s in sy])
    elec = base["청구액"] - bill
    tco2 = (base["수전kWh"] - grid_kwh) / 1000 * co2
    carb = tco2 * carbon_price
    om_pv = np.full(YEARS, c_pv * pv_om)
    om_ess = np.full(YEARS, kwh * ESS_OM)
    om_pl = np.full(YEARS, platform * PLATFORM_SW_SHARE * plat_om_rate)
    repl_cf = np.zeros(YEARS)
    for yr in ESS_REPL_YEARS:
        repl_cf[yr - 1] += kwh * repl
    resid = np.zeros(YEARS)
    resid[-1] = ESS_RESID * kwh * repl
    if pcs_repl:
        for yr in PCS_REFURB_YEARS:
            repl_cf[yr - 1] += P * ESS_PCS * PCS_REFURB_FRAC
    cf = elec + carb - om_pv - om_ess - om_pl - repl_cf + resid
    d = DF[1:]
    npv = -capex0 + sub + float((cf * d).sum())
    npv_elec = npv - float((carb * d).sum())
    cum = np.concatenate([[-capex0 + sub], -capex0 + sub + np.cumsum(cf)])
    dcum = np.concatenate([[-capex0 + sub], -capex0 + sub + np.cumsum(cf * d)])

    def payback(arr):
        k = np.where(arr >= 0)[0]
        if len(k) == 0:
            return np.nan
        k = k[0]
        if k == 0:
            return 0.0
        return (k - 1) + (-arr[k - 1]) / (arr[k] - arr[k - 1])

    def irr():
        flows = np.concatenate([[-capex0 + sub], cf])
        if flows[0] >= 0:
            return np.nan
        lo, hi = -0.9, 5.0
        npv_at = lambda r_: float((flows / (1 + r_) ** np.arange(len(flows))).sum())
        if npv_at(lo) * npv_at(hi) > 0:
            return np.nan
        for _ in range(200):
            mid = (lo + hi) / 2
            if npv_at(lo) * npv_at(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    scc = {k: float((tco2 * v * d).sum()) for k, v in SCC.items()}
    return dict(초기투자=capex0, 국비=sub, 학교초기부담=capex0 - sub,
                PV_ESS투자=capex_ess, PV_PV투자=capex_pv,
                PV_전기요금절감=float((elec * d).sum()), PV_배출권절감=float((carb * d).sum()),
                PV_OM=float(((om_pv + om_ess + om_pl) * d).sum()), PV_교체=float((repl_cf * d).sum()),
                PV_잔존=float((resid * d).sum()),
                NPV=npv, NPV_전기요금만=npv_elec, 회수기간=payback(cum), 할인회수기간=payback(dcum),
                IRR=irr(), 절감_1년차_전기요금=float(elec[0]), 절감_1년차_배출권=float(carb[0]),
                감축_1년차_tCO2=float(tco2[0]), 감축_25년_tCO2=float(tco2.sum()), **scc)


# =====================================================================
# 6. 실행
# =====================================================================


def run_design(task):
    """설비 결정(S_REP) + 연차 평가 2점(1년차, 25년차). 반환: 설계·3점 운영결과."""
    kw = dict(task.get("kw", {}))
    scen, frac, h = task["scen"], task["frac"], task["hours"]
    c_pv = PV_MAX_KW * frac
    basis = "self" if scen == "C-2" and task.get("perspective", "self") == "self" else "total"
    if scen in ("C-1", "C-2") and "mandate" not in kw:
        kw["mandate"] = MANDATE_LEVEL
    t0 = time.time()
    o_rep = solve(c_pv, h, scen, s=S_REP, basis=basis, **kw)
    pts = [(S_REP, o_rep)]
    if c_pv > 0:
        ev = {k: v for k, v in kw.items() if k not in ("platform", "plat_eligible", "sub_cap", "repl", "pcs_repl", "ess_cost")}
        for s in (1.0, S_END):
            pts.append((s, solve(c_pv, h, scen, s=s, p_fix=o_rep["P_ess_kW"], **ev)))
    pts.sort(key=lambda p: p[0])
    print(f"  [{task['tag']}] {scen} PV {frac:.0%} {h}h -> ESS {o_rep['P_ess_kW']:,.0f}kW "
          f"({time.time() - t0:.0f}초)", flush=True)
    return dict(task=task, C_PV_kW=c_pv, P_ess_kW=o_rep["P_ess_kW"], hours=h, points=pts)


def baseline(tou="split", demand="monthly"):
    inp = inputs()
    price, mask = tariff(inp, tou)
    out = bill_of(inp["L"], price, inp["month"], demand, mask)
    out["배출권비용"] = out["수전kWh"] / 1000 * CO2 * CARBON_PRICE
    return out


def validate():
    inp = inputs()
    month = inp["month"]
    actual = 13_885_735_690
    print("실측 부하로 청구액 재현 — 원장 실제 청구액 12개월(2025.09~2026.08) 13,885,735,690원\n")
    rows = []
    allmask = np.ones(len(month), bool)
    for tag, grid, dem, taxf, full in (
            ("종전 코드(라벨시각·주말규칙 없음, x12, 1.132)", "legacy", "x12", 1.132, True),
            ("시간대만 교정(x12, 1.132)", "split", "x12", 1.132, True),
            ("+ 월별 적용전력 + 1.127, 수요는 전 구간", "split", "monthly", TAX_F, True),
            ("+ 별표3 2호: 중간·최대부하 수요만 (새 모델)", "split", "monthly", TAX_F, False),
            ("  참고: 1년 전체 개편 전 시간대", "old", "monthly", TAX_F, False),
            ("  참고: 1년 전체 개편 후 시간대", "new", "monthly", TAX_F, False)):
        price, mask = tariff(inp, grid)
        o = bill_of(inp["L_meas"], price, month, dem, allmask if full else mask)
        total = o["청구액"] / TAX_F * taxf
        rows.append(dict(방식=tag, 청구액_억=total / 1e8, 오차pct=(total / actual - 1) * 100,
                         전력량요금_억=o["전력량요금"] / 1e8, 기본요금_억=o["기본요금"] / 1e8))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    b = bands(inp, "split")
    share = pd.Series(inp["L_meas"]).groupby(b).sum() / inp["L_meas"].sum() * 100
    print(f"\n새 시간대 수전량 비중: 경부하 {share.get(0, 0):.1f}% / 중간 {share.get(1, 0):.1f}% / "
          f"최대 {share.get(2, 0):.1f}%")
    price, mask = tariff(inp, "split")
    bd, pk = billing_demand(inp["L_meas"], month, "monthly", mask)
    _, pk_all = billing_demand(inp["L_meas"], month, "monthly", allmask)
    print("월별 피크, 중간·최대부하만(kW):", " ".join(f"{m}월 {pk[m - 1]:,.0f}" for m in range(1, 13)))
    print("월별 피크, 전 구간(kW):       ", " ".join(f"{m}월 {pk_all[m - 1]:,.0f}" for m in range(1, 13)))
    print("월별 요금적용전력(kW):", " ".join(f"{m}월 {bd[m - 1]:,.0f}" for m in range(1, 13)))
    reg = bill_of(inp["L"], price, month, "monthly", mask)
    meas = bill_of(inp["L_meas"], price, month, "monthly", mask)
    print(f"\n회귀 present 부하 청구액 {reg['청구액'] / 1e8:,.3f}억 (실측 대비 "
          f"{(reg['청구액'] / meas['청구액'] - 1) * 100:+.2f}%)")
    return df


def quick():
    t0 = time.time()
    inputs()
    print(f"입력 {time.time() - t0:.1f}초")
    o = solve(PV_MAX_KW, 5, "C-1", mandate=MANDATE_LEVEL)
    print(f"C-1 PV100% 5h: ESS {o['P_ess_kW']:,.1f}kW / {o['ESS_kWh']:,.0f}kWh, LP {o['LP초']:.1f}초, "
          f"청구액 {o['청구액'] / 1e8:,.3f}억, EFC {o['EFC_년']:.0f}/년, 동시충방전 {o['동시충방전']}")
    try:
        import psutil
        print(f"peak RSS {psutil.Process().memory_info().peak_wset / 2 ** 30:.2f} GB")
    except Exception:
        pass


def main_tasks():
    T = []
    for f in PV_FRACS:
        for h in (1, 2, 3, 4, 5):
            T.append(dict(tag="main", scen="A", frac=f, hours=h))
        T.append(dict(tag="main", scen="C-1", frac=f, hours=5))
        T.append(dict(tag="main", scen="C-2", frac=f, hours=5))
    return T


def sens_tasks(a_h):
    """민감도 — PV 100% 에서. a_h = A 의 기준 duration (본 실행에서 NPV 최대)."""
    T = []
    for h in (1, 2, 3, 4, 5):
        T.append(dict(tag="BOS출력비례", scen="A", frac=1.0, hours=h, kw=dict(ess_cost="alt")))
    for scen, h in (("A", a_h), ("C-1", 5), ("C-2", 5)):
        for tag, kw in (("열화비10원", dict(c_deg=C_DEG_OLD)),
                        ("자기방전5%월", dict(self_dis=SELF_DIS_OLD)),
                        ("교체비15만", dict(repl=150_000.0)),
                        ("교체비20만", dict(repl=200_000.0)),
                        ("PCS정비30%(PNNL)", dict(pcs_repl=True)),
                        ("기본요금x12", dict(demand="x12")),
                        ("시간대_개편전", dict(tou="old")),
                        ("시간대_개편후", dict(tou="new")),
                        ("시간대_종전코드", dict(tou="legacy")),
                        ("배출계수0.45941", dict(co2=CO2_OLD))):
            T.append(dict(tag=tag, scen=scen, frac=1.0, hours=h, kw=kw))
    for scen in ("C-1", "C-2"):
        for lv in (0.0, 0.5):
            T.append(dict(tag=f"의무충전{lv}", scen=scen, frac=1.0, hours=5, kw=dict(mandate=lv)))
    for pl in (0.0, 5e8, 20e8):
        T.append(dict(tag=f"플랫폼{pl / 1e8:.0f}억", scen="C-2", frac=1.0, hours=5, kw=dict(platform=pl)))
    T.append(dict(tag="플랫폼국비제외", scen="C-2", frac=1.0, hours=5, kw=dict(plat_eligible=False)))
    return T


def run_fixed(task):
    """설계 고정 운전만(BIPV 추가, 연차 정확계산 검증용)."""
    pts = [(s, solve(task["c_pv"], task["hours"], task["scen"], s=s, p_fix=task["p_fix"], **task.get("kw", {})))
           for s in task["s_list"]]
    print(f"  [{task['tag']}] {task['scen']} PV {task['c_pv']:,.0f}kW s={task['s_list'][0]:.4f}"
          f"{'...' if len(task['s_list']) > 1 else ''}", flush=True)
    return dict(task=task, points=pts)


def _work(task):
    try:
        return run_fixed(task) if task.get("kind") == "fixed" else run_design(task)
    except Exception as e:                                   # 한 건 실패가 전체를 죽이지 않게
        return dict(task=task, error=repr(e))


def pv_yield():
    return float(inputs()["p_unit"].sum() * DT_H)            # kWh/kW·년 (1년차, 열화 전)


def headline_designs(res_main, a_h):
    """PV 100% 설계 — A 는 최적 duration."""
    out = {}
    for r in res_main:
        t = r["task"]
        if t["frac"] == 1.0 and (t["scen"] != "A" or t["hours"] == a_h):
            out[t["scen"]] = r
    return out


def extra_tasks(designs):
    # 원장 등가용량 902.54kW(= 1,153MWh / 1,277.5h). 옥상 10,305.4kW 와 같은 관례(용량은 발표자료, 발전량은
    # 우리 물리모형)라 우리 모형에서는 연 1,223MWh 가 된다 — 옥상과 같은 +6.1% 비율.
    bipv_eq = BIPV_EQ_KW
    T = []
    for scen, r in designs.items():
        kw = {} if scen == "A" else {"mandate": MANDATE_LEVEL}
        T.append(dict(kind="fixed", tag="BIPV", scen=scen, c_pv=PV_MAX_KW + bipv_eq, hours=r["hours"],
                      p_fix=r["P_ess_kW"], s_list=[S_END, S_REP, 1.0], kw=kw))
    chk = max(designs.values(), key=lambda r: r["P_ess_kW"])   # ESS 가 가장 큰 설계로 보간 검증
    kw = {} if chk["task"]["scen"] == "A" else {"mandate": MANDATE_LEVEL}
    for s in PV_F:
        T.append(dict(kind="fixed", tag="연차정확", scen=chk["task"]["scen"], c_pv=PV_MAX_KW,
                      hours=chk["hours"], p_fix=chk["P_ess_kW"], s_list=[float(s)], kw=kw))
    return T


def run_pool(tasks, workers):
    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(_work, tasks, chunksize=1)
    return [_work(t) for t in tasks]


def evaluate(r, base, **over):
    """설계 1건 -> NPV 결과 행. over 로 사후 민감도(가격·O&M·플랫폼 등)를 바꾼다."""
    task = r["task"]
    kw = task.get("kw", {})
    scen = task["scen"]
    if "tou" in kw or "demand" in kw:          # 요금 체계 민감도는 기준선도 같은 요금으로 청구해야 절감액이 맞다
        base = baseline(kw.get("tou", "split"), kw.get("demand", "monthly"))
    f = curve(r["points"])
    args = dict(pv_price=PV_PRICE[scen], pv_om=PV_OM["high"], platform=kw.get("platform", PLATFORM_BASE),
                plat_om_rate=PLATFORM_OM_RATE["10%"], subsidy=(scen == "C-2"),
                plat_eligible=kw.get("plat_eligible", True), repl=kw.get("repl", ESS_REPL),
                pcs_repl=kw.get("pcs_repl", False), ess_cost=kw.get("ess_cost", "base"),
                co2=kw.get("co2", CO2))
    args.update(over)
    cf = cashflow(r, f, base, **args)
    o1 = dict(r["points"])[1.0] if r["C_PV_kW"] > 0 else r["points"][0][1]
    orep = dict(r["points"])[S_REP]
    row = dict(시나리오=scen, 민감도=task["tag"], PV비율=task["frac"], PV_kW=r["C_PV_kW"],
               duration_h=r["hours"], ESS_kW=r["P_ess_kW"], ESS_kWh=r["P_ess_kW"] * r["hours"],
               청구액_1년차=o1["청구액"], 피크_1년차=o1["연간피크kW"], EFC_년=orep["EFC_년"],
               동시충방전=max(p[1]["동시충방전"] for p in r["points"]),
               출력제한kWh=orep["출력제한kWh"], 청구액_x12_대표=orep["청구액_x12"],
               청구액_월별_대표=orep["청구액"])
    row.update(cf)
    if args["subsidy"]:                         # C-2: 같은 설계를 국비 없이 = 총사업비 관점
        row["NPV_총사업비관점"] = cashflow(r, f, base, **dict(args, subsidy=False))["NPV"]
    return row


def run_all(workers):
    os.makedirs(OUT, exist_ok=True)
    inputs()
    base = baseline()
    print(f"기준선(PV0+ESS0+MG0): 청구액 {base['청구액'] / 1e8:,.3f}억/년, "
          f"배출권 {base['배출권비용'] / 1e8:,.3f}억/년, 피크 {base['연간피크kW']:,.0f}kW")
    print(f"S_REP={S_REP:.4f}, S_END={S_END:.4f}, CRF={CRF:.5f}")

    t0 = time.time()
    res_main = run_pool(main_tasks(), workers)
    errs = [r for r in res_main if "error" in r]
    for e in errs:
        print("실패:", e["task"], e["error"])
    res_main = [r for r in res_main if "error" not in r]
    pd.to_pickle(res_main, os.path.join(OUT, "_raw_main.pkl"))
    print(f"본 실행 {len(res_main)}건 {time.time() - t0:.0f}초")

    main_rows = [evaluate(r, base) for r in res_main]
    main_df = pd.DataFrame(main_rows)
    a100 = main_df[(main_df["시나리오"] == "A") & (main_df["PV비율"] == 1.0)]
    a_h = int(a100.loc[a100["NPV"].idxmax(), "duration_h"])
    print(f"A 의 PV 100% 최적 duration = {a_h}h")

    t0 = time.time()
    res_sens = run_pool(sens_tasks(a_h), workers)
    for e in [r for r in res_sens if "error" in r]:
        print("실패:", e["task"], e["error"])
    res_sens = [r for r in res_sens if "error" not in r]
    pd.to_pickle(res_sens, os.path.join(OUT, "_raw_sens.pkl"))
    print(f"민감도 {len(res_sens)}건 {time.time() - t0:.0f}초")

    t0 = time.time()
    res_extra = run_pool(extra_tasks(headline_designs(res_main, a_h)), workers)
    for e in [r for r in res_extra if "error" in r]:
        print("실패:", e["task"], e["error"])
    res_extra = [r for r in res_extra if "error" not in r]
    pd.to_pickle(res_extra, os.path.join(OUT, "_raw_extra.pkl"))
    print(f"BIPV·연차검증 {len(res_extra)}건 {time.time() - t0:.0f}초")
    pd.to_pickle(dict(base=base, a_h=a_h), os.path.join(OUT, "_raw_meta.pkl"))
    report(res_main, res_sens, res_extra, base, a_h)


def bipv_and_check(res_main, res_extra, base, a_h):
    designs = headline_designs(res_main, a_h)
    inp = inputs()
    pu = inp["p_unit"]
    price = tou_price(inp, "split")
    v_energy = float((pu * (price + CLIMATE_WON + FUEL_WON) * TAX_F).sum() / pu.sum())   # 원/kWh, PV가중
    kwh_lb = BIPV_EQ_KW * pv_yield() * PV_F                  # 두 방법이 같은 발전량을 쓰고 시점만 다르게
    rows = []
    bipv = {r["task"]["scen"]: r for r in res_extra if r["task"]["tag"] == "BIPV"}
    for scen, r in designs.items():
        f_d, f_b = curve(r["points"]), curve(bipv[scen]["points"])
        d_bill = np.array([f_d("청구액", s) - f_b("청구액", s) for s in PV_F])
        d_kwh = np.array([f_d("수전kWh", s) - f_b("수전kWh", s) for s in PV_F])
        for method, elec, kwh in (("옥상 프로파일(피크 기여 포함)", d_bill, d_kwh),
                                  ("연간 발전량 x PV가중 전력량단가(피크 기여 0)", kwh_lb * v_energy, kwh_lb)):
            for p in BIPV_PRICE:
                for omk, om in BIPV_OM.items():
                    carb = kwh / 1000 * CO2 * CARBON_PRICE
                    cf = elec + carb - om * BIPV_RATED_KW
                    rows.append(dict(시나리오=scen, 방법=method, 단가_만원kW=p / 1e4, OM=omk,
                                     BIPV투자_억=p * BIPV_RATED_KW / 1e8,
                                     절감_1년차_억=(elec[0] + carb[0]) / 1e8,
                                     dNPV_억=(-p * BIPV_RATED_KW + float((cf * DF[1:]).sum())) / 1e8,
                                     dNPV_배출권제외_억=(-p * BIPV_RATED_KW
                                                     + float(((elec - om * BIPV_RATED_KW) * DF[1:]).sum())) / 1e8))
    bipv_df = pd.DataFrame(rows)

    ex = sorted((p for r in res_extra if r["task"]["tag"] == "연차정확" for p in r["points"]), key=lambda p: p[0])
    chk = None
    if ex:
        scen = next(r["task"]["scen"] for r in res_extra if r["task"]["tag"] == "연차정확")
        d = designs[scen]
        exact = {float(s): o for s, o in ex}
        f_ex = lambda key, s: exact[float(s)][key]
        args = dict(pv_price=PV_PRICE[scen], pv_om=PV_OM["high"], platform=PLATFORM_BASE,
                    plat_om_rate=PLATFORM_OM_RATE["10%"], subsidy=(scen == "C-2"))
        n_int = cashflow(d, curve(d["points"]), base, **args)["NPV"]
        n_ex = cashflow(d, f_ex, base, **args)["NPV"]
        bill_err = max(abs(curve(d["points"])("청구액", s) - exact[float(s)]["청구액"]) for s in PV_F)
        chk = dict(시나리오=scen, ESS_kW=d["P_ess_kW"], NPV_3점보간_억=n_int / 1e8, NPV_25년정확_억=n_ex / 1e8,
                   차이_억=(n_int - n_ex) / 1e8, 연간청구액_최대보간오차_원=bill_err)
    return bipv_df, chk, v_energy


def report(res_main, res_sens, res_extra, base, a_h):
    """결과 표 -> results/econ/*.csv"""
    main_df = pd.DataFrame([evaluate(r, base) for r in res_main])
    main_df.to_csv(os.path.join(OUT, "main.csv"), index=False, encoding="utf-8-sig")
    sens_df = pd.DataFrame([evaluate(r, base) for r in res_sens])
    sens_df.to_csv(os.path.join(OUT, "sens_rerun.csv"), index=False, encoding="utf-8-sig")

    # 사후 민감도 (설계 고정, 비용만 변경)
    rows = []
    for r in res_main:
        t = r["task"]
        if t["frac"] != 1.0 or (t["scen"] == "A" and t["hours"] != a_h):
            continue
        for tag, over in (("PV O&M 15,000", dict(pv_om=PV_OM["low"])),
                          ("PV O&M 20,000(기준)", {}),
                          ("PV O&M 27,196", dict(pv_om=PV_OM["ext"])),
                          ("플랫폼 O&M 15%", dict(plat_om_rate=PLATFORM_OM_RATE["15%"])),
                          ("배출권 0원(현금편익 미확정)", dict(carbon_price=0.0))):
            rows.append(dict(evaluate(r, base, **over), 사후민감도=tag))
        if t["scen"] == "A":
            rows.append(dict(evaluate(r, base, pv_price=PV_PRICE_A_HIGH), 사후민감도="A 시장단가 150만"))
        for pl in PLATFORM_LEVELS:
            rows.append(dict(evaluate(r, base, platform=pl), 사후민감도=f"플랫폼 {pl / 1e8:.0f}억(설계 고정)"))
    post_df = pd.DataFrame(rows)
    post_df.to_csv(os.path.join(OUT, "sens_posthoc.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([base]).to_csv(os.path.join(OUT, "baseline.csv"), index=False, encoding="utf-8-sig")
    bipv_df, chk, v_energy = bipv_and_check(res_main, res_extra, base, a_h)
    bipv_df.to_csv(os.path.join(OUT, "bipv.csv"), index=False, encoding="utf-8-sig")
    meta = dict(S_REP=S_REP, S_END=S_END, CRF=CRF, AF=AF, TAX_F=TAX_F, A_best_h=a_h,
                PV_yield_kWh_per_kW=pv_yield(), BIPV_eq_kW=BIPV_EQ_KW,
                BIPV_MWh_model=BIPV_EQ_KW * pv_yield() / 1000,
                BIPV_v_energy=v_energy, **({f"check_{k}": v for k, v in chk.items()} if chk else {}))
    pd.Series(meta).to_csv(os.path.join(OUT, "meta.csv"), header=["값"], encoding="utf-8-sig")
    print("저장:", OUT)
    return main_df, sens_df, post_df


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False, buffering=1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.rebuild and os.path.exists(INPUTS):
        os.remove(INPUTS)
    if a.validate:
        validate()
    if a.quick:
        quick()
    if a.all:
        run_all(a.workers)
