# -*- coding: utf-8 -*-
"""
data/15분 수전량 2025.09~.xlsx 의 각 (날짜, 시간) 구간에
data/combined_sorted_weather_data.csv (1분 관측, 지점 119) 를 매칭해
data/data.csv 를 생성한다.

수전량의 '시간'은 구간 종료 시각(interval-end) 라벨이다.
  날짜=2025-09-04, 시간=00:15  ->  (00:00, 00:15] 구간 (1분 관측 00:01~00:15)
  날짜=2026-09-02, 시간=24:00  ->  (23:45, 익일 00:00] 구간

집계 방식 (구간에 속한 최대 15개 1분 관측):
  기온/습도/현지기압/해면기압/풍속        : 구간 평균
  풍향(deg)                              : 벡터(원형) 평균
  일사_15분 / 일조_15분 / 강수량_15분     : 15분 적산
      원자료(일사·일조·누적강수량)는 매일 00:01 리셋되는 '하루 누적값'이므로
      분단위 증가분(리셋으로 인한 음수는 0)을 합산해 해당 15분 값으로 환산한다.
  누적강수량(mm)                         : 구간 종료 시각의 원값(그날 누적)
  특보/주의보 15종                       : 구간 내 1분이라도 발효 시 1
  관측분수                               : 실제 매칭된 1분 관측 수(0~15, 결측 진단용)

캘린더 파생 컬럼 (날짜 기준, 하루 96구간 공통):
  day_of_week   : 요일 영문명 (Monday ~ Sunday)
  weekend       : 토·일이면 1
  holiday       : 대한민국 공휴일(대체공휴일 포함) 또는 일요일이면 1
  vacation      : 방학 기간이면 1  (VACATION_RANGES)
  exam_period   : 시험 기간이면 1  (EXAM_RANGES)

출력(data.csv)의 컬럼명은 COLUMN_RENAME 에 따라 전부 영어(snake_case)로,
인코딩은 BOM 없는 UTF-8 로 저장한다. 원본 한글 헤더 대응은 COLUMN_RENAME 참조.
컬럼 순서: 수전량 9 -> datetime -> 캘린더 5 -> 기상 26 (총 41열).

사용법:  python scripts/merge_weather.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "15분 수전량 2025.09~.xlsx"
WCSV = ROOT / "data" / "combined_sorted_weather_data.csv"
OUT = ROOT / "data" / "data.csv"

FLAG_COLS = ["강풍경보", "강풍주의보", "건조주의보", "대설경보", "대설주의보", "열대야", "태풍경보",
             "태풍주의보", "폭염경보", "폭염주의보", "한파경보", "한파주의보", "호우경보", "호우주의보", "황사경보"]
MEAN_COLS = ["기온(°C)", "습도(%)", "현지기압(hPa)", "해면기압(hPa)", "풍속(m/s)"]
ACC_COLS = [("일사(MJ/m^2)", "일사_15분(MJ/m^2)"),
            ("일조(Sec)", "일조_15분(Sec)"),
            ("누적강수량(mm)", "강수량_15분(mm)")]
WEATHER_ORDER = (["기온(°C)", "습도(%)", "풍향(deg)", "풍속(m/s)", "현지기압(hPa)", "해면기압(hPa)",
                  "일사_15분(MJ/m^2)", "일조_15분(Sec)", "누적강수량(mm)", "강수량_15분(mm)"]
                 + FLAG_COLS + ["관측분수"])

CALENDAR_COLS = ["day_of_week", "weekend", "holiday", "vacation", "exam_period"]

# 방학 / 시험 기간 (양끝 포함). 데이터 범위(2025-09-04~2026-09-02) 밖은 자동 무시.
VACATION_RANGES = [("2025-06-24", "2025-08-31"),
                   ("2025-12-20", "2026-03-01"),
                   ("2026-06-20", "2026-08-30")]
EXAM_RANGES = [("2025-10-06", "2025-10-24"),
               ("2025-12-01", "2025-12-19"),
               ("2026-04-06", "2026-04-24"),
               ("2026-06-01", "2026-06-19")]

# 대한민국 공휴일 (2025-09 ~ 2026-09). `holidays` 0.103 이 반영한 법정 공휴일 + 대체공휴일.
# 주: '근로자의날'은 관공서 공휴일이 아닌 근로기준법상 유급휴일이나 캠퍼스 저활동일로 보아 포함.
#     '제헌절'은 2026년 재지정(공휴일법 개정) 반영.
KR_HOLIDAYS = {
    "2025-10-03": "개천절",
    "2025-10-05": "추석 연휴",
    "2025-10-06": "추석",
    "2025-10-07": "추석 연휴",
    "2025-10-08": "추석 대체공휴일",
    "2025-10-09": "한글날",
    "2025-12-25": "성탄절",
    "2026-01-01": "신정",
    "2026-02-16": "설날 연휴",
    "2026-02-17": "설날",
    "2026-02-18": "설날 연휴",
    "2026-03-01": "삼일절",
    "2026-03-02": "삼일절 대체공휴일",
    "2026-05-01": "근로자의날",
    "2026-05-05": "어린이날",
    "2026-05-24": "부처님오신날",
    "2026-05-25": "부처님오신날 대체공휴일",
    "2026-06-03": "제9회 전국동시지방선거일",
    "2026-06-06": "현충일",
    "2026-07-17": "제헌절",
    "2026-08-15": "광복절",
    "2026-08-17": "광복절 대체공휴일",
}

# 출력 컬럼명: 한글 -> 영어 (인코딩/툴 호환성). 캘린더 컬럼은 이미 영어.
COLUMN_RENAME = {
    "날짜": "date",
    "시간": "time",
    "사용량(kWh)": "usage_kwh",
    "최대수요(kW)": "peak_demand_kw",
    "무효전력_지상(kVarh)": "reactive_lag_kvarh",
    "무효전력_진상(kVarh)": "reactive_lead_kvarh",
    "CO2(tCO2)": "co2_tco2",
    "역률_지상(%)": "power_factor_lag_pct",
    "역률_진상(%)": "power_factor_lead_pct",
    "일시": "datetime",
    "기온(°C)": "temp_c",
    "습도(%)": "humidity_pct",
    "풍향(deg)": "wind_dir_deg",
    "풍속(m/s)": "wind_speed_ms",
    "현지기압(hPa)": "pressure_local_hpa",
    "해면기압(hPa)": "pressure_sea_hpa",
    "일사_15분(MJ/m^2)": "solar_rad_15min_mj_m2",
    "일조_15분(Sec)": "sunshine_15min_sec",
    "누적강수량(mm)": "precip_cumulative_mm",
    "강수량_15분(mm)": "precip_15min_mm",
    "강풍경보": "strong_wind_warning",
    "강풍주의보": "strong_wind_advisory",
    "건조주의보": "dry_advisory",
    "대설경보": "heavy_snow_warning",
    "대설주의보": "heavy_snow_advisory",
    "열대야": "tropical_night",
    "태풍경보": "typhoon_warning",
    "태풍주의보": "typhoon_advisory",
    "폭염경보": "heat_wave_warning",
    "폭염주의보": "heat_wave_advisory",
    "한파경보": "cold_wave_warning",
    "한파주의보": "cold_wave_advisory",
    "호우경보": "heavy_rain_warning",
    "호우주의보": "heavy_rain_advisory",
    "황사경보": "yellow_dust_warning",
    "관측분수": "obs_minutes",
}


def load_power(path: Path) -> pd.DataFrame:
    p = pd.read_excel(path, sheet_name="Sheet1", dtype=str)
    p.columns = [str(c).strip() for c in p.columns]
    date_col, time_col = p.columns[0], p.columns[1]
    for c in p.columns[2:]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    d = pd.to_datetime(p[date_col], format="%Y-%m-%d")
    hh = p[time_col].str.slice(0, 2).astype(int)
    mm = p[time_col].str.slice(3, 5).astype(int)
    p["interval_end"] = d + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
    return p


def load_weather(path: Path, start, end) -> pd.DataFrame:
    w = pd.read_csv(path)
    w.columns = [str(c).strip() for c in w.columns]
    w["일시"] = pd.to_datetime(w["일시"])
    w = w[(w["일시"] > start) & (w["일시"] <= end)].sort_values("일시").reset_index(drop=True)
    w["label"] = w["일시"].dt.ceil("15min")  # 1분 관측 -> 소속 15분 구간(종료 시각)
    return w


def aggregate(w: pd.DataFrame) -> pd.DataFrame:
    g = w.groupby("label")
    agg = pd.DataFrame(index=pd.Index(sorted(w["label"].unique()), name="label"))

    for c in MEAN_COLS:
        agg[c] = g[c].mean()

    rad = np.deg2rad(w["풍향(deg)"])
    w = w.assign(_sin=np.sin(rad), _cos=np.cos(rad))
    gs = w.groupby("label")[["_sin", "_cos"]].mean()
    wd = np.rad2deg(np.arctan2(gs["_sin"], gs["_cos"])) % 360.0
    wd[(gs["_sin"].abs() < 1e-9) & (gs["_cos"].abs() < 1e-9)] = np.nan
    agg["풍향(deg)"] = wd

    for src, dst in ACC_COLS:
        inc = w[src].diff()
        inc[inc < 0] = 0.0
        inc = inc.fillna(0.0)
        s = w.assign(_inc=inc).groupby("label")["_inc"].sum()
        s[g[src].count() == 0] = np.nan  # 구간 전체 결측이면 NaN
        agg[dst] = s

    agg["누적강수량(mm)"] = g["누적강수량(mm)"].last()
    for c in FLAG_COLS:
        agg[c] = g[c].max()
    agg["관측분수"] = g.size()
    return agg[WEATHER_ORDER]


def _in_ranges(d: pd.Series, ranges) -> pd.Series:
    hit = pd.Series(False, index=d.index)
    for lo, hi in ranges:
        hit |= (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    return hit.astype(int)


def add_calendar(merged: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = pd.to_datetime(merged[date_col], format="%Y-%m-%d")
    ymd = d.dt.strftime("%Y-%m-%d")
    merged["day_of_week"] = d.dt.day_name()
    merged["weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    merged["holiday"] = (d.dt.dayofweek.eq(6) | ymd.isin(KR_HOLIDAYS)).astype(int)
    merged["vacation"] = _in_ranges(d, VACATION_RANGES)
    merged["exam_period"] = _in_ranges(d, EXAM_RANGES)
    return merged


def main() -> None:
    p = load_power(XLSX)
    start = p["interval_end"].min() - pd.Timedelta(minutes=15)
    end = p["interval_end"].max()
    print(f"수전량 : {p['interval_end'].min()} ~ {p['interval_end'].max()}  ({len(p)}행)")

    w = load_weather(WCSV, start, end)
    print(f"기상   : {w['일시'].min()} ~ {w['일시'].max()}  ({len(w)}행, 지점 {sorted(w['지점'].unique())})")

    agg = aggregate(w)
    date_col = p.columns[0]
    power_cols = list(p.columns[:-1])  # interval_end 제외

    merged = p.merge(agg, left_on="interval_end", right_index=True, how="left")
    merged["일시"] = merged["interval_end"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged = add_calendar(merged, date_col)
    merged = merged[power_cols + ["일시"] + CALENDAR_COLS + WEATHER_ORDER]

    miss = int(merged["기온(°C)"].isna().sum())
    part = int(((merged["관측분수"] < 15) & (merged["관측분수"] > 0)).sum())
    days = merged.groupby(date_col)[["weekend", "holiday", "vacation", "exam_period"]].first().sum()

    merged = merged.rename(columns=COLUMN_RENAME)
    merged.to_csv(OUT, index=False, encoding="utf-8")

    print(f"저장   : {OUT}  ({len(merged)}행 x {merged.shape[1]}열, 컬럼명 영어)")
    print(f"진단   : 기온 결측 {miss}구간, 부분관측(1~14분) {part}구간")
    print(f"캘린더 : 주말 {int(days['weekend'])}일, 휴일 {int(days['holiday'])}일, "
          f"방학 {int(days['vacation'])}일, 시험 {int(days['exam_period'])}일 "
          f"(공휴일 {sum(1 for k in KR_HOLIDAYS if p[date_col].eq(k).any())}일 포함)")


if __name__ == "__main__":
    main()
