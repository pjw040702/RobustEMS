# -*- coding: utf-8 -*-
"""
data/data_long.csv 생성 — 일 단위(daily) 장기 데이터셋 (2021-09 ~ 2026-09).

입력 (둘 다 data/source/ 안):
  일일 수전량 2021.09~.csv           : 하루 총 사용량(kWh)
  combined_sorted_weather_data.csv  : 1분 기상 관측(지점 119) -> 일 단위로 집계

data/data.csv(15분) 와 같은 형식·컬럼명 규칙(영어 snake_case, BOM 없는 UTF-8,
day_of_week 는 영문명, weekend/holiday/vacation/exam_period 는 0/1).
15분 전용 컬럼은 일 단위에 맞게 이름을 바꾼다:
  solar_rad_15min_mj_m2 -> solar_rad_mj_m2   (일 적산)
  sunshine_15min_sec    -> sunshine_sec      (일 적산, 0~86400)
  precip_15min_mm        -> precip_mm         (일 적산)
  precip_cumulative_mm   -> (없음, 일 단위에선 precip_mm 와 동일)
time / datetime / peak_demand_kw 등 15분·부가 지표는 일일 원본에 없어 제외.

기상 집계 (하루 최대 1440분):
  기온/습도/현지기압/해면기압/풍속 : 일 평균
  풍향(deg)                        : 벡터(원형) 평균
  일사/일조/강수량                 : 그날 마지막(≈23:59) 누적값 = 일 적산.
      원자료는 매일 00:01 리셋되는 하루 누적값이라 마지막 값이 곧 일 합계.
      (분단위 증가분 합은 원자료의 순간 스파이크에 취약해 사용하지 않음)
  특보·주의보 15종                 : 하루 중 1분이라도 발효 시 1
  obs_minutes                     : 실제 매칭된 1분 관측 수(0~1440)

학사일정 (사용자 규칙):
  봄학기 = 3/2 이 포함된 주, 가을학기 = 9/1 이 포함된 주가 1주차.
  단 3/2(또는 9/1)이 주말이면 그 다음 월요일이 1주차 월요일.
  수업 16주 = 1주차 월 ~ 16주차 금 (개강일 + 109일).  그 밖은 방학.
  시험기간 = 6~8주차, 14~16주차 (각 3주, 월~금 경계, 중간 주말 포함).

사용법:  python scripts/build_data_long.py
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "source" / "일일 수전량 2021.09~.csv"
WCSV = ROOT / "data" / "source" / "combined_sorted_weather_data.csv"
OUT = ROOT / "data" / "data_long.csv"

FLAG_SRC = ["강풍경보", "강풍주의보", "건조주의보", "대설경보", "대설주의보", "열대야", "태풍경보",
            "태풍주의보", "폭염경보", "폭염주의보", "한파경보", "한파주의보", "호우경보", "호우주의보", "황사경보"]
FLAG_EN = ["strong_wind_warning", "strong_wind_advisory", "dry_advisory", "heavy_snow_warning",
           "heavy_snow_advisory", "tropical_night", "typhoon_warning", "typhoon_advisory",
           "heat_wave_warning", "heat_wave_advisory", "cold_wave_warning", "cold_wave_advisory",
           "heavy_rain_warning", "heavy_rain_advisory", "yellow_dust_warning"]
MEAN_SRC = {"기온(°C)": "temp_c", "습도(%)": "humidity_pct", "현지기압(hPa)": "pressure_local_hpa",
            "해면기압(hPa)": "pressure_sea_hpa", "풍속(m/s)": "wind_speed_ms"}
ACC_SRC = {"일사(MJ/m^2)": "solar_rad_mj_m2", "일조(Sec)": "sunshine_sec", "누적강수량(mm)": "precip_mm"}

OUT_ORDER = (["date", "usage_kwh", "day_of_week", "weekend", "holiday", "vacation", "exam_period",
              "temp_c", "humidity_pct", "wind_dir_deg", "wind_speed_ms",
              "pressure_local_hpa", "pressure_sea_hpa", "solar_rad_mj_m2", "sunshine_sec", "precip_mm"]
             + FLAG_EN + ["obs_minutes"])

# 대한민국 공휴일 2021~2027 (`holidays` 0.103, 법정 공휴일 + 대체공휴일 + 임시공휴일).
# holiday 컬럼 = (이 집합에 속함) OR (일요일).  data/data.csv 와 동일 기준.
KR_HOLIDAYS = frozenset({
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01", "2021-05-05",
    "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16", "2021-09-20", "2021-09-21",
    "2021-09-22", "2021-10-03", "2021-10-04", "2021-10-09", "2021-10-11", "2021-12-25",
    "2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-03-01", "2022-03-09",
    "2022-05-05", "2022-05-08", "2022-06-01", "2022-06-06", "2022-08-15", "2022-09-09",
    "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-03", "2022-10-09", "2022-10-10",
    "2022-12-25", "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06", "2023-08-15",
    "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02", "2023-10-03", "2023-10-09",
    "2023-12-25", "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15", "2024-06-06",
    "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03",
    "2024-10-09", "2024-12-25", "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29",
    "2025-01-30", "2025-03-01", "2025-03-03", "2025-05-05", "2025-05-06", "2025-06-03",
    "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07",
    "2025-10-08", "2025-10-09", "2025-12-25", "2026-01-01", "2026-02-16", "2026-02-17",
    "2026-02-18", "2026-03-01", "2026-03-02", "2026-05-01", "2026-05-05", "2026-05-24",
    "2026-05-25", "2026-06-03", "2026-06-06", "2026-07-17", "2026-08-15", "2026-08-17",
    "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05", "2026-10-09",
    "2026-12-25", "2027-01-01", "2027-02-06", "2027-02-07", "2027-02-08", "2027-02-09",
    "2027-03-01", "2027-05-01", "2027-05-03", "2027-05-05", "2027-05-13", "2027-06-06",
    "2027-07-17", "2027-07-19", "2027-08-15", "2027-08-16", "2027-09-14", "2027-09-15",
    "2027-09-16", "2027-10-03", "2027-10-04", "2027-10-09", "2027-10-11", "2027-12-25",
})


# ─────────────────────────────────────────────── 학사일정
def semester_start(year: int, kind: str) -> dt.date:
    anchor = dt.date(year, 3, 2) if kind == "spring" else dt.date(year, 9, 1)
    wd = anchor.weekday()                                   # Mon=0 .. Sun=6
    if wd >= 5:                                             # 주말이면 다음 월요일
        return anchor + dt.timedelta(days=7 - wd)
    return anchor - dt.timedelta(days=wd)                   # 그 주의 월요일


def academic_flags(d: dt.date) -> tuple[int, int]:
    """(vacation, exam_period) 반환."""
    for year, kind in ((d.year - 1, "fall"), (d.year, "spring"), (d.year, "fall")):
        off = (d - semester_start(year, kind)).days
        if 0 <= off <= 109:                                 # 1주차 월 ~ 16주차 금 : 학기 중
            exam = int(35 <= off <= 53 or 91 <= off <= 109)  # 6~8주차, 14~16주차
            return 0, exam
    return 1, 0


# ─────────────────────────────────────────────── 입력
def load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={d.columns[0]: "date", "사용량(kWh)": "usage_kwh"})
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d")
    d["usage_kwh"] = pd.to_numeric(d["usage_kwh"], errors="coerce")
    return d[["date", "usage_kwh"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def aggregate_weather_daily(path: Path, start, end) -> pd.DataFrame:
    w = pd.read_csv(path)
    w.columns = [c.strip() for c in w.columns]
    w["일시"] = pd.to_datetime(w["일시"])
    w = w[(w["일시"] >= start) & (w["일시"] < end + pd.Timedelta(days=1))].sort_values("일시").reset_index(drop=True)
    w["date"] = w["일시"].dt.normalize()
    g = w.groupby("date")

    agg = pd.DataFrame(index=pd.Index(sorted(w["date"].unique()), name="date"))
    for src, dst in MEAN_SRC.items():
        agg[dst] = g[src].mean()

    rad = np.deg2rad(w["풍향(deg)"])
    gs = w.assign(_s=np.sin(rad), _c=np.cos(rad)).groupby("date")[["_s", "_c"]].mean()
    wd = np.rad2deg(np.arctan2(gs["_s"], gs["_c"])) % 360.0
    wd[(gs["_s"].abs() < 1e-9) & (gs["_c"].abs() < 1e-9)] = np.nan
    agg["wind_dir_deg"] = wd

    for src, dst in ACC_SRC.items():                        # 누적형 -> 그날 마지막 값(= 일 적산)
        s = g[src].last()
        s[g[src].count() == 0] = np.nan
        agg[dst] = s

    for src, dst in zip(FLAG_SRC, FLAG_EN):
        agg[dst] = g[src].max()
    agg["obs_minutes"] = g.size()
    return agg.reset_index()


# ─────────────────────────────────────────────── main
def main() -> None:
    daily = load_daily(DAILY)
    print(f"일일 수전량 : {daily['date'].min().date()} ~ {daily['date'].max().date()}  ({len(daily)}일)")

    wx = aggregate_weather_daily(WCSV, daily["date"].min(), daily["date"].max())
    print(f"기상(일집계): {wx['date'].min().date()} ~ {wx['date'].max().date()}  ({len(wx)}일)")

    m = daily.merge(wx, on="date", how="left")

    dd = m["date"].dt.date
    m["day_of_week"] = m["date"].dt.day_name()
    m["weekend"] = m["date"].dt.dayofweek.isin([5, 6]).astype(int)
    m["holiday"] = (m["date"].dt.dayofweek.eq(6) | dd.astype(str).isin(KR_HOLIDAYS)).astype(int)
    ve = dd.map(academic_flags)
    m["vacation"] = [x[0] for x in ve]
    m["exam_period"] = [x[1] for x in ve]

    m["date"] = m["date"].dt.strftime("%Y-%m-%d")
    for c in FLAG_EN + ["obs_minutes"]:                     # 결측 없으면 int 유지
        if m[c].notna().all():
            m[c] = m[c].astype(int)
    m = m[OUT_ORDER]
    m.to_csv(OUT, index=False, encoding="utf-8")

    ndays = len(m)
    print(f"저장 : {OUT}  ({ndays}행 x {m.shape[1]}열)")
    print(f"결측 : usage_kwh {int(m['usage_kwh'].isna().sum())}일, temp_c {int(m['temp_c'].isna().sum())}일")
    dsum = m.groupby(m['date'].str.slice(0, 4))[["weekend", "holiday", "vacation", "exam_period"]].sum()
    print("연도별 일수 합:")
    print(dsum.to_string())


if __name__ == "__main__":
    main()
