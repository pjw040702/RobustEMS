# RobustEMS

**AI 인프라 확대와 이상기후 하에서 캠퍼스 에너지관리시스템(EMS)의 운영 한계 검증**
— ESS 버퍼링과 계층적 예측·최적화 기반 스트레스 테스트 연구

---

## 개요

| 항목 | 내용 |
| --- | --- |
| 연구 주제 | AI 전력 수요 증가와 폭염 복합 스트레스 하에서 기존 캠퍼스 EMS의 경제성·안정성·친환경성 유지 한계(failure boundary) 검증 |
| 연구 유형 | 수요예측·최적화·시뮬레이션 통합 스트레스 테스트 (신규 최적화 알고리즘 개발이 아닌 기존 벤치마크 EMS의 성능 검증) |
| 연구 기간 | 12개월 (계획) |
| 핵심어 | 캠퍼스 마이크로그리드, EMS, ESS, AI workload, 폭염, Model Predictive Control, 예측 불확실성, failure boundary |

본 연구의 초점은 **더 좋은 최적화 알고리즘을 만드는 것이 아니라**, 이미 충분히 최적화된 기존 EMS가
AI 인프라 확대라는 새로운 수요환경에서도 기존의 경제성·안정성·친환경성을 얼마나 유지할 수 있는지를
검증하는 데 있다. 검증된 최적화 모델을 benchmark로 고정하고, 수요환경(AI workload 침투율, 폭염 강도)을
체계적으로 변화시켜 성능이 급락하는 임계 조건을 찾는다.

---

## 1. 연구 배경 및 필요성

대학 캠퍼스는 스마트 에너지 캠퍼스 사업 등을 통해 ESS, 태양광(PV), 통합관제 기반의 에너지관리 인프라를
구축해 왔다. 성균관대학교 역시 한국전력공사와의 협력으로 K-SEM 기반 인프라를 구축한 바 있다. 그러나
이러한 인프라는 **과거의 정상적인 수요 패턴을 전제로 설계·최적화**된 것으로, AI 인프라 확대와 이상기후로
변화하는 전력수요 환경에서 기존 운영 방식이 어느 수준까지 유효한지는 검증되지 않았다.

네 가지 문제의식에서 본 연구를 계획한다.

- **AI 학습과 전력 스파이크** — 초고성능 GPU 수십~수백 개의 동시 학습은 순간 전력 사용량을 설비 설계
  용량의 상당 비중까지 끌어올릴 수 있으며, 핵심 인프라(냉방·배터리)를 조기에 소모시킬 수 있다.
- **전기요금 구조의 함정** — 최대수요전력(피크)이 계약전력 요금 산정 기준이므로, 짧은 스파이크 한 번이
  장기간의 요금 부담을 크게 늘릴 수 있다.
- **정전·설비 리스크** — 과전류·용량 초과 스파이크는 전력계통 안정성을 저하시키고, 여름철 냉방 부하와
  겹치면 서버 및 연구 장비가 손해를 입을 수 있다.
- **이행의 불확실성** — 최신 AI 장비·데이터센터형 부하의 도입 속도와 규모는 예측이 어렵다. 학교 전력
  시스템은 지속적이고 불확실한 수요 전환(demand transition)이라는 구조적 문제에 직면해 있다.

기존 마이크로그리드·EMS 연구의 다수는 더 나은 최적화 알고리즘 개발에 집중해 왔다. 반면 이미 충분히
최적화된 EMS가 AI 수요 전환과 폭염이라는 복합 스트레스 하에서 **언제, 어떤 방식으로 성능이 급락하는지**를
체계적으로 검증한 연구는 확인되지 않는다. 본 연구는 이 공백을 겨냥한다.

---

## 2. 연구 질문

| 구분 | 연구 질문 |
| --- | --- |
| **RQ1** | AI workload 증가와 폭염 등 extreme weather가 기존 캠퍼스 EMS의 peak 및 operational robustness를 얼마나 악화시키는가? |
| **RQ2** | 이러한 demand transition이 기존 EMS의 경제성과 탄소저감·재생에너지 활용 효과를 어느 수준까지 훼손하는가? |
| **RQ3** | Forecast uncertainty를 고려한 ESS 운영 및 flexible AI workload scheduling을 통해 경제성·안정성·친환경성의 저하를 얼마나 완화할 수 있는가? |

---

## 3. 연구 목표 및 범위

ESS의 SOC 및 충·방전 제약, 전력수급 균형, 피크, 전력구매비용, 재생에너지 활용 등을 충분히 고려한
검증된 최적화 모델을 benchmark로 선정·고정하고, 수요환경(AI workload 침투율, 폭염 강도)을 체계적으로
변화시킨다.

**최종 산출물**

- AI 수요 증가와 폭염 강도의 2차원 공간에서 peak·비용·탄소배출 등 성능이 급격히 악화되는
  **failure boundary의 식별**
- 불확실성을 고려한 ESS 운영과 flexible AI workload scheduling이 이 boundary를 어느 정도까지 확장할 수
  있는지에 대한 **정량 평가**

---

## 4. 연구 내용 및 방법

### 4.1 수요예측 및 불확실성 모델링

과거 전력수요와 기온·습도·일사량 등 기상 변수, 시간·요일 등 캘린더 변수를 이용하여 **LSTM/GRU**를 기본
예측 모델로 구축하고, 필요시 Transformer 계열과 성능을 비교한다. 점 예측뿐 아니라 prediction interval
또는 예측오차 분포를 추정하여 forecast uncertainty를 함께 모델링한다. 특히 AI 부하 중첩 이후 예측구간
캘리브레이션(실측 포함 비율)의 변화를 별도로 평가하여, 수요 전환이 예측 품질에 미치는 영향을 정량화한다.

### 4.2 스트레스 시나리오 생성

예측된 baseline demand에 지속적 workload 증가와 순간적 workload spike를 단계적으로 추가하고, 폭염 등
기후조건을 결합하여 stress scenario를 생성한다 (**Normal / AI Growth / AI Spike / Heatwave / AI+Heatwave**).
임의의 합성 데이터 생성은 원칙적으로 배제하며, AI workload는 공개된 실측 GPU 클러스터·HPC 트레이스
(Alibaba PAI, MIT Supercloud, PM100 등)를 전력으로 환산해 캠퍼스 부하에 중첩하고, 폭염은 실제 관측된
폭염 연도의 기상 자료를 사용한다.

### 4.3 벤치마크 EMS와 계층적 제어 구조

제어는 단기·초단기 두 계층으로 구성한다.

- **단기 MPC (전략 계획)** — 향후 24~48시간 수요예측을 바탕으로 ESS 충·방전 계획, 한전 구매전력 계획,
  SOC 궤적, 피크 관리 계획 등 전체 운영계획 수립
- **초단기 제어 (실시간 보정)** — 실제 수요가 예측과 달라졌을 때 실측값 기반으로 ESS 출력·구매전력을
  빠르게 보정
- **실측 피드백** — 새로운 실측값을 다시 예측·최적화에 반영하는 receding horizon 방식으로 두 계층을 연결

제약조건: 전력수급 균형, ESS 용량·SOC·충방전 한계, 계약전력·과전류·계기 한계 등 운영 제약.

### 4.4 시뮬레이션 및 평가지표

동일한 benchmark EMS에 각 시나리오를 투입하여 반복 시뮬레이션하고, AI 수요와 이상기후 강도가 증가함에
따른 시스템 성능 변화를 분석한다. 평가지표는 세 축으로 구성한다.

| 평가 축 | 지표 |
| --- | --- |
| **안정성** | 최대수요전력(peak), peak violation 횟수·지속시간, 제약 위반(constraint violation), 미공급전력(ENS) |
| **경제성** | 전력구매비용, 계약전력(기본)요금, ESS 열화비용, 수요반응 효과 등을 포함한 총 운영비용 |
| **친환경성** | 탄소배출량, 재생에너지 활용률(자가소비율, 잉여 축소) |

### 4.5 성능 열화의 분해 설계

동일 시나리오를 **(a) perfect-foresight EMS**(실측 수요를 그대로 아는 이상적 조건)와 **(b) 예측 기반 EMS**에
각각 투입해 비교한다. (a)의 열화는 수요의 구조적 증가 자체에 기인하고, (b)와 (a)의 차이는 예측 불확실성
악화가 추가로 기여한 부분이다. 이 분해로 *AI 수요가 커서 문제인지, 예측이 어려워져서 문제인지*를 구분하고,
RQ3의 완화 전략 선택(ESS 운영 개선 vs. workload 유연화)의 근거를 제공한다.

### 4.6 완화 전략 평가 (RQ3)

두 가지 완화 수단의 효과를 boundary 확장 관점에서 평가한다.

- **불확실성 고려 ESS 운영** — 예측구간·오차분포를 반영한 보수적 SOC 여유 확보, 피크 대비 예비 방전 용량 유지 등
- **Flexible AI workload scheduling** — 지연 가능한 AI 학습 작업의 시점 이동과 전력 상한(power capping)을 통한 피크 회피

workload 유연화 비율과 ESS 용량을 축으로 failure boundary 확장 정도를 곡선으로 제시하고, ESS 증설과
workload 유연화 간의 등가 교환관계를 도출한다.

---

## 5. 기존 연구와의 차별성

| 기존 연구의 한계 | 본 연구의 확장 |
| --- | --- |
| 정상 수요환경 전제, 과거 데이터 기반 forecast error/bound 중심의 불확실성 설정 | AI workload spike, 폭염 등 비정상적 demand shock을 명시적으로 고려 |
| 수요 분포 유지 전제 하의 유사 수요 패턴 가정 | AI 인프라 확대로 수요 수준·변동성이 구조적으로 변하는 상황을 평가 |
| 경제성 분석이 전력구매비 중심 | 계약전력 요금, ESS 열화비용, 수요반응 효과를 포함한 총비용 관점 평가 |
| 평균적 성능(운영비, 제약 만족) 중심 평가 | 극단 조건에서 peak·비용·탄소·재생에너지 활용 악화 정도의 stress test |
| 예측 불확실성과 구조적 수요 변화의 구분 부족 | 예측오차 기인 열화와 AI 기인 구조적 수요 증가 열화의 분해 |
| 시스템 한계점 분석 부족 | 제약 위반과 비용 급증이 시작되는 failure boundary의 명시적 탐색 |

근접 선행연구로 AI/데이터센터 수요 충격을 다룬 분포적 강건 최적화 연구(arXiv:2601.14665)와 대규모 언어모델
학습이 유발하는 전력 과도현상을 실측한 연구(arXiv:2409.11416)가 있으나, 전자는 신규 알고리즘 제안과 계통
스케일에 초점을 두고 후자는 EMS 운영 평가를 포함하지 않는다. 검증된 EMS를 고정하고 수요환경을 변화시켜
캠퍼스 스케일의 운영 한계를 검증하는 본 연구의 설계는 문헌에서 확인되지 않는 공백에 해당한다.

---

## 6. 데이터 및 벤치마크 확보 계획

### 6.1 교내 EMS(K-SEM) 확인 사항

성균관대학교는 과거 한국전력공사와 스마트 에너지 캠퍼스 사업을 진행한 바 있으나, 해당 EMS의 현재 운영
여부는 확인이 필요하다. 현재 운영 중이라면 교내 EMS의 실제 운영방식을 benchmark로 두고 AI 수요·이상기후
시나리오를 적용하는 것이 가장 적절하다. 다음 사항을 문의·확인할 계획이다.

- EMS(K-SEM)의 현재 운영 여부와 운영 방식
- 제공 예정 전력 데이터의 성격: 실제 캠퍼스 전력계측 부하인지, ESS·태양광 제어가 반영된 한전 수전량인지 (가능하면 둘 다)
- PV 발전량, ESS 충·방전량 및 SOC 등 운영 데이터의 제공 가능 여부
- 데이터 해상도(15분 또는 1시간)와 기간(여름철 2회 이상 포함)
- 계약전력·요금제 적용 이력 (피크 요금 산정에 필요)
- 연구 목적의 데이터 이용 및 논문 게재 허가 범위

### 6.2 대체 경로: 공개 실측 데이터 기반 벤치마크

교내 EMS가 운영되고 있지 않거나 데이터 확보가 어려운 경우, 검증된 EMS 및 MPC 모델을 캠퍼스 규모에 맞게
구성하고 아래 공개 실측 데이터를 결합하여 벤치마크 시스템을 구축한다. 교내 데이터는 확보되는 경우 한국
사례 검증으로 활용한다.

| 역할 | 데이터 | 내용 |
| --- | --- | --- |
| 폭염 축 | BDG2 (Fox 사이트, 애리조나주립대) | 대학 건물 다수의 2년간 시간별 실측 전력. 최고기온 48.3℃ 수준의 폭염 구간 포함. 지역냉방(chilled water) 부하의 전력 환산(COP 가정) 필요 |
| 설비 축 | UCSD 캠퍼스 마이크로그리드 공개 DB | 캠퍼스 총부하·자가발전·계통수전·요금기준수요와 배터리·축열·PV 실측 포함. 벤치마크 EMS 설비 구성·파라미터 캘리브레이션에 활용 (이용조건 확인 예정) |
| 일본 축 | 기타큐슈시립대 캠퍼스 데이터셋 (Scientific Data, CC BY 4.0) | 20년 장기의 시간별 태양광 발전, 계통 수전, 냉방부하 직접 계측, 기상 자료. 외적 타당성 검증에 활용 |
| AI 부하 축 | 공개 GPU/HPC 실측 트레이스 (Alibaba PAI, MIT Supercloud, PM100 등) | 실측 작업 트레이스를 전력으로 환산하여 AI workload 시나리오에 사용. 임의의 합성 스파이크 생성은 배제 |

---

## 7. 추진 일정 (3개월 기준)

| 단계 | 내용 |
| --- | --- |
| 1 | 교내 EMS 운영 여부·데이터 문의, 공개 데이터 정비, 문헌 검토 및 벤치마크 EMS 모델 확정 |
| 2 | 수요예측 모델(LSTM/GRU, 필요시 Transformer) 구축 및 prediction interval 추정, 예측 성능 검증 |
| 3 | AI workload·폭염 스트레스 시나리오 생성 (실측 트레이스 환산, 폭염 관측자료 결합) |
| 4 | 계층적 제어(단기 MPC + 초단기 보정) 구현, 벤치마크 EMS 시뮬레이션 환경 구축 및 검증 |
| 5 | 시나리오 스윕 실험: failure boundary 탐색, 열화 분해(perfect-foresight 대비), KPI 평가 |
| 6 | 완화 전략(불확실성 고려 ESS 운영, workload scheduling) 실험, 결과 정리 및 논문 초고 작성 |

---

## 8. 기대 성과 및 활용 방안

- AI 인프라 확대와 폭염 복합 조건에서 캠퍼스 EMS 성능이 급락하는 임계 조건(failure boundary)의 최초 정량화
- 성능 열화의 원인 분해: 구조적 수요 증가 기인 vs. 예측 불확실성 기인
- ESS 증설과 AI workload 유연화 간 등가 교환관계 등 설비 투자·운영 정책에 직접 활용 가능한 정량 근거
- 대학·연구기관의 AI 인프라 도입 계획 수립 시 전력 인프라 측면의 의사결정 지원
- 에너지 분야 국제 학술지(예: Applied Energy, Sustainable Cities and Society, Energy 급) 논문 1~2편 투고

---

## 9. 저장소 구조

```
RobustEMS/
├── README.md
├── TalkFile_연구계획서_캠퍼스EMS_AI스트레스_v1.docx   # 원본 연구계획서
├── scripts/
│   ├── merge_weather.py         # 15분 수전량 × 1분 기상 × 캘린더 → data/data.csv
│   └── build_data_long.py       # 일일 수전량 × 1분 기상(일집계) × 캘린더 → data/data_long.csv
└── data/
    ├── data.csv                 # 15분 통합 데이터셋 (2025-09 ~ 2026-09, 34,944행 × 41열) — git 추적
    ├── data_long.csv            # 일 단위 장기 데이터셋 (2021-09 ~ 2026-09, 1,826행 × 32열) — gitignore
    └── source/                  # 원본 (gitignore, 로컬 전용)
        ├── 15분 수전량 2025.09~.xlsx              # 15분 수전량 (2025-09-04 ~ 2026-09-02, 2026-04-07 누락)
        ├── 일일 수전량 2021.09~.csv               # 일 단위 총 사용량 (2021-09-03 ~ 2026-09-02)
        ├── combined_sorted_weather_data.csv      # 1분 기상 관측 (지점 119, 2021-09 ~ 2026-09)
        └── 전기사용량_시간대별(20260407)_15분.xls  # 2026-04-07 하루치 보강분 (HTML 표) — merge_weather.py 가 자동 병합
```

> `data/source/` 와 `data/data_long.csv` 는 `.gitignore` 로 제외(대용량·로컬 전용). `data/data.csv` 만 추적한다.
> `merge_weather.py` 재실행 시 `data/source/전기사용량_시간대별(20260407)_15분.xls` 가 없으면 2026-04-07 96행이 빠진다.

### `data/data.csv` — 수전량 × 기상 × 캘린더 통합 데이터셋

`15분 수전량 …xlsx`의 각 (날짜, 시간) 구간에 `combined_sorted_weather_data.csv`(1분 관측, 지점 119)와
캘린더 파생 변수를 결합한 파일. **34,944행 × 41열** (364일 × 96구간), 인코딩 **BOM 없는 UTF-8**,
컬럼명은 전부 영어 snake_case(값도 순수 ASCII).
컬럼 순서: 수전량(9) → `datetime` → 캘린더(5) → 기상(26).

- **매칭 기준** — 수전량의 `time` 은 구간 **종료 시각** 라벨이다. `00:15` → `(00:00, 00:15]` 구간
  (1분 관측 `00:01`–`00:15`), `24:00` → 익일 `00:00`. `datetime` 열에 구간 종료 시각을 timestamp로 추가.
- **2026-04-07** — 원본 xlsx에 없던 하루. `전기사용량_시간대별(20260407)_15분.xls`(HTML 표)의 15분 표에서
  수전량 9열을 읽어 `merge_weather.py` 가 자동 병합하며, 기상·캘린더는 다른 날과 동일하게 산출된다.
  (일 사용량 235,511.64 kWh / 일 피크 11,363.04 kW — 원본 파일 요약과 일치.)
- **집계 방식** (구간당 최대 15개 1분 관측)

  | 그룹 | 열 | 방법 |
  | --- | --- | --- |
  | 수전량 (원본) | `date` `time` `usage_kwh` `peak_demand_kw` `reactive_lag/lead_kvarh` `co2_tco2` `power_factor_lag/lead_pct` | 그대로 |
  | 기상 (순간값) | `temp_c` `humidity_pct` `wind_speed_ms` `pressure_local_hpa` `pressure_sea_hpa` | 구간 평균 |
  | 기상 (방향) | `wind_dir_deg` | 벡터(원형) 평균 |
  | 기상 (적산) | `solar_rad_15min_mj_m2` `sunshine_15min_sec` `precip_15min_mm` | 15분 적산 — 원자료가 매일 00:01 리셋되는 **하루 누적값**이라, 분단위 증가분(리셋·음수는 0)을 합산해 해당 15분 값으로 환산 |
  | 기상 (원값) | `precip_cumulative_mm` | 구간 종료 시각의 그날 누적 강수량 |
  | 특보·주의보 14종 | `strong_wind_*` `heavy_snow_*` `typhoon_*` `heavy_rain_*` (각 `_warning`/`_advisory`), `dry_advisory` `cold_wave_*` `yellow_dust_warning` | 구간 내 1분이라도 발효 시 `1` |
  | 조건 | `tropical_night` | 열대야이면 `1` |
  | 진단 | `obs_minutes` | 실제 매칭된 1분 관측 수 (0–15) |

- **캘린더 파생** (`date` 기준, 하루 96구간 공통)

  | 열 | 정의 |
  | --- | --- |
  | `day_of_week` | 요일 영문명 `Monday`–`Sunday` |
  | `weekend` | 토·일이면 `1` |
  | `holiday` | 대한민국 공휴일(대체공휴일 포함) **또는 일요일**이면 `1` — 범위 내 공휴일 22일 |
  | `vacation` | 방학이면 `1` (아래 학사일정 규칙) — 데이터 범위 내 `2025-12-20~2026-03-01`, `2026-06-20~2026-08-30` |
  | `exam_period` | 시험기간이면 `1` — `2025-10-06~10-24`, `2025-12-01~12-19`, `2026-04-06~04-24`, `2026-06-01~06-19` |

<a name="academic-calendar"></a>
  **학사일정 규칙** (`data.csv` · `data_long.csv` 공통):
  - 봄학기 1주차 = **3/2 이 포함된 주**, 가을학기 1주차 = **9/1 이 포함된 주**. 단 3/2(9/1)이 토·일이면
    그 다음 월요일이 1주차 월요일.
  - 수업 16주 = 1주차 월 ~ 16주차 금 (개강일 + 109일). 그 밖의 날은 `vacation = 1`.
  - `exam_period = 1` : **6~8주차**, **14~16주차** (각 3주, 1주차 월 기준 개강일 + 35~53일 / 91~109일).
  - `holiday` = 대한민국 공휴일(대체·임시공휴일 포함, `holidays` 라이브러리 기준) **또는 일요일**.
    `근로자의날`·`제헌절`도 포함. 목록은 각 스크립트 상단 `KR_HOLIDAYS`.

- **결측** (원자료 관측 공백): `temp_c` 4구간(센서 개별 결측, `obs_minutes`는 15), `humidity_pct` 155,
  `pressure_sea_hpa` 3, `precip_cumulative_mm`·`precip_15min_mm` 35구간. `obs_minutes < 15` 인 구간 26개.
- **재생성** — `python scripts/merge_weather.py` (`data/source/` 의 xlsx·기상 CSV·2026-04-07 xls 필요)

#### 컬럼명 대응 (원본 한글 → 출력 영어)

| 한글 | 영어 | · | 한글 | 영어 |
| --- | --- | --- | --- | --- |
| 날짜 | `date` | · | 일사_15분(MJ/m^2) | `solar_rad_15min_mj_m2` |
| 시간 | `time` | · | 일조_15분(Sec) | `sunshine_15min_sec` |
| 사용량(kWh) | `usage_kwh` | · | 누적강수량(mm) | `precip_cumulative_mm` |
| 최대수요(kW) | `peak_demand_kw` | · | 강수량_15분(mm) | `precip_15min_mm` |
| 무효전력_지상(kVarh) | `reactive_lag_kvarh` | · | 강풍경보 / 주의보 | `strong_wind_warning` / `_advisory` |
| 무효전력_진상(kVarh) | `reactive_lead_kvarh` | · | 대설경보 / 주의보 | `heavy_snow_warning` / `_advisory` |
| CO2(tCO2) | `co2_tco2` | · | 호우경보 / 주의보 | `heavy_rain_warning` / `_advisory` |
| 역률_지상(%) | `power_factor_lag_pct` | · | 태풍경보 / 주의보 | `typhoon_warning` / `_advisory` |
| 역률_진상(%) | `power_factor_lead_pct` | · | 한파경보 / 주의보 | `cold_wave_warning` / `_advisory` |
| 일시 | `datetime` | · | 건조주의보 | `dry_advisory` |
| 기온(°C) | `temp_c` | · | 황사경보 | `yellow_dust_warning` |
| 습도(%) | `humidity_pct` | · | 열대야 | `tropical_night` |
| 풍향(deg) | `wind_dir_deg` | · | 관측분수 | `obs_minutes` |
| 풍속(m/s) | `wind_speed_ms` | · | | |
| 현지기압(hPa) | `pressure_local_hpa` | · | | |
| 해면기압(hPa) | `pressure_sea_hpa` | · | | |

### `data/data_long.csv` — 일 단위 장기 데이터셋

`일일 수전량 2021.09~.csv`(하루 총 사용량)에 `combined_sorted_weather_data.csv`를 **일 단위로 집계**해
결합. **1,826행 × 32열**, `2021-09-03 ~ 2026-09-02`. 인코딩·컬럼명 규칙은 `data/data.csv`와 동일.

- **컬럼 차이** (일 단위라 15분·부가 지표 없음): `time` `datetime` 및 `peak_demand_kw` `reactive_*`
  `co2_tco2` `power_factor_*` 제외. 15분 적산 컬럼은 개명 —
  `solar_rad_15min_mj_m2 → solar_rad_mj_m2`, `sunshine_15min_sec → sunshine_sec`(0~86400),
  `precip_15min_mm → precip_mm`, `precip_cumulative_mm` 제외. `obs_minutes` 는 0~1440.
- **컬럼 순서**: `date` `usage_kwh` → 캘린더(5) → 기상(`temp_c`~`precip_mm`, 9) → 특보·주의보(15) → `obs_minutes`.
- **기상 일집계**: 기온·습도·기압·풍속 = 일 평균, `wind_dir_deg` = 벡터 평균,
  `solar_rad_mj_m2`·`sunshine_sec`·`precip_mm` = 그날 마지막(≈23:59) 누적값(= 일 합계),
  특보·주의보 = 하루 1분이라도 발효 시 `1`.
- **캘린더**: [학사일정 규칙](#academic-calendar) 그대로 전 연도 적용. `data.csv`와 겹치는 364일은 5개
  캘린더 컬럼·`usage_kwh` 일합이 완전히 일치.
- **결측**: `usage_kwh` 4일(원본 공백: 2023-07-24, 08-18~20), `precip_mm` 1일(2023-07-31 강수 관측 공백).
- **재생성** — `python scripts/build_data_long.py` (`data/source/` 의 일일 수전량 CSV·기상 CSV 필요).

---

## 10. 주요 참고문헌

1. *A Two-Stage Risk-Averse DRO-MILP Methodological Framework for Managing AI/Data Center Demand Shocks.* arXiv:2601.14665.
2. *The Unseen AI Disruptions for Power Grids: LLM-Induced Transients.* arXiv:2409.11416.
3. Miller, C. et al. (2020). *The Building Data Genome Project 2: energy meter data from the ASHRAE Great Energy Predictor III competition.* Scientific Data 7, 368.
4. Silwal, S. et al. (2021). *An open-source multi-year power and energy dataset from the UC San Diego microgrid.* Journal of Renewable and Sustainable Energy 13, 025301.
5. Liao, W. et al. (2024). *기타큐슈시립대학 캠퍼스 에너지 실측 데이터셋.* Scientific Data (figshare: 10.6084/m9.figshare.24978645).
6. *PM100: A Job Power Consumption Dataset of a Large-scale Production HPC System.* SC'23 Workshops, ACM.
7. *MIT Supercloud Dataset: GPU/CPU 이용률·전력·온도 실측 로그* (2021).

> 주: 참고문헌의 서지사항은 투고 전 원문 대조를 통해 최종 확정한다.
