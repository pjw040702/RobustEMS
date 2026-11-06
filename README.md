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
| **RQ3** | SOC 여유 확보 규칙과 flexible AI workload scheduling을 통해 경제성·안정성·친환경성의 저하를 얼마나 완화할 수 있는가? *(2026-11-06: 당초 "Forecast uncertainty를 고려한 ESS 운영"에서 재정의 — 4.1절 참고)* |

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

### 4.1 제어 구조 — 반응형 DPNN (2026-11-06 재정의)

> **변경 결정.** 당초 LSTM/GRU 수요예측기를 두는 설계였으나, 벤치마크를 **DPNN**(Liu, Zhao & He
> 2026, *Energy Economics* 154, 109104)으로 확정하면서 **예측기를 두지 않는 반응형 구조**로
> 재정의했다. DPNN은 예측기가 아니라 `동적 파라미터 → 최적해`를 즉시 내놓는 **최적화 대리모델**
> 이므로, 예측해서 계획하는 대신 **실측에 즉시 반응**하는 아키텍처가 된다. 논문의 셀링포인트
> (*"respond immediately to dynamic parameter changes"*)와 일치한다.
>
> 이 재정의는 RQ3·4.5절의 서술을 바꾼다(각 절 참고). **지도교수 확인 필요 사항**이다.

따라서 별도의 수요예측 모델은 구축하지 않는다. 대신 **lookahead의 가치 자체를 상한으로 측정**한다.

| 제어 팔 | 정보 | 구현 |
| --- | --- | --- |
| **반응형**(하한) | 현재 실측 $(E_t,L_t,PV_t)$ + 시간대 기반 SOC 목표 규칙. lookahead 없음 | DPNN |
| **완전예지**(상한) | 1년 전체를 미리 아는 이상적 조건 | 연간 LP (`dpnn_benchmark.ipynb` 3-1) |

두 팔의 격차가 **어떤 예측기라도 벌 수 있는 최대치**다. 특정 LSTM 하나와 비교하는 것보다
**예측기에 무관한 상한**이라는 점에서 더 일반적인 결과이며, 예측기 구축 공수도 들지 않는다.

반응형 팔에 SOC 목표 규칙이 필요한 이유: 단일시점 문제는 비용이 선형이면 해가 자명해진다
(차익거래는 본질적으로 시간 간 결합이므로 한 시점 안에서는 유인이 존재할 수 없다).
`dpnn_feasibility.md` 5절에서 확인한 사항이다.

### 4.2 스트레스 시나리오 생성

예측된 baseline demand에 지속적 workload 증가와 순간적 workload spike를 단계적으로 추가하고, 폭염 등
기후조건을 결합하여 stress scenario를 생성한다 (**Normal / AI Growth / AI Spike / Heatwave / AI+Heatwave**).
임의의 합성 데이터 생성은 원칙적으로 배제하며, AI workload는 공개된 실측 GPU 클러스터·HPC 트레이스
(Alibaba PAI, MIT Supercloud, PM100 등)를 전력으로 환산해 캠퍼스 부하에 중첩하고, 폭염은 실제 관측된
폭염 연도의 기상 자료를 사용한다.

### 4.3 벤치마크 EMS와 계층적 제어 구조

4.1절 재정의에 따라 **예측 기반 계획 계층을 두지 않는다.** 두 팔을 각각 독립 제어기로 구성한다.

- **완전예지 팔 (상한)** — 1년 전체를 한 번에 푸는 LP. 비교 기준선 (`dpnn_benchmark.ipynb` 3-1에서
  구현·검증 완료). 48h receding-horizon MPC가 이것과 비용이 동일함을 확인했으므로 둘 중 어느 쪽을
  써도 같다
- **반응형 팔 (하한)** — 매 15분 현재 실측 $(E_t,L_t,PV_t)$와 시간대 기반 SOC 목표만 보고 DPNN이
  즉시 출력. lookahead 없음. 피크 목표는 계약전력·과거 실적 기반 고정값으로 하달

제약조건: 전력수급 균형, ESS 용량·SOC·충방전 한계, 계약전력·과전류·계기 한계 등 운영 제약.
반응형 팔에서 이 제약들이 **전부 박스 제약으로 환원**되는 것이 DPNN 적용의 전제다
(`reference/dpnn_feasibility.md` 3~4절).

제약조건: 전력수급 균형, ESS 용량·SOC·충방전 한계, 계약전력·과전류·계기 한계 등 운영 제약.

### 4.4 시뮬레이션 및 평가지표

동일한 benchmark EMS에 각 시나리오를 투입하여 반복 시뮬레이션하고, AI 수요와 이상기후 강도가 증가함에
따른 시스템 성능 변화를 분석한다. 평가지표는 세 축으로 구성한다.

| 평가 축 | 지표 |
| --- | --- |
| **안정성** | 최대수요전력(peak), peak violation 횟수·지속시간, 제약 위반(constraint violation), 미공급전력(ENS) |
| **경제성** | 전력구매비용, 계약전력(기본)요금, ESS 열화비용, 수요반응 효과 등을 포함한 총 운영비용 |
| **친환경성** | 탄소배출량, 재생에너지 활용률(자가소비율, 잉여 축소) |

### 4.5 성능 열화의 분해 설계 (2026-11-06 재정의)

동일 시나리오를 **(a) 완전예지 EMS**(1년 전체를 아는 이상적 조건)와 **(b) 반응형 EMS**(현재 실측만
보는 DPNN)에 각각 투입해 비교한다.

- (a)의 열화 = **수요충격 자체**에 기인한 부분. AI 부하·폭염이 커지면 완전예지로도 피할 수 없다
- (b)−(a) 격차 = **lookahead 부재**가 추가로 기여한 부분. 어떤 예측기라도 이 격차 이상은 못 번다

이 분해로 *AI 수요가 커서 문제인지, 미래를 못 봐서 문제인지*를 구분한다.

> **당초 설계와의 차이.** 원래는 (b)를 "예측 기반 EMS"로 두고 예측오차 기인 열화를 보려 했으나,
> 4.1절 재정의로 예측기를 두지 않게 되어 **lookahead 부재 기인**으로 바꿨다. 특정 예측기 성능에
> 의존하지 않는 **상한**이라 더 일반적이지만, "예측 품질이 수요 전환으로 얼마나 나빠지는가"라는
> 원래 질문에는 답하지 못한다. 그 질문이 필요하면 예측기를 별도로 붙여야 한다 — 지도교수 확인 필요.
>
> **이 분해가 연구의 전부라는 점에 유의.** `dpnn_benchmark.ipynb` 3-1에서 확인한 대로, 완전예지
> 조건에서 48h MPC는 연간 LP와 비용이 **정확히 같다**(괴리 0.000%). ESS가 캠퍼스 규모 대비 작아
> 다일 차익거래 여지가 없고 유일한 전역 결합이 연간 피크이기 때문이다. 따라서 **수요충격이나
> lookahead 부재 없이는 보여줄 열화가 존재하지 않는다.**

### 4.6 완화 전략 평가 (RQ3)

두 가지 완화 수단의 효과를 boundary 확장 관점에서 평가한다.

- **SOC 여유 확보 규칙** — 반응형 제어가 피크를 놓치지 않도록 예비 방전 용량을 남기는 규칙
  (당초의 "예측구간 기반 보수적 운영"을 예측기 없는 형태로 재정의)
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
| 예측 불확실성과 구조적 수요 변화의 구분 부족 | **수요충격 기인 열화와 lookahead 부재 기인 열화의 분해** (2026-11-06 재정의: 특정 예측기에 의존하지 않는 상한 형태) |
| 시스템 한계점 분석 부족 | 제약 위반과 비용 급증이 시작되는 failure boundary의 명시적 탐색 |

근접 선행연구로 AI/데이터센터 수요 충격을 다룬 분포적 강건 최적화 연구(arXiv:2601.14665)와 대규모 언어모델
학습이 유발하는 전력 과도현상을 실측한 연구(arXiv:2409.11416)가 있으나, 전자는 신규 알고리즘 제안과 계통
스케일에 초점을 두고 후자는 EMS 운영 평가를 포함하지 않는다. 검증된 EMS를 고정하고 수요환경을 변화시켜
캠퍼스 스케일의 운영 한계를 검증하는 본 연구의 설계는 문헌에서 확인되지 않는 공백에 해당한다.

---

## 6. 데이터 및 벤치마크 확보 계획

### 6.1 교내 EMS(K-SEM) 확인 사항 — **종결: 운영되지 않음**

> **확인 완료.** 교내 EMS(K-SEM)는 **현재 운영되지 않는 것으로 확인**되어, 설비 파라미터를
> 문헌 가정으로 두는 것으로 결론났다(`reference/parameters.md`). 따라서 아래 6.1 경로는 종결하고
> **6.2 대체 경로**를 따른다. 교내 **부하** 실측 데이터는 확보돼 있으므로(15분 수전량 + 한전 청구서)
> 부하 축은 실측 기반이고, 설비 축만 문헌 가정이다.
>
> 귀결 두 가지 — ① ESS·PV 운영 실측이 없어 디스패치 결과를 실측과 대조할 수 없으므로 검증은
> 부하모형↔청구서 대조 / 구현↔solver 해 일치 / **파라미터 민감도**에 의존한다(연구 한계로 명시).
> ② 교내 EMS가 없으므로 "기존 EMS"가 아니라 **"문헌 검증 모델로 구성한 벤치마크 EMS"**로 서술해야
> 한다. 상세는 `reference/team_review_and_plan.md`.

(이하 당초 문의 계획 — 기록용)

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

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 1 | 교내 EMS 운영 여부 확인, 데이터 정비, 벤치마크 EMS 모델 확정 | **완료** — K-SEM 미운영 확인(6.1절), DPNN 확정 |
| 2 | ~~수요예측 모델 구축~~ → 2026-11-06 재정의로 삭제. DPNN 반응형 제어기 완성 | **부분** — PNN·MPC 검증 완료, DPNN 정확도 미달 |
| 3 | AI workload·폭염 스트레스 시나리오 생성 | 폭염(SSP) 보유 / **AI 트레이스 미확보** |
| 4 | 벤치마크 EMS 시뮬레이션 환경 구축·검증 | **완료** — `dpnn_benchmark.ipynb` |
| 5 | 시나리오 스윕: failure boundary 탐색, 열화 분해, KPI 평가 | 미착수 |
| 6 | 완화 전략 실험, 결과 정리, 논문 초고 | 미착수 |

진행 상세와 담당은 `reference/team_review_and_plan.md` 참고.

---

## 8. 기대 성과 및 활용 방안

- AI 인프라 확대와 폭염 복합 조건에서 캠퍼스 EMS 성능이 급락하는 임계 조건(failure boundary)의 최초 정량화
- 성능 열화의 원인 분해: **수요충격 기인 vs. lookahead 부재 기인**(4.5절)
- ESS 증설과 AI workload 유연화 간 등가 교환관계 등 설비 투자·운영 정책에 직접 활용 가능한 정량 근거
- 대학·연구기관의 AI 인프라 도입 계획 수립 시 전력 인프라 측면의 의사결정 지원
- 에너지 분야 국제 학술지(예: Applied Energy, Sustainable Cities and Society, Energy 급) 논문 1~2편 투고

---

## 9. 저장소 구조

```
RobustEMS/
├── README.md
├── temperature_scenario_analysis.ipynb   # ① 기온-수요 모형 확정, SSP 시나리오, 복합재해 검토
├── ess_pv_pipeline.ipynb                 # ② ESS·PV 최적화 파이프라인 (트랙 A: 경제성·공고 준수)
├── dpnn_benchmark.ipynb                  # ③ DPNN 벤치마크 EMS (트랙 B: 연구 본체)
├── scripts/
│   ├── merge_weather.py          # 15분 수전량 × 1분 기상 × 캘린더 → data/data.csv
│   └── build_data_long.py        # 일일 수전량 × 1분 기상(일집계) × 캘린더 → data/data_long.csv
├── data/
│   ├── data.csv                  # 15분 통합 데이터셋 (34,944행 × 41열) — 추적
│   ├── data_long.csv             # 일 단위 장기 데이터셋 (1,826행 × 32열) — 추적
│   ├── bench_inputs.npz          # ③ 입력 캐시 (부하·PV단위발전·TOU) — 자동 생성
│   ├── bench_mpcref.npz          # ③ MPC 기준계획 — 자동 생성
│   ├── dpnn_model.pt             # ③ 학습된 DPNN 가중치 — 자동 생성
│   └── source/                   # 원본 (gitignore, 로컬 전용)
│       ├── 15분 수전량 2025.09~.xlsx
│       ├── 일일 수전량 2021.09~.csv
│       └── combined_sorted_weather_data.csv
└── reference/                    # gitignore (단, *.md 와 campus_pv_potential.csv 는 추적)
    ├── parameters.md             # ★ 파라미터 단일 진실 공급원 (출처·적용범위·민감도 대상)
    ├── optimization.md           # ② LP 목적함수·제약 설계와 수정 이력
    ├── dpnn_feasibility.md       # ③ DPNN 적용 타당성·설계 결정·진행 상황
    ├── team_review_and_plan.md   # 조원 자료 정합성 검토 + 전체 로드맵
    ├── soc_patch_for_teammate.md # 조원 코드 SOC 버그 패치 안내
    ├── campus_pv_potential.csv   # 자과캠 24개 건물 PV 설치 가능 데이터 (②가 런타임에 읽음)
    ├── 자과캠_PV_ESS_경제성분석.xlsx        # 조원 경제성 분석 본체 (15시트)
    ├── 진우님코드_수정본.txt 외 코드 .txt   # 조원이 수정한 ② 코드
    ├── 공고문·발표자료·논문 PDF             # 수치 근거 원문
    └── _archive/                 # 현재 미사용 (삭제 후보)
```

### 노트북 3개의 역할

세 노트북은 **① → ② → ③** 순으로 산출물을 넘긴다. 각자 독립 실행 가능하지만 모형·파라미터는
`reference/parameters.md` 하나를 공유한다.

#### ① `temperature_scenario_analysis.ipynb` — 기온-수요 반응 분석

**하는 일**: 일 단위 장기 데이터로 전력수요의 기온 반응을 추정하고, 그 모형으로 기후 시나리오를 만든다.

| 절 | 내용 |
| --- | --- |
| 1 | temperature-response 모델 확정 — 자연 3차 스플라인(당일·전일·전전일 기온 df=7) + 습도·강수·캘린더, Newey-West HAC, Duan smearing |
| 2 | 기온상승 시나리오 — 임의 +1~+4℃ 및 **SSP1-2.6/2-4.5/3-7.0/5-8.5 × 2030s/2050s**(수원 격자값) |
| 3 | 캘린더 시나리오 — 공휴일·방학 효과 |
| 4 | 전기요금 구조와 시나리오별 비용 |
| 5 | **복합재해 유의성 검토** — 폭염주의보·경보·열대야·대설주의보 더미를 추가해 유의성 확인 |

**산출**: 최종 모형 `res_multi`(R²=0.917), SSP 기온상승폭. ②가 이 스펙을 그대로 재적합해 쓴다.

#### ② `ess_pv_pipeline.ipynb` — ESS·PV 최적화 파이프라인 (트랙 A)

**하는 일**: ①의 모형으로 기후 시나리오별 15분 부하를 만들고, ESS·PV 디스패치와 PV 설비용량을
동시에 최적화한다. **융합연구학점제 산출물**이자 공고 준수성 검증.

| 단계 | 내용 | 산출 |
| --- | --- | --- |
| 1 | 실측 일중 비중 × 시나리오별 일별 총량 → 15분 부하 | `load_15min` |
| 2 | 일사량·기온 → PV 발전량 | `G_t_Wm2` |
| 3 | ESS 물리 제약·비용 파라미터 | `ESS_PARAMS`, `COST_PARAMS` |
| 4 | 디스패치 LP — 9 기후 시나리오 × 5 PV 설비단계 = **45개 조합** + ESS 효과 분해 | `dispatch_results`, `ess_dec` |
| 5 | **Rainflow 사후검증** (지도교수 자문 2단계) | `rainflow_df` |
| 6 | **경부하기 의무충전 제약** (공고 준수성) | `mandate_df` |
| 7 | **탄소 가격 민감도·베네핏 분해** (25,500 / 51 / 76 USD) | `carbon_df` |
| 8 | **민감도 통합** (PV CAPEX·O&M·BIPV 단가·옥상 설치비율) | `sens_pv`, `sens_roof` |
| 9 | **기후 시나리오 종합** (PNAS 스타일 ΔT 반응) | 그림 |

**범위 주의**: 목적함수는 청구서 전 항목(TOU + 기후환경 + 연료비조정, ×1.132) + 탄소 + 열화 +
PV 설비비다. 다만 **투자비·국비(70%)·NPV는 없고** ESS 용량은 조원 모델의 LP 최적값을 고정값으로
받아 쓴다. 사업 타당성은 조원 경제성 모델이 담당한다. 모든 인용 수치는
`reference/parameters.md` **(14)절**이 기준이다.

#### ③ `dpnn_benchmark.ipynb` — DPNN 벤치마크 EMS (트랙 B)

**하는 일**: 벤치마크 논문(Liu, Zhao & He 2026, *Energy Economics* 154)의 **Deep Projection
Neural Network**를 우리 문제에 맞춰 구현·검증한다. 연구 본체.

| 절 | 내용 | 산출 |
| --- | --- | --- |
| 1 | 연간 완전예지 LP + 48h receding-horizon MPC, 반증 테스트 | `ANN`, `MPCREF` |
| 2 | 초단기 보정문제(5변수) 정식화 + PNN 적분기 + QP 대조군 | `pnn_solve`, `qp_solve` |
| 3 | DPNN 학습(식 14·20, 비지도) + 교대투영 + 스윕 속도 비교 | `MODEL` |

**핵심 결과**: 완전예지에서 MPC는 연간 LP와 비용이 동일(**myopia 0.000%**, 임계는 ESS가
일평균 부하의 256%)이고, PNN은 QP와 목적함수 초과 **+0.0000%**로 일치하며, DPNN은 교대투영 후
목적함수 초과 중앙 **+0.96%**, 스윕 가속 **3,000~5,000배**다.

### 데이터셋

> `data/source/` 는 `.gitignore` 로 제외(대용량·로컬 전용). `data/*.csv` 는 추적하고,
> `*.npz`·`*.pt` 는 노트북이 자동 생성하므로 추적하지 않아도 된다.

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
