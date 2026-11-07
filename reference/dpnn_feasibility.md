# DPNN 적용 설계 — 우리 LP를 PNN 투영형으로 쓰는 방법

대상 논문: Liu, Zhao & He (2026), *Energy Economics* 154, 109104, "Economic emission dispatching
strategy considering dynamic parameter effects: A novel approach based on projection neural networks
and deep learning". 구현은 `dpnn_benchmark.ipynb`.

---

## 결론

**DPNN은 초단기 보정 계층에 배치한다. 호라이즌을 통째로 출력하는 방식은 불가능하다.**

단순히 편해서가 아니다. **PNN이 풀 수 있는 문제 형태가 "선형 등식 + 박스 제약"뿐**이고 우리 LP의
기본요금 항은 그 형태에 들어가지 않는다. 초단기 계층으로 내리면 그 항이 **박스 제약으로 전환**되어
비로소 호환된다. 추가로 **목적함수를 2차 추종(tracking) 형태로 둬야 한다**는 설계 결정이 따라
나온다(5절).

---

## 1. PNN이 풀 수 있는 문제 형태 (논문 2.1~2.2절)

논문은 EED를 다음 형태로 정리한 뒤(식 10) PNN을 적용한다.

$$\min f(P_G) \quad \text{s.t.}\quad A P_G = D,\qquad P_G \in \Omega$$

PNN 상태방정식(식 11~12):

$$\frac{dz}{dt}=\lambda\big(-Ug(z)-(I-U)(z-g(z)+\nabla f((I-U)g(z)+q))+q\big),\qquad P_G=g(z)$$

여기서 $U=A^\top(AA^\top)^{-1}A$, $q=A^\top(AA^\top)^{-1}D$, $g(\cdot)$는 **투영 연산자**다.

**결정적 제약**: 논문이 쓰는 $g(\cdot)$는 **성분별 clamp**다.

$$\Omega=\prod_{i=1}^{N}[P_{G,i,\min},\,P_{G,i,\max}],\qquad
g(z_i)=\begin{cases} P_{G,i,\min} & z_i<P_{G,i,\min}\\ z_i & \text{(범위 내)}\\ P_{G,i,\max} & z_i>P_{G,i,\max}\end{cases}$$

즉 $\Omega$가 **박스(구간의 곱)** 여야 clamp가 성립한다. 등식 제약은 투영행렬 $U$가 처리한다.

> **정리: PNN은 "선형 등식 + 박스 경계"만 다룬다. 일반 부등식 제약은 들어갈 자리가 없다.**
> (논문도 결론에서 PNN의 한계를 "convex feasible region 기반 투영 연산자"라고 명시한다.)

## 2. 우리 LP와의 대조

`ess_pv_pipeline.ipynb`의 `solve_dispatch()` 기준.

| 구성 | 형태 | PNN 호환 |
| --- | --- | --- |
| 전력수지 $P_{grid,t}-P_{curt,t}+P_{dis,t}-P_{ch,t}+C_{PV}p_{unit,t}=L_t$ | 선형 등식 ($T$개) | ✓ |
| SOC 동역학 $E_k=(1-\delta)E_{k-1}+\eta\Delta t P_{ch,k-1}-\frac{\Delta t}{\eta}P_{dis,k-1}$ | 선형 등식 ($T$개) | ✓ |
| $P_{ch},P_{dis},P_{grid},P_{curt},E$ 상·하한 | 박스 | ✓ |
| **기본요금** $P_{grid,t}-P_{peak}\le 0$ (∀t) | **일반 부등식** ($T$개) | **✗** |
| **PV curtailment** $P_{curt,t}-C_{PV}p_{unit,t}\le 0$ | **일반 부등식** ($T$개) | **✗** |
| **의무충전 SOC 상승** $E_{t_0}-E_{t_1+1}\le -\delta$ | **일반 부등식** | **✗** |

**호라이즌 통째 방식이 막히는 지점**: 호라이즌을 그대로 DPNN에 넘겨도 위 세 부등식은 박스가 되지
않는다. 특히 기본요금은 $P_{peak}$가 **결정변수**이고 전 구간을 묶는 epigraph 제약이라 호라이즌을
키우든 줄이든 형태가 바뀌지 않는다. 투영 연산자를 일반 다면체용으로 새로 설계해야 하는데, 그건
**PNN을 가져다 쓰는 게 아니라 새 알고리즘을 만드는 일**이라 연구 범위(README: "신규 최적화
알고리즘 개발이 아님")를 벗어난다.

## 3. 초단기 보정 계층이 형태를 맞추는 방식

상위 **단기 MPC**(기존 LP, 24~48h)가 계획을 세우고 DPNN은 **매 15분 시점의 보정량**만 출력한다.
이때 세 가지가 바뀐다.

1. **SOC를 변수에서 입력 파라미터로 내린다** → SOC 동역학 등식과 $E$ 경계가 그 시점 문제에서
   사라진다. 현재 SOC는 그냥 알려진 숫자다.
2. **$P_{peak}$가 변수에서 파라미터가 된다** — MPC가 피크 목표 $\bar P_{peak}$를 내려주므로
   $P_{grid,t}\le P_{peak}$가 **$P_{grid,t}$의 박스 상한**이 된다. ★ 이게 핵심이다.
3. **$C_{PV}$와 $p_{unit,t}$가 모두 상수** → $P_{curt,t}\le C_{PV}p_{unit,t}$도 **박스 상한**이 된다.
   의무충전 구간의 방전 금지도 $P_{dis,t}\le 0$, 즉 박스 상한이다.

결과적으로 시점 $t$의 문제는 **정확히 논문 식 (10)의 형태**가 된다.

$$\min_{x_t} f(x_t)\quad\text{s.t.}\quad A\,x_t=D,\qquad x_t\in\Omega$$

$A$가 $1\times n$ 행벡터라 $U=A^\top(AA^\top)^{-1}A$도 닫힌 형태로 즉시 계산된다. 구현에서
$P_{grid}$를 둘로 쪼개 변수 5개를 쓰는 이유는 8절 ①에 있다.

**MPC는 DPNN의 대안이 아니라 DPNN이 들어가 앉을 자리다.** 연구안 4.3이 이미 *단기 MPC(24~48h
계획) + 초단기 실시간 보정 + receding horizon* 2계층이므로 억지 설계가 아니다.

## 4. SOC 경계 처리

SOC를 파라미터로 내리면 $0.2C_{ESS}\le E_t\le 0.9C_{ESS}$를 그 시점에서 지킬 방법이 필요하다.
이것도 **박스로 환원된다** — 현재 SOC $E_t$가 주어졌으므로 다음 한 스텝에서 가능한 충·방전량이
산술로 정해진다.

$$P_{ch,t}\le\min\Big(C_{PCS},\ \frac{0.9C_{ESS}-E_t}{\eta_{ch}\Delta t}\Big),\qquad
P_{dis,t}\le\min\Big(C_{PCS},\ \frac{(E_t-0.2C_{ESS})\,\eta_{dis}}{\Delta t}\Big)$$

즉 SOC 경계가 **매 시점 박스 상한을 좁히는 형태**로 흡수된다. 추가 부등식이 생기지 않는다.
구현에서 이 두 주장(피크 목표 → `P_grid` 박스 상한, 현재 SOC → 충·방전 박스 상한)이 확인됐고
**추가 부등식 제약이 하나도 필요하지 않았다.**

## 5. 목적함수는 2차 추종형이어야 한다

이 문제는 변수가 적고 등식이 1개뿐이라 **목적함수가 선형이면 해가 자명해진다**(충·방전 둘 다
$c_{deg}$만 들고 아무 이득이 없으므로 $P_{ch}=P_{dis}=0$, $P_{grid}=L_t-PV$). 시점 단위에서는
TOU 차익거래 유인이 존재할 수 없다 — 그건 본질적으로 시간 간 결합이기 때문이다.

따라서 초단기 계층의 목적함수는 **MPC 기준계획 추종 + 편차 패널티**의 2차 형태여야 한다.

$$f(x_t)=\underbrace{w_1\|P_{ch,t}-\bar P_{ch,t}\|^2+w_1\|P_{dis,t}-\bar P_{dis,t}\|^2
+w_2\|E_{t+1}-\bar E_{t+1}\|^2}_{\text{MPC 계획 추종}}
+\underbrace{\pi_t\Delta t\,P_{grid,t}}_{\text{실제 전력비용}}$$

이게 오히려 **논문과 더 잘 맞는다** — 논문의 $f$도 2차($C_i=a_iP^2+b_iP+c_i$)이고 강볼록이라
해가 유일하고 $\nabla f$가 매끄러워 PNN 수렴 이론이 그대로 적용된다. 선형 LP였다면 $\nabla f$가
상수여서 PNN 궤적이 경계에서 잘 정의되지 않는다. 실시간 보정 계층을 추종형으로 두는 것은 MPC
문헌의 표준 관행이다.

## 6. Dynamic parameter 공간 $\Phi$

DPNN 입력 $\phi$는 논문의 $\phi$(수요·기후인자)에 우리 스트레스 축을 더한 것이다.

| 구분 | 변수 | 비고 |
| --- | --- | --- |
| 실측 상태 | 현재 SOC $E_t$, 실측 부하 $L_t$, 실측 PV $p_{unit,t}C_{PV}$ | 논문의 "power demand", "climatic factors" 대응 |
| MPC 기준 | $\bar P_{ch,t},\bar P_{dis,t},\bar E_{t+1},\bar P_{peak}$ | 상위 계층 출력 |
| 요금 | $\pi_t$ (TOU 시간대) | |
| **스트레스 축** | **AI workload 침투율**, **기온 상승폭 $\Delta T$** | README RQ1·RQ2의 2차원 공간 |

학습은 논문 식 (20)의 **unsupervised PNN-guided MSE 손실**을 쓴다 —
$L(G;W)=\frac{1}{|G|}\sum(\partial\hat z/\partial t-\Psi_{\phi}(\hat z))^2$.
궤적은 $\hat z(s,\phi;W)=z_0(\phi)+(1-e^{-\kappa s})\,\mathrm{NN}(s,\phi;W)$로 두고 시간미분은
`torch.func.jvp`로 계산한다. **미리 풀어둔 해 데이터셋이 필요 없다**는 점이 구현 부담을 크게 줄인다.

## 7. 속도 논거의 위치

이 구조에서 한 번의 문제는 변수 5개 QP다. 어떤 solver로도 마이크로초 단위라서 **"한 번 푸는
속도"에서는 DPNN의 이득이 없다.** 논문의 속도 주장을 그대로 가져오면 반박당한다.

우리 쪽 속도 논거는 **스윕 규모**에 있다. 스트레스 테스트는
`AI 침투율 × 폭염 강도 × 기후 시나리오`를 격자로 훑으며 각 조합마다 34,944개 시점을 돌린다.
조합 100개면 **약 350만 번의 반복 해**가 필요하고, 이것이 논문이 지적한 *"computational
inefficiency caused by repeated solving when dealing with EED involving continuous changes in
dynamic parameters"* 바로 그 상황이다. DPNN은 파라미터가 바뀌어도 재학습·반복 없이 forward
pass 한 번(배치 처리 가능)이다.

→ 보고서·논문에서는 **"단일 디스패치 가속"이 아니라 "파라미터 공간 스윕 가속"**으로 서술한다.

## 8. 수치 설계

구현에서 확정된 네 가지. 어느 하나라도 빠지면 수렴하지 않는다.

### ① $P_{grid}$를 둘로 쪼갠다 — 실행가능성 보장

실측 부하가 피크 목표를 넘으면 $\Omega$가 공집합이 된다. 2차 힌지 패널티
$\tfrac{w}{2}(P_{grid}-\bar P_{peak})_+^2$로 완화하면 정규화 좌표에서 기울기가 $10^9$까지 커져
ODE가 극도로 stiff해진다. **$P_{grid}=P_{base}+P_{over}$로 분할한다.**

$$x=[P_{base},P_{over},P_{curt},P_{dis},P_{ch}]^\top,\quad A=[1,1,-1,1,-1],\quad
P_{base}\in[0,\bar P_{peak}],\ P_{over}\in[0,\,C_{contract}-\bar P_{peak}]$$

초과분이 **독립 변수 + 선형비용** $M_{over}P_{over}$가 되어 (i) 항상 실행가능, (ii) 전부 박스 제약
유지, (iii) 기울기가 상수라 stiff하지 않다. **전 샘플에서 $P_{over}^*=0$** — 하드 피크 제약과
동일한 해를 주면서 실행가능성만 확보한다. $M_{over}$의 경제적 정확값은
$6{,}980\times12\times1.132=94{,}816$원/kW이나 조건수 때문에 5,000원/kW를 쓴다(초과가 최후수단인
한 해가 같으므로 무해하고, 위 검증으로 확인했다).

### ② 비차원화 + 목적함수 **샘플별** 정규화

변수 크기가 $P_{grid}\sim10^4$ vs $P_{dis}\sim10^2$로 벌어져 물리 단위로는 조건수가 나쁘다.
박스 폭으로 나눠 단위박스 $y\in[0,1]^5$로 보낸다. $\tilde A=A\odot w$가 rank-1이므로
$U=\tilde A^\top\tilde A/\lVert\tilde A\rVert^2$, $q=\tilde A^\top D/\lVert\tilde A\rVert^2$로
닫힌 형태다(배치 계산 가능).

목적함수는 **샘플별 기울기 inf-norm으로 나눈다.** 고정 상수로 정규화하면 설비 규모가 바뀔 때
$\Psi$의 크기가 함께 변해 RK4 안정조건을 깨뜨린다 — `C_PCS`가 커지자 $\Psi$가 $O(30)$이 되어
PNN 상대오차가 $2.8\times10^{-14}$에서 **0.31**로 무너졌고 스텝을 늘려도 회복되지 않았다.
KKT 해는 목적함수의 양수배 스케일링에 불변이므로 샘플별 정규화는 해를 바꾸지 않는다.

> **파라미터를 바꿀 때는 수치 설정도 함께 검증해야 한다.**

### ③ 가상시간 $\lambda T$와 RK4 안정조건

명시적 RK4는 $\lambda h L\lesssim 2.8$를 넘으면 발산한다(NaN). 수렴에는 충분한 가상시간이
필요하다 — $\lambda T$가 작으면 일부 $\Phi$가 미수렴해 등식 잔차가 수천 kW까지 난다.

**채택: $\lambda=1{,}000$, $T=24$, `n_steps` $=24{,}000$** ($\lambda T=24{,}000$, $\lambda h=1$).

### ④ 논문 손실함수는 우리 규모에서 느슨한 대리지표다

$\Psi$의 등식 성분이 $q-Ug=\tilde A(D-\tilde A\!\cdot\!g)/\lVert\tilde A\rVert^2$이므로

$$\lvert Ax-D\rvert=\lVert\tilde A\rVert\cdot\lVert q-Ug\rVert,\qquad
\lVert\tilde A\rVert=O(10^4)$$

즉 $\Psi$ 잔차 $10^{-5}$가 등식 위반 약 0.2kW에 해당한다. 등식을 1kW로 맞추려면 손실이
$10^{-9}$까지 내려가야 하는데 학습이 도달하는 값보다 여섯 자릿수 아래다. 손실을 120배 줄여도
해 정확도가 개선되지 않는다. **논문의 발전기 10기 $O(1\!\sim\!100)$ 테스트케이스에서는 드러나지
않는 제약이다 — 이 단계의 핵심 발견이다.**

**처방: 등식을 손실이 아니라 구조로 강제한다.** 등식이 단일 선형 제약이라 초평면 투영이 닫힌
형태이므로 박스 투영과 교대 적용한다(Dykstra식). 재학습 없이 후처리만으로 등식 잔차가 기계정밀도
(3.6e-12 kW)까지 떨어지고 해 정확도가 수백 배 개선된다.

## 9. 검증 결과

| 항목 | 결과 |
| --- | --- |
| PNN vs scipy QP | 상대오차 중앙 **2.75e-14** / 최대 2.86e-04, 목적함수 초과 **+0.0000%** |
| 등식·박스 | 등식 잔차 ≤3.6e-12 kW, 박스 위반 0 |
| DPNN + 교대투영 | 목적함수 초과 중앙 **+0.96%** / 최대 +3.87% |
| 단기 MPC(48h) vs 연간 완전예지 LP | 괴리 **−0.000000%** |
| 스윕 가속 | Φ 1천~2만개에서 **약 3,000~3,500배** (측정 시점마다 변동) |

**MPC myopia 비용이 0인 것은 구현 오류가 아니다.** ESS를 ×100·×1000으로 키우는 반증 테스트에서
괴리가 나타나며, 임계는 ESS가 **일평균 부하의 256%**다(캠퍼스는 2.56%). ESS가 캠퍼스 규모 대비
작아 다일 차익거래 여지가 없고 **이 문제의 유일한 전역 결합은 연간 피크**다.

→ 완전예지에서 계층적 MPC = 연간 LP이므로 성능 열화는 **예측·lookahead 부재에서만** 나온다
(연구안 4.5절 분해의 근거). 따라서 **DPNN의 가치는 "더 좋은 디스패치"가 아니라 "스윕 가속 +
고정 벤치마크"다.**

## 10. 남은 것

1. 교대투영을 forward pass에 내장해 재학습 — 네트워크가 등식 맞추기에 용량을 쓰지 않게 된다
2. AI workload 시나리오 생성 (Alibaba PAI / MIT Supercloud / PM100)
3. Stress test → failure boundary, 성능 열화 분해(완전예지 vs 반응형)

### 미해결 위험

**기본요금의 귀속.** 연간 최대피크가 12개월 요금을 결정하므로 MPC가 내려주는 $\bar P_{peak}$를
어떻게 정할지가 별도 문제다. 현재는 **연간 완전예지 LP의 최적 피크를 목표로 그대로 사용**한다
(이러면 Phase 5의 분해 기준선과 자동으로 정합한다). 월·일 단위 분배 규칙은 lookahead 부재를
도입할 때 재검토가 필요하다.
