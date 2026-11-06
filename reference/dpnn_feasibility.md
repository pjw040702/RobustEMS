# DPNN 적용 타당성 노트 — 우리 LP를 PNN 투영형으로 쓸 수 있는가

작성 2026-11-05. 대상 논문: Liu, Zhao & He (2026), *Energy Economics* 154, 109104,
"Economic emission dispatching strategy considering dynamic parameter effects: A novel approach
based on projection neural networks and deep learning".
결정 사항: **(a) 호라이즌 통째 vs (b) 초단기 보정 계층** 중 택일.

---

## 결론

**(b) 초단기 보정 계층이 맞고, (a)는 사실상 불가능하다.**

단순히 (b)가 편해서가 아니다. **PNN이 풀 수 있는 문제 형태가 "선형 등식 + 박스 제약"뿐**이고,
우리 LP의 기본요금 항은 그 형태에 들어가지 않는다. (b)로 가면 그 항이 **박스 제약으로 전환**되어
비로소 호환된다. 아래가 근거다.

추가로 **목적함수를 2차 추종(tracking) 형태로 둬야 한다**는 설계 결정이 따라 나온다(5절).

---

## 1. PNN이 풀 수 있는 문제 형태 (논문 2.1~2.2절)

논문은 EED를 다음 형태로 정리한 뒤(식 10) PNN을 적용한다.

$$\min f(P_G) \quad \text{s.t.}\quad A P_G = D,\qquad P_G \in \Omega$$

PNN 상태방정식(식 11~12):

$$\frac{dz}{dt}=\lambda\big(-Ug(z)-(I-U)(z-g(z)+\nabla f((I-U)g(z)+q))+q\big),\qquad P_G=g(z)$$

여기서 $U=A^\top(AA^\top)^{-1}A$, $q=A^\top(AA^\top)^{-1}D$, 그리고 $g(\cdot)$는 **투영 연산자**다.

**결정적 제약**: 논문이 쓰는 $g(\cdot)$는 **성분별 clamp**다.

$$\Omega=\prod_{i=1}^{N}[P_{G,i,\min},\,P_{G,i,\max}],\qquad
g(z_i)=\begin{cases} P_{G,i,\min} & z_i<P_{G,i,\min}\\ z_i & \text{(범위 내)}\\ P_{G,i,\max} & z_i>P_{G,i,\max}\end{cases}$$

즉 $\Omega$가 **박스(구간의 곱)** 여야 clamp가 성립한다. 등식 제약은 투영행렬 $U$가 처리한다.

> **정리: PNN은 "선형 등식 + 박스 경계"만 다룬다. 일반 부등식 제약은 들어갈 자리가 없다.**
> (논문도 결론에서 PNN의 한계를 "convex feasible region 기반 투영 연산자"라고 명시한다.)

## 2. 현재 우리 LP와의 대조

`ess_pv_pipeline.ipynb` 4-2 `solve_dispatch()` 기준.

| 구성 | 형태 | PNN 호환 |
| --- | --- | --- |
| 전력수지 $P_{grid,t}-P_{curt,t}+P_{dis,t}-P_{ch,t}+C_{PV}p_{unit,t}=L_t$ | 선형 등식 (T개) | ✓ |
| SOC 동역학 $E_k=E_{k-1}+\eta\Delta t P_{ch,k-1}-\frac{\Delta t}{\eta}P_{dis,k-1}$ | 선형 등식 (T개) | ✓ |
| $P_{ch},P_{dis},P_{grid},P_{curt},E$ 상·하한 | 박스 | ✓ |
| **기본요금** $P_{grid,t}-P_{peak}\le 0$ (∀t) | **일반 부등식** (T개) | **✗** |
| **PV curtailment** $P_{curt,t}-C_{PV}p_{unit,t}\le 0$ | **일반 부등식** (T개) | **✗** |
| **의무충전 SOC 상승** $E_{t_0}-E_{t_1+1}\le -\delta$ | **일반 부등식** | **✗** |

**(a)가 막히는 지점**: 호라이즌을 통째로 DPNN에 넘겨도 위 세 부등식은 박스가 되지 않는다.
특히 기본요금은 $P_{peak}$가 **결정변수**이고 전 구간을 묶는 epigraph 제약이라, 호라이즌을
키우든 줄이든 형태가 바뀌지 않는다. 투영 연산자를 일반 다면체용으로 새로 설계해야 하는데,
그건 **PNN을 가져다 쓰는 게 아니라 새 알고리즘을 만드는 일**이라 연구 범위(README:
"신규 최적화 알고리즘 개발이 아님")를 벗어난다.

## 3. (b)가 형태를 맞추는 방식

(b)에서는 상위 **단기 MPC**(기존 LP, 24~48h)가 계획을 세우고, DPNN은 **매 15분 시점의 보정량**만
출력한다. 이때 세 가지가 바뀐다.

1. **SOC를 변수에서 입력 파라미터로 내린다** → SOC 동역학 등식과 $E$ 경계가 그 시점 문제에서
   사라진다. 현재 SOC는 그냥 알려진 숫자다.
2. **$P_{peak}$가 변수에서 파라미터가 된다** — MPC가 피크 목표 $\bar P_{peak}$를 내려주므로,
   $P_{grid,t}\le P_{peak}$가 **$P_{grid,t}$의 박스 상한**이 된다. ★ 이게 핵심이다.
3. **$C_{PV}$와 $p_{unit,t}$가 모두 상수** → $P_{curt,t}\le C_{PV}p_{unit,t}$도 **박스 상한**이 된다.
   의무충전 구간의 방전 금지도 $P_{dis,t}\le 0$, 즉 박스 상한이다.

결과적으로 시점 $t$의 문제는 다음이 된다 — **정확히 논문 식 (10)의 형태**다.

$$\min_{x_t} f(x_t)\quad\text{s.t.}\quad
\underbrace{[\,1,\,-1,\,1,\,-1\,]}_{A}\,x_t=\underbrace{L_t-C_{PV}p_{unit,t}}_{D},\qquad x_t\in\Omega$$

$$x_t=[P_{grid,t},\,P_{curt,t},\,P_{dis,t},\,P_{ch,t}]^\top,\qquad
\Omega=[0,\bar P_{peak}]\times[0,C_{PV}p_{unit,t}]\times[0,\bar P_{dis,t}]\times[0,C_{PCS}]$$

변수 4개, 등식 1개, 박스 4개. $A$가 $1\times4$라 $U=A^\top(AA^\top)^{-1}A$도 닫힌 형태로 즉시 계산된다.

## 4. SOC 경계는 어디로 가는가 (처리 필요)

SOC를 파라미터로 내리면 $0.2C_{ESS}\le E_t\le 0.9C_{ESS}$를 그 시점에서 지킬 방법이 필요하다.
다행히 이것도 **박스로 환원된다** — 현재 SOC $E_t$가 주어졌으므로 다음 한 스텝에서 가능한
충·방전량이 산술로 정해진다.

$$P_{ch,t}\le\min\Big(C_{PCS},\ \frac{0.9C_{ESS}-E_t}{\eta_{ch}\Delta t}\Big),\qquad
P_{dis,t}\le\min\Big(C_{PCS},\ \frac{(E_t-0.2C_{ESS})\,\eta_{dis}}{\Delta t}\Big)$$

즉 SOC 경계가 **매 시점 박스 상한을 좁히는 형태**로 흡수된다. 추가 부등식이 생기지 않는다.

## 5. 따라 나오는 설계 결정 — 목적함수를 2차 추종형으로

(b)의 문제는 변수 4개·등식 1개뿐이라, **목적함수가 선형이면 해가 자명해진다**(충·방전 둘 다
$c_{deg}$만 들고 아무 이득이 없으므로 $P_{ch}=P_{dis}=0$, $P_{grid}=L_t-PV$). 시점 단위에서는
TOU 차익거래 유인이 존재할 수 없다 — 그건 본질적으로 시간 간 결합이기 때문이다.

따라서 초단기 계층의 목적함수는 **MPC 기준계획 추종 + 편차 패널티**의 2차 형태여야 한다.

$$f(x_t)=\underbrace{w_1\|P_{ch,t}-\bar P_{ch,t}\|^2+w_1\|P_{dis,t}-\bar P_{dis,t}\|^2
+w_2\|E_{t+1}-\bar E_{t+1}\|^2}_{\text{MPC 계획 추종}}
+\underbrace{\pi_t\Delta t\,P_{grid,t}}_{\text{실제 전력비용}}$$

이게 오히려 **논문과 더 잘 맞는다** — 논문의 $f$도 2차($C_i=a_iP^2+b_iP+c_i$)이고, 강볼록이라
해가 유일하고 $\nabla f$가 매끄러워 PNN 수렴 이론이 그대로 적용된다. 선형 LP였다면 $\nabla f$가
상수여서 PNN 궤적이 경계에서 잘 정의되지 않는 문제가 있었다.

실시간 보정 계층을 추종형으로 두는 것은 MPC 문헌의 표준 관행이므로 억지 설계가 아니다.

## 6. Dynamic parameter 공간 $\Phi$ 정의

DPNN 입력 $\phi$는 다음으로 둔다. 논문의 $\phi$(수요·기후인자)에 우리 스트레스 축을 더한 것이다.

| 구분 | 변수 | 비고 |
| --- | --- | --- |
| 실측 상태 | 현재 SOC $E_t$, 실측 부하 $L_t$, 실측 PV $p_{unit,t}C_{PV}$ | 논문의 "power demand", "climatic factors" 대응 |
| MPC 기준 | $\bar P_{ch,t},\bar P_{dis,t},\bar E_{t+1},\bar P_{peak}$ | 상위 계층 출력 |
| 요금 | $\pi_t$ (TOU 시간대) | |
| **스트레스 축** | **AI workload 침투율**, **기온 상승폭 $\Delta T$** | README RQ1·RQ2의 2차원 공간 |

학습은 논문 식 (20)의 **unsupervised PNN-guided MSE 손실**을 쓴다 —
$L(G;W)=\frac{1}{|G|}\sum(\partial\hat z/\partial t-\Psi_{\phi}(\hat z))^2$.
**미리 풀어둔 해 데이터셋이 필요 없다**는 점이 구현 부담을 크게 줄인다(PyTorch 자동미분).

## 7. 정직하게 짚어둘 것 — 속도 논거의 위치

(b)에서 한 번의 문제는 변수 4개 QP다. 이건 어떤 solver로도 마이크로초 단위라서,
**"한 번 푸는 속도"에서는 DPNN의 이득이 없다.** 논문의 속도 주장을 그대로 가져오면 반박당한다.

우리 쪽 속도 논거는 **스윕 규모**에 있다. 스트레스 테스트는
`AI 침투율 × 폭염 강도 × 기후 시나리오`를 격자로 훑으며 각 조합마다 34,944개 시점을 돌린다.
조합 100개면 **약 350만 번의 반복 해**가 필요하고, 이것이 논문이 지적한
*"computational inefficiency caused by repeated solving when dealing with EED involving
continuous changes in dynamic parameters"* 바로 그 상황이다. DPNN은 파라미터가 바뀌어도
재학습·반복 없이 forward pass 한 번(배치 처리 가능)이다.

→ 보고서·논문에서는 **"단일 디스패치 가속"이 아니라 "파라미터 공간 스윕 가속"**으로 서술해야 한다.

## 8. 진행 상황 (2026-11-05, `dpnn_benchmark.ipynb`)

| 단계 | 상태 |
| --- | --- |
| 1. PNN 재현 (식 11~13) | **완료** — 경계 활성 6케이스에서 QP와 1.8e-06 이내, 배치 RK4로 4.2e-08 |
| 2. 단기 MPC 재구성 | **완료** — 48h receding horizon, 연간 완전예지 LP 대비 **괴리 0.000%** |
| 3. DPNN 학습 (식 14·20) | **부분 완료** — 아래 ③ 참고 |
| 4. 검증 (품질·스윕 속도) | **완료** — 스윕 가속 1,300~2,100배(Φ 1만~5만) |
| 5. AI workload → stress test | 미착수 (Phase 4~5) |

**① 배치 RK4 안정조건.** 명시적 RK4는 $\lambda h\lesssim 2.8$를 넘으면 발산한다(NaN).
$\lambda=1000,T=8$에서 `n_steps` 2,000(λh=4) → NaN, 4,000(λh=2) → 8.4e-03,
**8,000(λh=1) → 4.2e-08**. 8,000 채택.

**② MPC myopia 비용이 0.** 반증 테스트로 ESS를 100배 키우면 괴리가 +1.89%로 나타나므로
구현 오류가 아니다. ESS 428kWh가 일평균 부하 264MWh의 0.16%라 다일 차익거래 여지가 없고,
**이 문제의 유일한 전역 결합은 연간 피크**다. → 완전예지에서 계층적 MPC = 연간 LP이므로
성능 열화는 **예측오차에서만** 나온다(연구안 4.5절이 차별성의 전부).
→ **DPNN의 가치는 "더 좋은 디스패치"가 아니라 "스윕 가속 + 고정 벤치마크"뿐이다.**

**③ 논문 손실함수가 우리 규모에서는 느슨한 대리지표다 — 이 단계의 핵심 발견.**
$\Psi$의 등식 성분이 $q-Ug=\tilde A(D-\tilde A\!\cdot\!g)/\lVert\tilde A\rVert^2$이므로

$$\lvert Ax-D\rvert=\lVert\tilde A\rVert\cdot\lVert q-Ug\rVert,\qquad \lVert\tilde A\rVert\approx 2.24\times10^4$$

즉 $\Psi$ 잔차 $10^{-5}$ = 등식 위반 0.22kW. 학습이 도달한 $\lVert\Psi\rVert\approx0.5$는
**등식 위반 12,740kW**다. 등식을 1kW로 맞추려면 손실이 $10^{-9}$까지 내려가야 하는데 달성값보다
여섯 자릿수 아래다. 손실은 $0.96\to7.8\times10^{-3}$(120배), $\lVert\Psi\rVert$는 700배 줄었지만
해 정확도는 전혀 개선되지 않았다(1차 시도와 소수점 7자리까지 동일). 논문의 발전기 10기
$O(1\!\sim\!100)$ 테스트케이스에서는 드러나지 않는 제약이다.

**처방: 등식을 손실이 아니라 구조로 강제한다.** 등식이 단일 선형 제약이라 초평면 투영이 닫힌
형태이므로 박스 투영과 교대 적용한다(Dykstra식).

| 후처리 | 상대오차 중앙 | 최대 | 등식잔차 최대 |
| --- | --- | --- | --- |
| clamp만 (논문 그대로) | 9.00e-01 | 1.75e+00 | 1.27e+04 kW |
| + 교대투영 1회 | 1.82e-01 | 3.51e-01 | 7.3e-12 kW |
| + 교대투영 3회 | 1.04e-02 | 1.94e-02 | 3.6e-12 kW |
| **+ 교대투영 10회** | **4.47e-03** | 3.11e-02 | 3.6e-12 kW |

**재학습 없이 후처리만으로 200배 개선**된다. 다음 개선은 이 투영을 forward pass 안에 넣고
재학습하는 것(네트워크가 등식 맞추기에 용량을 쓰지 않게 된다).

## 9. 다음 단계

1. 투영을 forward pass에 내장해 DPNN 재학습 → 목표 상대오차 $10^{-4}$ 수준
2. AI workload 시나리오 생성 (Alibaba PAI / MIT Supercloud / PM100)
3. 예측기(LSTM/GRU + prediction interval) → 예측오차를 $\Phi$에 주입
4. Stress test → failure boundary, 성능 열화 분해

### 구현 중 해결된 사항 (2026-11-05, `dpnn_benchmark.ipynb` Phase 3)

**① $\bar P_{peak}$ 초과 문제 → $P_{grid}$ 분할로 해결.** 실측이 목표를 넘으면 $\Omega$가 공집합이
되는 문제가 실제로 발생했다(무작위 $\Phi$ 샘플 중 일부에서 상대오차 1.0). 2차 힌지 패널티
$\tfrac{w}{2}(P_{grid}-\bar P_{peak})_+^2$로 완화하려 했으나, 정규화 좌표에서 기울기가 $10^9$까지
커져 ODE가 극도로 stiff해졌다. **해결: $P_{grid}=P_{base}+P_{over}$로 분할한다.**

$$x=[P_{base},P_{over},P_{curt},P_{dis},P_{ch}]^\top,\quad A=[1,1,-1,1,-1],\quad
P_{base}\in[0,\bar P_{peak}],\ P_{over}\in[0,\,C_{contract}-\bar P_{peak}]$$

초과분이 **독립 변수 + 선형비용** $M_{over}P_{over}$가 되어 (i) 항상 실행가능, (ii) 전부 박스 제약
유지, (iii) 기울기가 상수라 stiff하지 않다. 검증 결과 **전 샘플에서 $P_{over}^*=0$** — 하드 피크
제약과 동일한 해를 주면서 실행가능성만 확보한다. $M_{over}$의 경제적 정확값은
$6{,}980\times12\times1.132=94{,}816$원/kW이나 조건수 때문에 5,000원/kW를 쓴다(초과가 최후수단인
한 해가 같으므로 무해, 위 검증으로 확인).

**② 수렴에 필요한 가상시간 $\lambda T$.** $\lambda=20,T=60$에서는 일부 $\Phi$가 미수렴해
등식 잔차가 4,130kW까지 났다. $\lambda T\approx 8{,}000$이 필요하며 **$\lambda=1000,\,T=8$**
(중앙오차 8.7e-14, 최대 4.7e-10, 등식잔차 ≤1.2e-10)을 채택했다.

**③ 비차원화 필수.** 변수 크기가 $P_{grid}\sim10^4$ vs $P_{dis}\sim10^2$로 벌어져 물리 단위로는
조건수가 나쁘다. 박스 폭으로 나눠 단위박스 $y\in[0,1]^5$로 보내고 목적함수를 $F_{scale}=10^5$로
나눈다. $\tilde A=A\odot w$가 rank-1이므로 $U=\tilde A^\top\tilde A/\lVert\tilde A\rVert^2$,
$q=\tilde A^\top D/\lVert\tilde A\rVert^2$로 닫힌 형태다(배치 계산 가능).

### 미해결 위험

- **기본요금의 귀속**: 연간 최대피크가 12개월 요금을 결정하므로, MPC가 내려주는
  $\bar P_{peak}$를 어떻게 정할지가 별도 문제다. 현재는 **연간 완전예지 LP의 최적 피크를
  목표로 그대로 사용**한다(이러면 Phase 5의 분해 기준선과 자동으로 정합). 월·일 단위 분배 규칙은
  예측오차를 도입할 때 재검토 필요.
