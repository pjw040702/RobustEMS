# 조원 코드 SOC 인덱스 패치 (2026-11-05)

대상: `자과캠_PV_ESS_최적화_재현코드.txt` / `진우님코드_수정본.txt` 의 `solve()`
(내 `ess_pv_pipeline.ipynb` 4-2에서 그대로 옮겨간 버그라 양쪽 모두 해당)

## 증상

SOC 동역학 제약이 `for t in range(1,T)`인데 `iE`가 **T개**라서, **t=0의 충·방전이 어떤
에너지수지 식에도 등장하지 않는다.** `E[0]`은 등식으로 `0.5×5×P_ess`에 고정돼 있으므로,
LP가 **SOC를 전혀 바꾸지 않고 PCS 최대치로 방전**하는 '공짜 에너지'를 만들어낸다.

내 코드(C_ESS 428kWh 고정)에서 실측한 결과:

```
수정 전:  P_dis = [142.7, 0, 0, ...]   E = [214, 214, 214, ...]   <- SOC 불변
          t=0 에너지수지 미반영분 = +39.16 kWh/년
수정 후:  P_dis = [0, 0, 0, ...]       미반영분 = 0.00 kWh
```

**금액 영향은 무시할 수준**이다. P_ess 1kW당 이득이 약 15.5원인데 투자비 연간환산이
약 284,000원/kW(=449,800×5×CRF(4.5%,10))이므로 **최적 ESS 용량은 바뀌지 않을 것**이다.
그래도 에너지 보존 위반이라 심사에서 짚이면 곤란하다.

## 패치 (5곳)

`E`를 구간 경계의 상태량으로 보아 **T+1개**(E[0]~E[T])로 늘리고, T개 구간의 충·방전이
모두 반영되게 한다.

```python
# 1) 변수 개수  5*T + 3  ->  5*T + 4
N = 5*T + 4

# 2) 인덱스: iE 를 T+1개로, 이후 스칼라 인덱스 +1
ich, idis, igrid, icurt = (np.arange(k*T, (k+1)*T) for k in range(4))
iE = np.arange(4*T, 5*T + 1)                 # <- T+1개
ipk, iP, iZ = 5*T + 1, 5*T + 2, 5*T + 3      # <- 각각 +1

# 3) bounds 의 E 블록을 T+1개로 (5번째 [(0,None)]*T 만 수정)
bounds = ([(0,None)]*(2*T) + [(0,CONTRACT_KW)]*T + [(0,None)]*T
          + [(0,None)]*(T+1)                 # <- E: T -> T+1
          + [(0,None), (0,p_ess_max), (0,None)])

# 4) SOC 동역학: 모든 T개 구간의 충·방전을 반영
for k in range(1, T+1):                      # <- range(1,T) -> range(1,T+1)
    R += [r_]*4
    C += [iE[k], iE[k-1], ich[k-1], idis[k-1]]   # <- ich[k]/idis[k] -> [k-1]
    V += [1, -1, -ETA_CH*DT_H, DT_H/ETA_DIS]
    b.append(0.0); r_ += 1

# 5) 초기/말기 SOC 고정: 끝점이 T-1 -> T
for k in (0, T):                             # <- (0, T-1) -> (0, T)
    R += [r_]*2; C += [iE[k], iP]; V += [1, -SOC_INIT*ESS_HOURS]
    b.append(0.0); r_ += 1
```

그리고 부등식 쪽 SOC 범위도 T+1개 전부에 걸어야 한다.

```python
for t in range(T):
    add([igrid[t],ipk],[1,-1],0.0)
    add([icurt[t]],[1],c_pv*p_unit[t])
    add([ich[t],iP],[1,-1],0.0); add([idis[t],iP],[1,-1],0.0)
    # SOC 범위는 아래 별도 루프로 분리 (E가 T+1개이므로)

for k in range(T+1):                         # <- range(T) -> range(T+1)
    add([iE[k],iP],[ 1, -SOC_MAX*ESS_HOURS], 0.0)   # E <= 0.9*5P
    add([iE[k],iP],[-1,  SOC_MIN*ESS_HOURS], 0.0)   # E >= 0.2*5P
```

## 검증 방법

수정 후 다음이 성립해야 한다.

```python
# t=0 구간의 에너지수지가 닫히는지
lhs = o["E"][1] - o["E"][0]
rhs = ETA_CH*DT_H*o["P_ch"][0] - DT_H/ETA_DIS*o["P_dis"][0]
assert abs(lhs - rhs) < 1e-6        # 수정 전에는 P_dis[0]>0 인데 lhs=0 이라 실패
```

## 내 쪽 결과 변화 (참고)

총비용은 억원 단위에서 불변(127.52 / 127.18 / 112.62억원)이었고,
Rainflow 괴리율만 -0.46% → **-0.43%**로 갱신됐다(허수 방전이 throughput에서 빠짐).
