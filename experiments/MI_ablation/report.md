# MI Ablation 결과 레포트

## 실험 개요

DAG-KD의 Disentanglement MI(상호정보량 최소화) 컴포넌트를 구성하는 세 가지 축에 대한 ablation 결과입니다.

- **데이터셋**: LibriSpeech train-clean-100 / dev-clean / test-clean
- **Teacher**: `stt_en_conformer_ctc_small`
- **공통 구성**: CTC + LogitKD + Flow(8 steps) + DiffKD + SpkCE
- **Disent 레이어**: spk=layer4, txt=layer16
- **에포크**: 100, 배치 크기: 32

---

## Table 1 — MI 쌍 조합 Ablation

> 가정: txt↔spk(ts), txt↔pros(tp), pros↔spk(ps) 세 쌍을 모두 사용할수록 성능이 향상될 것이다.

| 실험 | MI 쌍 | Phys | Rec | dev WER | test WER |
|------|--------|------|-----|---------|----------|
| E0 (baseline) | 없음 | ✗ | ✗ | 0.1335 | 0.3242 |
| E1 | ts | ✓ | ✓ | 0.1369 | **0.3223** |
| E2 | tp | ✓ | ✓ | 0.1368 | 0.3302 |
| E3 | ts, tp | ✓ | ✓ | 0.1355 | 0.3271 |
| E4 | ts, tp, ps | ✓ | ✓ | 0.1394 | 0.3303 |

**결과 분석**:
- 가정과 달리, MI 쌍이 많을수록 성능이 향상되지 않았다.
- `ts` (txt↔spk) 단독 사용(E1)이 test WER 0.3223으로 최고 성능.
- `tp` (txt↔pros) 추가 시 오히려 성능이 하락하며, 전체 MI(E4)는 베이스라인(E0)보다도 낮은 성능.
- **해석**: pros 표현이 ASR 태스크에서 txt 표현과의 독립성 강제가 역효과를 낼 가능성. txt↔spk 분리만으로도 충분한 개선이 이루어지며, pros 관련 MI는 학습을 불안정하게 만들 수 있음.

---

## Table 2 — Phys(SL) 기여도 Ablation

> 가정: Phys(F0/Energy/VUV supervision)이 pros 표현 학습에 필수적이어서, Phys 제거 시 성능이 하락할 것이다.

| 비교 | MI 쌍 | Phys | dev WER | test WER | Δ test WER |
|------|--------|------|---------|----------|------------|
| E2 (with Phys) | tp | ✓ | 0.1368 | 0.3302 | — |
| E5 (w/o Phys) | tp | ✗ | 0.1388 | 0.3261 | **-0.0041** |
| E4 (with Phys) | ts, tp, ps | ✓ | 0.1394 | 0.3303 | — |
| E6 (w/o Phys) | ts, tp, ps | ✗ | 0.1372 | 0.3255 | **-0.0048** |

**결과 분석**:
- 가정과 반대로, Phys 제거 시 성능이 오히려 개선됨(Δ ≈ -0.004~-0.005 WER).
- **해석**: Phys loss(F0/Energy/VUV)는 pros 표현을 물리량에 과도하게 구속하여, MI 최소화와 함께 사용될 때 학습에 부정적인 간섭을 일으킬 수 있음. pros 임베딩이 물리량과 ASR 성능 사이에서 최적화 충돌이 발생하는 것으로 보임.

---

## Table 3 — Rec(txt) 기여도 Ablation

> 가정: txt AE reconstruction 손실이 txt 표현 품질을 높여 Rec 없는 것보다 성능이 좋을 것이다.

| 비교 | MI 쌍 | Rec | dev WER | test WER | Δ test WER |
|------|--------|-----|---------|----------|------------|
| E1 (with Rec) | ts | ✓ | 0.1369 | 0.3223 | — |
| E7 (w/o Rec) | ts | ✗ | 0.1388 | 0.3246 | +0.0023 |
| E2 (with Rec) | tp | ✓ | 0.1368 | 0.3302 | — |
| E8 (w/o Rec) | tp | ✗ | 0.1369 | 0.3269 | **-0.0033** |
| E3 (with Rec) | ts, tp | ✓ | 0.1355 | 0.3271 | — |
| E9 (w/o Rec) | ts, tp | ✗ | 0.1368 | 0.3233 | **-0.0038** |
| E4 (with Rec) | ts, tp, ps | ✓ | 0.1394 | 0.3303 | — |
| E10 (w/o Rec) | ts, tp, ps | ✗ | 0.1367 | 0.3254 | **-0.0049** |

**결과 분석**:
- ts 단독(E1 vs E7): Rec 있을 때가 유일하게 성능이 좋음 (+0.0023 WER 불이익).
- tp 포함 설정(E2→E8, E3→E9, E4→E10): Rec 제거 시 성능이 일관되게 개선됨.
- **해석**: pros 관련 MI가 개입될 때, txt AE reconstruction이 txt 표현에 과도한 재건 제약을 부과하여 pros MI와 충돌하는 것으로 보임. ts 단독 설정에서만 Rec이 효과적.

---

## 종합 결론

| 컴포넌트 | 단독 효과 | 종합 권고 |
|----------|-----------|-----------|
| MI(ts) | ✅ 유효 (WER -0.002) | 유지 |
| MI(tp) | ❌ 역효과 | 제거 고려 |
| MI(ps) | ❌ 역효과 (tp 없이 단독 미검증) | 제거 고려 |
| Phys(SL) | ❌ 역효과 | 제거 권고 |
| Rec(txt) | ⚠️ 조건부 (ts 단독에서만 유효) | ts 단독 설정에서만 유지 |

**최적 설정**: E1 (MI=ts, Phys=✓, Rec=✓) — test WER 0.3223

> 다음 실험 방향: ts MI만 유지하고 Phys 제거 변형(E1 + no Phys)을 추가로 검증하면 추가 개선 가능성 있음.
