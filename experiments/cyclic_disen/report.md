# Cyclic Disentanglement 실험 결과 레포트

**실험 브랜치**: `exp/cyclic-disen`
**데이터셋**: LibriSpeech train-clean-100 / dev-clean / dev-other / test-clean / test-other
**Student**: NeMo EncDecCTCModelBPE
**Teacher**: `stt_en_conformer_ctc_small`
**공통 설정**: epochs=100, batch_size=32, flow_steps=8, disent_spk_layers=4, disent_txt_layers=16, cyclic_weight=1e-2, cyclic_grl_alpha=0.1, cyclic_hidden_dim=128

---

## 실험 설계

| 실험 | cyclic 쌍 | CLUB MI | 가설 |
|------|-----------|---------|------|
| E1_cyclic_ts | ts | ✗ | cyclic adversarial이 CLUB MI(ts)보다 안정적 |
| E2_cyclic_tp | tp | ✗ | cyclic tp로 txt↔pros 분리 |

---

## 실험 결과

### WER 비교표

| 실험 | dev_clean | dev_other | test_clean | test_other |
|------|----------:|----------:|-----------:|-----------:|
| DAG-KD Baseline (S-DisKD E1) | 14.00% | 31.95% | 14.08% | 32.56% |
| MI_ablation E1 (CLUB ts) | 13.69% | — | — | 32.23% |
| E1_cyclic_ts | 13.74% | 31.93% | 14.35% | 32.60% |
| E2_cyclic_tp | 13.71% | 31.73% | **13.96%** | **32.46%** |

### 보조 지표

| 실험 | CKA(ts) | CKA(tp) | txt_spk probe acc |
|------|--------:|--------:|------------------:|
| E1_cyclic_ts | 0.277 | — | 0.674 |
| E2_cyclic_tp | 0.232 | 0.025 | 0.671 |

---

## 분석

### E1_cyclic_ts — 가설 불지지

- 가설: "cyclic adversarial이 CLUB MI보다 안정적"
- 결과: test_other 32.60% vs CLUB ts 32.23% → cyclic ts가 오히려 더 나쁨
- test_clean도 14.35%로 베이스라인(14.08%) 대비 악화
- CKA(ts) = 0.277, probe acc = 0.674 → txt↔spk 분리 불충분

### E2_cyclic_tp — 예상 외 호전

- test_clean 13.96%로 베이스라인(14.08%) 대비 소폭 개선
- E1 대비 test_clean 기준 0.39%p 더 좋음
- CKA(tp) = 0.025로 매우 낮음 → txt↔pros 분리 효과적
- MI_ablation에서 CLUB tp는 역효과였으나, cyclic tp는 유효 → 직접 최소화 대신 순환 구조로 간접 분리하는 것이 더 안전

---

## 종합 결론

- **cyclic ts**: 효과 없음. CLUB MI ts보다 분리 효율 낮고 ASR 성능 저하.
- **cyclic tp**: 유효. CLUB tp가 역효과였던 것과 대조적으로 베이스라인 소폭 개선.

---

## 다음 실험 제안

1. **cyclic ts+tp 조합**: tp의 긍정 효과 + ts GRL 압력 병행
2. **cyclic_weight / grl_alpha 튜닝**: ts의 weight(1e-2)가 너무 강해 ASR loss와 충돌 가능성
3. **CLUB ts + cyclic tp 혼합**: pair별로 효과적인 방법 선택 적용
