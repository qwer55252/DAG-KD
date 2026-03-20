# DAG-KD 프로젝트 — Claude Code 지침서

## 프로젝트 개요

DAG-KD는 자동 음성 인식(ASR)을 위한 **Teacher-Student Knowledge Distillation** 프레임워크입니다.

### 핵심 아이디어
- ASR 표현의 계층적 성질 활용: 하위 레이어(speaker/prosody) → 상위 레이어(linguistic)
- Generative KD (Flow Matching, DiffKD)로 Teacher feature 분포를 Student에 전이
- 상호정보량(MI) 최소화 (CLUB)로 Text / Speaker / Prosody 표현을 서로 독립적으로 분리

### 핵심 모듈
| 모듈 | 역할 |
|------|------|
| `FlowMatchingModule` | Student → Teacher feature를 Rectified Flow로 정렬 |
| `DiffKDModule` | Diffusion 방식의 iterative denoising으로 feature KD |
| `GlobalProsodyReferenceEncoder` + `GlobalStyleTokenLayer` | 운율(prosody) 임베딩 추출 |
| `ARClubGaussian` / `ClubGaussian` | 상호정보량 상한 추정 (CLUB 기반) |
| Speaker Backbone (TDNN) | 화자 임베딩 + 분류기 |
| Text Speaker Probe | Text 표현의 화자 불변성 검증 |

### 최종 목표
가장 좋은 ASR 성능을 기록하는 실험 설정을 찾는 것. 필요하다면 핵심 아이디어 자체를 수정하는 방향도 허용.

---

## 실험 워크플로우

### Step 1 — Worktree 생성
- Git branch로 독립된 작업 공간을 생성한다.
- 각 branch는 하나의 실험 테이블 단위로 구성한다.
- 실험을 위한 코드 검토(step4)까지 완료 후 해당 branch에서 main branch로 PR을 생성한다.

### Step 2 — 테이블 주제 설계 (AI 주도)
- AI 스스로 성능 개선 가능성이 있는 아이디어를 찾아 실험을 설계한다.
- 실험마다 **가정(hypothesis)** 이 명확해야 한다 — "왜 성능이 개선될 것인가?"
- 비교군(baseline)은 해당 논문을 직접 확인하고 정확히 구현한다.
- 실험 계획을 **레포트로 작성**하여 사용자에게 제출하고 확인을 받는다.
- 각 테이블을 위한 실험 디렉토리를 별도로 생성하여 테이블 단위로 관리한다.

### Step 3 — 코드 수정 및 스크립트 작성
- 어떤 파일을 수정해야 하는지, 어떤 순서로 작업할지를 직접 분석하고 계획한다.
- 계획에 따라 스크립트를 수정하고 **모든 수정은 커밋**으로 남긴다 (세이브 포인트).
- 결과 분석에 필요한 내용을 로그로 충분히 기록한다 (Step 5 분석용).
- 하나의 테이블을 위한 개별 실험은 **각각의 `.sh` 파일**로 분리한다.
- 수정한 스크립트가 정확히 실행되는지 테스트하고, 에러 발생 시 디버깅한다.
- 테스트 완료 후 실험 시작 전에 코드를 사용자에게 검사받는다.

### Step 4 — 코드 검토 및 승인
- 사용자가 코드와 레포트를 검토한다.
  - 기존 코드와의 일관성 확인
  - 실험 정상 실행 여부 확인
  - 실험 설계의 적절성 확인

### Step 5 — PR 생성 → CI → 코드 리뷰 반영
- 사용자가 승인 하면 PR을 신청한다. 승인하지 않으면 Step2 부터 다시 진행한다.
- 신청된 PR은 앤트로픽의 Code Reviewer가 코드를 확인하고 통과될 때까지 수정한다.
- 문제 없으면 main 브랜치와 merge한다.
- merge된 main 브랜치에서 전체 실험을 진행한다.

### Step 6 — 실험 결과 분석
- 모든 실험 완료되면 결과를 종합하여 분석한다.
- 가정(hypothesis)에 기반한 해석을 작성한다.
- 제안 방법이 비교군보다 낮은 성능이면 개선 방향을 제시하고 Step 2부터 다시 반복한다.

---

## 필수 지시사항

- **커밋은 각 단계마다**: 게임 세이브 포인트처럼, 작업 내용을 중간중간 커밋으로 저장한다.
- **사용자 확인 필수 시점**:
  - Step 2 완료 후 (실험 레포트 제출)
  - Step 3 완료 후 (코드 검사)
  - Step 5 완료 후 (결과 분석 보고)
- **비교군 구현**: 반드시 원 논문을 참조하여 정확히 구현한다.
- **로그 관리**: 결과 분석(Step 5)에 충분한 로그를 남긴다.
- **실험 디렉토리 구조**: 각 테이블은 독립된 디렉토리 + 개별 `.sh` 파일로 관리한다.
- **Self-review + 테스트**: 기획 대비 빠진 항목이 없는지 확인하고 필요한 테스트를 실행하여 검증한다.
