# DAG-KD 프로젝트 — Claude Code 지침서

## 1. 프로젝트 개요

DAG-KD는 자동 음성 인식(ASR)을 위한 **Teacher-Student Knowledge Distillation** 프레임워크이다.  
이 프로젝트의 목표는 **작고 효율적인 ASR student 모델이, teacher 표현 전체를 무차별적으로 따라하는 대신 ASR에 실제로 중요한 정보만 선택적으로 전달받도록 만드는 것**이다.

---

## 2. 메인 아이디어

### 문제 인식
ASR 모델의 representation에는 크게 두 종류의 정보가 함께 섞여 있을 가능성이 높다.

1. **Content / Linguistic 정보**
   - 음소, 발음 내용, 단어, 문맥 등
   - 최종 ASR 성능에 직접적으로 기여하는 정보

2. **Non-linguistic 정보**
   - 화자 특성(speaker identity), prosody, timbre, recording condition, channel, style 등
   - 일부는 보조적으로 유용할 수 있으나, **최종 ASR 관점에서는 기여도가 낮거나 불필요한 정보일 수 있음**

기존 KD는 대체로 teacher의 hidden representation이나 output을 **통째로 student가 모사하도록 학습**시킨다.  
하지만 이 방식은 student가 작은 상황에서 다음 문제를 만든다.

- 작은 student의 제한된 capacity가 **ASR에 불필요한 정보 재현**에 낭비될 수 있음
- teacher representation 안에 content와 non-content가 섞여 있으면,  
  student는 **무엇이 핵심 정보인지 구분하지 못한 채 전부 따라하게 됨**
- 결과적으로 student가 **정작 인식 성능에 중요한 linguistic 정보**를 충분히 학습하지 못할 수 있음

### 우리가 하려는 일
DAG-KD는 **teacher representation 내부의 정보를 분리(disentanglement)** 하고,  
그 중 **ASR에 효과적인 content 중심 정보만 student에 distillation** 하여  
**더 작고 효율적인 student 모델**을 만드는 것을 목표로 한다.

즉, DAG-KD의 핵심은 다음 한 문장으로 정리된다.

> **“작은 teacher에서 content와 non-content를 분리한 뒤, ASR에 유효한 정보만 student로 전달하여 효율적인 ASR 모델을 학습한다.”**

---

## 3. 핵심 모듈

| 모듈 | 역할 |
|------|------|
| `FlowMatchingModule` | Student → Teacher feature를 rectified flow 방식으로 정렬 |
| `DiffKDModule` | Diffusion 기반 iterative denoising으로 teacher feature 구조 전달 |
| `GlobalProsodyReferenceEncoder` + `GlobalStyleTokenLayer` | Prosody / style 관련 정보 추출 |
| `ARClubGaussian` / `ClubGaussian` | 표현 간 상호정보량(MI) 상한 추정 |
| Speaker Backbone (TDNN) | 화자 정보 추출 및 speaker-related supervision |
| Text Speaker Probe | Text/content 표현에 speaker 정보가 남아 있는지 검증 |

---

## 4. 최종 목표

최종 목표는 다음과 같다.

> ** teacher의 representation을 ASR 관점에서 분해하고,  
> ASR에 유효한 정보만 작은 student에 효율적으로 전달하는 KD 프레임워크를 확립하는 것**

보다 구체적으로는:

- baseline KD보다 더 좋은 **student ASR 성능 + 정보 전달 효율** 확보
- disentanglement와 generative KD가 **언제/왜 효과적인지 설명 가능한 실험 결과** 확보
- 필요하다면 초기 아이디어를 수정하더라도,  
  최종적으로는 **“효율적인 ASR student를 만드는 원리”**를 분명히 남기는 것

---

## 5. 실험 워크플로우

## Step 1 — Worktree 생성
- Git branch로 독립된 작업 공간을 생성한다.
- 각 branch는 논문에 사용될 실험 테이블 또는 하나의 핵심 가설 검증 단위로 구성한다.
- Step 4의 코드 검토까지 완료되면 해당 branch에서 main branch로 PR을 생성한다.

## Step 2 — 실험 주제 설계 (AI 주도)
- AI 스스로 성능 개선 가능성이 있는 아이디어를 찾고 실험을 설계한다.
  - 아이디어는 기존에 신뢰도가 높은 논문을 참고하여 DAG-KD에 반영할 만한 아이디어를 찾는 방식으로
- 실험 설계는 아이디어의 타당성을 증명할 수 있도록 철저한 변인 통제로 대조군 실험들을 설계한다.
- 모든 실험에는 가설이 명확해야 한다.
- 테이블의 모든 실험이 마치면 가설을 증명할 수 있는 하나의 테이블이 완성되고 이것에 대한 분석을 할 수 있어야 한다.
- 실험 계획은 반드시 **레포트 형태로 작성**하여 사용자에게 제출한다. 
  - 구성 요소: 실험 목적, 가설, 변경할 모듈, 비교군, 예상 리스크, 로그 계획, 실행 스크립트 구성 계획
- 각 테이블을 위한 실험은 별도 디렉토리로 관리한다.

## Step 3 — 코드 수정 및 스크립트 작성
- 어떤 파일을 수정해야 하는지, 어떤 순서로 작업할지를 직접 분석하고 계획한다.
- 계획에 따라 스크립트를 수정하고 **모든 수정은 커밋**으로 남긴다 (세이브 포인트).
- 결과 분석에 필요한 내용을 로그로 충분히 기록한다 (Step 5 분석용).
- 하나의 테이블을 위한 개별 실험은 **각각의 `.sh` 파일**로 분리한다.
- 수정한 스크립트가 정확히 실행되는지 테스트하고, 에러 발생 시 디버깅한다.
- 테스트 완료 후 실험 시작 전에 코드를 사용자에게 검사받는다.
- Step 3 필수 체크
  - 학습 시작 전 dry-run 또는 짧은 sanity check 수행
  - loss / tensor shape / logging / checkpoint 저장 정상 여부 확인
  - 새 모듈이 inference path에 불필요한 overhead를 만들지 확인
  - baseline과 제안 방법의 차이가 로그상 명확히 드러나도록 구성

### Step 4 — 코드 검토 및 승인
- 사용자가 코드와 레포트를 검토한다.
- 기존 코드와의 일관성 확인
- 실험 정상 실행 여부 확인
- 실험 설계의 적절성 확인

**승인 전에는 대규모 실험을 시작하지 않는다.**

### Step 5 — PR 생성 → CI → 코드 리뷰 반영
- 사용자가 승인 하면 PR을 신청한다. 승인하지 않으면 Step2 부터 다시 진행한다.
- 신청된 PR은 앤트로픽의 Code Reviewer가 코드를 확인하고 통과될 때까지 수정한다.
- 문제 없으면 main 브랜치와 merge한다.
- merge된 main 브랜치에서 전체 실험을 진행한다.

## Step 6 — 실험 결과 분석
- 모든 실험이 끝나면 결과를 종합해 분석한다.
- 반드시 **초기 가설 기준**으로 해석한다.
- 단순히 “올랐다/내렸다”가 아니라 다음을 설명해야 한다.
  - 어떤 정보 분리가 실제로 도움이 되었는지
  - 어떤 제약이 과했는지
  - 어떤 KD 경로가 효과적이었는지
  - 성능 변화와 representation 분석이 일관되는지
- 제안 방법이 비교군보다 낮은 성능이면,  
  원인을 추정하고 개선 방향을 정리한 뒤 Step 2부터 다시 반복한다.
- 분석 내용을 "/workspace/DAG-KD/experiments" 경로에 실험명/report.md 파일로 저장한다.
  - 포함되어야 할 내용: 핵심 결과 요약, baseline 대비 변화, 가설 검증 여부, 실패 원인 또는 성공 요인

---

## 6. 필수 지시사항
- **커밋은 각 단계마다**: 게임 세이브 포인트처럼, 작업 내용을 중간중간 커밋으로 저장한다.
- **사용자 확인 필수 시점**:
  - Step 2 완료 후 (실험 레포트 제출)
  - Step 3 완료 후 (코드 검사)
  - Step 5 완료 후 (결과 분석 보고)
- **로그 관리**: 결과 분석(Step 5)에 충분한 로그를 남긴다.
- **실험 디렉토리 구조**: 각 테이블은 독립된 디렉토리 + 개별 `.sh` 파일로 관리한다.
- **Self-review + 테스트**: 기획 대비 빠진 항목이 없는지 확인하고 필요한 테스트를 실행하여 검증한다.
- 저위험군 명령어는 물어보지 말고 수행한다.

---

## 7. 의사결정 원칙

실험 중 선택지가 여러 개일 때는 아래 우선순위로 판단한다.

1. ASR 성능 개선 가능성이 높은가
2. 작은 student의 efficiency 관점에서 타당한가
3. 전달할 정보가 더 명확해지는가
4. 결과 해석 가능성이 높은가
5. 구현 복잡도 대비 실험 가치가 충분한가

즉,  
“복잡하지만 멋있어 보이는 방법”보다  
“작은 student가 왜 더 잘 배우는지 설명 가능한 방법”**을 우선한다.
