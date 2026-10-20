# Step2 3주차 결과

작성자: 오성빈 / 실험 및 보고서 날짜: 2026-10-20

- [결과보고서 PDF](Step2_week3_report_OhSeongbin.pdf): 7페이지, 강의의 제출 항목 모두 포함.
- [편집 가능한 DOCX](Step2_week3_report_OhSeongbin.docx): 본문·표 편집 가능. PDF는 렌더링 검수를 완료했으나 DOCX 페이지 배치는 렌더러 부재로 별도 검증하지 못했다.
- [실행 결과 노트북](../../week3_model_comparison.ipynb)
- [실험 코드](../../week3.py) / [저장 결과 검증](../../verify_week3.py)

## 결과 파일

모든 실험 결과는 [outputs/week3](../../outputs/week3)에 있다.

| 파일 | 내용 |
|---|---|
| `holdout_metrics.csv` | 세 모델의 정확도·Macro Precision·Recall·F1 및 계산 비용 |
| `holdout_predictions.csv` | 모델별 테스트 예측과 시행 ID |
| `cv_scores.csv` / `cv_summary.csv` | fold별 지표 및 평균·표준편차 |
| `cv_trial_assignments.csv` | 학습 시행별 검증 fold 배정 |
| `paired_tests.csv` | 대응표본 t-검정 원 p값 및 Holm 보정 p값 |
| `confusion_SVM/RF/XGB.csv` 및 `.png` | 모델별 혼동행렬 |
| `confusion8.png` | 강의 제출 파일명에 맞춘 RF 혼동행렬 복사본 |
| `inference_SVM/RF/XGB.csv` | 단일 창 추론 100회 원시 측정값 |
| `results.json` | 전체 결과와 클래스별 오류 분석 |
| `config.json` | 입력 SHA-256, 버전, 파라미터, 측정 환경 |

## 재현

저장소 루트에서 실행한다. 원자료 다운로드와 2주차 데이터 생성 절차는 루트 README를 따른다.

```bash
python -m pip install -r requirements-week3.txt
python -m unittest -v test_week2 test_week3
python week3.py
python verify_week3.py
```

기존 출력의 동일 설정 CV 결과는 재사용한다. 전체 재실행은 `python week3.py --output outputs/week3_new`로 별도 폴더를 지정한다.
모델별 학습은 순차 실행되며 RF와 XGBoost만 n_jobs=4로 설정했다. 숫자·시간은 라이브러리와 장비에 따라 달라질 수 있다.

보고서 재생성은 `requirements-report.txt`를 설치한 뒤 `python build_week3_report.py`를 실행한다.
보고서 생성기는 Windows의 Malgun Gothic 폰트 파일을 사용한다. `python make_week3_notebook.py`는 출력이 없는 노트북을 새로 만들므로 기존 실행 결과를 덮어쓸 수 있다.

## 해석

RF의 홀드아웃 정확도는 65.87%, Macro F1은 65.90%다. 강의에 소개된 논문 수치를 그대로 재현하지 못했으며 이를 숨기거나 설정을 테스트 결과에 맞추지 않았다.
동일 피험자·기록이 양쪽에 존재하는 시행 분할, 잠정 분석 구간, 상관된 창과 CV fold, 소수 대응쌍, 계산 환경의 한계를 보고서에 명시했다.
