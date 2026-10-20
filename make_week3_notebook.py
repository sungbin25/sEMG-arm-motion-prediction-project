from pathlib import Path
import nbformat as nbf
nb=nbf.v4.new_notebook()
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
nb.cells=[
md('''# Step2 3주차 모델 비교와 통계 검정

오성빈 · 2026-10-20

이 노트북은 `python week3.py`로 생성한 실제 실험 결과를 읽어 재현 가능한 표와 그림으로 확인한다.
모델을 처음부터 학습하려면 `python -m pip install -r requirements-week3.txt` 후 `python week3.py --output outputs/week3_new`를 실행하고 아래 OUT을 변경한다.
기본 결과는 SVM·RF·XGBoost, 고정 파라미터, 동일 시행 분할, 학습 부분만의 5-fold 교차검증으로 생성했다.
'''),
code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nOUT=Path('outputs/week3')\nr=json.loads((OUT/'results.json').read_text(encoding='utf-8'))\nconfig=json.loads((OUT/'config.json').read_text())\ndisplay(config)"),
md('## 홀드아웃 평가\nPrecision·Recall·F1은 macro 평균이다. 윈도우 단위 지표이며 테스트 창은 클래스당 912개다.'),
code("display(pd.read_csv(OUT/'holdout_metrics.csv'))"),
md('## 학습 데이터 내부 시행 단위 교차검증\n같은 시행의 19개 창은 동일 fold에 배정한다. StandardScaler는 각 fold의 학습 부분에서만 fit한다. SD는 ddof=0이며 신뢰구간이 아니다.'),
code("display(pd.read_csv(OUT/'cv_scores.csv'))\ndisplay(pd.read_csv(OUT/'cv_summary.csv'))\ndisplay(Image(filename=str(OUT/'cv_accuracy.png')))"),
md('## 대응표본 t 검정\n같은 5개 fold의 정확도를 짝지었다. 원 p값과 세 쌍 비교의 Holm 보정 p값을 함께 제시한다. fold 간 학습 데이터가 겹치므로 독립성이 보장되지 않으며 탐색적으로 해석한다. p≥0.05는 동등성의 증거가 아니다.'),
code("display(pd.read_csv(OUT/'paired_tests.csv'))"),
md('## 혼동행렬과 오류 구조\n행은 실제 클래스, 열은 예측 클래스다. 동작과 각도 의미를 분리하여 오류를 집계했다.'),
code("for name in ['SVM','RF','XGB']:\n    print(name)\n    display(Image(filename=str(OUT/f'confusion_{name}.png')))\n    display(r['errors'][name])"),
md('## 한계와 결과 추적\n동일 피험자·기록의 다른 시행은 양쪽 분할에 존재한다. 실제 동작 정렬은 미확인이고, 고정된 강의 파라미터를 사용했다. 논문 수치와의 차이를 성능 실패 또는 누수의 단독 증거로 해석하지 않는다. 결과보고서의 환경·통계 검정·계산 비용 한계를 함께 읽어야 한다.\n\n`python verify_week3.py`는 저장된 예측으로 지표와 혼동행렬, 통계 검정을 재계산하고 시행 분리·라벨 대응·입력 해시를 검증한다.')]
nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
nb.metadata['language_info']={'name':'python'}
nbf.write(nb,'week3_model_comparison.ipynb')
