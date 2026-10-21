"""Improve the submission structure using local results; append unchanged LOSO."""
from pathlib import Path
import csv,json,statistics
ROOT=Path(__file__).parent
OUT=ROOT/'reports/submission_revised';OUT.mkdir(parents=True,exist_ok=True)
ref='https://github.com/Oseojin/DCU_DeepLearningProgramming_SEMG'
ns={'__file__':str(ROOT/'build_submission_report.py')}
exec((ROOT/'build_submission_report.py').read_text(encoding='utf-8').split('# Append every content block')[0],ns)
pages=[[]]
for b in ns['blocks']:
    if b[0]=='page':pages.append([])
    else:pages[-1].append(b)
assert len(pages)==7
def read(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
comparison=read('outputs/final/comparison.csv')
cv=read('outputs/week3/cv_scores.csv')
result=json.loads((ROOT/'outputs/week3/results.json').read_text())
config=json.loads((ROOT/'outputs/week3/config.json').read_text())
def value(p,u,m,k):return float(next(r[k] for r in comparison if r['protocol']==p and r['unit']==u and r['model']==m))
def pct(x):return f'{x*100:.2f}'
blocks=[]
def text(s,k='body'):blocks.append((k,s))
def table(r,w):blocks.append(('table',(r,w)))
def page():blocks.append(('page',None))
def append_page(p):blocks.extend(p);page()
cms={}
classrows=[];pairrows=[]
for m in ['SVM','RF','XGB']:
    raw=list(csv.reader((ROOT/f'outputs/week3/confusion_{m}.csv').open()))
    labels=raw[0][1:];cm=[[int(x) for x in r[1:]] for r in raw[1:]];cms[m]=cm
    for i,l in enumerate(labels):
        support=sum(cm[i]);pred=sum(r[i] for r in cm);tp=cm[i][i]
        classrows.append(dict(model=m,label=l,correct=tp,support=support,errors=support-tp,precision=tp/pred,recall=tp/support))
    pairs=sorted([(cm[i][j]+cm[j][i],i,j) for i in range(8) for j in range(i+1,8)],reverse=True)
    errors=sum(sum(r) for r in cm)-sum(cm[i][i] for i in range(8))
    for count,i,j in pairs[:3]:pairrows.append(dict(model=m,pair=labels[i]+' ↔ '+labels[j],forward=cm[i][j],reverse=cm[j][i],total=count,error_share=count/errors))

text('AI 모델 재현 및 성능 비교','title')
text('sEMG 손동작과 팔꿈치 각도 분류','h1')
text('오성빈  |  1~3주차 종합 결과와 LOSO 추가 실험','small')
text('8명의 5채널 근전도에서 손동작 4종과 팔꿈치 각도 2종을 조합한 8클래스를 예측했다. 동일한 특징과 시행 분할에서 SVM, Random Forest, XGBoost를 비교하고, 저장된 예측의 시행별 집계와 피험자 제외 평가로 결과의 적용 범위를 확인했다.')
text('결과 한눈에 보기','h2')
table([['평가 조건','모델','정확도','해석'],['학습 시행 5 fold','RF','63.40 ± 1.04%','기존 학습 자료 안의 검증'],['기존 테스트 7,296창','RF','65.87%','같은 사람의 다른 시행'],['기존 테스트 384시행 투표','RF','96.09%','2초 구간을 모아 판단'],['동일 테스트 창 LOSO','RF','23.08%','학습에서 제외한 사람'],['동일 테스트 시행 LOSO 투표','RF','31.51%','제외한 사람의 시행 판단']],[170,40,110,187])
text('위 수치는 서로 다른 평가 조건이다. 96.09%는 단일 창 성능이나 새로운 사람에 대한 성능이 아니다. LOSO는 기존 자료를 재구성한 사후 탐색 실험이며 독립적인 새 측정 검증은 아니다. 교차검증의 ±는 5개 fold의 SD로 ddof=0이다.','small')
text('보고서 구성','h2')
text('1 코드 설명에서는 데이터 흐름과 실제 실행 절차를, 2 모델 성능 비교에서는 평가 방식과 비용을, 3 Confusion Matrix 분석에서는 오류의 위치와 비중을 설명한다. 4 최종 결과에서 확인한 사실과 한계를 연결한 뒤 LOSO 실험 보고서 8쪽 전체를 이어 수록한다.')
text('작성 형식 참고','h2')
text('Oseojin의 DCU_DeepLearningProgramming_SEMG README에서 결과 요약, 평가 조건 구분, 오류의 수량화, 근거와 추정의 분리 방식을 참고했다. 해당 저장소의 사용자 식별 과제와 이번 손동작 및 각도 과제는 다르므로 타인의 모델 설정과 성능을 이번 실험 결과로 가져오지 않았다.','small')
text(ref,'small');page()
# Keep source-derived week 1-3 content, but avoid a second cover.
p0=pages[0];start=next(i for i,b in enumerate(p0) if b==('h1','1 코드 설명'))
append_page(p0[start:]);append_page(pages[1])
text('1 코드 설명과 재현 범위','h1')
text('입력부터 평가까지의 처리 흐름','h2')
text('64개 CSV → 마커 상승점과 시행 ID → 시행 단위 층화 분할 → 1.5~3.5초 추출 → 200ms 창 19개 → 시간영역 특징 25개 → 학습 자료로 표준화 → 모델 학습 → 고정된 테스트 예측 → 지표와 혼동행렬 저장.')
text('특징이 표현하는 정보','h2')
table([['특징','계산과 해석'],['RMS','제곱평균의 제곱근으로 신호 크기를 요약'],['MAV','절대값 평균으로 진폭 수준을 요약'],['WL','인접 표본 차이의 절대값 합으로 변화량을 요약'],['ZC','부호 변화 중 표본 차이 0.01 이상인 교차 수'],['SSC','연속 기울기 부호 변화 수로 파형 변화를 요약']],[65,442])
text('StandardScaler는 전체 자료에서 먼저 학습하지 않았다. 기존 평가는 학습 부분, 교차검증은 각 fold 학습 부분, LOSO는 제외 대상 이외 7명에서만 적합했다. 같은 시행의 창을 분리하지 않는 처리와 학습 데이터만 사용하는 표준화는 서로 다른 누수 경로를 차단한다.')
text('실행 환경과 보존된 근거','h2')
text(f"Python {config['python']}, {config['platform']}, 논리 CPU {config['logical_cpus']}개. RF와 XGB는 n_jobs=4이고 GPU는 사용하지 않았다. "+', '.join(k+' '+v for k,v in config['versions'].items())+'.','small')
table([['경로','보존된 근거'],['outputs/week2','데이터셋, 시행과 창 위치, 입력 해시 및 설정'],['outputs/week3','fold 배정, 지표, 예측, 혼동행렬, 추론 시간'],['outputs/loso','피험자별 24개 학습 결과와 전체 예측'],['outputs/final','다수결 예측, 동률 여부, 평가별 비교표']],[145,362])
text('저장된 결과의 확인은 verify_week3.py, verify_loso.py, verify_final.py로 구분한다. 처음부터 학습할 경우 각 평가 프로토콜의 입력과 패키지 버전을 함께 보존해야 한다. 이번 보완은 저장된 결과의 재집계와 문서 작성이며 모델 재학습이나 신규 측정은 하지 않았다.','small');page()
append_page(pages[2])
text('2 평가 방식별 성능과 해석','h1')
text('교차검증의 네 가지 지표','h2')
table([['모델','Accuracy','Precision','Recall','F1']]+[[m]+[f"{pct(statistics.mean(float(r[k]) for r in cv if r['model']==m))} ± {pct(statistics.pstdev(float(r[k]) for r in cv if r['model']==m))}" for k in ['accuracy','precision_macro','recall_macro','f1_macro']] for m in ['SVM','RF','XGB']],[47,115,115,115,115])
text('단위는 %이며 5개 fold의 평균 ± SD(ddof=0)다. P R F1은 fold별 macro 평균을 먼저 계산했다. 홀드아웃 점수와 섞어 평균하지 않는다. fold는 시행 단위로 나누지만 같은 피험자와 연속 기록의 다른 시행은 공유할 수 있다.','small')
text('판단 단위를 바꾸면 달라지는 결과','h2')
table([['모델','기존 창','기존 시행','LOSO 동일 창','LOSO 동일 시행']]+[[m,pct(value('baseline','window',m,'accuracy')),pct(value('baseline','trial',m,'accuracy')),pct(value('loso_matched','window',m,'accuracy')),pct(value('loso_matched','trial',m,'accuracy'))] for m in ['SVM','RF','XGB']],[47,110,110,120,120])
text('모든 값은 정확도 %다. 창은 기존 테스트 7,296개, 시행은 그 창이 속한 384개를 평가한다. 시행 판단은 19개 결합 예측의 최다 득표를 선택하며 동률이면 고정된 클래스 순서에서 앞선 값을 쓴다. 전체 LOSO 1,920시행 결과는 후첨 보고서에 별도로 제시했다.')
text('RF는 기존 조건에서 창 65.87%가 시행 96.09%로 높아졌다. 하지만 같은 평가 시행의 LOSO 투표는 31.51%였다. 시행 내 예측을 모으는 효과와 새 사람에게 일반화하는 능력은 구분해야 한다. 학습량도 기존 29,184창과 LOSO 31,920창으로 달라 피험자 차이만의 인과 효과로 볼 수 없다.')
text('운영 관점의 제한','h2')
text('시행 판단에는 2초 분석 구간 전체가 필요하고 마커 기준으로는 3.5초까지 기다려야 한다. 기존 단일 창 추론 시간은 이 관찰 대기와 특징 계산 시간을 제외한 값이다. 서로 겹친 창은 독립적인 19표가 아니므로 집계로 항상 오류가 제거되는 것도 아니다.')
text('RF의 높은 점수를 보고 새로운 설정을 탐색하거나 별도의 최종 모델을 다시 학습하지 않았다. 따라서 본 보고서의 최고 성능 모델은 동일한 고정 설정 비교에서의 관찰 결과이며, 모델 선택 이후 독립 검증까지 완료했다는 뜻은 아니다.','small');page()
for i,m in enumerate(['SVM','RF','XGB']):
    p=list(pages[3+i])
    best=max((r for r in classrows if r['model']==m),key=lambda r:r['recall'])
    worst=min((r for r in classrows if r['model']==m),key=lambda r:r['recall'])
    for j,(k,s) in enumerate(p):
        if isinstance(s,str) and s.startswith('가장 잘 분류된 클래스:'):
            p[j]=(k,f"가장 잘 분류된 클래스: {best['label']}. 재현율 {pct(best['recall'])}% ({best['correct']}/{best['support']}), 정밀도 {pct(best['precision'])}%다.")
        if isinstance(s,str) and s.startswith('가장 많이 오분류된 클래스:'):
            p[j]=(k,f"가장 많이 오분류된 클래스: {worst['label']}. 오류 {worst['errors']}건, 재현율 {pct(worst['recall'])}%, 정밀도 {pct(worst['precision'])}%다. 클래스별 표본 수는 같다.")
    append_page(p)
text('3 모델 사이에 남는 오류 구조','h1')
text('클래스별 재현율 비교','h2')
table([['클래스','SVM','RF','XGB']]+[[l]+[pct(next(r['recall'] for r in classrows if r['model']==m and r['label']==l)) for m in ['SVM','RF','XGB']] for l in labels],[198,103,103,103])
deltas=[(cms['RF'][i][i]-cms['SVM'][i][i])/912*100 for i in range(8)]
total_svm=result['errors']['SVM']['errors'];total_rf=result['errors']['RF']['errors']
text(f"RF는 SVM보다 8개 클래스 모두에서 재현율이 높았다. 향상 폭은 {min(deltas):.2f}~{max(deltas):.2f}pp이며, 총 오류는 {total_svm:,}건에서 {total_rf:,}건으로 {(total_svm-total_rf)/total_svm*100:.2f}% 줄었다. 단일 클래스만 좋아져 전체 지표가 오른 것은 아니다.")
text('상위 양방향 혼동 쌍','h2')
table([['모델','혼동 쌍','정방향','역방향','합계','오류 비중']]+[[r['model'],r['pair'],r['forward'],r['reverse'],r['total'],pct(r['error_share'])+'%'] for r in pairrows[:2]+pairrows[3:5]+pairrows[6:8]],[43,230,58,58,58,60])
text('정방향은 표에 적힌 왼쪽 → 오른쪽, 역방향은 반대 방향이다. 오류 비중은 해당 모델 전체 오류 중 두 방향의 합이 차지하는 비율이다. 클래스 오류율과 분모가 다르다. 한 방향 순위와 양방향 쌍 순위도 다를 수 있다.','small')
text('상위 혼동 쌍은 모델을 바꾼 후에도 일부 남는다. 이 결과는 공통적으로 어려운 구분이 존재한다는 관찰이다. 특정 근육이나 각도의 생리적 유사성이 원인인지 확인하는 실험은 수행하지 않았다.','small');page()
text('3 오류 해석을 뒷받침하는 관찰','h1')
text('진폭 차이를 보는 것과 분류하는 것은 다르다','h2')
table([['1주차 관찰','30도 RMS','60도 RMS','60도 대 30도'],['a 주먹 Biceps','0.01079','0.01636','1.516배'],['a 주먹 ED','0.04479','0.03445','0.769배'],['a 손 펴기 Biceps','0.00968','0.01457','1.504배']],[225,90,90,102])
text('각 조건 30개의 5초 관찰 구간 RMS 중앙값이다. 일부 채널의 각도 차이가 관찰되지만 1명의 예시이고 휴식도 포함한다. 실제 모델은 다른 길이의 구간과 여러 특징을 사용하므로 이 표가 모델의 각도 판별 단서를 직접 입증하지는 않는다.')
text('2주차 안정성 분석의 연결','h2')
st=read('outputs/stability/segment_summary.csv')
table([['특징','0.5~2.5초 비교','1.0~3.0초 비교']]+[[f]+[f"{float(next(r['median'] for r in st if int(r['start'])==s and r['feature']==f)):.2f}%" for s in [500,1000]] for f in ['RMS','MAV','WL','ZC','SSC']],[107,200,200])
text('기준은 1.5~3.5초다. 학습 시행만 사용했으며 200×|비교−기준|/(|비교|+|기준|)의 중앙값을 요약했다. 두 값이 0이면 변화율을 0으로 처리했다. 시행 내 창 평균 후 기록 내 시행 중앙값, 채널 중앙값, 기록 간 중앙값 순서로 집계했다.','small')
text('SSC가 덜 민감했다는 사실은 안정성 관찰이다. SSC만 사용한 모델 비교는 하지 않았으므로 분류 성능 기여로 해석하지 않는다. 구간을 이동한 모델의 LOSO도 수행하지 않았기 때문에 구간 민감도가 일반화 저하를 만들었다고 결론내릴 수 없다.')
text('확인된 결과와 미확인 원인','h2')
text('확인된 결과는 RF의 클래스 전반 개선, 남아 있는 각도 및 동작 혼동, 구간에 따른 특징 변화, 새 사람에서의 성능 하락이다. 전극 위치, 근활성의 개인차, 마커 정렬은 원인 후보다. 원인을 분리한 통제 실험이나 특징 제거 실험은 아직 없으며, 분석 결과를 근피로나 과적합의 확정 증거로 쓰지 않는다.');page()
# Final section remains in the required four-section framework.
p=list(pages[6])
for i,(k,s) in enumerate(p):
    if isinstance(s,str) and s.startswith('근거:'):
        p[i]=(k,s+' 작성 구조 참고: '+ref+' . 수치와 해석은 이번 프로젝트 자료에서 산출했다.')
append_page(p)
loso={'__file__':str(ROOT/'build_loso_report.py')}
exec((ROOT/'build_loso_report.py').read_text(encoding='utf-8').split("source=(ROOT/'build_week3_report.py')")[0],loso)
blocks.extend(loso['blocks'])
source=(ROOT/'build_week3_report.py').read_text(encoding='utf-8')
exec(source.split('ROOT=')[0])
renderer=source.split('# Native editable Word paragraphs and tables; figures remain images.')[1]
renderer=renderer.replace('Step2_week3_report_OhSeongbin','Step2_final_revised_with_LOSO_OhSeongbin').replace('Step2 3주차 결과보고서','AI 모델 재현 및 성능 비교 보완 보고서').replace('Step2 3주차 |','종합 보완 및 LOSO |').replace('tmp/week3_report_qa','tmp/revised_report_qa')
renderer=renderer.replace('Pt(595.28)','Pt(612)').replace('Pt(841.89)','Pt(792)').replace('595.28,841.89','612,792')
renderer=renderer.replace("('Normal',9.5)","('Normal',11)").replace("('body',9.5,15,8)","('body',11,16,8)")
renderer=renderer.replace('font.size=Pt(8)','font.size=Pt(9)').replace("('small',8,12,6)","('small',9,13,6)")
exec(renderer)
for name,rs in [('class_metrics',classrows),('top_confusion_pairs',pairrows)]:
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
md=['# AI 모델 재현 및 성능 비교 보완 보고서','', '[편집용 Word](Step2_final_revised_with_LOSO_OhSeongbin.docx) · [PDF](Step2_final_revised_with_LOSO_OhSeongbin.pdf)','']
for kind,v in blocks:
    if kind=='page':continue
    if kind=='table':
        rs=v[0];md+=['|'+'|'.join(map(str,rs[0]))+'|','|'+'|'.join(['---']*len(rs[0]))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs[1:]]+['']
    elif kind=='image':md+=['![실험 결과](../../'+Path(v[0]).relative_to(ROOT).as_posix()+')','']
    else:md+=[('## ' if kind in ['h1','title'] else '### ' if kind=='h2' else '')+v,'']
(OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
