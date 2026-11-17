# -*- coding: utf-8 -*-
"""DepoNav AI • Dört sezon özellikli sınıflandırıcı + Exact TSP + forklift."""
from __future__ import annotations
from datetime import timedelta
import hashlib
import inspect
import json
import numpy as np
import pandas as pd
import sklearn
import xgboost
import joblib
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import roc_curve, precision_recall_curve
from product_demand import ROOT,source_matrix,future_features,BASE,SEASON
from warehouse_core import load_project,wave_locations,parse_saved_route,route_details,solve_selected
from forklift_simulator import render_simulator

st.set_page_config(page_title='DepoNav AI | Ürün Talebi & Rota',page_icon='🚜',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp { background:linear-gradient(135deg,#0a1524 0%,#101e30 42%,#0c1829 100%);color:#eef6ff }
.block-container {max-width:1470px;padding-top:1.2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#101e30}
[data-testid="stMetric"] {background:#162b40;border:1px solid #294660;border-radius:15px;padding:14px 15px}
[data-testid="stMetricValue"] {color:#f7fbff;font-weight:780}
.hero {background:linear-gradient(108deg,#113448,#172940 58%,#152847);border-radius:19px;border:1px solid #2e5d6b;padding:26px 29px;margin-bottom:20px}
.hero .tag {font-weight:700;font-size:11px;letter-spacing:.13em;color:#6ff0c6}
.hero h1 {color:#f1f8ff;margin:9px 0 10px;font-size:31px;line-height:1.15}
.hero p {color:#bfd3df;font-size:14px;margin:0;max-width:980px}
.mini {color:#9cbaca;font-size:12px}
.panel {background:#162a41;border:1px solid #34546a;border-radius:14px;padding:18px}
.panel strong {color:#d8fcf3;font-size:17px}
.panel p {color:#bcd1df;font-size:13px;margin:8px 0 0}
</style>''',unsafe_allow_html=True)

# Streamlit eski ve yeni surumlerinde genislik parametresi farklidir.
# Eski Streamlit: use_container_width=True
# Yeni Streamlit: width='stretch'
# st.button, genislik parametresi icin uyumluluk kontrolu olarak kullanilir.
_FULL_WIDTH = (
    {'width': 'stretch'}
    if 'width' in inspect.signature(st.button).parameters
    else {'use_container_width': True}
)

PAGES=['🏠 Genel Bakış','🛒 Ürün Talep Tahmini','📊 ML Performansı','🚜 Rota & Simülasyon','🧭 Yeni Rota Planı','📘 Veri & Metodoloji']

@st.cache_resource(show_spinner=False)
def get_models():
    meta=json.loads((ROOT/'outputs'/'urun_talep_deney_ozeti.json').read_text(encoding='utf-8'))
    if meta['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(
            'scikit-learn surum uyumsuzlugu: model='
            + str(meta['sklearn_version']) + ', mevcut=' + sklearn.__version__
            + '. requirements.txt ile uyumlu surumu yukleyin.'
        )
    bundles = {}
    for label, slug in MODEL_SLUGS.items():
        # XGBoost farkli surumdeyken pickle dosyasini yuklemiyoruz.
        # Diger uc model guvenle kullanilabilir durumda kalir.
        if label == 'XGBoost' and meta.get('xgboost_version') != xgboost.__version__:
            continue
        bundles[label] = joblib.load(ROOT/'models'/f'urun_talep_{slug}.joblib')
    return bundles

MODEL_SLUGS={
    'Logistic Regression (Baseline)':'logistic_regression',
    'Random Forest':'random_forest',
    'Decision Tree':'decision_tree',
    'XGBoost':'xgboost'
}
COLORS={
    'Logistic Regression (Baseline)':'#A6B8FF',
    'Random Forest':'#52DBB4',
    'Decision Tree':'#FFBD70',
    'XGBoost':'#DDA7FF',
    '30 Günlük Sıklık (Kural)':'#90A1B4'
}

def prob_col(name):
    return 'prob_'+MODEL_SLUGS[name]

@st.cache_data(show_spinner=False)
def get_data():
    root=ROOT/'outputs'
    meta=json.loads((root/'urun_talep_deney_ozeti.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(root/'urun_talep_unseen_test_metrikleri.csv')
    predictions=pd.read_csv(root/'urun_talep_unseen_test_tahminleri.csv',parse_dates=['date'])
    validation=pd.read_csv(root/'urun_talep_model_secimi_validasyon.csv')
    cv=pd.read_csv(root/'urun_talep_cv_ozeti.csv')
    folds=pd.read_csv(root/'urun_talep_cv_fold_metrikleri.csv')
    fi=pd.read_csv(root/'urun_talep_ozellik_onemi.csv')
    work=pd.read_csv(root/'urun_talep_gunluk_ozet.csv',parse_dates=['date'])
    coverage=pd.read_csv(root/'urun_talep_kapsam_analizi.csv',parse_dates=['date'])
    pivot,_=source_matrix()
    return meta,metrics,predictions,validation,cv,folds,fi,work,coverage,pivot

@st.cache_resource(show_spinner=False)
def get_route_project():return load_project(ROOT)

@st.cache_data(show_spinner=False)
def backdrop():
    d=pd.read_csv(ROOT/'data'/'Storage_Location.csv')
    q=d.loc[pd.to_numeric(d.z,errors='coerce').eq(1),['x','y']]
    return q.dropna().astype(float).values.tolist()

@st.cache_data(show_spinner=False)
def wave_choices():
    d=pd.read_csv(ROOT/'outputs'/'tsp_dalga_ozetleri.csv',usecols=['waveNumber','benzersiz_raf','first_date'])
    return {int(r.waveNumber):f'{int(r.waveNumber)} · {int(r.benzersiz_raf)} raf · {str(r.first_date)[:10]}' for r in d.itertuples()}

def fmt(v,n=0):return f'{float(v):,.{n}f}'.replace(',','~').replace('.',',').replace('~','.')
def navigate(page):st.session_state['nav']=page

def hero(title,subtitle,tag='DEPONAV AI · KARAR DESTEK PROTOTİPİ'):
    st.markdown(f'<div class="hero"><div class="tag">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

def plot_style(fig,height=360):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      height=height,margin=dict(l=14,r=14,t=30,b=33),legend=dict(orientation='h',y=1.15))
    return fig

def overview():
    meta,metrics,preds,val,cv,folds,fi,work,cov,pivot=get_data()
    project=get_route_project()
    chosen=meta['selected_model']; chosen_row=metrics.loc[metrics.model.eq(chosen)].iloc[0]
    hero('Hangi ürün talep görecek? Hangi rotayla toplanacak?',
         'Dört sezon özellikli sınıflandırıcıyı karşılaştır; ertesi kayıtlı operasyon günü için ürün hazırlığına destek ol. Kesinleşmiş dalgaları Exact TSP ile optimize et ve forklift simülasyonunu izle.')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Geçmiş SKU',fmt(meta['n_products_total_historical']))
    c2.metric('Kayıtlı operasyon günü',fmt(meta['n_recorded_days']))
    c3.metric('Seçilen model F2 (unseen)',fmt(chosen_row.F2,3))
    c4.metric('Optimumluğu kanıtlı rota',fmt(project['summary']['proven_optimal_wave_count']))
    st.caption(f"Seçilen model: **{chosen}** · Bağımsız test: **{meta['test_start']} – {meta['test_end']}** · Rota mesafeleri yaklaşık koridor modelindedir")
    a,b=st.columns([1.4,1],gap='large')
    with a:
        st.markdown('#### Kaydedilmiş günlerde talep edilen ürün çeşidi')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=work.date,y=work.unique_skus,name='Farklı SKU',line=dict(color='#61e0b8',width=2)))
        fig.add_trace(go.Scatter(x=work.date,y=work.unique_skus.rolling(10,min_periods=2).mean(),name='10 gün ortalama',line=dict(color='#B0A7F9',dash='dash')))
        st.plotly_chart(plot_style(fig,380),**_FULL_WIDTH)
    with b:
        st.markdown('#### İki ayrı karar desteği')
        st.markdown('<div class="panel"><strong>🛒 Sınıflandırma</strong><p>Bir sonraki kayıtlı operasyon gününde ürünün sipariş edilme ihtimali. Recall, precision ve hazırlık kapasitesi değerlendirilir.</p></div>',unsafe_allow_html=True)
        st.write('')
        st.markdown('<div class="panel"><strong>🚜 Exact TSP</strong><p>Kesinleşmiş ürün ve raf listesi için optimum ziyaret sırası ve forkliftin şematik hareketi.</p></div>',unsafe_allow_html=True)
        st.write('')
        st.button('Ürün talebi ekranını aç',on_click=navigate,args=(PAGES[1],),type='primary',**_FULL_WIDTH)
        st.button('Forklift simülasyonunu aç',on_click=navigate,args=(PAGES[3],),**_FULL_WIDTH)
    st.info('Basit başlangıç: Logistic Regression. Ek modeller: Decision Tree, Random Forest ve XGBoost. Model seçimi eğitim CV ve doğrulamayla yapıldı; görülmemiş test sonuçları seçim için kullanılmadı.')


def forecast_page():
    meta,metrics,preds,val,cv,folds,fi,work,cov,pivot=get_data()
    try:
        bundles = get_models()
    except RuntimeError as exc:
        st.error(str(exc))
        st.code('python -m pip install -r requirements.txt', language='powershell')
        st.stop()
    hero('Gelecek operasyon günü: ürün talep ihtimali',
         'Dört sezon özellikli sınıflandırıcıdan birini seç; geçmişte görülen ürünleri sıralayıp hangi ürünlere hazırlık yapılabileceğini incele.',
         'ML · ÜRÜN BAZLI SINIFLANDIRMA')
    st.warning('Sipariş kayıtları 19 Ekim 2023 tarihinde bitiyor. Bu uygulama gerçek 2026 talebi için canlı tahmin üretmez; yalnız tarihsel bir prototip gösterimidir.')
    if 'XGBoost' not in bundles:
        st.warning(
            'XGBoost geçici olarak kullanılamıyor: kayıtlı model '
            + str(meta.get('xgboost_version')) + ', mevcut kütüphane '
            + xgboost.__version__ + '. Ürün tahmininde diğer üç model kullanılabilir. '
            'XGBoost için aynı Python ortamında '
            '`python -m pip install xgboost==' + str(meta.get('xgboost_version')) + '` çalıştırın.'
        )
    last=pivot.index[-1]
    c1,c2,c3=st.columns([1.4,1,1])
    with c1:
        available_models = [name for name in MODEL_SLUGS if name in bundles]
        default_model = meta['selected_model'] if meta['selected_model'] in bundles else available_models[0]
        model_name = st.selectbox('Sınıflandırma modeli', available_models, index=available_models.index(default_model))
    with c2:
        date=st.date_input('Planlanan sonraki operasyon günü',value=(last+pd.Timedelta(days=1)).date(),min_value=(last+pd.Timedelta(days=1)).date(),max_value=(last+pd.Timedelta(days=30)).date())
    with c3:
        top_k=st.selectbox('Önerilecek ürün adedi (Top-K)',[10,20,30,50],index=1)
    X=future_features(pivot,date)
    bundle=bundles[model_name]
    prob=bundle['model'].predict_proba(X[bundle['features']])[:,1]
    pred=X[['Reference','freq_7','freq_30','mean_units_30','month','season_code']].copy()
    pred['ML_probability']=prob
    pred['Tahmin']=np.where(prob>=bundle['threshold'],'Talep bekleniyor','Talep beklenmiyor')
    pred['30_gun_adet']=(pred.mean_units_30*30).round().astype(int)
    pred=pred.sort_values('ML_probability',ascending=False).reset_index(drop=True)
    pred['Sira']=np.arange(1,len(pred)+1)
    best=pred.head(top_k)
    a,b,c,d=st.columns(4)
    a.metric('Bilinen ürün adayı',len(pred))
    b.metric('Top-K ortalama model skoru',fmt(best.ML_probability.mean()*100,1)+'%')
    c.metric('CV ile seçilen karar eşiği',fmt(bundle['threshold']*100,1)+'%')
    d.metric('Pozitif sınıflanan ürün',int((prob>=bundle['threshold']).sum()))
    st.caption('Model skorları kalibre edilmiş gerçekleşme olasılığı garantisi değildir. SKU referansından bot/terlik bilgisi çıkarılmamıştır.')
    l,r=st.columns([1.25,1],gap='large')
    with l:
        st.markdown('#### Önceliklendirilmiş ürün listesi')
        show=best[['Reference','ML_probability','freq_30','30_gun_adet','Tahmin']].copy()
        show.columns=['Ürün referansı','Model skoru (%)','Son 30 gün sıklığı (%)','Son 30 gün ürün adedi','Sınıf kararı']
        show['Model skoru (%)']=(show['Model skoru (%)']*100).round(1)
        show['Son 30 gün sıklığı (%)']=(show['Son 30 gün sıklığı (%)']*100).round(1)
        st.dataframe(show,hide_index=True,**_FULL_WIDTH,height=420)
        st.download_button('Ürün hazırlık listesini CSV indir',show.to_csv(index=False).encode('utf-8-sig'),file_name='urun_hazirlik_listesi.csv',mime='text/csv')
    with r:
        st.markdown('#### En yüksek skorlu 15 ürün')
        plot=best.head(15).iloc[::-1]
        fig=go.Figure(go.Bar(x=plot.ML_probability*100,y=plot.Reference,orientation='h',marker_color=COLORS[model_name]))
        fig.update_layout(xaxis_title='Model skoru (%)',yaxis_title='Ürün referansı',xaxis_range=[0,100])
        st.plotly_chart(plot_style(fig,440),**_FULL_WIDTH)
    st.markdown('#### Seçilen ürünün tarihsel siparişleri')
    product=st.selectbox('Referans',best.Reference.to_list()+[x for x in sorted(pivot.columns) if x not in set(best.Reference)])
    series=pivot[product].tail(80)
    fig=go.Figure(go.Bar(x=series.index,y=series.values,marker_color='#42c7b3',name='Gerçek ürün adedi'))
    fig.add_trace(go.Scatter(x=series.index,y=series.rolling(7,min_periods=1).mean(),name='7 kayıt günü ort.'))
    fig.update_layout(xaxis_title='Kayıtlı operasyon günü',yaxis_title='Sipariş adedi')
    st.plotly_chart(plot_style(fig,315),**_FULL_WIDTH)
    st.info('Sezon özellikleri (ay, mevsim, yıl içi sinüs/kosinüs, geçmiş aynı sezondaki gözlemler) dört modelde de kullanılmaktadır. Sadece Ocak–Ekim 2023 gözlemlerinden kesin yıllık sezon etkisi çıkarılamaz.')


def perf_page():
    meta,metrics,preds,val,cv,folds,fi,work,cov,pivot=get_data()
    chosen=meta['selected_model']
    mr=metrics.loc[metrics.model.eq(chosen)].iloc[0]
    hero('Model performansı · dört sınıflandırıcı',
         'Zaman sıralı 3 fold CV → ayrı doğrulama → model seçimi → son 26 kayıtlı gün üzerinde hiç kullanılmamış test. Dört model de aynı sezon özelliklerini kullanır.',
         'CV · UNSEEN TEST · PRECISION / RECALL')
    a,b,c,d=st.columns(4)
    a.metric('Seçili model',chosen)
    b.metric('Unseen Recall',fmt(mr.Recall*100,1)+'%')
    c.metric('Unseen Precision',fmt(mr.Precision*100,1)+'%')
    d.metric('Unseen Accuracy',fmt(mr.Accuracy*100,1)+'%')
    st.caption(f"Seçili modelin unseen F2 skoru: {mr.F2:.3f} · Balanced accuracy: {mr.BalancedAccuracy:.3f}")
    st.markdown('#### İş açısından karar: Recall öncelikli, precision kontrol altında')
    st.info('**Accuracy**, tüm doğru sınıflandırmaların oranıdır. Yalnız Accuracy ile seçim yapmıyoruz: negatiflerin çok olduğu bir veri kümesinde talep edecek ürünleri kaçıran model de yüksek Accuracy alabilir. Bu yüzden Balanced Accuracy, Recall, Precision ve F2 birlikte gösterilir.')
    st.markdown('**Yanlış negatif (FN):** Talep gelecek ürünü atlarız; hazırlıksız kalabiliriz. **Yanlış pozitif (FP):** Talep gelmeyen ürüne gereksiz hazırlık yapılır. Bu nedenle **F2**, recall’a F1’den daha fazla ağırlık verir. CV döneminde precision için en az %60 hedefi konuldu; bu eşik unseen testte garanti değildir.')
    if float(mr.Precision)<float(meta['precision_floor']):
        st.warning(f"Seçili model, unseen testte hedef precision alt sınırını karşılamadı: {mr.Precision:.1%} < {meta['precision_floor']:.0%}. Gerçek kullanım öncesi eşik/kapasite planı gözden geçirilmelidir.")
    st.markdown('#### Gerçek test sonuçları — dört model + basit kural')
    fields=['model','Accuracy','BalancedAccuracy','ROC_AUC','PR_AUC','Precision','Recall','F2','F1','Precision_at_20','Recall_at_20','FP','FN','threshold']
    shown=metrics[fields].copy().rename(columns={'model':'Model','Accuracy':'Accuracy','BalancedAccuracy':'Balanced Accuracy','ROC_AUC':'ROC-AUC','PR_AUC':'PR-AUC','Precision':'Precision','Recall':'Recall','F2':'F2','F1':'F1','Precision_at_20':'Precision@20','Recall_at_20':'Recall@20','FP':'FP','FN':'FN','threshold':'Eşik'})
    st.dataframe(shown.style.format({c:'{:.3f}' for c in ['Accuracy','Balanced Accuracy','ROC-AUC','PR-AUC','Precision','Recall','F2','F1','Precision@20','Recall@20','Eşik']}),hide_index=True,**_FULL_WIDTH)
    st.caption(f"Bağımsız test: {meta['test_start']} – {meta['test_end']}, {meta['n_test_sku_days']:,} ürün-gün. Test pozitif oranı: {meta['test_prevalence']:.1%}. Kural sütunu beşinci ML modeli değildir.")
    st.markdown('#### Modellerin Accuracy, Recall ve F2 karşılaştırması')
    cmp=metrics[metrics.model.isin(MODEL_SLUGS)].copy()
    fig=go.Figure()
    for key,label,color in [('Accuracy','Accuracy','#A6B8FF'),('Recall','Recall','#52DBB4'),('F2','F2','#FFBD70')]:
        fig.add_trace(go.Bar(x=cmp.model,y=cmp[key],name=label,marker_color=color))
    fig.update_layout(barmode='group',yaxis=dict(title='Skor',range=[0,1]),xaxis_title='',legend=dict(orientation='h',y=1.20))
    st.plotly_chart(plot_style(fig,390),**_FULL_WIDTH)
    st.markdown('#### Eğitim dönemindeki zaman sıralı CV')
    cvview=cv[['model','setting','cv_mean_PR_AUC','cv_mean_ROC_AUC','Accuracy','Precision','Recall','F2','threshold']].rename(columns={'model':'Model','setting':'Ayar','cv_mean_PR_AUC':'Ortalama PR-AUC','cv_mean_ROC_AUC':'Ortalama ROC-AUC','Accuracy':'Accuracy','threshold':'Eşik'})
    st.dataframe(cvview.style.format({c:'{:.3f}' for c in ['Ortalama PR-AUC','Ortalama ROC-AUC','Accuracy','Precision','Recall','F2','Eşik']}),hide_index=True,**_FULL_WIDTH)
    with st.expander('CV fold ayrıntıları (yalnız eğitim)',expanded=False):
        st.dataframe(folds[['model','setting','fold','train_days','valid_days','Accuracy','PR_AUC','ROC_AUC']],hide_index=True,**_FULL_WIDTH)
    st.markdown('#### Ayrı doğrulamada model seçimi')
    st.dataframe(val[['model','setting','Accuracy','PR_AUC','Precision','Recall','F2','threshold']].sort_values('F2',ascending=False),hide_index=True,**_FULL_WIDTH)
    st.caption('Model parametreleri eğitim içi CV PR-AUC ile, her modelin eşiği CV OOF tahminlerinde F2 ile seçildi. Nihai model ailesi doğrulama F2 + precision koşuluna göre seçildi. Unseen test bu kararlara katılmadı.')
    left,right=st.columns(2,gap='large')
    with left:
        st.markdown('#### ROC eğrileri')
        fig=go.Figure()
        for name in list(MODEL_SLUGS)+['30 Günlük Sıklık (Kural)']:
            column='naive_frequency' if name.startswith('30 Günlük') else prob_col(name)
            x,y,_=roc_curve(preds.target,preds[column])
            fig.add_trace(go.Scatter(x=x,y=y,name=name,line=dict(color=COLORS[name],width=2)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(color='#8895A4',dash='dash'),name='Rastgele'))
        fig.update_layout(xaxis_title='Yanlış pozitif oranı',yaxis_title='Doğru pozitif oranı')
        st.plotly_chart(plot_style(fig,390),**_FULL_WIDTH)
    with right:
        st.markdown('#### Precision–Recall eğrileri')
        fig=go.Figure()
        for name in list(MODEL_SLUGS)+['30 Günlük Sıklık (Kural)']:
            column='naive_frequency' if name.startswith('30 Günlük') else prob_col(name)
            precision,recall,_=precision_recall_curve(preds.target,preds[column])
            fig.add_trace(go.Scatter(x=recall,y=precision,name=name,line=dict(color=COLORS[name],width=2)))
        fig.add_hline(y=float(preds.target.mean()),line_color='#8B97A9',line_dash='dash')
        fig.update_layout(xaxis_title='Recall',yaxis_title='Precision')
        st.plotly_chart(plot_style(fig,390),**_FULL_WIDTH)
    st.markdown('#### Seçilen modelin hataları (test)')
    c1,c2,c3,c4=st.columns(4)
    for col,lab,num in zip([c1,c2,c3,c4],['TP · doğru talep','FN · kaçırılan talep','FP · gereksiz hazırlık','TN · doğru ret'],[mr.TP,mr.FN,mr.FP,mr.TN]):
        col.metric(lab,int(num))
    threshold=float(mr.threshold)
    compare=preds.copy()
    col=prob_col(chosen)
    compare['prediction']=(compare[col]>=threshold).astype(int)
    fn=compare[(compare.target==1)&(compare.prediction==0)].nsmallest(15,col)
    fp=compare[(compare.target==0)&(compare.prediction==1)].nlargest(15,col)
    l,r=st.columns(2)
    with l:
        st.markdown('##### Yanlış negatif örnekleri')
        st.dataframe(fn[['date','Reference','target',col]].rename(columns={col:'Model skoru'}),hide_index=True,**_FULL_WIDTH)
    with r:
        st.markdown('##### Yanlış pozitif örnekleri')
        st.dataframe(fp[['date','Reference','target',col]].rename(columns={col:'Model skoru'}),hide_index=True,**_FULL_WIDTH)
    st.markdown('#### Günlük Top-20 ürün listesinin isabeti')
    daily=[]
    for date,g in preds.groupby('date'):
        row={'date':date}
        for name in MODEL_SLUGS:
            row[name]=float(g.nlargest(20,prob_col(name)).target.mean())*100
        daily.append(row)
    daily=pd.DataFrame(daily)
    fig=go.Figure()
    for name in MODEL_SLUGS:
        fig.add_trace(go.Scatter(x=daily.date,y=daily[name],name=name,line=dict(color=COLORS[name])))
    fig.update_layout(yaxis_title='Precision@20 (%)',xaxis_title='Test günü',yaxis_range=[0,100])
    st.plotly_chart(plot_style(fig,370),**_FULL_WIDTH)
    st.markdown('#### Seçilen modelin özellik önemleri')
    data=fi.head(15).iloc[::-1]
    fig=go.Figure(go.Bar(x=data.importance,y=data.feature,orientation='h',marker_color='#64D9B5'))
    st.plotly_chart(plot_style(fig,370),**_FULL_WIDTH)
    st.caption('Bu grafik, seçili modelin algoritmaya özgü katsayı/önem büyüklüğüdür; nedensellik veya modeller arası doğrudan karşılaştırma değildir.')
    st.download_button('Unseen testte dört modelin tahminleri',preds.to_csv(index=False).encode('utf-8-sig'),file_name='unseen_dort_model_tahminleri.csv',mime='text/csv')

def show_route(route,locs,geometry,expected=None):
    details=route_details(route,locs,geometry)
    if expected is not None and abs(float(expected)-float(details['distance']))>.16:
        st.error('Kayıtlı rota uzunluğu ile çizilen mesafe uyuşmuyor.');st.stop()
    a,b,c=st.columns(3)
    a.metric('Raf durağı',details['total_stops'])
    b.metric('Toplanacak ürün',details['total_units'])
    c.metric('Minimum rota (model)',fmt(details['distance'],1)+' koordinat')
    st.markdown('#### 🚜 Forklift toplama simülasyonu')
    components.html(render_simulator(details,backdrop()),height=790,scrolling=False)
    st.caption('Şematik forklift animasyonu; fiziksel forklift sürüşü değildir. Koridor ve giriş/çıkış koordinatı modele dayalı varsayımlardır; kesin optimum yalnız bu matematiksel mesafe modeli için geçerlidir.')
    with st.expander('Raf ziyaret sırası ve etap mesafeleri',expanded=True):
        st.write(' → '.join(route))
        frame=pd.DataFrame([{'Adım':i,'Başlangıç':a['from'],'Varış':a['to'],'Toplanacak ürün':a['units'],
                            'Etap (koordinat)':round(a['distance'],1),'Birikimli (koordinat)':round(a['cumulative'],1)}
                            for i,a in enumerate(details['legs'],1)])
        st.dataframe(frame,hide_index=True,**_FULL_WIDTH)
        st.download_button('Rota CSV indir',data=frame.to_csv(index=False).encode('utf-8-sig'),file_name='rota_adimlari.csv')
    return details


def historical_route():
    p=get_route_project()
    hero('Exact TSP rotası & forklift simülasyonu',
         'Toplama dalgasını seç, önceden optimumluğu kanıtlanmış raf sırasını incele ve forkliftin temsili hareketini izle.',
         'KESİN OPTİMİZASYON · FİZİKSEL MODEL VARSAYIMLI')
    choices=wave_choices();ids=sorted(choices);initial=ids.index(33168) if 33168 in ids else 0
    wave=st.selectbox('Toplama dalgası',ids,index=initial,format_func=lambda k:choices[k])
    row=p['waves'].loc[p['waves'].waveNumber.eq(wave)].iloc[0]
    solver=p['solver'].loc[p['solver'].waveNumber.eq(wave)].iloc[0]
    a,b,c=st.columns(3)
    a.metric('Minimum model rotası',fmt(row.optimal_distance,1)+' koordinat')
    b.metric('Optimumluk',str(solver.status))
    c.metric('Kayıtlı çözüm süresi',fmt(solver.solve_seconds,3)+' sn')
    show_route(parse_saved_route(p,wave),wave_locations(p,wave),p['geometry'],row.optimal_distance)


def new_route():
    p=get_route_project()
    hero('Yeni toplama planı','Bildiğin rafları seç ve ürün adetlerini gir. Exact TSP en kısa ziyaret sırasını oluşturur; ML bu aşamada rota hesaplamaz.')
    st.info('Yalnız koordinatı bilinen z≤2 raflar seçilebilir. RC-01 gibi eşleşmeyen adresler için rota hesaplanmaz.')
    storage=p['storage'].loc[p['storage'].z.le(2)].dropna(subset=['x','y','z','originalLocation']).copy()
    all_racks=sorted(storage.originalLocation.astype(str).unique())
    available=set(p['waves'].waveNumber)
    suggestions=[x for x in (33168,33169,41760,41761) if x in available]
    with st.expander('Hazır dalgadan rafları aktar',expanded=True):
        if suggestions:
            ex=st.selectbox('Örnek dalga',suggestions)
            st.button('Bu rafları kullan',on_click=lambda:st.session_state.update({'selected_racks':wave_locations(p,ex).locations.tolist(),'fresh_route_result':None}))
    selected=st.multiselect('Raf seç (en fazla 25)',all_racks,key='selected_racks',max_selections=25,placeholder='Raf ara…')
    if not selected:
        st.warning('Önce raf seç veya örnek dalgayı aktar.');return
    loc=storage.loc[storage.originalLocation.isin(selected)].copy().rename(columns={'originalLocation':'locations'}).sort_values('locations')
    table=pd.DataFrame({'Raf':loc.locations.tolist(),'Adet':[1]*len(loc)})
    edited=st.data_editor(table,hide_index=True,**_FULL_WIDTH,num_rows='fixed',disabled=['Raf'],
                          column_config={'Raf':st.column_config.TextColumn('Raf'),
                                         'Adet':st.column_config.NumberColumn('Adet',min_value=1,max_value=1000,step=1,required=True)},
                          key='edit_'+hashlib.sha1('|'.join(sorted(selected)).encode()).hexdigest()[:10])
    qty=pd.to_numeric(edited.Adet,errors='coerce')
    if qty.isna().any() or qty.le(0).any() or not np.allclose(qty,np.rint(qty)):
        st.error('Adetler pozitif tam sayı olmalı.');return
    loc=loc.merge(pd.DataFrame({'locations':edited.Raf,'toplam_urun':qty.astype(int)}),on='locations',validate='one_to_one')
    signature=json.dumps(loc[['locations','toplam_urun']].to_dict('records'),sort_keys=True)
    if st.button('Exact TSP ile optimum rotayı hesapla',type='primary'):
        with st.spinner('Optimum sıra aranıyor…'):
            answer,route=solve_selected(p,loc,limit_sec=20.0,prefer_cp_sat=True)
        st.session_state['fresh_route_result']={'signature':signature,'solver':answer,'route':route}
    saved=st.session_state.get('fresh_route_result')
    if saved and saved.get('signature')==signature:
        res=saved['solver'];route=saved['route']
        if res['status']=='OPTIMAL':st.success(f"Model optimumu: {fmt(res['distance'],1)} koordinat · {res['backend']} · {fmt(res['elapsed'],3)} sn")
        elif res['status']=='FEASIBLE':st.warning('Geçerli rota bulundu ama optimumluğu kanıtlanamadı.')
        else:st.error('Çözücü durumu: '+res['status'])
        if route:show_route(route,loc,p['geometry'],res['distance'])
    elif saved:st.info('Raflar veya adetler değişmiş; yeni optimizasyon başlat.')


def methodology():
    meta,metrics,preds,val,cv,folds,fi,work,cov,pivot=get_data()
    hero('Veri, CV ve yöntem','Dört sezon özellikli sınıflandırıcı, model/eşik seçimi, bağımsız unseen test, Exact TSP ve fiziksel model sınırlılıkları.')
    st.markdown('#### Sistem akışı')
    st.code('Customer_Order → Tarih × SKU → Geçmiş talep + sezon özellikleri → 4 sınıflandırıcı → walk-forward CV → doğrulama → kilitli unseen test → ürün hazırlık listesi\n\nPicking_Wave + Storage_Location → Koordinatları bilinen toplama dalgaları → varsayımlı koridor mesafe matrisi → Exact TSP → forklift simülasyonu',language='text')
    st.markdown('#### Seçilen veri ve model kararı')
    st.write('Aynı 3 CSV kullanılır. Dört model: **Logistic Regression (baseline)**, **Decision Tree**, **Random Forest** ve **XGBoost**. Her modelde takvim özellikleri (ay/çeyrek/mevsim, sinüs/kosinüs) ve yalnız geçmişteki ürün davranışı kullanılır. Ek olarak 30 günlük talep sıklığı algoritmasız referanstır.')
    st.write(f"30 başlangıç günü, ardından **{meta['train_days']} eğitim**, **{meta['validation_days']} doğrulama**, **{meta['test_days']} görülmemiş test** operasyon günü. Test aralığı: **{meta['test_start']} – {meta['test_end']}**.")
    st.write('Eğitim içinde 3 expanding-window fold: 36, 56, 76 günlük eğitim ve her birinden sonra 20 günlük doğrulama. Parametreler eğitim içi CV PR-AUC ile; eşik CV out-of-fold tahminleriyle seçildi. Ayrı doğrulamada precision tabanı ve F2 kullanılarak model seçildi. Son testte tekrar ayar yapılmadı.')
    st.markdown('#### Recall mı, precision mı?')
    st.write('**Recall öncelikli.** Gerçekte talep edilen SKU’yu düşük tahmin etmek (FN), operasyon hazırlığını kaçırabilir. Fakat gereksiz hazırlık (FP) maliyetlidir. Bu nedenle F2 ve en az %60 precision hedefi birlikte kullanıldı. Test döneminde hedefin karşılanması garanti edilmez; Top-K listesinde kapasite sınırı ayrıca gösterilir.')
    st.markdown('#### Sezon ve veri sınırları')
    st.write('Kaynak yalnız 2023 Ocak–Ekim dönemini içerir. Tek yıl, ay/mevsim sinyallerini tam güvenilir yıllık sezonsallık olarak doğrulamaz. Referans kodunda bot/terlik gibi ürün sınıfları yok; kurgusal ürün türü eklenmedi.')
    st.write('İşlenmeyen takvim günleri otomatik sıfır sayılmaz. Model yalnızca daha önce görülmüş SKU’ları sıralar; yeni ürünler için cold-start vardır. Güncel stok bulunmadığı için stok tükenme veya gerçek ikmal maliyeti modellenmez.')
    st.markdown('#### Optimizasyonun sınırı')
    st.write('Exact TSP yalnızca tanımlanan yaklaşık depo koridor mesafesi matrisi için optimumluğu kanıtlar; gerçek fiziksel yürüyüş mesafesi olarak doğrulanmadı. Forklift 2B temsili animasyondur; gerçek araç telemetrisi değildir.')


def main():
    st.sidebar.markdown('### 🚜 DepoNav AI')
    st.sidebar.caption('4 sınıflandırıcı + Exact TSP + forklift')
    if 'nav' not in st.session_state: st.session_state['nav']=PAGES[0]
    page=st.sidebar.radio('Gezinme',PAGES,key='nav',label_visibility='collapsed')
    st.sidebar.divider()
    st.sidebar.caption('2023 veri seti · Son 26 operasyon günü unseen test')
    st.sidebar.caption('Mesafe: yaklaşık koridor modelinin koordinat ölçeği')
    if page==PAGES[0]:overview()
    elif page==PAGES[1]:forecast_page()
    elif page==PAGES[2]:perf_page()
    elif page==PAGES[3]:historical_route()
    elif page==PAGES[4]:new_route()
    else:methodology()
    st.divider()
    st.caption('Yapay Zekâ Sistem Tasarımı ve Yaşam Döngüsü · Dört sezon özellikli ML sınıflandırıcı · Exact TSP · 2B forklift simülasyonu')

if __name__=='__main__':main()
