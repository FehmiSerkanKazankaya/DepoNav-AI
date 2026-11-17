# -*- coding: utf-8 -*-
"""Dört sezon özellikli sınıflandırıcı: zaman sıralı CV + bağımsız unseen test.

Modeller:
  1) Logistic Regression (baseline), 2) Random Forest,
  3) Decision Tree, 4) XGBoost.
Yalnız eğitim döneminde 3 walk-forward CV fold; ayrı doğrulama; kilitli son test.
Öncelik: pozitif talebi kaçırmamak (recall, F2), en az %60 precision hedefi.
Bu dosyayı çalıştırmadan önce outputs/tsp_* dosyaları yeniden hesaplanmaz.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, fbeta_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix,
)
from product_demand import ROOT, BASE, SEASON, source_matrix, make_training_frame, precision_at_k

OUT = ROOT / 'outputs'
MODELS = ROOT / 'models'
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
FEATURES = BASE + SEASON  # Bütün modeller aynı sezon özelliklerini kullanır.
TRAIN_END = 126  # [30,126) eğitim: 96 operasyon günü
VALID_END = 152  # [126,152) doğrulama: 26 operasyon günü
HISTORY_START = 30
PRECISION_FLOOR = 0.60
SEED = 42


def make_model(name: str, setting: str = 'default'):
    """Her sınıflandırma ailesi için bir veya iki CV aday parametresi."""
    if name == 'Logistic Regression (Baseline)':
        c = 1.0 if setting == 'default' else 0.2
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=1200, random_state=SEED))
    if name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=150 if setting == 'default' else 200,
            max_depth=9 if setting == 'default' else 12,
            min_samples_leaf=15 if setting == 'default' else 8,
            max_features='sqrt', n_jobs=-1, random_state=SEED)
    if name == 'Decision Tree':
        return DecisionTreeClassifier(
            max_depth=7 if setting == 'default' else 11,
            min_samples_leaf=45 if setting == 'default' else 18,
            random_state=SEED)
    if name == 'XGBoost':
        return XGBClassifier(
            n_estimators=180 if setting == 'default' else 260,
            max_depth=3 if setting == 'default' else 4,
            learning_rate=0.055 if setting == 'default' else 0.035,
            subsample=0.90, colsample_bytree=0.9, min_child_weight=15,
            reg_lambda=2.0, objective='binary:logistic',
            eval_metric='logloss', tree_method='hist',
            n_jobs=2, random_state=SEED)
    raise ValueError(name)


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def metrics(y, probability, threshold=0.50):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        'Accuracy': float(accuracy_score(y,pred)),
        'ROC_AUC': safe_auc(y,p),
        'PR_AUC': float(average_precision_score(y,p)),
        'Precision': float(precision_score(y,pred,zero_division=0)),
        'Recall': float(recall_score(y,pred,zero_division=0)),
        'F1': float(f1_score(y,pred,zero_division=0)),
        'F2': float(fbeta_score(y,pred,beta=2,zero_division=0)),
        'BalancedAccuracy': float(balanced_accuracy_score(y,pred)),
        'Brier': float(brier_score_loss(y,p)),
        'threshold': float(threshold), 'Prevalence': float(y.mean()),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }


def rank_metrics(table: pd.DataFrame, col: str, top_k: int):
    precision_values = []
    recall_values = []
    for _, day in table.groupby('date', sort=True):
        k = min(top_k, len(day))
        sample = day.nlargest(k, col)
        precision_values.append(sample['target'].mean())
        positives = int(day['target'].sum())
        recall_values.append(float(sample['target'].sum()) / positives if positives else np.nan)
    return {
        f'Precision_at_{top_k}': float(np.nanmean(precision_values)),
        f'Recall_at_{top_k}': float(np.nanmean(recall_values))
    }


def select_threshold(y, p, floor=PRECISION_FLOOR):
    """Önce Precision>=floor; bunun altında aday yoksa en yüksek F2.
    CV dışındaki bilgileri kullanmaz.
    """
    out=[]
    for threshold in np.round(np.linspace(0.15,0.85,36),3):
        met=metrics(y,p,float(threshold))
        out.append({'threshold':float(threshold),'Precision':met['Precision'],
                    'Recall':met['Recall'],'F2':met['F2']})
    df=pd.DataFrame(out)
    eligible=df[df.Precision>=floor]
    if eligible.empty:
        eligible=df
        rule='Precision tabanı karşılanamadı: yalnız F2 maksimumu'
    else:
        rule=f'Precision >= {floor:.2f} koşulu altında maksimum F2'
    choice=eligible.sort_values(['F2','Recall','Precision'],ascending=False).iloc[0]
    return float(choice.threshold), rule


def time_cv_splits():
    """Eğitim dönemi [30,126) içinde 3 kronolojik expanding-window fold.
    Her fold 20 test-günü, öncesinde 36/56/76 eğitim-günü kullanır.
    """
    return [(30,66,66,86),(30,86,86,106),(30,106,106,126)]


def prepare_data():
    pivot, raw=source_matrix()
    frame, coverage=make_training_frame(pivot, min_history=HISTORY_START)
    train=frame.loc[frame.day_idx<TRAIN_END].copy()
    valid=frame.loc[(frame.day_idx>=TRAIN_END)&(frame.day_idx<VALID_END)].copy()
    unseen=frame.loc[frame.day_idx>=VALID_END].copy()
    assert train.date.max()<valid.date.min()<unseen.date.min()
    assert valid.date.max()<unseen.date.min()
    assert len(pivot)==178, 'Eğitim dönemleri değişti; ayrım tarihlerini gözden geçirin.'
    assert len(train.day_idx.unique())==96 and len(valid.day_idx.unique())==26 and len(unseen.day_idx.unique())==26
    return pivot, raw, frame, coverage, train, valid, unseen


def run():
    t0=time.perf_counter()
    pivot,raw,frame,coverage,train,valid,unseen=prepare_data()
    candidates=[
        ('Logistic Regression (Baseline)','default'),
        ('Random Forest','default'),
        ('Random Forest','deep'),
        ('Decision Tree','default'),
        ('Decision Tree','deep'),
        ('XGBoost','default'),
        ('XGBoost','alt'),
    ]
    fold_rows=[]
    cv_rows=[]
    cv_oof=[]
    for name,setting in candidates:
        fold_predictions=[]
        for fold,(start,tr_end,va_start,va_end) in enumerate(time_cv_splits(),1):
            tr=frame.loc[(frame.day_idx>=start)&(frame.day_idx<tr_end)]
            va=frame.loc[(frame.day_idx>=va_start)&(frame.day_idx<va_end)]
            assert tr.date.max()<va.date.min() and va.date.max()<valid.date.min()
            model=make_model(name,setting)
            model.fit(tr[FEATURES],tr.target)
            p=model.predict_proba(va[FEATURES])[:,1]
            fold_pred=va[['date','day_idx','Reference','target']].copy()
            fold_pred['probability']=p
            fold_pred['model']=name
            fold_pred['setting']=setting
            fold_pred['fold']=fold
            fold_predictions.append(fold_pred)
            m=metrics(va.target,p,.5)
            m.update(model=name,setting=setting,fold=fold,train_days=tr.day_idx.nunique(),
                     valid_days=va.day_idx.nunique())
            fold_rows.append(m)
        oof=pd.concat(fold_predictions,ignore_index=True)
        threshold,rule=select_threshold(oof.target,oof.probability)
        cv_score=metrics(oof.target,oof.probability,threshold)
        cv_score.update(model=name,setting=setting,threshold_rule=rule,
                        cv_mean_PR_AUC=float(np.mean([r['PR_AUC'] for r in fold_rows if r['model']==name and r['setting']==setting])),
                        cv_mean_ROC_AUC=float(np.mean([r['ROC_AUC'] for r in fold_rows if r['model']==name and r['setting']==setting])))
        cv_score.update(rank_metrics(oof,'probability',20))
        cv_rows.append(cv_score)
        cv_oof.append(oof)
        print(f'CV {name} ({setting}) | PR-AUC={cv_score["cv_mean_PR_AUC"]:.4f} | F2={cv_score["F2"]:.4f} | Recall={cv_score["Recall"]:.4f} | Precision={cv_score["Precision"]:.4f} | eşik={threshold:.2f}',flush=True)

    cv=pd.DataFrame(cv_rows)
    # Her ailenin hiperparametre seçimi YALNIZ eğitim dönemindeki CV ile yapılır.
    selected_configs=(cv.sort_values(['cv_mean_PR_AUC','F2'],ascending=False)
                        .drop_duplicates('model')[['model','setting','threshold','threshold_rule','cv_mean_PR_AUC','F2']])
    validation=[]
    final_specs={}
    for item in selected_configs.itertuples(index=False):
        name,setting,threshold=item.model,item.setting,float(item.threshold)
        model=make_model(name,setting)
        model.fit(train[FEATURES],train.target)
        p=model.predict_proba(valid[FEATURES])[:,1]
        tab=valid[['date','day_idx','Reference','target']].copy()
        tab['probability']=p
        scores=metrics(valid.target,p,threshold)
        scores.update(rank_metrics(tab,'probability',20))
        scores.update(model=name,setting=setting,threshold_rule=item.threshold_rule,cv_mean_PR_AUC=item.cv_mean_PR_AUC)
        validation.append(scores)
        final_specs[name]={'setting':setting,'threshold':threshold}
        print(f'VALID {name}: F2={scores["F2"]:.4f}, Precision={scores["Precision"]:.4f}, Recall={scores["Recall"]:.4f}, AP={scores["PR_AUC"]:.4f}',flush=True)

    val=pd.DataFrame(validation)
    # İş gerekçesi: Recall önemli, ama fazla yalancı alarmın hazırlığı zorlaştırması önlenmeli.
    qualified=val[val.Precision>=PRECISION_FLOOR]
    if qualified.empty:
        qualified=val
        selection_note='Hiçbir model doğrulama precision tabanını karşılamadı; en iyi F2 seçildi.'
    else:
        selection_note=f'Doğrulamada Precision>={PRECISION_FLOOR:.2f}; ardından F2, sonra PR-AUC.'
    chosen=str(qualified.sort_values(['F2','PR_AUC'],ascending=False).iloc[0]['model'])

    train_val=frame.loc[frame.day_idx<VALID_END].copy()
    test_predictions=unseen[['date','day_idx','Reference','target','freq_30']].copy().reset_index(drop=True)
    test_predictions['naive_frequency']=unseen.freq_30.to_numpy(dtype=float)
    test_summary=[]
    for name in final_specs:
        setting=final_specs[name]['setting']
        threshold=final_specs[name]['threshold']
        model=make_model(name,setting)
        model.fit(train_val[FEATURES],train_val.target)
        p=model.predict_proba(unseen[FEATURES])[:,1]
        test_predictions[f'prob_{slug(name)}']=p
        test_predictions[f'pred_{slug(name)}']=(p>=threshold).astype(int)
        row=metrics(unseen.target,p,threshold)
        row.update(rank_metrics(test_predictions.assign(probability=p),'probability',20))
        row.update(rank_metrics(test_predictions.assign(probability=p),'probability',50))
        row.update(model=name,setting=setting,chosen_for_app=name==chosen)
        test_summary.append(row)
        joblib.dump({'model':model,'features':FEATURES,'threshold':float(threshold),
                     'choice':name,'sklearn_version':sklearn.__version__,
                     'xgboost_version':xgboost.__version__,
                     'baseline_model':name.startswith('Logistic'),
                     'selection':'Eğitim CV + ayrı doğrulama; görülmemiş teste göre seçim yapılmadı.'},
                    MODELS/f'urun_talep_{slug(name)}.joblib',compress=3)
        print(f'UNSEEN {name}: AP={row["PR_AUC"]:.4f}, F2={row["F2"]:.4f}, Precision={row["Precision"]:.4f}, Recall={row["Recall"]:.4f}',flush=True)

    # 30 günlük geçmiş sıklık: beşinci MODEL DEĞİL, basit kural referansı.
    naive_threshold=.5  # yalnız referans, optimize edilmiyor
    naive=metrics(unseen.target,test_predictions.naive_frequency,naive_threshold)
    naive.update(rank_metrics(test_predictions,'naive_frequency',20))
    naive.update(rank_metrics(test_predictions,'naive_frequency',50))
    naive.update(model='30 Günlük Sıklık (Kural)',setting='no_model',chosen_for_app=False)
    test_summary.append(naive)
    result=pd.DataFrame(test_summary)
    selected_bundle=joblib.load(MODELS/f'urun_talep_{slug(chosen)}.joblib')
    joblib.dump(selected_bundle,MODELS/'urun_talep_secili_model.joblib',compress=3)

    pd.DataFrame(fold_rows).to_csv(OUT/'urun_talep_cv_fold_metrikleri.csv',index=False)
    cv.to_csv(OUT/'urun_talep_cv_ozeti.csv',index=False)
    pd.concat(cv_oof,ignore_index=True).to_csv(OUT/'urun_talep_cv_oof_tahminleri.csv',index=False)
    val.to_csv(OUT/'urun_talep_model_secimi_validasyon.csv',index=False)
    result.to_csv(OUT/'urun_talep_unseen_test_metrikleri.csv',index=False)
    test_predictions.to_csv(OUT/'urun_talep_unseen_test_tahminleri.csv',index=False)
    coverage.to_csv(OUT/'urun_talep_kapsam_analizi.csv',index=False)
    pivot.rename_axis('date').to_csv(OUT/'urun_talep_gunluk_urun_matrisi.csv')
    work=raw.groupby('date').agg(total_units=('units','sum'),order_lines=('Reference','size'),unique_skus=('Reference','nunique'))
    work.to_csv(OUT/'urun_talep_gunluk_ozet.csv')
    # Korelasyona dayalı önem ölçümlerini bütün aday modeller için karşılaştırılabilir nedensel çıkarım olarak sunmuyoruz.
    if hasattr(selected_bundle['model'],'feature_importances_'):
        fimp=selected_bundle['model'].feature_importances_
    elif hasattr(selected_bundle['model'],'coef_'):
        fimp=np.abs(selected_bundle['model'].coef_[0])
    elif hasattr(selected_bundle['model'],'steps'):
        fimp=np.abs(selected_bundle['model'][-1].coef_[0])
    else:
        fimp=np.full(len(FEATURES),np.nan)
    pd.DataFrame({'feature':FEATURES,'importance':fimp}).sort_values('importance',ascending=False).to_csv(OUT/'urun_talep_ozellik_onemi.csv',index=False)

    meta={
        'task':'Bir sonraki kayıtlı/planlı operasyon gününde geçmişte bilinen bir SKU sipariş edilir mi?',
        'date_start':str(pivot.index.min().date()),'date_end':str(pivot.index.max().date()),
        'n_recorded_days':int(len(pivot)),'n_products_total_historical':int(len(pivot.columns)),
        'history_days':HISTORY_START,'train_end_index_exclusive':TRAIN_END,'val_end_index_exclusive':VALID_END,
        'train_days':int(train.day_idx.nunique()),'validation_days':int(valid.day_idx.nunique()),
        'test_days':int(unseen.day_idx.nunique()),
        'train_start':str(train.date.min().date()),'train_end':str(train.date.max().date()),
        'val_start':str(valid.date.min().date()),'val_end':str(valid.date.max().date()),
        'test_start':str(unseen.date.min().date()),'test_end':str(unseen.date.max().date()),
        'test_prevalence':float(unseen.target.mean()),'n_test_sku_days':int(len(unseen)),
        'feature_columns':FEATURES,'seasonal_features':SEASON,
        'cv_scheme':'3 expanding-window fold, 20 days each, on train only: 36/56/76 train days',
        'cv_splits':time_cv_splits(),'cv_hyperparameter_selection':'mean PR-AUC within training CV',
        'threshold_selection':'F2 maximal under precision floor, using OOF CV predictions; without test',
        'precision_floor':PRECISION_FLOOR,
        'selected_model':chosen,'selected_model_slug':slug(chosen),
        'threshold_from_cv':float(final_specs[chosen]['threshold']),
        'validation_selection_metric':'F2 within precision floor; PR-AUC tiebreaker',
        'selection_note':selection_note,
        'model_selection_uses_unseen':False,
        'baseline_classifier':'Logistic Regression (Baseline)',
        'reference_rule':'30 Günlük Sıklık (Kural)',
        'models_compared':list(final_specs),
        'test_results':result.to_dict('records'),
        'candidate_coverage_test':float(coverage.loc[coverage.day_idx>=VALID_END,'positives_covered'].sum()/coverage.loc[coverage.day_idx>=VALID_END,'active_skus'].sum()),
        'new_sku_events_test':int(coverage.loc[coverage.day_idx>=VALID_END,'new_skus'].sum()),
        'sklearn_version':sklearn.__version__,
                     'xgboost_version':xgboost.__version__,
        'notes':['Tek kısmi 2023 yılı; bot/terlik gibi ürün sınıfı kodlardan çıkarılmamıştır.',
                 'Her SKU ve tarih için yalnızca o tarihten önceki operasyon günleri kullanılmıştır.',
                 'Kaydı olmayan takvim günleri sıfır adet sayılmaz; planlı operasyon gününü kullanıcı seçer.',
                 'Toplam dört sınıflandırıcı her zaman sezon özellikleri kullanır.',
                 'Accuracy raporlanır ama recall/F2 ve precision olmadan tek başına model seçim ölçütü değildir.',
                 'Tüm modellerin eşikleri eğitim dönemindeki CV ile seçilir; testte tekrar ayarlanmaz.',
                 'Recall öncelikli; ancak yanlış alarm nedeniyle Precision alt sınırı vardır.',
                 'Yanlış pozitif = gereksiz hazırlık; yanlış negatif = tahmini beklenmeyen ürüne hazırlıksız yakalanma.',
                 'Güncel stok yok; gerçek stok tükenmesi / personel maliyeti çıkarımı yapılamaz.'],
    }
    (OUT/'urun_talep_deney_ozeti.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print('SELECTED FOR APP:',chosen,'—',selection_note)
    print(result[['model','Accuracy','BalancedAccuracy','ROC_AUC','PR_AUC','F2','F1','Precision','Recall','Precision_at_20','Recall_at_20']].to_string(index=False))
    print('Done in',round(time.perf_counter()-t0,1),'sec')
    return meta


def slug(name: str) -> str:
    return {'Logistic Regression (Baseline)':'logistic_regression',
            'Random Forest':'random_forest',
            'Decision Tree':'decision_tree',
            'XGBoost':'xgboost'}[name]

if __name__=='__main__':run()
