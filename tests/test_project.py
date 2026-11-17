# -*- coding: utf-8 -*-
"""4 sınıflandırıcı + 3-fold zaman CV + rota + simülasyon bütünlük testleri."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, fbeta_score, recall_score, precision_score, accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from product_demand import BASE,SEASON,source_matrix,features_for_day,make_training_frame
from train_models import FEATURES,make_model,time_cv_splits,prepare_data,slug
from warehouse_core import load_project,wave_locations,parse_saved_route,route_details,solve_selected
from forklift_simulator import render_simulator


def test_time_order_and_features_without_future_leak():
    pivot,_=source_matrix()
    i=140
    before=features_for_day(pivot,i).sort_values('Reference').reset_index(drop=True)
    altered=pivot.copy()
    altered.iloc[i:,:]=99999
    after=features_for_day(altered,i).sort_values('Reference').reset_index(drop=True)
    pd.testing.assert_frame_equal(before,after)
    assert set(SEASON).issubset(set(FEATURES))
    assert len(FEATURES)==len(set(FEATURES))


def test_temporal_folds_and_unseen_split():
    _,_,frame,coverage,tr,va,te=prepare_data()
    assert tr.date.max()<va.date.min()<te.date.min()
    assert va.date.max()<te.date.min()
    for a,b,c,d in time_cv_splits():
        assert a<b<=c<d<=126
        assert frame.loc[frame.day_idx.eq(b-1),'date'].max()<frame.loc[frame.day_idx.eq(c),'date'].min()
    meta=json.loads((ROOT/'outputs'/'urun_talep_deney_ozeti.json').read_text(encoding='utf8'))
    assert meta['model_selection_uses_unseen'] is False
    assert meta['train_days']==96 and meta['validation_days']==26 and meta['test_days']==26
    assert len(meta['models_compared'])==4


def test_four_models_reproduce_saved_unseen_predictions():
    _,_,_,_,_,_,test=prepare_data()
    preds=pd.read_csv(ROOT/'outputs'/'urun_talep_unseen_test_tahminleri.csv')
    result=pd.read_csv(ROOT/'outputs'/'urun_talep_unseen_test_metrikleri.csv')
    assert len(preds)==len(test)
    for name in ['Logistic Regression (Baseline)','Random Forest','Decision Tree','XGBoost']:
        model=joblib.load(ROOT/'models'/f'urun_talep_{slug(name)}.joblib')
        assert model['features']==FEATURES
        p=model['model'].predict_proba(test[model['features']])[:,1]
        assert np.allclose(p,preds['prob_'+slug(name)])
        row=result[result.model.eq(name)].iloc[0]
        assert abs(average_precision_score(test.target,p)-row.PR_AUC)<1e-10
        assert abs(accuracy_score(test.target, p>=row.threshold)-row.Accuracy)<1e-10
        assert abs(fbeta_score(test.target,p>=row.threshold,beta=2)-row.F2)<1e-10


def test_selected_model_only_validation():
    meta=json.loads((ROOT/'outputs'/'urun_talep_deney_ozeti.json').read_text(encoding='utf8'))
    val=pd.read_csv(ROOT/'outputs'/'urun_talep_model_secimi_validasyon.csv')
    qualified=val[val.Precision>=meta['precision_floor']]
    if qualified.empty:qualified=val
    expected=qualified.sort_values(['F2','PR_AUC'],ascending=False).iloc[0]['model']
    assert meta['selected_model']==expected
    assert (ROOT/'models'/'urun_talep_secili_model.joblib').is_file()


def test_cold_start_rule():
    pivot,_=source_matrix()
    for idx in [30,70,152,177]:
        present=set(pivot.columns[(pivot.iloc[:idx].to_numpy()>0).any(axis=0)])
        got=set(features_for_day(pivot,idx).Reference)
        assert present==got


def test_routes_and_simulation():
    p=load_project(ROOT)
    route=parse_saved_route(p,33168)
    loc=wave_locations(p,33168)
    details=route_details(route,loc,p['geometry'])
    expected=p['waves'].loc[p['waves'].waveNumber.eq(33168),'optimal_distance'].iloc[0]
    assert abs(details['distance']-expected)<.16
    html=render_simulator(details,[[1,2],[3,4]])
    assert '__ROUTE_JSON__' not in html
    assert str(details['total_stops']) in html


def test_exact_tsp_for_new_racks():
    p=load_project(ROOT)
    loc=wave_locations(p,33168).head(5)
    ans,route=solve_selected(p,loc,limit_sec=15,prefer_cp_sat=True)
    assert ans['status']=='OPTIMAL'
    assert len(route)==len(loc)+2


def test_streamlit_code_no_removed_parameters():
    script=(ROOT/'app.py').read_text(encoding='utf8')
    assert 'use_container_width' not in script
    assert 'sezonsuz' not in script.lower()
    assert 'Precision_at_20' in script
    assert 'Accuracy' in script
    assert 'XGBoost' in script
    assert 'Decision Tree' in script
    assert 'Extra Trees' not in script
    assert 'HistGradientBoosting' not in script
