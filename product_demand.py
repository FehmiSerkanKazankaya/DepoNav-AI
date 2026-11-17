# -*- coding: utf-8 -*-
"""Ürün bazlı bir sonraki planlı operasyon günü talep olasılığı.
Yalnız geçmiş operasyon günleri; bilinmeyen referanslar otomatik tahmin edilmez.
2023 tek yıl: sezonsallık henüz doğrulanmış değildir.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = [
    'sku_past_days','days_since_first','days_since_last','days_since_prev_operation',
    'seen_previous','seen_2ago','seen_3ago','qty_previous','historical_rate',
    'freq_3','freq_7','freq_14','freq_30','freq_60',
    'mean_units_7','mean_units_14','mean_units_30','mean_units_60',
    'mean_units_when_sold_30','max_units_30','trend_7_30','trend_14_60',
]
SEASON = ['month','quarter','day_of_week','day_of_year_sin','day_of_year_cos',
          'month_sin','month_cos','season_code','season_history_rate','season_history_count']


def source_matrix(path: Path|None=None):
    """Kayıtsız günü sıfır saymadan, 2023 tarihlerinde kayıtlı gün x ürün matrisi."""
    filename=Path(path) if path is not None else ROOT/'data'/'Customer_Order.csv'
    df=pd.read_csv(filename,sep=';',low_memory=False)
    df.columns=df.columns.str.strip()
    needed={'Reference','creationDate','quantity (units)'}
    if not needed.issubset(df.columns):raise ValueError(f'Eksik sipariş sütunları: {needed-set(df.columns)}')
    df['Reference']=df['Reference'].astype('string').str.strip().str.upper()
    df['date']=pd.to_datetime(df.creationDate,dayfirst=True,errors='coerce').dt.normalize()
    df['units']=pd.to_numeric(df['quantity (units)'],errors='coerce')
    df=df.loc[df.date.notna() & df.Reference.notna() & df.units.notna()].copy()
    if df.units.lt(0).any():raise ValueError('Negatif miktarlar varsa önce doğrulayın.')
    pivot=df.groupby(['date','Reference'])['units'].sum().unstack(fill_value=0).sort_index().sort_index(axis=1)
    if pivot.empty:raise ValueError('Geçerli ürün ve tarih gözlemi yok.')
    return pivot, df


def season_index(ts:pd.Timestamp)->int:
    """0 kış, 1 ilkbahar, 2 yaz, 3 sonbahar; ülkeye özgü ürün tipi çıkarımı yok."""
    return (int(ts.month)%12)//3


def features_for_day(pivot:pd.DataFrame, idx:int, date:pd.Timestamp|None=None):
    """Tahmin günü satırları: yalnız [:idx] kullanır; idx günü etiket için ayrı tutulur."""
    if idx<30:raise ValueError('En az 30 geçmiş kayıtlı operasyon günü gerekiyor.')
    date=pd.Timestamp(date if date is not None else pivot.index[idx]).normalize()
    history=pivot.iloc[:idx].to_numpy(dtype=float)
    hit=(history>0).astype(float)
    n,s=history.shape
    seen=hit.sum(axis=0)
    candidate=seen>0  # bilinmeyen SKU'lar listeye hiç dahil edilmez
    last=np.full(s, n+1, dtype=float)
    first=np.full(s,n+1,dtype=float)
    for k in range(s):
        wh=np.flatnonzero(hit[:,k]>0)
        if len(wh):
            last[k]=n-1-wh[-1]
            first[k]=n-1-wh[0]
    feats={
        'sku_past_days': seen, 'days_since_first': first, 'days_since_last':last,
        'days_since_prev_operation': np.full(s, (date-pivot.index[idx-1]).days,dtype=float),
        'seen_previous':hit[-1], 'seen_2ago':hit[-2], 'seen_3ago':hit[-3],
        'qty_previous':history[-1], 'historical_rate':hit.mean(axis=0),
    }
    for window in [3,7,14,30,60]:
        hh=hit[-window:]; qq=history[-window:]
        if window in (3,7,14,30,60):feats[f'freq_{window}']=hh.mean(axis=0)
        if window in (7,14,30,60):feats[f'mean_units_{window}']=qq.mean(axis=0)
        if window==30:
            feats['max_units_30']=qq.max(axis=0)
            feats['mean_units_when_sold_30']=qq.sum(axis=0)/np.maximum(1,hh.sum(axis=0))
    feats['trend_7_30']=feats['freq_7']-feats['freq_30']
    feats['trend_14_60']=feats['freq_14']-feats['freq_60']
    day=int(date.dayofyear);mn=int(date.month);sc=season_index(date)
    feats.update({
        'month':np.full(s,mn),'quarter':np.full(s,date.quarter),
        'day_of_week':np.full(s,date.dayofweek),
        'day_of_year_sin':np.full(s,np.sin(2*np.pi*day/365.25)),
        'day_of_year_cos':np.full(s,np.cos(2*np.pi*day/365.25)),
        'month_sin':np.full(s,np.sin(2*np.pi*(mn-1)/12)),
        'month_cos':np.full(s,np.cos(2*np.pi*(mn-1)/12)),
        'season_code':np.full(s,sc),
    })
    # Aynı sezonda *geçmişte* görülen günler, şimdiki gün hariç; tek yıl için sınırlı kanıt.
    dates=pivot.index[:idx];same=np.array([season_index(d)==sc for d in dates],bool)
    season_n=int(same.sum())
    # Veri yoksa global geçmiş oran; sahte sezonsallık üretmez.
    # Küçük örnekleme dayanıklı 5 kayıt günü prior shrinkage.
    sr=(hit[same].sum(axis=0)+5*hit.mean(axis=0))/(season_n+5)
    feats['season_history_rate']=sr
    feats['season_history_count']=np.full(s,season_n)
    features=pd.DataFrame(feats,index=pivot.columns)
    features.index.name='Reference';features=features.loc[candidate].reset_index()
    return features


def make_training_frame(pivot:pd.DataFrame,min_history:int=30):
    parts=[];coverage=[]
    for i in range(min_history,len(pivot)):
        f=features_for_day(pivot,i)
        truth=(pivot.iloc[i]>0).astype(int)
        f['target']=f['Reference'].map(truth).astype(int)
        f['date']=pivot.index[i]
        f['day_idx']=i
        parts.append(f)
        candidates=set(f.Reference)
        actual=set(truth.index[truth.gt(0)])
        coverage.append({'date':pivot.index[i],'day_idx':i,'seen_skus':len(candidates),
                         'new_skus':len(actual-candidates),'active_skus':len(actual),
                         'positives_covered':len(actual&candidates)})
    return pd.concat(parts,ignore_index=True),pd.DataFrame(coverage)


def future_features(pivot:pd.DataFrame,target_date)->pd.DataFrame:
    date=pd.Timestamp(target_date).normalize()
    if date<=pivot.index[-1]:raise ValueError('Tahmin tarihi son kayıt tarihinden sonraki bir planlı operasyon günü olmalı.')
    if (date-pivot.index[-1]).days>60:raise ValueError('Bu tarih son kayıttan 60 günden daha uzak; eğitim verisiyle güvenilir gösterim yapılamaz.')
    return features_for_day(pivot,len(pivot),date=date)


def precision_at_k(frame:pd.DataFrame,prob_col:str,k:int=20):
    vals=[]
    for _,g in frame.groupby('date'):
        x=g.nlargest(min(k,len(g)),prob_col)
        vals.append(float(x.target.sum())/len(x) if len(x) else np.nan)
    return float(np.nanmean(vals))
