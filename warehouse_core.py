# -*- coding: utf-8 -*-
"""DepoNav AI: mevcut dalgalar için rota geometrisi ve kesin TSP çözümü.

Optimalite yalnızca yaklaşık, akademik koridor modeli içindir; gerçek forklift
sürüş mesafesi/ölçülmüş metre olarak yorumlanamaz.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'


def load_project(root:Path=ROOT)->dict:
    root=Path(root);out=root/'outputs'
    required=['tsp_dalga_raf_konumlari.csv','tsp_rotalari.csv','tsp_dalga_ozetleri.csv',
              'tsp_sonuclari.csv','tsp_geometri_varsayimlari.json','warehouse_summary.json']
    missing=[name for name in required if not (out/name).exists()]
    if missing:raise FileNotFoundError('Eksik rota dosyaları: '+', '.join(missing))
    p={
      'waves':pd.read_csv(out/'tsp_dalga_ozetleri.csv',parse_dates=['first_date']),
      'routes':pd.read_csv(out/'tsp_rotalari.csv'),
      'locations':pd.read_csv(out/'tsp_dalga_raf_konumlari.csv'),
      'solver':pd.read_csv(out/'tsp_sonuclari.csv'),
      'summary':json.loads((out/'warehouse_summary.json').read_text(encoding='utf-8')),
      'geometry':json.loads((out/'tsp_geometri_varsayimlari.json').read_text(encoding='utf-8')),
      'storage':pd.read_csv(root/'data'/'Storage_Location.csv'),
    }
    if (out/'tsp_eslesmeyen_raf_adresleri.csv').exists():
      p['unmatched']=pd.read_csv(out/'tsp_eslesmeyen_raf_adresleri.csv')
    p['storage']['originalLocation']=p['storage']['originalLocation'].astype('string').str.strip().str.upper()
    for c in ('x','y','z'):
      p['storage'][c]=pd.to_numeric(p['storage'][c],errors='coerce')
    for c in ('waves','routes','locations','solver'):
      p[c]['waveNumber']=p[c]['waveNumber'].astype(int)
    if set(p['waves'].waveNumber)-set(p['routes'].waveNumber):
      raise ValueError('Birkaç dalganın kayıtlı rotası eksik')
    return p

def geometry_from(project: dict) -> tuple[np.ndarray, np.ndarray, tuple[float,float], int]:
    p = project['geometry']
    return np.array(p['spines'], float), np.array(p['aisles'], float), tuple(map(float, p['IO'])), int(p['cost_scale'])


def access_y(y: float, aisles: np.ndarray) -> float:
    return float(aisles[np.argmin(np.abs(aisles - float(y)))])


def corridor_distance(a: tuple[float,float], b: tuple[float,float], spines: np.ndarray) -> float:
    x,y = map(float,a); u,v=map(float,b)
    if np.isclose(y,v):
        return abs(x-u)
    return float(min(abs(x-s)+abs(y-v)+abs(u-s) for s in spines))


def leg_points(a: tuple[float,float], b: tuple[float,float], spines: np.ndarray) -> list[list[float]]:
    x,y=map(float,a);u,v=map(float,b)
    if np.isclose(y,v):
        points=[[x,y],[u,v]]
    else:
        s=float(min(spines,key=lambda z:abs(x-z)+abs(y-v)+abs(u-z)))
        points=[[x,y],[s,y],[s,v],[u,v]]
    unique=[points[0]]
    for p in points[1:]:
        if not np.allclose(unique[-1],p):
            unique.append(p)
    return unique


def distance_matrix(points: list[tuple[float,float]], spines: np.ndarray, scale: int) -> np.ndarray:
    n=len(points); mat=np.zeros((n,n),dtype=np.int64)
    for i in range(n):
        for j in range(i+1,n):
            mat[i,j]=mat[j,i]=round(scale*corridor_distance(points[i],points[j],spines))
    return mat


def wave_locations(project: dict, wave_id: int) -> pd.DataFrame:
    return project['locations'].loc[project['locations'].waveNumber.eq(int(wave_id))].copy()


def parse_saved_route(project: dict, wave_id: int) -> list[str]:
    selected=project['routes'].loc[project['routes'].waveNumber.eq(int(wave_id))]
    if selected.empty: raise KeyError(f'{wave_id} dalgasının optimum rotası bulunamadı')
    route=[v.strip() for v in selected.iloc[0]['rota'].split('|')]
    if not (route[0]=='I/O' and route[-1]=='I/O' and len(route)>=3):
        raise ValueError('Kaydedilmiş rota biçimi hatalı.')
    return route


def route_details(route: list[str], locations: pd.DataFrame, geometry: dict) -> dict:
    spines=np.array(geometry['spines'],dtype=float)
    aisles=np.array(geometry['aisles'],dtype=float)
    io=tuple(map(float,geometry['IO']))
    frame=locations.copy()
    frame['locations']=frame['locations'].astype(str).str.strip().str.upper()
    if frame.locations.duplicated().any():raise ValueError('Rota rafları tekrarlı konum içeriyor.')
    position={r.locations:(float(r.x),access_y(r.y,aisles)) for r in frame.itertuples()}
    physical={r.locations:[float(r.x),float(r.y)] for r in frame.itertuples()}
    units={r.locations:int(r.toplam_urun) for r in frame.itertuples()}
    position['I/O']=io
    if route[0]!='I/O' or route[-1]!='I/O':raise ValueError('Rota I/O noktasına dönmeli.')
    if len(route)!=len(position)+1 or set(route[1:-1])!=set(units):
        raise ValueError('Rota tüm rafları birer kez ziyaret etmelidir.')
    legs=[];cumulative=0.0
    for a,b in zip(route,route[1:]):
        pts=leg_points(position[a],position[b],spines)
        dist=corridor_distance(position[a],position[b],spines)
        cumulative+=dist
        legs.append({'from':a,'to':b,'points':pts,'distance':dist,
                     'cumulative':cumulative,'units':units.get(b,0)})
    return {
        'route':route, 'legs':legs, 'distance':cumulative,
        'io':[float(io[0]),float(io[1])],
        'racks':[{'name':name,'x':float(frame.loc[frame.locations.eq(name),'x'].iloc[0]),
                  'y':float(frame.loc[frame.locations.eq(name),'y'].iloc[0]),
                  'z':float(frame.loc[frame.locations.eq(name),'z'].iloc[0]),
                  'access':list(position[name]),'units':units[name]}
                 for name in route[1:-1]],
        'total_units':int(sum(units.values())),
        'total_stops':len(units),
        'geometry':geometry,
    }


def cycles(successors: dict[int,int], n: int) -> list[list[int]]:
    remaining=set(range(n));result=[]
    while remaining:
        start=min(remaining);current=start;order=[]
        while current not in order:
            order.append(current);remaining.discard(current)
            current=successors[current]
        result.append(order)
    return result


def ordered_route(successors:dict[int,int],n:int)->list[int]:
    seq=[0];current=successors[0]
    while current!=0 and len(seq)<=n:
        seq.append(current);current=successors[current]
    seq.append(0)
    if len(seq)!=n+1 or set(seq[:-1])!=set(range(n)):
        raise ValueError('Geçersiz TSP çözümü')
    return seq


def solve_exact_tsp(mat: np.ndarray, scale:int, limit_sec:float=10.0, prefer_cp_sat:bool=True)->dict:
    """Aynı Exact TSP problemi: CP-SAT veya SciPy MILP + subtour cut.
    status='OPTIMAL' yalnızca optimumluk kanıtlandığında döner.
    """
    start=time.perf_counter();n=len(mat)
    if n<2 or not np.array_equal(mat,mat.T) or not np.array_equal(np.diag(mat),np.zeros(n)):
        raise ValueError('Geçersiz mesafe matrisi')
    if n==2:
        return {'status':'OPTIMAL','order':[0,1,0],
                'distance':float(mat[0,1]+mat[1,0])/scale,
                'elapsed':time.perf_counter()-start,'backend':'analitik'}
    if prefer_cp_sat:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            pass
        else:
            model=cp_model.CpModel();variables={}
            arcs=[]
            for i in range(n):
                for j in range(n):
                    if i!=j:
                        var=model.NewBoolVar(f'a_{i}_{j}')
                        arcs.append((i,j,var));variables[(i,j)]=var
            model.AddCircuit(arcs)
            model.Minimize(sum(int(mat[i,j])*v for (i,j),v in variables.items()))
            solver=cp_model.CpSolver()
            solver.parameters.max_time_in_seconds=float(limit_sec)
            solver.parameters.num_search_workers=1
            solver.parameters.random_seed=42
            r=solver.Solve(model)
            state={cp_model.OPTIMAL:'OPTIMAL',cp_model.FEASIBLE:'FEASIBLE',cp_model.INFEASIBLE:'INFEASIBLE'}.get(r,'TIME_LIMIT')
            order=None;dist=None
            if r in (cp_model.OPTIMAL,cp_model.FEASIBLE):
                succ={i:j for (i,j),v in variables.items() if solver.Value(v)}
                order=ordered_route(succ,n)
                dist=float(sum(mat[a,b] for a,b in zip(order,order[1:])))/scale
            return dict(status=state,order=order,distance=dist,elapsed=time.perf_counter()-start,backend='OR-Tools CP-SAT')
    # SciPy MILP: exactly the same TSP optimality objective; lazily cut subtours.
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j]
    indices={a:k for k,a in enumerate(arcs)}
    z=len(arcs);costs=np.array([mat[i,j] for i,j in arcs],float)
    A=lil_matrix((2*n,z),dtype=float)
    for k,(i,j) in enumerate(arcs):A[i,k]=1;A[n+j,k]=1
    A=A.tocsr();low=np.ones(2*n);high=np.ones(2*n);cuts=set()
    for _ in range(100):
        left=float(limit_sec)-(time.perf_counter()-start)
        if left<=0.01:break
        res=milp(costs,integrality=np.ones(z),bounds=Bounds(0,1),
                 constraints=LinearConstraint(A,low,high),
                 options={'time_limit':max(0.01,left),'mip_rel_gap':0.0})
        if res.status!=0 or res.x is None:
            return dict(status='TIME_LIMIT' if res.status==1 else 'ERROR', order=None,
                        distance=None,elapsed=time.perf_counter()-start,backend='SciPy MILP')
        succ={i:j for k,(i,j) in enumerate(arcs) if res.x[k]>.5}
        parts=cycles(succ,n)
        if len(parts)==1:
            order=ordered_route(succ,n)
            return dict(status='OPTIMAL',order=order,
                        distance=float(sum(mat[a,b] for a,b in zip(order,order[1:])))/scale,
                        elapsed=time.perf_counter()-start,backend='SciPy MILP')
        new_rows=[];new_upper=[]
        for cyc in parts:
            key=frozenset(cyc)
            if key in cuts:continue
            cuts.add(key)
            row=lil_matrix((1,z),dtype=float)
            for i in cyc:
                for j in cyc:
                    if i!=j:row[0,indices[(i,j)]]=1
            new_rows.append(row.tocsr());new_upper.append(len(cyc)-1)
        if not new_rows:break
        A=vstack([A]+new_rows,format='csr')
        low=np.r_[low,np.full(len(new_rows),-np.inf)]
        high=np.r_[high,np.array(new_upper)]
    return dict(status='TIME_LIMIT',order=None,distance=None,
                elapsed=time.perf_counter()-start,backend='SciPy MILP')


def solve_selected(project:dict, locations:pd.DataFrame, limit_sec:float=10.,prefer_cp_sat:bool=True)->tuple[dict,list[str]]:
    g=locations.sort_values('locations').copy()
    spines,aisles,io,scale=geometry_from(project)
    labels=['I/O']+g.locations.tolist()
    points=[io]+[(float(r.x),access_y(r.y,aisles)) for r in g.itertuples()]
    mat=distance_matrix(points,spines,scale)
    result=solve_exact_tsp(mat,scale,limit_sec,prefer_cp_sat)
    route=[labels[i] for i in result['order']] if result['order'] else []
    return result,route


