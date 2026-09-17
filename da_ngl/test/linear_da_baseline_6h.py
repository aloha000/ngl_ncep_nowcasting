# -*- coding: utf-8 -*-
"""线性 Kalman 增益基线，lead 6 h 版本（HANDOFF 11.4 的 6h 对照）。

和 ``linear_da_baseline.py``（lead 24 h）逐行一致，只有三处不同：

* 背景/算子 store 换成 6 h 的那两个（``fuxi_europe_0p25.zarr``、
  ``ztd_fuxi_europe_0p25_6h.zarr``）；
* 6 h store 没有 ``--init-pad-hours``，它的 ``init`` 轴与 label 轴同起点，
  所以 label 下标 ``i`` 对应的背景是 ``i-1``（``init = T - 6h``）；
* innovation 用逐站去静态偏差后的 ``(obs - b_s) - H(bg)``，与当前 CNN 跑法一致。

    x_a = x_bg + K·d,   d = (obs - b_s - H(bg)) / sigma_ztd
"""
import sys, os, numpy as np, pandas as pd, zarr, multiprocessing as mp
from pathlib import Path

BASE = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/dataset'
PRE = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/preprocessing'
C = 69
sys.path.insert(0, PRE)
from common import decode_time_axis

_G = {}
def init_globals():
    global BG, LB, HBG, OBS, IY, IX, W1, ZM, ZS, FR, BS
    BG = zarr.open(f'{BASE}/fuxi_europe_0p25.zarr', 'r')['z']          # lead 6 h
    LB = zarr.open(f'{BASE}/label_europe_0p25.zarr', 'r')['label']
    HBG = zarr.open(f'{BASE}/ztd_fuxi_europe_0p25_6h.zarr', 'r')['ztd_fuxi']
    ngl = zarr.open(f'{BASE}/ngl_europe_0p25_5min.zarr', 'r'); OBS = ngl['ztd']
    ZM, ZS = float(ngl['ztd_train_mean'][0]), float(ngl['ztd_train_std'][0])
    gm = pd.read_parquet(f'{BASE}/ngl_europe_0p25_80x120_station_grid_map.parquet')
    lat = np.round(np.arange(36.50, 56.25 + 1e-9, 0.25), 6)
    lon = np.round(np.arange(-5.25, 24.50 + 1e-9, 0.25), 6)
    iy = np.searchsorted(lat, gm['lat'].values); ix = np.searchsorted(lon, gm['lon'].values)
    st = ~gm['mask'].astype(bool).values
    IY, IX = iy[st], ix[st]
    w = np.cos(np.deg2rad(lat)); w /= w.mean(); W1 = w[IY]
    lab_t = decode_time_axis(Path(f'{BASE}/label_europe_0p25.zarr'), 'time')
    obs_t = decode_time_axis(Path(f'{BASE}/ngl_europe_0p25_5min.zarr'), 'time')
    FR = np.searchsorted(obs_t.values, lab_t.values)
    BS = np.load(f'{BASE}/obs_debias_lead6h.npz')['bias_mm'].astype('float32')

def _pair(i):
    bg = np.asarray(BG[i - 1, 0, :C], 'float32')[:, IY, IX]      # init = T - 6h
    lb = np.asarray(LB[i, :C], 'float32')[:, IY, IX]
    hb = np.asarray(HBG[i], 'float32')[IY, IX]
    ob = np.asarray(OBS[FR[i]], 'float32')[IY, IX] * ZS + ZM
    m = np.isfinite(ob) & np.isfinite(hb)
    d = np.where(m, ((ob - BS) - hb) / ZS, 0.0)                  # de-biased innovation
    e = (lb - bg).astype('float64')
    return e, d, m

def task_fit(idx):
    sd2 = np.zeros(IX.size); sde = np.zeros((C, IX.size)); sae = np.zeros((C, IX.size)); n = 0
    for i in idx:
        if i < 1:
            continue
        e, d, m = _pair(i)
        if not m.any():
            continue
        em, dd = e[:, m], d[m]
        sd2[m] += dd * dd; sde[:, m] += em * dd[None, :]; sae[:, m] += np.abs(em)
        n += 1
    return sd2, sde, sae, n

def task_eval(args):
    idx, KV = args
    base = np.zeros(C); acc = {k: np.zeros(C) for k in KV}; n = 0
    sw = 0.0
    for i in idx:
        if i < 1:
            continue
        e, d, m = _pair(i)
        if not m.any():
            continue
        em, dd = e[:, m], d[m]
        wm = W1[m]
        base += (np.abs(em) * wm[None, :]).sum(axis=1)
        for k, K in KV.items():
            if K.ndim == 1:
                acc[k] += (np.abs(em - K[:, None] * dd[None, :]) * wm[None, :]).sum(axis=1)
            else:
                acc[k] += (np.abs(em - K[:, m] * dd[None, :]) * wm[None, :]).sum(axis=1)
        sw += float(wm.sum())
        n += 1
    return base, acc, sw

def wmae(x, wsum):
    return x / max(float(wsum), 1e-9)

def main():
    init_globals()
    lab_t = decode_time_axis(Path(f'{BASE}/label_europe_0p25.zarr'), 'time')
    sp = {'train': ('2022-01-01', '2024-05-01'), 'test': ('2025-01-01', '2025-10-01')}
    idx = {k: np.where((lab_t >= a) & (lab_t < b))[0] for k, (a, b) in sp.items()}
    with mp.get_context('fork').Pool(32, initializer=init_globals) as pool:
        blocks = [idx['train'][i:i + 100] for i in range(0, len(idx['train']), 100)]
        res = pool.map(task_fit, blocks)
        sd2 = sum(r[0] for r in res); sde = sum(r[1] for r in res)
        sae = sum(r[2] for r in res); n_tr = sum(r[3] for r in res)

        Kpool = sde.sum(axis=1) / max(sd2.sum(), 1e-12)
        Kstn = sde / np.maximum(sd2[None, :], 1e-12)
        KV = {'pooled': Kpool, 'station': Kstn}
        for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
            KV[f'station_l{lam:g}'] = Kstn / (1.0 + lam)
        print(f'[fit] train times {n_tr}, station cells {sd2.size}')
        print(f'[fit] |K| 均值: pooled {np.abs(Kpool).mean():.4f} | per-station {np.abs(Kstn).mean():.4f}')

        out = {}
        for split in ('train', 'test'):
            blocks = [idx[split][i:i + 100] for i in range(0, len(idx[split]), 100)]
            res = pool.map(task_eval, [(b, KV) for b in blocks])
            base = sum(r[0] for r in res); acc = {k: sum(r[1][k] for r in res) for k in KV}
            sw = sum(r[2] for r in res)
            out[split] = (wmae(base, sw), {k: wmae(v, sw) for k, v in acc.items()}, int(sw))

    print(f'\n{"方法":22s} {"训练集 MAE":>11s} {"相对背景":>9s} | {"测试集 MAE":>11s} {"相对背景":>9s}')
    tr, te = out['train'], out['test']
    def line(nm, a_tr, a_te):
        print(f'{nm:22s} {a_tr:11.5f} {100*(tr[0].mean()-a_tr)/tr[0].mean():+8.3f}% | '
              f'{a_te:11.5f} {100*(te[0].mean()-a_te)/te[0].mean():+8.3f}%')
    line('背景（恒等）', tr[0].mean(), te[0].mean())
    for k, v in te[1].items():
        line(f'线性 DA: {k}', tr[1][k].mean(), v.mean())

    ch = ['z50','z100','z150','z200','z250','z300','z400','z500','z600','z700','z850','z925','z1000',
          't50','t100','t150','t200','t250','t300','t400','t500','t600','t700','t850','t925','t1000',
          'u50','u100','u150','u200','u250','u300','u400','u500','u600','u700','u850','u925','u1000',
          'v50','v100','v150','v200','v250','v300','v400','v500','v600','v700','v850','v925','v1000',
          'r50','r100','r150','r200','r250','r300','r400','r500','r600','r700','r850','r925','r1000',
          't2m','u10','v10','msl']
    best = 'station_l1'
    imp = 100*(te[0]-te[1][best])/te[0]
    order = np.argsort(-imp)
    print(f'\n逐通道改善 %（测试集，{best}）— 前 8 / 后 5：')
    for i in list(order[:8]) + list(order[-5:]):
        print(f'   {ch[i]:6s} {imp[i]:+6.3f}%   (背景 {te[0][i]:.4f} -> 分析 {te[1][best][i]:.4f})')

if __name__ == '__main__':
    main()
