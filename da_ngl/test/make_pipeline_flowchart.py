import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

W, TOP, BOT = 188.0, 296.0, -99.0
FIGW, FIGH, DPI = 15.5, 42.5, 108
fig = plt.figure(figsize=(FIGW, FIGH), dpi=DPI)
ax = fig.add_axes([0.005, 0.004, 0.99, 0.992])
ax.set_xlim(0, W); ax.set_ylim(BOT, TOP); ax.axis('off')
R = fig.canvas.get_renderer()

CB, CE = '#dbeafe', '#2563eb'
GB, GE = '#dcfce7', '#16a34a'
OB, OE = '#ffedd5', '#ea580c'
YB, YE = '#fef9c3', '#ca8a04'
PB, PE = '#f3f4f6', '#9ca3af'
VB, VE = '#ede9fe', '#7c3aed'
INK = '#1f2937'
WARN = []

def th(t):
    bb = t.get_window_extent(R)
    inv = ax.transData.inverted()
    (_, y0), (_, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return abs(y1 - y0)

def tw(t):
    bb = t.get_window_extent(R)
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return abs(x1 - x0)

def box(x0, x1, y_top, title, body='', fc='#ffffff', ec=INK, ls='-', lw=1.5,
        tfs=13.0, bfs=9.6, pad=2.8, gap=2.2, check=True, name=''):
    cx = (x0 + x1) / 2
    t1 = ax.text(cx, y_top - pad, title, ha='center', va='top', fontsize=tfs,
                 fontweight='bold', color=INK, zorder=3)
    h = th(t1); wmax = tw(t1)
    if body:
        yb = y_top - pad - h - gap
        t2 = ax.text(cx, yb, body, ha='center', va='top', fontsize=bfs,
                     color='#111827', zorder=3, linespacing=1.55)
        h += gap + th(t2); wmax = max(wmax, tw(t2))
    height = pad + h + pad
    bottom = y_top - height
    if check and wmax > (x1 - x0) - 2 * pad:
        WARN.append(f'WIDE {name or title[:20]!r}: text {wmax:.0f} > box {x1-x0-2*pad:.0f}')
    ax.add_patch(FancyBboxPatch((x0, bottom), x1 - x0, height,
                                boxstyle='round,pad=0.4,rounding_size=1.8',
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    return bottom, y_top

def diamond(cx, cy, w, h, lines):
    ax.add_patch(Polygon([(cx, cy - h/2), (cx + w/2, cy), (cx, cy + h/2), (cx - w/2, cy)],
                         closed=True, fc=YB, ec=YE, lw=1.6, zorder=2))
    ax.text(cx, cy, lines, ha='center', va='center', fontsize=11.5,
            fontweight='bold', color=INK, zorder=3, linespacing=1.5)
    return cy - h/2, cy + h/2

def arrow(p, q, color='#374151', ls='-', lw=1.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=17, lw=lw,
                                 color=color, linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))

def elbow(pts, color='#374151', ls='-', lw=1.7):
    for i in range(len(pts) - 2):
        ax.add_line(Line2D([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                           color=color, lw=lw, linestyle=ls, zorder=1, solid_capstyle='round'))
    arrow(pts[-2], pts[-1], color=color, ls=ls, lw=lw)

def lab(x, y, text, fs=11, color='#374151', weight='normal'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=color,
            fontweight=weight, zorder=4, bbox=dict(fc='white', ec='none', pad=1.5, alpha=0.93))

ax.text(W/2, TOP - 1.5, 'da_ngl training pipeline   (GNSS ZTD + FuXi  ->  ERA5 assimilation)',
        ha='center', va='top', fontsize=23, fontweight='bold', color=INK)
ax.text(W/2, TOP - 7.5,
        'blue = data build   |   green = training   |   orange = evaluation & plots   |   yellow = decision   |   grey = optimiser   |   purple = outputs   |   dashed = optional',
        ha='center', va='top', fontsize=11, color='#4b5563')

SX0, SX1 = 6, 168
y = TOP - 12

b, t = box(SX0, SX1, y, '(1) RAW SOURCES   (read-only, under /cpfs01/.../public)',
    'NGL GNSS ZTD 5 min :  ngl_ztd_all_downloaded/.../Western_and_central_Europe\n'
    'FuXi forecasts     :  Fuxi_pred_2017_2024  +  FuXi_Pred_2024_2025  +  FuXi_Pred_2025_other\n'
    '                      z[init, step = 6, 12, ... 180 h, channel(70), lat(720), lon(1440)] ,  6-hourly initialisations\n'
    'ERA5 2017_2025 (global 720x1440)   |   IMERG tp (zarr_25_720_more)   |   mean_era5.npy / std_era5.npy   |   ETOPO2 elevation',
    fc=CB, ec=CE, name='raw')
raw_b = b
y = b - 7

b, t = box(SX0, SX1, y, '(2) UNIFIED GRID + STATION MAP    -    preprocessing/map_stations_to_grid.py',
    'target grid 0.25 deg :  80 x 120 ,  lat 36.50-56.25 ,  lon -5.25-24.50 ;  every source field resampled by nearest neighbour (Region: lon %360 wrap, lat flip)\n'
    'ngl_europe_0p25_80x120_station_grid_map.parquet :  1378 cells with a station (one per cell, seed=2021) ,  8222 cells masked\n'
    'ngl_europe_stations.parquet :  station heights (used by the ZHD term of the ZTD operator)',
    fc=CB, ec=CE, name='grid')
grid_b = b
arrow((SX0 + (SX1 - SX0) / 2, raw_b), (SX0 + (SX1 - SX0) / 2, t))
y = b - 7

row_top = y
x_ranges = [(4, 46), (49, 91), (94, 136), (139, 183)]
stores = [
    ('NGL observation store',
     'ngl_europe_0p25_5min.zarr\n'
     'ztd[394272, 80, 120]\n'
     '5-min native sampling\n'
     'standardised with the\n'
     'train-split global mean/std\n'
     '  (2333.645 / 119.763 mm)\n'
     'mask / station fields\n'
     'ztd_train_mean/std\n'
     'build_ngl_zarr.py', CB, CE, '-'),
    ('FuXi background store',
     'fuxi_europe_0p25.zarr\n'
     '      step=[6]   5476 inits\n'
     'fuxi_europe_0p25_24h.zarr\n'
     '      step=[24]  5484 inits\n'
     'z[init, step, 69, 80, 120]\n'
     'already standardised\n'
     '(ERA5 mean/std)\n'
     'build_fuxi_zarr.py\n'
     '   --leads 24\n'
     '   --init-pad-hours 24', CB, CE, '-'),
    ('ZTD operator store (optional)',
     'ztd_fuxi_europe_0p25_6h.zarr\n'
     'H(FuXi) : station-cell ZTD [mm]\n'
     '(ZHD / ZWD also stored)\n'
     'ztd_operator.py\n'
     '  ZHD = 0.0022768 p_s /(...)\n'
     '    p_s = msl reduced to h_sta\n'
     '  ZWD = 1e-6 R_d/g *\n'
     '    int(k2 e/T + k3 e/T^2) dp\n'
     '  13 levels + a node at p_s\n'
     'validated vs NGL: RMSE 14.3 mm\n'
     'needed for obs_mode in\n'
     '{residual, both}', CB, CE, '--'),
    ('ERA5 label store',
     'label_europe_0p25.zarr\n'
     'label[5476, 71, 80, 120]\n'
     '0-68    ERA5 state\n'
     '69      IMERG tp\n'
     '70      ERA5 tp\n'
     'tp : clip(0) - log1p - z-score\n'
     'training reads the first 70\n'
     'evaluation reads all 71\n'
     'build_label_zarr.py', CB, CE, '-'),
]
heights = []
arts = []
for (x0, x1), (ti, bo, fc, ec, ls) in zip(x_ranges, stores):
    cx = (x0 + x1) / 2
    t1 = ax.text(cx, row_top - 2.8, ti, ha='center', va='top', fontsize=11.5,
                 fontweight='bold', color=INK, zorder=3)
    h1 = th(t1)
    t2 = ax.text(cx, row_top - 2.8 - h1 - 2.0, bo, ha='center', va='top', fontsize=8.8,
                 color='#111827', zorder=3, linespacing=1.5)
    hh = 2.8 + h1 + 2.0 + th(t2) + 2.8
    heights.append(hh)
    if tw(t2) > (x1 - x0) - 6:
        WARN.append(f'WIDE store {ti[:18]!r}: {tw(t2):.0f} > {x1-x0-6:.0f}')
    arts.append((x0, x1, fc, ec, ls))
row_h = max(heights)
for (x0, x1, fc, ec, ls) in arts:
    ax.add_patch(FancyBboxPatch((x0, row_top - row_h), x1 - x0, row_h,
                                boxstyle='round,pad=0.4,rounding_size=1.8',
                                fc=fc, ec=ec, lw=1.5, linestyle=ls, zorder=2))
for x0, x1 in x_ranges:
    arrow(((x0 + x1) / 2, grid_b), ((x0 + x1) / 2, row_top))
row_b = row_top - row_h
y = row_b - 7

b, t = box(SX0, SX1, y, '(3) SAMPLE CONSTRUCTION    -    AssimilationDataset (main_code/main/utils/utils_data.py)',
    'one sample = one analysis time T  (6-hourly, iterating configs.dates_*_range ; samples with a missing background / label / observation are skipped)\n'
    'background  bg    = FuXi[init = T - fcst_step x 6h ,  step = fcst_step x 6h]   ->  (69, 80, 120)     [fcst_step = 1 -> lead 6 h ;  = 4 -> lead 24 h]\n'
    'observations obs  = obs_frames = 73 consecutive NGL 5-min frames , window [T-6h, T] , NaN outside stations   ->  (73, C, 80, 120)\n'
    'label             = ERA5 channels 0-68 + the chosen tp (tp_label_source = era5 | imerg)   ->  (70, 80, 120)\n'
    'train 2022-01-01 -> 2024-05-01 (3403)    |    val -> 2025-01-01 (980)    |    test -> 2025-10-01 (1092)',
    fc=GB, ec=GE, name='sample')
sample_top, sample_b = t, b
arrow((87, row_b), (87, sample_top))
y = b - 6

b, t = box(16, 158, y, '(4) DataLoader',
    'batch 2/rank x world_size GPUs   |   num_workers 8   |   forkserver   |   persistent_workers\n'
    'DistributedSampler(shuffle=True) for train ,  sequential for val / test',
    fc=GB, ec=GE, tfs=12, bfs=9.2, name='loader')
loader_b, loader_t = b, t
arrow((87, sample_b), (87, loader_t))
y = b - 6

b, t = box(SX0, SX1, y, '(5) PER-BATCH PREPROCESSING / TO GPU    -    train_FSDP.py : process_bg / process_obs',
    'process_bg : move to GPU , replicate-pad H/W to a multiple of 16 (120 -> 128) , crop the network output back to 120 ; labels handled the same way\n'
    'process_obs : nan_to_num , per-frame finite mask , [zero_obs=True -> ZTD set to 0] , append mask + lat/lon channels , multiply the block by the mask (0 outside stations)\n'
    'obs channels : absolute = 1 | residual = 1 | both = 2 , plus mask + lat + lon  ->  4 or 5 channels x 73 frames           AMP : fp16 mixed precision',
    fc=GB, ec=GE, name='prep')
prep_b, prep_t = b, t
arrow((87, loader_b), (87, prep_t))
y = b - 6

b, t = box(16, 158, y, '(6) MODEL    AssimilationNetv6',
    'three encoders (background / observations / side info)  ->  FussionStackv2 stage by stage (depth 2,2,2)  ->  EnhanceStack  ->  DecoderBlock\n'
    'residual : out = decoder + bg on the first 69 channels ; tp (ch 69) has no background and is predicted directly  |  embed_dim 128, ~73.7 M params',
    fc=GB, ec=GE, tfs=12, bfs=9.2, name='model')
model_b, model_t = b, t
arrow((87, prep_b), (87, model_t))
y = b - 6

b, t = box(16, 158, y, '(7) LOSS    mae()',
    'latitude-weighted (cos lat, mean-normalised) , NaN-safe (masked where the label is not finite) , over all 70 channels (incl. tp)\n'
    'NOTE : there is no station mask yet - station cells carry 14% of the weighted loss, station-less cells 86%  (change under discussion)',
    fc=GB, ec=GE, tfs=12, bfs=9.2, name='loss')
loss_b, loss_t = b, t
arrow((87, model_b), (87, loss_t))
y = b - 8

decA_b, decA_t = diamond(87, y - 16, 66, 32, 'epoch data exhausted\nor iteration >= 20000 ?')
arrow((87, loss_b), (87, decA_t))

up_t = y - 16 + 12
up_b, _ = box(124, 183, up_t, 'PARAMETER UPDATE',
    'loss.backward()\n'
    'clip_grad_norm(max_norm=32)\n'
    'GradScaler.step -> AdamW\n'
    'lr 1e-4 , wd 0.1\n'
    'warmup 1000 steps (1e-8 -> 1e-4)\n'
    'then CosineAnnealing(T_max=19000)\n'
    'one log line per 100 iters (rank 0)',
    fc=PB, ec=PE, tfs=12, bfs=9.2, name='update')
elbow([(124, up_t - 12), (120, up_t - 12)], color=INK)
lab(122, up_t - 5.5, 'no', fs=10, weight='bold')
elbow([(183, up_t - 12), (186, up_t - 12), (186, prep_t - 10), (169, prep_t - 10)])
lab(179, (up_t + prep_t) / 2, 'next\nbatch', fs=9.5, color='#6b7280')
y = decA_b - 7

b, t = box(SX0, SX1, y, '(8) END OF AN EPOCH : val evaluation    -    evaluate()',
    '70-channel loss , dist.all_reduce across ranks  (the wrap-up evaluation must NOT sit inside an "if rank == 0" block, otherwise all_reduce deadlocks)\n'
    'best criterion : val must beat the historical best by min_delta = 0 ; every rank computes it, only rank 0 writes to disk',
    fc=GB, ec=GE, name='epochend')
epoch_b, epoch_t = b, t
arrow((87, decA_b), (87, epoch_t))
y = b - 9

decB_b, decB_t = diamond(87, y - 15, 66, 30, 'val improved ?')
arrow((87, epoch_b), (87, decB_t))
save_top = y - 3
save_b, _ = box(124, 183, save_top, 'SAVE val-best',
    'FSDP FULL_STATE_DICT\n'
    '(rank 0 writes, all ranks must\n'
    'enter the context together)\n'
    'model / optimizer / scheduler\n'
    '/ iteration / val_loss\n'
    '-> results/{model_id}_{exp_tag}/\n'
    '      {model_id}_{arch_tag}.pth',
    fc=PB, ec=PE, tfs=12, bfs=9.2, name='save')
elbow([(124, y - 15), (120, y - 15)], color=INK)
lab(122, y - 8.5, 'yes', fs=10, weight='bold')
lab(87, decB_b - 3.5, 'no (keep the old best)', fs=10, color='#6b7280')
elbow([(157, save_b), (157, decB_b - 16)])

y = decB_b - 9
decC_b, decC_t = diamond(87, y - 15, 66, 30, 'reached 20000 iters ?')
arrow((87, decB_b), (87, decC_t))
elbow([(54, y - 15), (2, y - 15), (2, prep_t - 22), (5.4, prep_t - 22)])
lab(31, y - 10, 'no  ->  next epoch', fs=10, color='#6b7280')
y = decC_b - 7
arrow((87, decC_b), (87, y))
lab(97, decC_b - 3.5, 'yes', fs=12, weight='bold')

b, t = box(SX0, SX1, y, '(9) TRAINING WRAP-UP    (still train_FSDP.py)',
    'reload the val-best checkpoint  ->  evaluate background-only loss (train / val, 69 channels)  ->  test evaluation (model 70 ch vs background 69 ch)\n'
    'write results/{model_id}_{exp_tag}/summary.json  (best val, per-epoch loss, lr, sample counts, test metrics)',
    fc=GB, ec=GE, name='wrap')
y = b - 7

b, t = box(SX0, SX1, y, '(10) EVALUATION AND PLOTS    -    plot_results.py  (chained automatically after training by train.sh, single GPU)',
    'rebuild the dataset from the same config + tags  ->  load the val-best checkpoint  ->  per-sample forward, collecting analysis / background / truth (71 ch)\n'
    'metrics : 69-ch and 70-ch MAE, climatology, per-channel metrics.csv, improve_pct, number of channels better than the background, mean |analysis - bg|\n'
    'figures : loss_curve.png  |  channel_metrics.png  |  maps_{z500,t850,r700,t2m,msl}.png  |  timeseries(_tp).png  |  tp_maps.png',
    fc=OB, ec=OE, name='plot')
y = b - 7

b, t = box(SX0, SX1, y, '(11) OUTPUTS    -    work_dir/results/{model_id}_{exp_tag}/',
    '{model_id}_{arch_tag}.pth  |  summary.json  |  metrics.csv / metrics.json  |  train_loss.npy  |  val_loss.npy  |  lr.npy  |  all *.png\n'
    'log file : da_ngl/logs/{model_id}_{arch_tag}.log   (one progress line per 100 iterations ; the stdout handler is WARNING-only)',
    fc=VB, ec=VE, tfs=12, bfs=9.2, name='out')

ax.text(SX0, b - 6, 'tags :  exp_tag = lead{fcst_step x 6}h_obs{obs_frames x 5 / 60}h_{tp_label_source}tp   (+_zeroobs / +_{obs_mode})        arch_tag = ed{embed_dim}_d{depth}',
        ha='left', va='top', fontsize=11, color='#374151')

fig.savefig('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/test/pipeline_flowchart.png', facecolor='white')
print('WARN:', WARN if WARN else 'none')
print('bottom of last box:', round(b, 1), ' canvas BOT:', BOT)
