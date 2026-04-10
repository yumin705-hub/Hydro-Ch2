"""
visualizer.py ─ HydroStatic AI v3.1
한글 폰트 완전 해결 + 3D/2D 시각화
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa
import io, os, glob
from typing import Dict

# ══════════════════════════════════════════════════════════
# 한글 폰트 — 동적 탐색 (□ 제거 핵심)
# ══════════════════════════════════════════════════════════
_KR_PROP = None
_KR_FONT = 'DejaVu Sans'

def _find_cjk_font():
    global _KR_PROP, _KR_FONT
    # 1) 직접 경로 탐색
    candidates = glob.glob('/usr/share/fonts/**/*CJK*', recursive=True) \
               + glob.glob('/usr/share/fonts/**/*Noto*KR*', recursive=True) \
               + glob.glob('/usr/share/fonts/**/*nanum*', recursive=True) \
               + glob.glob('/usr/share/fonts/**/*Baekmuk*', recursive=True) \
               + glob.glob('/usr/share/fonts/**/*UnBatang*', recursive=True) \
               + glob.glob('/usr/share/fonts/**/*Malgun*', recursive=True)
    for path in candidates:
        if os.path.isfile(path):
            try:
                fm.fontManager.addfont(path)
                _KR_PROP = fm.FontProperties(fname=path)
                _KR_FONT = _KR_PROP.get_name()
                return True
            except Exception:
                continue
    # 2) fontManager 내부 검색
    for f in fm.fontManager.ttflist:
        nl = f.name.lower()
        if any(k in nl for k in ['cjk', 'noto sans kr', 'nanumgothic', 'malgun',
                                   'unbatang', 'baekmuk', 'source han']):
            try:
                _KR_PROP = fm.FontProperties(fname=f.fname)
                _KR_FONT = f.name
                return True
            except Exception:
                continue
    return False

_find_cjk_font()

def _kfont(size=9):
    if _KR_PROP:
        return fm.FontProperties(fname=_KR_PROP.get_file(), size=size)
    return fm.FontProperties(size=size)

def _has_kr():
    return _KR_PROP is not None

_SHAPE_EN = {
    '직사각형': 'Rectangle', '삼각형': 'Triangle', '사다리꼴': 'Trapezoid',
    '원형': 'Circle', '반원형': 'Semicircle', '포물선 곡면': 'Parabola',
}

def _safe_kr(kr, en=None):
    return kr if _has_kr() else (en or kr)

BG     = '#040d14'
PANEL  = '#071523'
CYAN   = '#00c8ff'
TEAL   = '#00ff9f'
ORANGE = '#ff6b35'
GOLD   = '#ffd700'
TEXT   = '#c0d8e8'
GAMMA  = 9810

def _setup_rcparams():
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': PANEL,
        'text.color': TEXT, 'axes.labelcolor': TEXT,
        'xtick.color': TEXT, 'ytick.color': TEXT,
        'axes.titlecolor': CYAN, 'grid.color': '#122a3a',
        'grid.linewidth': 0.4, 'font.size': 8,
    })
    if _has_kr():
        plt.rcParams['font.family'] = _KR_FONT
        plt.rcParams['axes.unicode_minus'] = False

_setup_rcparams()

# ═══════════════════════════════════════════════════════
# 공개 API
# ═══════════════════════════════════════════════════════

def plot_3d_hydrostatic(shape: str, params: Dict, result: Dict) -> bytes:
    _setup_rcparams()
    fig = plt.figure(figsize=(15, 6.5), facecolor=BG)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    _style_ax(ax1, '3D Pressure Distribution')
    _style_ax(ax2, 'Resultant Force & Center of Pressure')

    try: _draw_pressure(ax1, shape, params, result)
    except Exception as e:
        ax1.text2D(0.5,0.5,f'Error: {e}',transform=ax1.transAxes,ha='center',color='red',fontsize=8)
    try: _draw_force_vector(ax2, shape, params, result)
    except Exception as e:
        ax2.text2D(0.5,0.5,f'Error: {e}',transform=ax2.transAxes,ha='center',color='red',fontsize=8)

    shape_kr = result.get('shape', shape)
    lbl = _safe_kr(shape_kr, _SHAPE_EN.get(shape_kr, shape))
    title = f"{lbl}  |  F = {result['F']/1000:.2f} kN  |  y_cp = {result['y_cp']:.3f} m"
    if _has_kr():
        fig.text(0.5,0.97,title,ha='center',color=CYAN,fontsize=11,fontweight='bold',fontproperties=_kfont(11))
    else:
        fig.text(0.5,0.97,title,ha='center',color=CYAN,fontsize=11,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.96])
    buf = io.BytesIO()
    fig.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor=BG,edgecolor='none')
    buf.seek(0); data = buf.read(); plt.close(fig)
    return data

def plot_2d_summary(shape: str, params: Dict, result: Dict) -> bytes:
    _setup_rcparams()
    fig, ax = plt.subplots(figsize=(6,6.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    _draw_2d_section(ax, shape, params, result)
    buf = io.BytesIO()
    fig.savefig(buf,format='png',dpi=110,bbox_inches='tight',facecolor=BG,edgecolor='none')
    buf.seek(0); data = buf.read(); plt.close(fig)
    return data

# ═══════════════════════════════════════════════════════
# 형상 헬퍼
# ═══════════════════════════════════════════════════════

def _dims(shape, params):
    if shape=='rectangle':    return max(params['width'],params['height']),params['height'],params['width']/2
    elif shape=='triangle':   return max(params['width'],params['height']),params['height'],params['width']/2
    elif shape=='trapezoid':  return max(params['b_bottom'],params['height']),params['height'],params['b_bottom']/2
    elif shape=='circle':     R=params['radius']; return R*2,R*2,R
    elif shape=='semicircle': R=params['radius']; return R*2,R,R
    elif shape=='parabola':   return max(params['width']*2,params['height']),params['height'],params['width']
    return 2.0,2.0,1.0

def _gate_mesh(shape, params, nx=40, nz=35):
    if shape=='rectangle':
        b,H=params['width'],params['height']
        X,Z=np.meshgrid(np.linspace(-b/2,b/2,nx),np.linspace(0,H,nz))
        return X,Z,np.ones_like(X,bool)
    elif shape=='triangle':
        b,H=params['width'],params['height']
        X,Z=np.meshgrid(np.linspace(-b/2,b/2,nx),np.linspace(0,H,nz))
        return X,Z,np.abs(X)<=b/2*(1-Z/H)
    elif shape=='trapezoid':
        bt,bb,H=params['b_top'],params['b_bottom'],params['height']
        X,Z=np.meshgrid(np.linspace(-bb/2,bb/2,nx),np.linspace(0,H,nz))
        return X,Z,np.abs(X)<=(bb/2+(bt/2-bb/2)*(Z/H))
    elif shape=='circle':
        R=params['radius']
        X,Z=np.meshgrid(np.linspace(-R,R,nx),np.linspace(0,2*R,nz))
        return X,Z,(X**2+(Z-R)**2)<=R**2
    elif shape=='semicircle':
        R=params['radius']
        X,Z=np.meshgrid(np.linspace(-R,R,nx),np.linspace(0,R,nz))
        return X,Z,(X**2+Z**2)<=R**2
    elif shape=='parabola':
        w,H=params['width'],params['height']; a=H/(w**2)
        X,Z=np.meshgrid(np.linspace(-w,w,nx),np.linspace(0,H,nz))
        return X,Z,Z>=(a*X**2)
    X,Z=np.meshgrid(np.linspace(-1,1,nx),np.linspace(0,2,nz))
    return X,Z,np.ones_like(X,bool)

def _outline(shape, params, n=80):
    if shape=='rectangle':
        b,H=params['width'],params['height']
        return np.array([[-b/2,0],[b/2,0],[b/2,H],[-b/2,H],[-b/2,0]])
    elif shape=='triangle':
        b,H=params['width'],params['height']
        return np.array([[-b/2,0],[b/2,0],[0,H],[-b/2,0]])
    elif shape=='trapezoid':
        bt,bb,H=params['b_top'],params['b_bottom'],params['height']
        return np.array([[-bb/2,0],[bb/2,0],[bt/2,H],[-bt/2,H],[-bb/2,0]])
    elif shape=='circle':
        R=params['radius']; t=np.linspace(0,2*np.pi,n)
        return np.column_stack([R*np.cos(t),R*np.sin(t)+R])
    elif shape=='semicircle':
        R=params['radius']; t=np.linspace(0,np.pi,n)
        return np.vstack([np.column_stack([R*np.cos(t),R*np.sin(t)]),[[-R,0],[R,0]]])
    elif shape=='parabola':
        w,H=params['width'],params['height']; a=H/(w**2)
        x=np.linspace(-w,w,n)
        return np.vstack([np.column_stack([x,a*x**2]),[[w,H],[-w,H],[-w,0]]])
    return np.zeros((2,2))

def _x_half(shape, params, z):
    if shape=='rectangle':   return params['width']/2
    elif shape=='triangle':
        b,H=params['width'],params['height']; return max(0.0,b/2*(1-z/H))
    elif shape=='trapezoid':
        bt,bb,H=params['b_top'],params['b_bottom'],params['height']
        return bb/2+(bt/2-bb/2)*(z/H)
    elif shape=='circle':
        R=params['radius']; return max(0.0,float(np.sqrt(max(0,R**2-(z-R)**2))))
    elif shape=='semicircle':
        R=params['radius']; return max(0.0,float(np.sqrt(max(0,R**2-z**2))))
    elif shape=='parabola':
        w,H=params['width'],params['height']; a=H/(w**2)
        return float(np.sqrt(max(0,z/a))) if a>0 else w
    return 0.0

# ═══════════════════════════════════════════════════════
# 3D 압력 분포
# ═══════════════════════════════════════════════════════

def _draw_pressure(ax, shape, params, result):
    hw=params.get('h_w',1.0)
    X,Z,mask=_gate_mesh(shape,params)
    ref,H_total,hw_max=_dims(shape,params)
    P_raw=GAMMA*(hw+Z)/1000.0
    P=np.where(mask,P_raw,np.nan)
    valid=P[~np.isnan(P)]
    if len(valid)==0: return
    P_min,P_max=float(np.nanmin(P)),float(np.nanmax(P))
    if P_max<=P_min: P_max=P_min+1.0
    norm=mcolors.Normalize(vmin=P_min,vmax=P_max)
    cmap=plt.get_cmap('plasma')
    Y_scale=ref*0.55
    Y_press=np.where(mask,(P-P_min)/(P_max-P_min)*Y_scale,0.0)
    ax.plot_surface(X,np.zeros_like(X),Z,alpha=0.10,color=CYAN,linewidth=0,shade=False)
    ax.plot_surface(X,Y_press,Z,facecolors=cmap(norm(np.where(mask,P,P_min))),
                    alpha=0.85,linewidth=0,antialiased=True,shade=True)
    ol=_outline(shape,params)
    ax.plot(ol[:,0],np.zeros(len(ol)),ol[:,1],color=CYAN,lw=2.0,alpha=0.95)
    for z_lv in np.linspace(0,H_total,6):
        xh=_x_half(shape,params,z_lv)
        if xh>0:
            p_n=(GAMMA*(hw+z_lv)/1000-P_min)/(P_max-P_min)*Y_scale
            ax.plot([-xh,xh],[p_n,p_n],[z_lv,z_lv],color='white',lw=0.5,alpha=0.28,linestyle='--')
    sm=plt.cm.ScalarMappable(cmap='plasma',norm=norm); sm.set_array([])
    cb=plt.colorbar(sm,ax=ax,shrink=0.48,pad=0.06,aspect=22)
    cb.set_label('Pressure [kPa]',color=TEXT,fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT,labelsize=6.5)
    plt.setp(plt.getp(cb.ax.axes,'yticklabels'),color=TEXT)
    xr=hw_max*0.9
    ax.plot([-xr,xr],[0,0],[H_total,H_total],color='#3399ff',lw=1.8,linestyle='-.',alpha=0.65)
    ax.text(xr,0,H_total,' WL',color='#3399ff',fontsize=7)
    ax.set_xlim(-hw_max*1.2,hw_max*1.2)
    ax.set_ylim(-0.02,Y_scale*1.15)
    ax.set_zlim(-0.05,H_total*1.15)
    ax.view_init(elev=24,azim=-50)
    ax.set_xlabel('X [m]',labelpad=4)
    ax.set_ylabel('Pressure',labelpad=4)
    ax.set_zlabel('Z [m]',labelpad=4)

# ═══════════════════════════════════════════════════════
# 합력 벡터
# ═══════════════════════════════════════════════════════

def _draw_force_vector(ax, shape, params, result):
    hw=params.get('h_w',1.0)
    Fh=result.get('F_horizontal',result.get('F',0.0))
    Fv=result.get('F_vertical',0.0)
    F_tot=max(result.get('F',1.0),1.0)
    y_cp=result.get('y_cp',hw)
    h_c=result.get('h_c',hw)
    X,Z,mask=_gate_mesh(shape,params)
    ref,H_total,hw_max=_dims(shape,params)
    P_raw=GAMMA*(hw+Z)/1000.0
    P=np.where(mask,P_raw,np.nan)
    P_min=float(np.nanmin(P)) if not np.all(np.isnan(P)) else 0
    P_max=float(np.nanmax(P)) if not np.all(np.isnan(P)) else 1
    if P_max<=P_min: P_max=P_min+1
    norm=mcolors.Normalize(vmin=P_min,vmax=P_max)
    ax.plot_surface(X,np.zeros_like(X),Z,
                    facecolors=plt.get_cmap('plasma')(norm(np.where(mask,P,P_min))),
                    alpha=0.38,linewidth=0,antialiased=True,shade=False)
    ol=_outline(shape,params)
    ax.plot(ol[:,0],np.zeros(len(ol)),ol[:,1],color=CYAN,lw=2.2,alpha=1.0)
    z_cp=float(np.clip(y_cp-hw,0.0,H_total))
    z_gc=float(np.clip(h_c-hw,0.0,H_total))
    arr=ref*0.52
    Fh_n,Fv_n=Fh/F_tot*arr,Fv/F_tot*arr
    ax.quiver(0,0,z_cp,0,Fh_n,Fv_n,color=ORANGE,linewidth=3.0,arrow_length_ratio=0.22,
              label=f'F = {F_tot/1000:.2f} kN')
    if abs(Fh)>1.0:
        ax.quiver(0,0,z_cp,0,Fh_n,0,color=GOLD,linewidth=2.2,arrow_length_ratio=0.22,
                  label=f'Fh = {Fh/1000:.2f} kN')
    if abs(Fv)>1.0:
        ax.quiver(0,0,z_cp,0,0,Fv_n,color=TEAL,linewidth=2.2,arrow_length_ratio=0.22,
                  label=f'Fv = {Fv/1000:.2f} kN')
    ax.scatter([0],[0],[z_cp],color=ORANGE,s=90,zorder=20,depthshade=False)
    ax.plot([-hw_max*0.7,hw_max*0.7],[0,0],[z_cp,z_cp],color=ORANGE,lw=1.3,linestyle='--',alpha=0.6)
    ax.text(hw_max*0.72,0,z_cp,f' y_cp\n {y_cp:.2f}m',color=ORANGE,fontsize=7.5,va='center')
    ax.scatter([0],[0],[z_gc],color=TEAL,s=55,marker='D',zorder=20,depthshade=False)
    ax.plot([-hw_max*0.7,hw_max*0.7],[0,0],[z_gc,z_gc],color=TEAL,lw=1.0,linestyle=':',alpha=0.5)
    ax.text(hw_max*0.72,0,z_gc,f' h_c\n {h_c:.2f}m',color=TEAL,fontsize=7.5,va='center')
    xr=hw_max*0.9
    ax.plot([-xr,xr],[0,0],[H_total,H_total],color='#3399ff',lw=1.8,linestyle='-.',alpha=0.65)
    ax.legend(loc='upper left',fontsize=8,facecolor='#071523',edgecolor=CYAN,labelcolor=TEXT,framealpha=0.85)
    ax.set_xlim(-hw_max*1.2,hw_max*1.2)
    ax.set_ylim(-0.02,arr*1.25)
    ax.set_zlim(-0.05,H_total*1.15)
    ax.view_init(elev=22,azim=-45)
    ax.set_xlabel('X [m]',labelpad=4)
    ax.set_ylabel('Force Scale',labelpad=4)
    ax.set_zlabel('Z [m]',labelpad=4)

def _style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for pane in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
        pane.fill=False; pane.set_edgecolor('#122a3a')
    ax.grid(True,linestyle='--',alpha=0.20,color='#122a3a')
    ax.set_title(title,color=CYAN,pad=8,fontsize=9.5)
    ax.tick_params(colors=TEXT,labelsize=6.5)

# ═══════════════════════════════════════════════════════
# 2D 단면도
# ═══════════════════════════════════════════════════════

def _draw_2d_section(ax, shape, params, result):
    hw=params.get('h_w',1.0)
    y_cp=result.get('y_cp',hw)
    h_c=result.get('h_c',hw)
    shape_kr=result.get('shape',shape)
    shape_en=_SHAPE_EN.get(shape_kr,shape)
    _,H_total,hw_max=_dims(shape,params)
    from matplotlib.patches import Polygon as MplPoly
    ol=_outline(shape,params)
    try:
        ax.add_patch(MplPoly(ol,closed=True,facecolor='#003d66',edgecolor=CYAN,linewidth=2.0,alpha=0.45,zorder=2))
    except Exception: pass
    p_max=GAMMA*(hw+H_total)/1000.0
    arrow_max=hw_max*0.55
    for z_lv in np.linspace(H_total*0.04,H_total*0.96,10):
        p=GAMMA*(hw+z_lv)/1000.0
        p_norm=p/p_max*arrow_max if p_max>0 else 0
        xh=_x_half(shape,params,z_lv)
        if xh<=0: continue
        c=plt.get_cmap('plasma')((z_lv/H_total)*0.85+0.1)
        ax.annotate('',xy=(xh+p_norm,z_lv),xytext=(xh,z_lv),arrowprops=dict(arrowstyle='->',color=c,lw=1.3),zorder=3)
        ax.annotate('',xy=(-xh-p_norm,z_lv),xytext=(-xh,z_lv),arrowprops=dict(arrowstyle='->',color=c,lw=1.3),zorder=3)
    z_line=np.linspace(0,H_total,60)
    p_sc=np.array([GAMMA*(hw+z)/1000 for z in z_line])
    p_sc=p_sc/p_max*arrow_max if p_max>0 else p_sc
    xr_l=np.array([_x_half(shape,params,z) for z in z_line])
    ax.plot(xr_l+p_sc,z_line,color=ORANGE,lw=1.4,alpha=0.6,label='Pressure envelope')
    ax.plot(-xr_l-p_sc,z_line,color=ORANGE,lw=1.4,alpha=0.6)
    z_cp=float(np.clip(y_cp-hw,0.0,H_total))
    ax.axhline(z_cp,color=ORANGE,lw=1.8,linestyle='--',alpha=0.88,zorder=5,label=f'y_cp = {y_cp:.3f} m')
    ax.scatter([0],[z_cp],color=ORANGE,s=70,zorder=10)
    z_gc=float(np.clip(h_c-hw,0.0,H_total))
    ax.axhline(z_gc,color=TEAL,lw=1.5,linestyle=':',alpha=0.80,zorder=5,label=f'h_c = {h_c:.3f} m')
    ax.scatter([0],[z_gc],color=TEAL,s=55,marker='D',zorder=10)
    ax.axhline(-hw,color='#4499ff',lw=2.0,linestyle='-',alpha=0.65,label=f'Water Level (h_w = {hw} m)')
    ax.fill_betweenx([-hw,0.0],-hw_max*0.8,hw_max*0.8,color='#002244',alpha=0.25)
    ax.set_xlim(-hw_max*1.55,hw_max*1.55)
    ax.set_ylim(-hw*1.25 if hw>0 else -0.5,H_total*1.25)
    ax.set_xlabel('Width [m]',color=TEXT,fontsize=9)
    ax.set_ylabel('Height Z [m]',color=TEXT,fontsize=9)
    lbl=_safe_kr(shape_kr,shape_en)
    if _has_kr():
        ax.set_title(f'{lbl} — 2D Pressure Distribution',color=CYAN,fontsize=10,fontproperties=_kfont(10))
    else:
        ax.set_title(f'{lbl} — 2D Pressure Distribution',color=CYAN,fontsize=10)
    ax.tick_params(colors=TEXT,labelsize=8)
    ax.grid(True,alpha=0.18,color='#122a3a',linestyle='--')
    ax.spines[:].set_color('#1a3a55')
    ax.legend(loc='lower right',fontsize=8,facecolor='#071523',edgecolor=CYAN,labelcolor=TEXT,framealpha=0.85)
