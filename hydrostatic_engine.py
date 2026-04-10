"""
hydrostatic_engine.py  v3.1
정수력 계산 엔진 - SymPy 기호 계산 + SciPy 수치 적분
6종 형상 + 전도 모멘트 계산
"""

import sympy as sp
import numpy as np
from scipy import integrate
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

GAMMA_WATER = 9810  # N/m³

# ─── SymPy 기호 정의 ─────────────────────────────────────
y, h, b, B, w, H, R, r_sym, a_sym = sp.symbols('y h b B w H R r a', positive=True)
gamma, h_c, A_sym, I_G = sp.symbols('gamma h_c A I_G', positive=True)
h_w = sp.Symbol('h_w', positive=True)


def symbolic_hydrostatic(shape: str, params: Dict[str, float]) -> Dict[str, Any]:
    result = {}

    if shape == 'rectangle':
        b_v, H_v, hw_v = params['width'], params['height'], params['h_w']
        A_expr   = b * H
        hc_expr  = h_w + H / 2
        F_expr   = gamma * hc_expr * A_expr
        IG_expr  = b * H**3 / 12
        ycp_expr = hc_expr + IG_expr / (hc_expr * A_expr)
        subs = {b: b_v, H: H_v, h_w: hw_v, gamma: GAMMA_WATER}
        A_val   = float(A_expr.subs(subs))
        hc_val  = float(hc_expr.subs(subs))
        F_val   = float(F_expr.subs(subs))
        ycp_val = float(ycp_expr.subs(subs))
        IG_val  = float(IG_expr.subs(subs))
        M_over  = F_val * ycp_val  # 전도 모멘트 (수면 기준)
        result = {
            'shape': '직사각형', 'shape_en': shape,
            'A_expr': sp.latex(A_expr), 'hc_expr': sp.latex(hc_expr),
            'F_expr': sp.latex(F_expr), 'ycp_expr': sp.latex(ycp_expr),
            'A': A_val, 'h_c': hc_val, 'F': F_val, 'y_cp': ycp_val,
            'params': params, 'IG': IG_val,
            'F_horizontal': F_val, 'F_vertical': 0.0,
            'M_overturn': M_over,
        }

    elif shape == 'triangle':
        b_v, H_v, hw_v = params['width'], params['height'], params['h_w']
        A_expr  = sp.Rational(1, 2) * b * H
        hc_expr = h_w + H / 3
        F_expr  = gamma * hc_expr * A_expr
        IG_expr = b * H**3 / 36
        ycp_expr = hc_expr + IG_expr / (hc_expr * A_expr)
        subs = {b: b_v, H: H_v, h_w: hw_v, gamma: GAMMA_WATER}
        A_val   = float(A_expr.subs(subs))
        hc_val  = float(hc_expr.subs(subs))
        F_val   = float(F_expr.subs(subs))
        ycp_val = float(ycp_expr.subs(subs))
        IG_val  = float(IG_expr.subs(subs))
        M_over  = F_val * ycp_val
        result = {
            'shape': '삼각형', 'shape_en': shape,
            'A_expr': sp.latex(A_expr), 'hc_expr': sp.latex(hc_expr),
            'F_expr': sp.latex(F_expr), 'ycp_expr': sp.latex(ycp_expr),
            'A': A_val, 'h_c': hc_val, 'F': F_val, 'y_cp': ycp_val,
            'params': params, 'IG': IG_val,
            'F_horizontal': F_val, 'F_vertical': 0.0,
            'M_overturn': M_over,
        }

    elif shape == 'trapezoid':
        b_v, B_v, H_v, hw_v = params['b_top'], params['b_bottom'], params['height'], params['h_w']
        A_expr  = (b + B) * H / 2
        ybar    = H / 3 * (2*B + b) / (B + b)
        hc_expr = h_w + ybar
        F_expr  = gamma * hc_expr * A_expr
        IG_expr = H**3 / 36 * (b**2 + 4*b*B + B**2) / (b + B)
        ycp_expr = hc_expr + IG_expr / (hc_expr * A_expr)
        subs = {b: b_v, B: B_v, H: H_v, h_w: hw_v, gamma: GAMMA_WATER}
        A_val   = float(A_expr.subs(subs))
        hc_val  = float(hc_expr.subs(subs))
        F_val   = float(F_expr.subs(subs))
        ycp_val = float(ycp_expr.subs(subs))
        IG_val  = float(IG_expr.subs(subs))
        M_over  = F_val * ycp_val
        result = {
            'shape': '사다리꼴', 'shape_en': shape,
            'A_expr': sp.latex(A_expr), 'hc_expr': sp.latex(hc_expr),
            'F_expr': sp.latex(F_expr), 'ycp_expr': sp.latex(ycp_expr),
            'A': A_val, 'h_c': hc_val, 'F': F_val, 'y_cp': ycp_val,
            'params': params, 'IG': IG_val,
            'F_horizontal': F_val, 'F_vertical': 0.0,
            'M_overturn': M_over,
        }

    elif shape == 'circle':
        R_v, hw_v = params['radius'], params['h_w']
        A_expr  = sp.pi * R**2
        hc_expr = h_w
        F_expr  = gamma * hc_expr * A_expr
        IG_expr = sp.pi * R**4 / 4
        ycp_expr = hc_expr + IG_expr / (hc_expr * A_expr)
        subs = {R: R_v, h_w: hw_v, gamma: GAMMA_WATER}
        A_val   = float(A_expr.subs(subs))
        hc_val  = float(hc_expr.subs(subs))
        F_val   = float(F_expr.subs(subs))
        ycp_val = float(ycp_expr.subs(subs))
        IG_val  = float(IG_expr.subs(subs))
        M_over  = F_val * ycp_val
        result = {
            'shape': '원형', 'shape_en': shape,
            'A_expr': sp.latex(A_expr), 'hc_expr': sp.latex(hc_expr),
            'F_expr': sp.latex(F_expr), 'ycp_expr': sp.latex(ycp_expr),
            'A': A_val, 'h_c': hc_val, 'F': F_val, 'y_cp': ycp_val,
            'params': params, 'IG': IG_val,
            'F_horizontal': F_val, 'F_vertical': 0.0,
            'M_overturn': M_over,
        }

    elif shape == 'semicircle':
        R_v, hw_v = params['radius'], params['h_w']
        A_expr  = sp.pi * R**2 / 2
        ybar    = sp.Rational(4, 1) * R / (3 * sp.pi)
        hc_expr = h_w + ybar
        F_expr  = gamma * hc_expr * A_expr
        IG_expr = (sp.pi/8 - sp.Rational(8,1)/(9*sp.pi)) * R**4
        ycp_expr = hc_expr + IG_expr / (hc_expr * A_expr)
        subs = {R: R_v, h_w: hw_v, gamma: GAMMA_WATER}
        A_val   = float(A_expr.subs(subs))
        hc_val  = float(hc_expr.subs(subs))
        F_val   = float(F_expr.subs(subs))
        ycp_val = float(ycp_expr.subs(subs))
        IG_val  = float(IG_expr.subs(subs))
        M_over  = F_val * ycp_val
        result = {
            'shape': '반원형', 'shape_en': shape,
            'A_expr': sp.latex(A_expr), 'hc_expr': sp.latex(hc_expr),
            'F_expr': sp.latex(F_expr), 'ycp_expr': sp.latex(ycp_expr),
            'A': A_val, 'h_c': hc_val, 'F': F_val, 'y_cp': ycp_val,
            'params': params, 'IG': IG_val,
            'F_horizontal': F_val, 'F_vertical': 0.0,
            'M_overturn': M_over,
        }

    elif shape == 'parabola':
        w_v, H_v, hw_v = params['width'], params['height'], params['h_w']
        a_v = H_v / (w_v**2)
        result = _parabola_hydrostatic(w_v, H_v, hw_v, a_v)

    else:
        raise ValueError(f"지원하지 않는 형상: {shape}")

    return result


def _parabola_hydrostatic(w_v, H_v, hw_v, a_v):
    gamma_val = GAMMA_WATER
    A_proj = 2 * w_v * H_v
    hc_proj = hw_v - H_v / 2
    if hc_proj < 0:
        hc_proj = hw_v / 2

    Fh = gamma_val * hc_proj * A_proj

    def integrand_volume(x):
        z_gate = a_v * x**2
        depth = hw_v - z_gate
        return max(0.0, depth) * H_v

    V_result, _ = integrate.quad(integrand_volume, -w_v, w_v)
    Fv = gamma_val * V_result
    F_total = np.sqrt(Fh**2 + Fv**2)
    angle   = np.degrees(np.arctan2(Fv, Fh))
    A_val = float(2 * w_v * H_v)
    M_over = F_total * hw_v

    return {
        'shape': '포물선 곡면', 'shape_en': 'parabola',
        'A': A_val, 'h_c': hc_proj,
        'F': F_total, 'y_cp': hw_v,
        'F_horizontal': Fh, 'F_vertical': Fv,
        'F_total': F_total, 'angle_deg': angle,
        'volume_above': V_result,
        'A_expr': r'2w \cdot H',
        'hc_expr': r'h_w - \frac{H}{2}',
        'F_expr': r'F = \sqrt{F_h^2 + F_v^2}',
        'ycp_expr': r'y_{cp} = h_w',
        'IG': 0.0,
        'params': {'width': w_v, 'height': H_v, 'h_w': hw_v, 'a_coeff': a_v},
        'M_overturn': M_over,
    }


def compute_pressure_distribution(shape, params, result):
    gamma_val = GAMMA_WATER
    hw_v = params.get('h_w', 1.0)

    if shape == 'rectangle':
        b_v, H_v = params['width'], params['height']
        x = np.linspace(-b_v/2, b_v/2, 30)
        z = np.linspace(0, H_v, 30)
        X, Z = np.meshgrid(x, z)
        depth = hw_v + Z
        P = gamma_val * depth / 1000
        Y = np.zeros_like(X)
        vertices = _rect_vertices(b_v, H_v)

    elif shape == 'triangle':
        b_v, H_v = params['width'], params['height']
        x = np.linspace(-b_v/2, b_v/2, 30)
        z = np.linspace(0, H_v, 30)
        X, Z = np.meshgrid(x, z)
        half_width = b_v/2 * (1 - Z/H_v)
        mask = np.abs(X) <= half_width
        depth = hw_v + Z
        P = np.where(mask, gamma_val * depth / 1000, 0)
        Y = np.zeros_like(X)
        vertices = _tri_vertices(b_v, H_v)

    elif shape == 'trapezoid':
        b_v, B_v, H_v = params['b_top'], params['b_bottom'], params['height']
        x = np.linspace(-B_v/2, B_v/2, 30)
        z = np.linspace(0, H_v, 30)
        X, Z = np.meshgrid(x, z)
        half_w_trap = b_v/2 + (B_v/2 - b_v/2) * (Z/H_v)
        mask = np.abs(X) <= half_w_trap
        depth = hw_v + Z
        P = np.where(mask, gamma_val * depth / 1000, 0)
        Y = np.zeros_like(X)
        vertices = _trap_vertices(b_v, B_v, H_v)

    elif shape == 'circle':
        R_v = params['radius']
        theta = np.linspace(0, 2*np.pi, 40)
        r = np.linspace(0, R_v, 20)
        T, Rv = np.meshgrid(theta, r)
        X = Rv * np.cos(T)
        Z = Rv * np.sin(T) + R_v
        depth = hw_v + Z - R_v
        P = gamma_val * np.maximum(depth, 0) / 1000
        Y = np.zeros_like(X)
        vertices = _circle_vertices(R_v)

    elif shape == 'semicircle':
        R_v = params['radius']
        theta = np.linspace(0, np.pi, 30)
        r = np.linspace(0, R_v, 20)
        T, Rv = np.meshgrid(theta, r)
        X = Rv * np.cos(T)
        Z = Rv * np.sin(T)
        depth = hw_v + Z
        P = gamma_val * depth / 1000
        Y = np.zeros_like(X)
        vertices = _semicircle_vertices(R_v)

    elif shape == 'parabola':
        w_v, H_v = params['width'], params['height']
        a_v = params.get('a_coeff', H_v / w_v**2)
        x = np.linspace(-w_v, w_v, 40)
        t = np.linspace(0, H_v, 30)
        X, T_grid = np.meshgrid(x, t)
        Z_3d = np.tile(a_v * x**2, (len(t), 1))
        X_3d = X
        depth_3d = hw_v + T_grid
        P = gamma_val * depth_3d / 1000
        Y = np.zeros_like(X_3d)
        vertices = _parabola_vertices(w_v, H_v, a_v)
        X, Z = X_3d, T_grid

    return {
        'X': X, 'Y': Y, 'Z': Z, 'P': P,
        'vertices': vertices,
        'F_h': result.get('F_horizontal', result.get('F', 0)),
        'F_v': result.get('F_vertical', 0),
        'y_cp': result.get('y_cp', hw_v),
        'h_c': result.get('h_c', hw_v),
    }


def _rect_vertices(b, H):
    return np.array([[-b/2,0,0],[b/2,0,0],[b/2,0,H],[-b/2,0,H],[-b/2,0,0]])
def _tri_vertices(b, H):
    return np.array([[-b/2,0,0],[b/2,0,0],[0,0,H],[-b/2,0,0]])
def _trap_vertices(b_top, b_bot, H):
    return np.array([[-b_bot/2,0,0],[b_bot/2,0,0],[b_top/2,0,H],[-b_top/2,0,H],[-b_bot/2,0,0]])
def _circle_vertices(R):
    theta = np.linspace(0, 2*np.pi, 60)
    return np.column_stack([R*np.cos(theta), np.zeros(60), R*np.sin(theta)+R])
def _semicircle_vertices(R):
    theta = np.linspace(0, np.pi, 40)
    pts = np.column_stack([R*np.cos(theta), np.zeros(40), R*np.sin(theta)])
    base = np.array([[-R,0,0],[R,0,0]])
    return np.vstack([pts, base[::-1]])
def _parabola_vertices(w, H, a):
    x = np.linspace(-w, w, 60)
    z = a * x**2
    return np.column_stack([x, np.zeros(60), z])


def calculate(shape: str, params: Dict[str, float]) -> Dict[str, Any]:
    result = symbolic_hydrostatic(shape, params)
    dist   = compute_pressure_distribution(shape, params, result)
    result['distribution'] = dist
    result['summary'] = _make_summary(result)
    return result


def _make_summary(r):
    lines = [
        f"**형상**: {r['shape']}",
        f"**면적 A**: {r['A']:.4f} m²",
        f"**도심 깊이 h_c**: {r['h_c']:.4f} m",
        f"**정수력 F**: {r['F']/1000:.2f} kN",
        f"**압력중심 y_cp**: {r['y_cp']:.4f} m",
        f"**전도 모멘트 M**: {r.get('M_overturn',0)/1000:.2f} kN·m",
    ]
    if r['shape_en'] == 'parabola':
        lines += [
            f"**수평 성분 Fh**: {r['F_horizontal']/1000:.2f} kN",
            f"**수직 성분 Fv**: {r['F_vertical']/1000:.2f} kN",
            f"**합력 방향**: {r.get('angle_deg',0):.1f}°",
        ]
    return "\n".join(lines)
