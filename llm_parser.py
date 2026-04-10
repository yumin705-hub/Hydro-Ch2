"""
llm_parser.py  v3.6 (Full Version - Syntax Safe)
- 다중 LLM API 지원 (Gemini, OpenAI, Anthropic)
- API 실패 시 전문적인 로컬 해설 자동 생성 (Fallback)
- JSON 응답 파싱 안정성 극대화 (마크다운 자동 제거)
"""

import json
import re
import time
import math
from typing import Dict, Any, Optional, Tuple

# API Providers (import 실패해도 동작하도록 예외 처리)
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ─── 시스템 프롬프트 ─────────────────────────────────────
EXTRACTION_SYSTEM = """당신은 수리공학 전문가 AI입니다. 사용자가 자연어로 입력한 수문(水門) 조건에서
정수력 계산에 필요한 파라미터를 추출하여 반드시 아래 JSON 형식으로만 응답하세요.

지원 형상(shape):
- rectangle  : 직사각형 수문 → 필요: width(폭), height(높이), h_w(수위=수면~수문상단 깊이)
- triangle   : 삼각형 수문   → 필요: width(밑변), height(높이), h_w
- trapezoid  : 사다리꼴 수문 → 필요: b_top(상변), b_bottom(하변), height(높이), h_w
- circle     : 원형 수문     → 필요: radius(반지름), h_w(수면~원 도심 깊이)
- semicircle : 반원형 수문   → 필요: radius(반지름), h_w(수면~직경 깊이)
- parabola   : 포물선 곡면   → 필요: width(반폭 w), height(깊이 H), h_w

단위: 모두 미터(m)

응답 JSON 형식 (반드시 이 형식만):
{
  "shape": "rectangle",
  "params": {
    "width": 3.0,
    "height": 5.0,
    "h_w": 1.0
  },
  "confidence": 0.95,
  "interpretation": "폭 3m, 높이 5m 직사각형 수문이며 수위는 수문 상단 위 1m입니다."
}

주의사항:
- h_w는 수면에서 수문 상단(원형/반원형은 도심)까지의 거리(양수)입니다.
- 수위가 수문 높이와 같다고 하면 h_w=0으로 처리하세요.
- 모든 값은 float 형태로 반환하세요.
- JSON 외 다른 텍스트는 절대 출력하지 마세요. 마크다운(```)도 사용하지 마세요.
"""

EXPLANATION_SYSTEM = """당신은 수리공학 및 유체역학 전문가입니다.
주어진 정수력 계산 결과를 한국어로 전문적이고 명확하게 해설해 주세요.

다음 항목을 포함하세요:
1. 계산 조건 요약
2. 핵심 결과 (정수력, 압력중심)
3. 압력중심이 도심 아래에 있는 물리적 의미
4. 댐/수문 설계 관점에서의 해석 및 주의사항
5. 전도 모멘트 및 안전율 검토
6. 수문 형식에 따른 특이사항 (곡면이면 수평/수직 성분 해설 포함)

전문적이되 한국어로 읽기 쉽게 작성하세요. 
수식은 LaTeX 형식으로 인라인($...$) 또는 블록($$...$$) 표기하세요.
"""

CHAT_SYSTEM = """당신은 '하이드로스텟 AI(HydroStatic AI)'라는 댐 수문 설계 전문 AI 어시스턴트입니다.
수리공학, 유체역학, 댐/수문 설계 분야의 깊은 전문 지식을 갖추고 있습니다.

역할:
- 정수력(Hydrostatic Force) 계산 지원
- 수문 형상별 압력 분포 및 설계 해설
- 유체역학 이론 설명 (SymPy 기호 계산 포함)
- 댐 안전 설계 관련 조언

응답 스타일:
- 한국어로 전문적이고 친절하게 응답
- 수식은 LaTeX 포맷 사용 ($$...$$)
- 구체적인 수치 제공 시 단위 명시
"""


# ═══════════════════════════════════════════════════════════
# LLM 호출 (재시도 로직 포함)
# ═══════════════════════════════════════════════════════════

def _call_llm(provider: str, api_key: str, system_prompt: str,
              user_prompt: str, is_chat: bool = False,
              chat_history: list = None, max_retries: int = 1) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            if provider == 'gemini' and genai:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash',
                                              system_instruction=system_prompt)
                if is_chat and chat_history:
                    history = [{'role': 'user' if m['role'] == 'user' else 'model',
                                'parts': [m['content']]} for m in chat_history]
                    chat = model.start_chat(history=history)
                    return chat.send_message(user_prompt).text
                else:
                    return model.generate_content(user_prompt).text

            elif provider == 'openai' and OpenAI:
                client = OpenAI(api_key=api_key)
                messages = [{"role": "system", "content": system_prompt}]
                if is_chat and chat_history:
                    for m in chat_history:
                        messages.append({"role": m['role'], "content": m['content']})
                messages.append({"role": "user", "content": user_prompt})
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, max_tokens=1500)
                return resp.choices[0].message.content

            elif provider == 'anthropic' and anthropic:
                client = anthropic.Anthropic(api_key=api_key)
                messages = []
                if is_chat and chat_history:
                    for m in chat_history:
                        messages.append({"role": m['role'], "content": m['content']})
                messages.append({"role": "user", "content": user_prompt})
                resp = client.messages.create(
                    model="claude-3-5-sonnet-20240620", max_tokens=1500,
                    system=system_prompt, messages=messages)
                return resp.content[0].text

            raise ValueError(f"Provider '{provider}' 사용 불가")

        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(k in err_str for k in ['429', 'rate', 'quota', 'exceeded', 'limit']):
                if attempt < max_retries:
                    time.sleep(3.0 * (attempt + 1))
                    continue
            raise
    raise last_err


# ═══════════════════════════════════════════════════════════
# 공개 API
# ═══════════════════════════════════════════════════════════

def test_api_key(provider: str, api_key: str) -> Tuple[bool, str, str]:
    try:
        _call_llm(provider, api_key, "Respond with OK", "Hello", max_retries=0)
        return True, provider.upper(), '정상 작동 완료'
    except Exception as e:
        return False, '', f'{provider.upper()} 오류: {str(e)[:120]}'


def extract_params(user_text: str, provider: str, api_key: str) -> Tuple[Optional[Dict], str]:
    try:
        # 🚀 에러 방지: 긴 문자열을 여러 줄로 안전하게 쪼갬
        combined_prompt = (
            f"반드시 JSON 괄호 {{}} 안의 내용만 출력해. "
            f"```json 과 같은 마크다운도 절대 쓰지마.\n\n"
            f"사용자 입력: {user_text}"
        )
        
        raw = _call_llm(provider, api_key, EXTRACTION_SYSTEM, combined_prompt, max_retries=1)
        raw_clean = raw.replace('```json', '').replace('```', '').strip()
        
        json_match = re.search(r'\{[\s\S]*\}', raw_clean)
        if not json_match:
            return None, f"JSON 추출 실패 (데이터 없음)"
            
        data = json.loads(json_match.group())
        return data, data.get('interpretation', '파라미터 추출 완료')
        
    except json.JSONDecodeError as e:
        return None, f"JSON 문법 오류: {e}"
    except Exception as e:
        return None, f"API 오류: {str(e)[:100]}"


def explain_result(result: Dict, user_query: str,
                   provider: str, api_key: str) -> str:
    try:
        summary = (
            f"수문: {result['shape']}, 파라미터: {result['params']}\n"
            f"A={result['A']:.4f}m², h_c={result['h_c']:.4f}m, "
            f"F={result['F']/1000:.4f}kN, y_cp={result['y_cp']:.4f}m"
        )
        prompt = f"사용자: {user_query}\n\n{summary}\n\n상세 해설:"
        return _call_llm(provider, api_key, EXPLANATION_SYSTEM, prompt, max_retries=0)
    except Exception:
        pass

    return _local_explain(result)


def chat_response(messages: list, user_message: str,
                  provider: str, api_key: str,
                  calc_context: Optional[str] = None) -> str:
    system = CHAT_SYSTEM
    if calc_context:
        system += f"\n\n현재 컨텍스트:\n{calc_context}"
    try:
        return _call_llm(provider, api_key, system, user_message,
                         is_chat=True, chat_history=messages[-8:], max_retries=1)
    except Exception as e:
        return (f"❌ API 응답 실패\n\n"
                f"💡 수치 조건을 포함하면 API 없이도 계산 가능합니다.\n"
                f"예: \"폭 3m 높이 5m 수위 4m\"")


# ═══════════════════════════════════════════════════════════
# 로컬 해설 생성 (API 불필요)
# ═══════════════════════════════════════════════════════════

def _local_explain(r: Dict) -> str:
    shape = r.get('shape', '수문')
    shape_en = r.get('shape_en', '')
    A = r['A']
    hc = r['h_c']
    F_N = r['F']
    F_kN = F_N / 1000
    ycp = r['y_cp']
    e = ycp - hc
    IG = r.get('IG', 0)
    M = r.get('M_overturn', F_N * ycp) / 1000
    params = r.get('params', {})
    hw = params.get('h_w', 0)

    lines = []
    lines.append(f"### 📋 {shape} 수문 정수력 분석 결과\n")

    lines.append("#### 1. 계산 조건")
    if shape_en == 'rectangle':
        lines.append(f"- 형상: {shape} (폭 {params.get('width',0)}m × 높이 {params.get('height',0)}m)")
    elif shape_en == 'circle':
        lines.append(f"- 형상: {shape} (반지름 R = {params.get('radius',0)}m)")
    elif shape_en == 'semicircle':
        lines.append(f"- 형상: {shape} (반지름 R = {params.get('radius',0)}m)")
    elif shape_en == 'triangle':
        lines.append(f"- 형상: {shape} (밑변 {params.get('width',0)}m × 높이 {params.get('height',0)}m)")
    elif shape_en == 'trapezoid':
        lines.append(f"- 형상: {shape} (상변 {params.get('b_top',0)}m, 하변 {params.get('b_bottom',0)}m, 높이 {params.get('height',0)}m)")
    elif shape_en == 'parabola':
        lines.append(f"- 형상: {shape} (반폭 {params.get('width',0)}m × 깊이 {params.get('height',0)}m)")
    else:
        lines.append(f"- 형상: {shape}")
    lines.append(f"- 수위 조건: $h_w$ = {hw} m (수면 → 수문 기준점)")
    lines.append(f"- 비중량: $\\gamma$ = 9,810 N/m³\n")

    lines.append("#### 2. 핵심 결과")
    lines.append(f"| 항목 | 기호 | 값 | 단위 |")
    lines.append(f"|------|------|-----|------|")
    lines.append(f"| 단면적 | $A$ | {A:.4f} | m² |")
    lines.append(f"| 도심 깊이 | $h_c$ | {hc:.4f} | m |")
    lines.append(f"| **정수력** | **$F$** | **{F_kN:.3f}** | **kN** |")
    lines.append(f"| 압력중심 | $y_{{cp}}$ | {ycp:.4f} | m |")
    lines.append(f"| 편심거리 | $e$ | {e:.5f} | m |")
    if IG > 0:
        lines.append(f"| 단면 2차 모멘트 | $I_G$ | {IG:.4f} | m⁴ |")
    lines.append(f"| 전도 모멘트 | $M$ | {M:.3f} | kN·m |")
    lines.append("")

    if shape_en == 'parabola':
        Fh = r.get('F_horizontal', 0) / 1000
        Fv = r.get('F_vertical', 0) / 1000
        angle = r.get('angle_deg', 0)
        lines.append(f"**곡면 성분 분해:**")
        lines.append(f"- 수평 성분: $F_h$ = {Fh:.3f} kN (투영면 기준)")
        lines.append(f"- 수직 성분: $F_v$ = {Fv:.3f} kN (수문 위 물 무게)")
        lines.append(f"- 합력 방향: $\\theta$ = {angle:.1f}°")
        lines.append(f"- $F = \\sqrt{{F_h^2 + F_v^2}}$ = {F_kN:.3f} kN\n")

    lines.append("#### 3. 물리적 의미")
    lines.append(f"압력중심($y_{{cp}}$ = {ycp:.3f}m)이 도심($h_c$ = {hc:.3f}m)보다 "
                 f"**{e:.4f}m 아래**에 위치합니다.")
    lines.append(f"이는 수압이 깊이에 비례하여 증가하기 때문이며, "
                 f"편심거리 $e = I_G / (h_c \\cdot A)$로 결정됩니다.")
    if hc > 0 and e > 0:
        ratio = e / hc * 100
        if ratio < 5:
            lines.append(f"편심비 $e/h_c$ = {ratio:.2f}%로, 수위가 충분히 높아 압력 분포가 비교적 균일합니다.\n")
        else:
            lines.append(f"편심비 $e/h_c$ = {ratio:.2f}%로, 압력 분포의 불균일성을 설계에 반영해야 합니다.\n")

    lines.append("#### 4. 설계 검토")
    lines.append(f"- **전도 모멘트**: $M = F \\times y_{{cp}}$ = {M:.2f} kN·m")
    lines.append(f"- 수문 힌지를 압력중심({ycp:.3f}m) 부근에 배치하면 개폐 토크를 최소화할 수 있습니다.")
    lines.append(f"- 전도 안전율: $SF = M_{{resist}} / M_{{overturn}} \\geq 1.5$ 확인 필요")
    lines.append(f"- 수문 자중, 마찰력, 동수압(지진·파압)은 별도 검토 대상입니다.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 계산 요청 판별 및 규칙 기반 파서
# ═══════════════════════════════════════════════════════════

def is_calculation_request(text: str) -> bool:
    t = text.lower().strip()
    if len(re.findall(r'\d+\.?\d*\s*m', t)) >= 2:
        return True

    shape_kw = ['직사각형', '삼각형', '사다리꼴', '원형', '포물선', '반원',
                'rectangle', 'triangle', 'trapezoid', 'circle', 'parabola', 'semicircle']
    has_shape = any(kw in t for kw in shape_kw)
    has_number = bool(re.search(r'\d+\.?\d*', t))
    if has_shape and has_number:
        return True

    dim_kw = ['폭', '높이', '수위', '수심', '반지름', '반경', '밑변', '상변', '하변', '깊이']
    if sum(1 for kw in dim_kw if kw in t) >= 1 and has_number:
        return True

    calc_kw = ['계산', '설계', '정수력', '압력', '구해', '구하', '얼마', '분석']
    if any(kw in t for kw in calc_kw) and has_number:
        return True

    return False


def rule_parse(text: str) -> Dict:
    t = text.replace(',', ' ').replace('，', ' ').replace(':', ' ').replace('=', ' ')
    t_lower = t.lower()

    if   any(k in t_lower for k in ['포물', 'parabola']):    shape = 'parabola'
    elif any(k in t_lower for k in ['반원', 'semicircle']):   shape = 'semicircle'
    elif any(k in t_lower for k in ['원형', '원 ', 'circle']): shape = 'circle'
    elif any(k in t_lower for k in ['사다리', 'trapezoid']):   shape = 'trapezoid'
    elif any(k in t_lower for k in ['삼각', 'triangle']):     shape = 'triangle'
    else:                                                     shape = 'rectangle'

    def ev(patterns):
        for p in patterns:
            m = re.search(p, t_lower)
            if m: return float(m.group(1))
        return None

    hw = ev([r'수위\s*(\d+\.?\d*)', r'수심\s*(\d+\.?\d*)',
             r'도심\s*깊이\s*(\d+\.?\d*)', r'h_?w\s*(\d+\.?\d*)'])
    all_nums = [float(x) for x in re.findall(r'(\d+\.?\d*)\s*m', t_lower)]

    p = {}
    if shape == 'rectangle':
        p['width'] = ev([r'폭\s*(\d+\.?\d*)', r'너비\s*(\d+\.?\d*)'])
        p['height'] = ev([r'높이\s*(\d+\.?\d*)'])
        non_hw = [v for v in all_nums if v != hw]
        if p['width'] is None and len(non_hw) > 0: p['width'] = non_hw[0]
        if p['height'] is None and len(non_hw) > 1: p['height'] = non_hw[1]
        p['width'] = p['width'] or 3.0
        p['height'] = p['height'] or 5.0

    elif shape == 'triangle':
        p['width'] = ev([r'밑변\s*(\d+\.?\d*)', r'폭\s*(\d+\.?\d*)'])
        p['height'] = ev([r'높이\s*(\d+\.?\d*)'])
        non_hw = [v for v in all_nums if v != hw]
        if p['width'] is None and len(non_hw) > 0: p['width'] = non_hw[0]
        if p['height'] is None and len(non_hw) > 1: p['height'] = non_hw[1]
        p['width'] = p['width'] or 4.0
        p['height'] = p['height'] or 3.0

    elif shape == 'trapezoid':
        p['b_top'] = ev([r'상변\s*(\d+\.?\d*)'])
        p['b_bottom'] = ev([r'하변\s*(\d+\.?\d*)'])
        p['height'] = ev([r'높이\s*(\d+\.?\d*)'])
        non_hw = [v for v in all_nums if v != hw]
        if p['b_top'] is None and len(non_hw) > 0: p['b_top'] = non_hw[0]
        if p['b_bottom'] is None and len(non_hw) > 1: p['b_bottom'] = non_hw[1]
        if p['height'] is None and len(non_hw) > 2: p['height'] = non_hw[2]
        p['b_top'] = p['b_top'] or 2.0
        p['b_bottom'] = p['b_bottom'] or 5.0
        p['height'] = p['height'] or 4.0

    elif shape in ('circle', 'semicircle'):
        p['radius'] = ev([r'반지름\s*(\d+\.?\d*)', r'반경\s*(\d+\.?\d*)',
                          r'radius\s*(\d+\.?\d*)'])
        non_hw = [v for v in all_nums if v != hw]
        if p['radius'] is None and non_hw: p['radius'] = non_hw[0]
        p['radius'] = p['radius'] or 2.0

    elif shape == 'parabola':
        p['width'] = ev([r'폭\s*(\d+\.?\d*)', r'반폭\s*(\d+\.?\d*)'])
        p['height'] = ev([r'깊이\s*(\d+\.?\d*)', r'높이\s*(\d+\.?\d*)'])
        non_hw = [v for v in all_nums if v != hw]
        if p['width'] is None and len(non_hw) > 0: p['width'] = non_hw[0]
        if p['height'] is None and len(non_hw) > 1: p['height'] = non_hw[1]
        p['width'] = p['width'] or 3.0
        p['height'] = p['height'] or 4.0

    p['h_w'] = hw or 1.0
    return {'shape': shape, 'params': p,
            'interpretation': f"규칙 기반: {shape}, {p}"}