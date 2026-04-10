"""
app.py — HydroStatic AI  v3.6 (Full Version)
- 빠른 계산 처리 (API 호출 50% 단축)
- 에러 원인 명확화 (마크다운 혼입 방어)
- CSS 및 UI 코드 완전 전개 (가독성 최적화)
"""

import streamlit as st
import numpy as np
import json, os, re
from datetime import datetime

from hydrostatic_engine import calculate, GAMMA_WATER
from visualizer import plot_3d_hydrostatic, plot_2d_summary
from llm_parser import (extract_params, explain_result, chat_response,
                         is_calculation_request, test_api_key, rule_parse, _local_explain)

# ═══════════════════════════════════════════════════════════
# Page Config & Initialization
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HydroStatic AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# MathJax 및 전역 CSS 설정
st.markdown("""
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] },
  svg: { fontCache: 'global' },
  startup: { pageReady: () => { MathJax.startup.defaultPageReady().then(() => {}); } }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg:     #040d14;
    --panel:  #071523;
    --card:   #0b1e30;
    --input:  #071a28;
    --cyan:   #00c8ff;
    --teal:   #00ff9f;
    --orange: #ff6b35;
    --text:   #c0d8e8;
    --dim:    #5a7a8a;
    --border: #1a3a55;
    --glow:   0 0 18px rgba(0,200,255,0.22);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stHeader"] { 
    background: transparent !important; 
}

.hydro-header {
    background: linear-gradient(135deg, #040d14, #071e38, #040d14);
    border-bottom: 2px solid var(--cyan);
    padding: 1.2rem 2rem 1rem;
    position: relative;
    overflow: hidden;
}

.hydro-header::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--teal), transparent);
    animation: scan 3s ease-in-out infinite;
}

@keyframes scan { 
    0%, 100% { opacity: .4 } 
    50% { opacity: 1 } 
}

.hydro-title {
    font-family:'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: .12em;
    text-shadow: 0 0 22px rgba(0, 200, 255, .5);
    margin: 0;
}

.hydro-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem;
    color: var(--dim);
    letter-spacing: .22em;
    margin-top: .2rem;
}

.chat-wrap {
    max-height: 60vh;
    overflow-y: auto;
    padding: .4rem 0;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.chat-wrap::-webkit-scrollbar { width: 4px; }
.chat-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.msg-user { display: flex; justify-content: flex-end; margin: .45rem 0; }
.msg-ai { display: flex; justify-content: flex-start; margin: .45rem 0; }

.bubble-user {
    background: linear-gradient(135deg, #00527a, #003d5c);
    border: 1px solid rgba(0, 200, 255, .3);
    border-radius: 12px 12px 2px 12px;
    padding: .65rem 1rem;
    max-width: 72%;
    font-size: .87rem;
    color: #e0f4ff;
    box-shadow: var(--glow);
}

.bubble-ai {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    border-radius: 2px 12px 12px 12px;
    padding: .7rem 1rem;
    max-width: 88%;
    font-size: .86rem;
    color: var(--text);
    line-height: 1.65;
}

.msg-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: .61rem;
    color: var(--dim);
    margin: .1rem .25rem;
}

.avatar-ai {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, var(--panel), #0a2033);
    border: 1px solid var(--cyan);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem;
    margin-right: .45rem;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(0, 200, 255, .18);
}

.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--teal);
    border-radius: 2px;
    padding: .9rem 1.1rem;
    margin: .5rem 0;
}

.result-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: .82rem;
    font-weight: 600;
    color: var(--teal);
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: .55rem;
    padding-bottom: .3rem;
    border-bottom: 1px solid var(--border);
}

.metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .45rem; }
.metric-item {
    background: rgba(0, 200, 255, .04);
    border: 1px solid rgba(0, 200, 255, .12);
    border-radius: 2px;
    padding: .45rem .6rem;
}

.metric-label { font-family: 'Share Tech Mono', monospace; font-size: .6rem; color: var(--dim); letter-spacing: .08em; text-transform: uppercase; }
.metric-value { font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; font-weight: 600; color: var(--cyan); margin-top: 1px; }
.metric-unit { font-size: .68rem; color: var(--dim); margin-left: 2px; }

.api-ok { background: rgba(0, 255, 159, .06); border: 1px solid rgba(0, 255, 159, .3); border-radius: 2px; padding: .45rem .7rem; font-size: .78rem; color: #00ff9f; }
.api-fail { background: rgba(255, 107, 53, .06); border: 1px solid rgba(255, 107, 53, .3); border-radius: 2px; padding: .45rem .7rem; font-size: .78rem; color: #ff9a7a; }
.api-none { background: rgba(90, 122, 138, .06); border: 1px solid rgba(90, 122, 138, .3); border-radius: 2px; padding: .45rem .7rem; font-size: .78rem; color: var(--dim); }

.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background: var(--input) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 2px !important;
    font-size: .87rem !important;
}

.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px rgba(0, 200, 255, .18) !important;
}

.stButton>button {
    background: linear-gradient(135deg, #003d5c, #00527a) !important;
    color: var(--cyan) !important; border: 1px solid rgba(0, 200, 255, .35) !important;
    border-radius: 2px !important; font-family: 'Rajdhani', sans-serif !important;
    font-size: .88rem !important; font-weight: 600 !important;
    letter-spacing: .07em !important; transition: all .18s !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #004f78, #00699e) !important;
    border-color: var(--cyan) !important; box-shadow: var(--glow) !important;
    transform: translateY(-1px) !important;
}

.stSelectbox>div>div { background: var(--input) !important; color: var(--text) !important; border-color: var(--border) !important; }
.stSlider>div>div>div { background: var(--border) !important; }
.stSlider>div>div>div>div { background: var(--cyan) !important; }

[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p { color: var(--text) !important; font-size: .82rem !important; }
[data-testid="stSidebar"] h3 {
    font-family: 'Rajdhani', sans-serif !important; color: var(--cyan) !important;
    font-size: .95rem !important; letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important; padding-bottom: .25rem !important;
}

.stTabs [data-baseweb="tab-list"] { background: var(--panel) !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--dim) !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: .84rem !important;
    font-weight: 600 !important; letter-spacing: .09em !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important; border-bottom: 2px solid var(--cyan) !important;
    background: rgba(0, 200, 255, .05) !important;
}

.section-label {
    font-family: 'Share Tech Mono', monospace; font-size: .66rem;
    color: var(--dim); letter-spacing: .18em; text-transform: uppercase;
    padding: .35rem 0 .15rem;
}

.info-box {
    background: rgba(0, 200, 255, .04); border: 1px solid rgba(0, 200, 255, .14);
    border-radius: 2px; padding: .65rem .85rem; font-size: .81rem;
    color: var(--dim); margin: .35rem 0;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
def _init():
    defs = {
        'messages': [], 'last_result': None, 'last_shape': None, 'last_params': None,
        'api_provider': 'gemini', 'api_key': '', 'api_status': 'none', 'api_model': '',
        'api_err_msg': '', 'calc_count': 0, 'img_3d': None, 'img_2d': None, 'pending_msg': '',
    }
    for k, v in defs.items():
        if k not in st.session_state: 
            st.session_state[k] = v

_init()

SHAPE_KR = {
    'rectangle': '직사각형', 'triangle': '삼각형', 
    'trapezoid': '사다리꼴', 'circle': '원형', 
    'semicircle': '반원형', 'parabola': '포물선 곡면'
}

def _get_api_key():
    k = st.session_state.api_key.strip()
    return k if k else os.environ.get('API_KEY','').strip() or None

def _run_calc(shape, params):
    result = calculate(shape, params)
    st.session_state.last_result = result
    st.session_state.last_shape  = shape
    st.session_state.last_params = params
    st.session_state.calc_count += 1
    st.session_state.img_3d = plot_3d_hydrostatic(shape, params, result)
    st.session_state.img_2d = plot_2d_summary(shape, params, result)
    return result

# ═══════════════════════════════════════════════════════════
# Sidebar UI
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.7rem 0 .4rem">
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.25rem; font-weight:700;color:#00c8ff;letter-spacing:.15em;">HYDROSTATIC AI</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.58rem; color:#5a7a8a;letter-spacing:.18em;margin-top:2px;">DAM GATE DESIGN v3.6</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### ⚙ API 설정")
    provider_options = {
        "Google Gemini (무료/추천)": "gemini", 
        "OpenAI (ChatGPT)": "openai", 
        "Anthropic (Claude)": "anthropic"
    }
    
    current_idx = list(provider_options.values()).index(st.session_state.api_provider)
    selected_provider_label = st.selectbox("AI 모델 선택", list(provider_options.keys()), index=current_idx)
    new_provider = provider_options[selected_provider_label]
    
    if new_provider != st.session_state.api_provider:
        st.session_state.api_provider = new_provider
        st.session_state.api_status = 'none'

    st.markdown("<div style='font-size:0.75rem; color:#00ff9f; margin-bottom:10px; margin-top:-5px;'>💡 한도 초과(429)시 2분 대기 또는 새 계정 필요!</div>", unsafe_allow_html=True)
    
    api_input = st.text_input("API Key 입력", value=st.session_state.api_key, type="password", placeholder="여기에 API 키를 붙여넣으세요")
    if api_input != st.session_state.api_key:
        st.session_state.api_key = api_input
        st.session_state.api_status = 'none'

    if st.button("🔍 API 키 검증", use_container_width=True):
        key = _get_api_key()
        if not key:
            st.session_state.api_status = 'fail'
            st.session_state.api_err_msg = 'API 키를 입력하세요'
        else:
            with st.spinner("검증 중..."):
                ok, model, msg = test_api_key(st.session_state.api_provider, key)
            if ok:
                st.session_state.api_status = 'ok'
                st.session_state.api_model  = model
                st.session_state.api_err_msg = msg
            else:
                st.session_state.api_status = 'fail'
                st.session_state.api_err_msg = msg

    status = st.session_state.api_status
    if status == 'ok': 
        st.markdown(f"""<div class="api-ok">✅ API 정상 작동<br><span style="font-size:.7rem;opacity:.8">모델: {st.session_state.api_model}<br>{st.session_state.api_err_msg}</span></div>""", unsafe_allow_html=True)
    elif status == 'fail': 
        st.markdown(f"""<div class="api-fail">❌ API 오류<br><span style="font-size:.7rem;opacity:.8">{st.session_state.api_err_msg}</span></div>""", unsafe_allow_html=True)
    else: 
        st.markdown('<div class="api-none">⬜ API 키 미검증 — 위 버튼으로 확인</div>', unsafe_allow_html=True)

    with st.expander("🔑 API 키 무료 발급 방법 보기"):
        st.markdown("""
        <div style="font-size:0.8rem; line-height:1.6; color:#c0d8e8;">
        <b>1. Google Gemini (무료 추천)</b><br>
        • <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#00c8ff;">Google AI Studio</a> 접속<br>
        • 구글 로그인 후 [Create API Key] 클릭<br>
        • 즉시 발급 및 <b>무료 사용 가능</b><br><br>
        <b>2. OpenAI (ChatGPT)</b><br>
        • <a href="https://platform.openai.com/api-keys" target="_blank" style="color:#00c8ff;">OpenAI Platform</a> 접속<br>
        • <i>주의: 계정에 크레딧 충전 필요</i><br><br>
        <b>3. Anthropic (Claude)</b><br>
        • <a href="https://console.anthropic.com/settings/keys" target="_blank" style="color:#00c8ff;">Anthropic Console</a> 접속<br>
        • <i>주의: 계정에 크레딧 충전 필요</i>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 형상 선택")
    sel_shape = st.selectbox("수문 형상", list(SHAPE_KR.keys()), format_func=lambda x: f"{SHAPE_KR[x]}  ({x})")

    st.markdown("---")
    st.markdown("### 📏 파라미터")
    params = {}
    hw_def = 1.0
    
    if sel_shape == 'rectangle':
        params['width'] = st.slider("폭 b [m]", 0.5, 15.0, 3.0, 0.1)
        params['height'] = st.slider("높이 H [m]", 0.5, 15.0, 5.0, 0.1)
    elif sel_shape == 'triangle':
        params['width'] = st.slider("밑변 b [m]", 0.5, 12.0, 4.0, 0.1)
        params['height'] = st.slider("높이 H [m]", 0.5, 12.0, 3.0, 0.1)
    elif sel_shape == 'trapezoid':
        params['b_top'] = st.slider("상변 b [m]", 0.5, 10.0, 2.0, 0.1)
        params['b_bottom'] = st.slider("하변 B [m]", 1.0, 15.0, 5.0, 0.1)
        params['height'] = st.slider("높이 H [m]", 0.5, 12.0, 4.0, 0.1)
    elif sel_shape == 'circle':
        params['radius'] = st.slider("반지름 R [m]", 0.3, 8.0, 2.0, 0.1)
        hw_def = params['radius']
    elif sel_shape == 'semicircle':
        params['radius'] = st.slider("반지름 R [m]", 0.3, 8.0, 2.5, 0.1)
    elif sel_shape == 'parabola':
        params['width'] = st.slider("반폭 w [m]", 0.5, 8.0, 3.0, 0.1)
        params['height'] = st.slider("깊이 H [m]", 0.5, 12.0, 4.0, 0.1)
        
    params['h_w'] = st.slider("수위 h_w [m]", 0.0, 20.0, hw_def, 0.1, help="수면→수문 상단(원형은 도심)까지 거리")

    st.markdown("---")
    calc_btn = st.button("▶ 정수력 계산 실행", use_container_width=True)

    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem; color:#3a5a6a;text-align:center;padding:.4rem 0;">
        γ = 9,810 N/m³  |  SI 단위계<br>
        SymPy · SciPy · Multi-LLM API<br>
        Provider: {st.session_state.api_provider.upper()}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hydro-header">
  <div style="display:flex;align-items:center;gap:.8rem;">
    <span style="font-size:1.7rem;">💧</span>
    <div>
      <div class="hydro-title">HYDROSTATIC  AI
        <span style="font-size:.65rem;background:rgba(0,200,255,.1);border:1px solid rgba(0,200,255,.3); color:#00c8ff;padding:2px 7px;border-radius:2px;letter-spacing:.12em;margin-left:.5rem;">v3.6</span>
      </div>
      <div class="hydro-sub">DAM GATE DESIGN · HYDROSTATIC FORCE ANALYSIS · AI-ASSISTED ENGINEERING</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_calc, tab_theory = st.tabs(["💬  AI 챗봇", "📊  3D 시각화 · 결과", "📖  이론 & 수식"])


# ─── TAB 1: AI 챗봇 ──────────────────────────────────────────
with tab_chat:
    col_chat, col_panel = st.columns([6, 4])
    
    with col_chat:
        st.markdown('<div class="section-label">💬 AI 채팅 인터페이스</div>', unsafe_allow_html=True)
        
        chat_html = '<div class="chat-wrap">'
        if not st.session_state.messages:
            chat_html += """
            <div style="text-align:center;padding:2.2rem 1rem;color:#3a5a6a;">
                <div style="font-size:1.8rem;margin-bottom:.6rem;">🌊</div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:#00c8ff; letter-spacing:.1em;margin-bottom:.35rem;">HYDROSTATIC AI 준비완료</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:.67rem;line-height:1.85;">
                    자연어로 수문 조건을 입력하세요<br>
                    예: "폭 3m, 높이 5m 직사각형 수문, 수위 4m"<br>
                    또는 이론/설계 질문을 하세요
                </div>
            </div>"""
        else:
            for msg in st.session_state.messages:
                ts = msg.get('time','')
                content = msg['content'].replace('\n','<br>')
                if msg['role'] == 'user': 
                    chat_html += f"""
                    <div class="msg-user">
                        <div>
                            <div class="bubble-user">{msg['content']}</div>
                            <div class="msg-meta" style="text-align:right">{ts}</div>
                        </div>
                    </div>"""
                else: 
                    chat_html += f"""
                    <div class="msg-ai">
                        <div class="avatar-ai">🤖</div>
                        <div>
                            <div class="bubble-ai">{content}</div>
                            <div class="msg-meta">{ts}</div>
                        </div>
                    </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

        examples = [
            ("직사각형", "폭 3m 높이 5m 직사각형 수문 수위 4m"), 
            ("원형", "반지름 2m 원형 수문 도심 깊이 6m"), 
            ("포물선", "폭 3m 깊이 4m 포물선 곡면 수위 5m"), 
            ("삼각형", "밑변 4m 높이 3m 삼각형 수문 수심 2m"), 
            ("사다리꼴", "상변 2m 하변 5m 높이 4m 사다리꼴 수문 수위 1m"), 
            ("반원형", "반지름 2.5m 반원형 수문 수위 3m")
        ]
        
        ecols = st.columns(3)
        for i, (lbl, txt) in enumerate(examples):
            with ecols[i % 3]:
                if st.button(f"▸ {lbl}", key=f"ex{i}", use_container_width=True):
                    st.session_state.pending_msg = txt
                    st.rerun()

        with st.form('chat_form', clear_on_submit=True):
            user_input = st.text_input("입력", value='', placeholder="수문 조건 또는 질문을 입력하세요... (Enter 또는 전송 버튼)", label_visibility='collapsed')
            fc1, fc2 = st.columns([5, 1])
            with fc1: send = st.form_submit_button("📡  전송", use_container_width=True)
            with fc2: clear = st.form_submit_button("🗑", use_container_width=True)

        if clear:
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.img_3d = None
            st.session_state.img_2d = None
            st.rerun()

        msg_txt = ''
        if st.session_state.get('pending_msg', ''):
            msg_txt = st.session_state.pending_msg.strip()
            st.session_state.pending_msg = ''
        elif send and user_input.strip():
            msg_txt = user_input.strip()

        if msg_txt:
            ts = datetime.now().strftime('%H:%M')
            st.session_state.messages.append({'role':'user', 'content':msg_txt, 'time':ts})

            api_key = _get_api_key()
            provider = st.session_state.api_provider
            is_calc = is_calculation_request(msg_txt)

            with st.spinner("🔄 쾌속 분석 및 계산 중..."):
                if is_calc:
                    # 🚀 속도 향상: 숫자 파싱을 로컬 규칙 기반 파서로 즉시 수행
                    parsed = rule_parse(msg_txt)
                    try:
                        result = _run_calc(parsed['shape'], parsed['params'])
                        if api_key:
                            try:
                                # 해설 작성에만 API 호출 (1회)
                                explanation = explain_result(result, msg_txt, provider, api_key)
                            except Exception:
                                explanation = _local_explain(result)
                        else:
                            explanation = _local_explain(result)
                            
                        reply = (f"✅ **{result['shape']} 수문 정수력 계산 완료**\n\n"
                                 f"---\n\n{explanation}\n\n---\n"
                                 f"> 📊 **[3D 시각화 탭]**에서 시각화를 확인하세요.")
                    except Exception as e:
                        reply = f"⚠️ 계산 오류: {e}\n\n입력하신 수치를 다시 확인해 주세요."
                else:
                    if api_key:
                        ctx = None
                        if st.session_state.last_result:
                            r0 = st.session_state.last_result
                            ctx = f"마지막 계산: {r0['shape']}, F={r0['F']/1000:.2f}kN"
                        reply = chat_response(st.session_state.messages[:-1], msg_txt, provider, api_key, ctx)
                    else:
                        reply = ("⚠️ **API 키 미연결**\n\n수치 조건을 포함하면 API 없이도 계산 가능합니다.\n예: \"폭 3m 높이 5m 수위 4m\"")

            st.session_state.messages.append({'role':'assistant', 'content':reply, 'time':datetime.now().strftime('%H:%M')})
            st.rerun()

    with col_panel:
        st.markdown('<div class="section-label">📈 최근 계산 결과</div>', unsafe_allow_html=True)
        r = st.session_state.last_result
        if r:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">▸ {r['shape']} 정수력 분석</div>
                <div class="metric-grid">
                    <div class="metric-item"><div class="metric-label">정수력 F</div><div class="metric-value">{r['F']/1000:.3f}<span class="metric-unit">kN</span></div></div>
                    <div class="metric-item"><div class="metric-label">도심 깊이 h<sub>c</sub></div><div class="metric-value">{r['h_c']:.3f}<span class="metric-unit">m</span></div></div>
                    <div class="metric-item"><div class="metric-label">압력중심 y<sub>cp</sub></div><div class="metric-value">{r['y_cp']:.3f}<span class="metric-unit">m</span></div></div>
                    <div class="metric-item"><div class="metric-label">면적 A</div><div class="metric-value">{r['A']:.3f}<span class="metric-unit">m²</span></div></div>
                    <div class="metric-item"><div class="metric-label">편심거리 e</div><div class="metric-value">{r['y_cp']-r['h_c']:.4f}<span class="metric-unit">m</span></div></div>
                    <div class="metric-item"><div class="metric-label">I<sub>G</sub></div><div class="metric-value">{r.get('IG',0):.4f}<span class="metric-unit">m⁴</span></div></div>
                    <div class="metric-item"><div class="metric-label">전도 모멘트 M</div><div class="metric-value">{r.get('M_overturn',0)/1000:.3f}<span class="metric-unit">kN·m</span></div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            if r['shape_en'] == 'parabola':
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">▸ 3D 정수력 벡터 (곡면)</div>
                    <div class="metric-grid">
                        <div class="metric-item"><div class="metric-label">수평 F<sub>h</sub></div><div class="metric-value">{r['F_horizontal']/1000:.3f}<span class="metric-unit">kN</span></div></div>
                        <div class="metric-item"><div class="metric-label">수직 F<sub>v</sub></div><div class="metric-value">{r['F_vertical']/1000:.3f}<span class="metric-unit">kN</span></div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-label">▸ SymPy 기호 공식</div>', unsafe_allow_html=True)
            st.latex(f"A = {r['A_expr']}")
            st.latex(f"h_c = {r['hc_expr']}")
            st.latex(f"F = {r['F_expr']}")
            st.latex(f"y_{{cp}} = {r['ycp_expr']}")

            st.markdown(f"""
            <div class="info-box">
                📌 계산 {st.session_state.calc_count}회  |  γ = {GAMMA_WATER:,} N/m³  |
                API: {'✅' if st.session_state.api_status=='ok' else '❌ 미연결'}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:2rem;">
                <div style="font-size:1.4rem;margin-bottom:.5rem;">⌛</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;">
                    계산 대기 중<br>자연어 입력 또는<br>사이드바 슬라이더 사용
                </div>
            </div>""", unsafe_allow_html=True)


# ─── TAB 2: 3D 시각화 및 세부 결과 ───────────────────────────────────
with tab_calc:
    if calc_btn:
        with st.spinner("⚙ 계산 및 3D 렌더링 중..."):
            try:
                _run_calc(sel_shape, params)
            except Exception as e:
                st.error(f"계산 오류: {e}")

    r = st.session_state.last_result
    if r:
        shp = st.session_state.last_shape
        prm = st.session_state.last_params

        st.markdown('<div class="section-label">▸ 계산 결과 요약</div>', unsafe_allow_html=True)
        mc = st.columns(6)
        metrics = [
            (mc[0], "정수력 F",     f"{r['F']/1000:.3f}",        "kN"),
            (mc[1], "면적 A",       f"{r['A']:.3f}",              "m²"),
            (mc[2], "도심 h_c",     f"{r['h_c']:.3f}",            "m"),
            (mc[3], "압력중심 y_cp",f"{r['y_cp']:.3f}",           "m"),
            (mc[4], "편심 e",       f"{r['y_cp']-r['h_c']:.4f}",  "m"),
            (mc[5], "전도 M",       f"{r.get('M_overturn',0)/1000:.2f}", "kN·m"),
        ]
        
        for col, lbl, val, unit in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-item" style="text-align:center;padding:.6rem">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value" style="font-size:1.2rem">{val}</div>
                    <div class="metric-unit">{unit}</div>
                </div>""", unsafe_allow_html=True)

        if shp == 'parabola':
            st.markdown('<br>', unsafe_allow_html=True)
            pc = st.columns(3)
            for col, lbl, val, unit in [
                (pc[0], "수평 F_h", f"{r['F_horizontal']/1000:.3f}", "kN"), 
                (pc[1], "수직 F_v", f"{r['F_vertical']/1000:.3f}", "kN"), 
                (pc[2], "방향 θ", f"{r.get('angle_deg',0):.1f}", "°")
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-item" style="text-align:center;padding:.6rem">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="font-size:1.2rem">{val}</div>
                        <div class="metric-unit">{unit}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        vc, dc = st.columns([7, 3])
        
        with vc:
            st.markdown('<div class="section-label">▸ 3D 압력분포 & 합력 벡터</div>', unsafe_allow_html=True)
            img3 = st.session_state.get('img_3d')
            if img3: 
                st.image(img3, use_container_width=True)
            else:
                with st.spinner("🎨 3D 렌더링 중..."):
                    try:
                        img3 = plot_3d_hydrostatic(shp, prm, r)
                        st.session_state.img_3d = img3
                        st.image(img3, use_container_width=True)
                    except Exception as e:
                        st.error(f"3D 오류: {e}")

        with dc:
            st.markdown('<div class="section-label">▸ SymPy 기호 공식</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="result-card"><div class="result-title">▸ {r['shape']} 유도 공식</div></div>""", unsafe_allow_html=True)
            st.latex(f"A = {r['A_expr']}")
            st.latex(f"h_c = {r['hc_expr']}")
            st.latex(f"F = {r['F_expr']}")
            st.latex(f"y_{{cp}} = {r['ycp_expr']}")

            st.markdown(f"""
            <div class="result-card" style="margin-top:.45rem">
                <div class="result-title">▸ 수치 결과</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:.74rem;color:#c0d8e8;line-height:2.1">
                    A = {r['A']:.4f} m²<br>
                    h<sub>c</sub> = {r['h_c']:.4f} m<br>
                    <span style="color:#00c8ff;font-weight:bold">F = {r['F']/1000:.4f} kN</span><br>
                    y<sub>cp</sub> = {r['y_cp']:.4f} m<br>
                    e = {r['y_cp']-r['h_c']:.5f} m<br>
                    I<sub>G</sub> = {r.get('IG',0):.4f} m⁴<br>
                    M = {r.get('M_overturn',0)/1000:.4f} kN·m
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">▸ 정면도 · 압력 분포 (2D)</div>', unsafe_allow_html=True)
        img2 = st.session_state.get('img_2d')
        if img2:
            c1, c2, c3 = st.columns([1, 3, 1])
            with c2: st.image(img2, use_container_width=True)
        else:
            with st.spinner("📐 2D 도면 생성 중..."):
                try:
                    img2 = plot_2d_summary(shp, prm, r)
                    st.session_state.img_2d = img2
                    c1, c2, c3 = st.columns([1, 3, 1])
                    with c2: st.image(img2, use_container_width=True)
                except Exception as e:
                    st.error(f"2D 오류: {e}")
    else:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:3rem">
            <div style="font-size:1.8rem;margin-bottom:.7rem">📊</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.05rem;color:#00c8ff;margin-bottom:.35rem;letter-spacing:.1em">시각화 대기 중</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#3a5a6a">
                사이드바 슬라이더 → [▶ 계산 실행]<br>또는 챗봇에서 자연어 입력
            </div>
        </div>""", unsafe_allow_html=True)


# ─── TAB 3: 이론 & 수식 ───────────────────────────────────────────
with tab_theory:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="result-card"><div class="result-title">▸ 정수력 기본 공식</div></div>""", unsafe_allow_html=True)
        st.markdown("**정수압**")
        st.latex(r"p = \gamma \cdot h = \rho \cdot g \cdot h")
        
        st.markdown("**수직 평면 합력**")
        st.latex(r"F = \gamma \cdot h_c \cdot A")
        st.markdown("$h_c$ = 수면에서 도심까지의 수직거리")
        
        st.markdown("**압력중심 (Center of Pressure)**")
        st.latex(r"y_{cp} = h_c + \frac{I_G}{h_c \cdot A}")
        st.markdown("$I_G$ = 도심 기준 단면 2차 모멘트")
        
        st.markdown("**편심거리**")
        st.latex(r"e = y_{cp} - h_c = \frac{I_G}{h_c \cdot A} > 0")
        st.markdown("압력중심은 항상 도심 아래에 위치")
        
        st.markdown("**전도 모멘트**")
        st.latex(r"M_{overturn} = F \cdot y_{cp}")

        st.markdown("---")
        st.markdown("""<div class="result-card"><div class="result-title">▸ 곡면 정수력 (SciPy 수치 적분)</div></div>""", unsafe_allow_html=True)
        st.markdown("**수평성분**")
        st.latex(r"F_h = \gamma \cdot h_{c}' \cdot A_{proj}")
        st.markdown("$A_{proj}$ = 연직 투영 면적")
        
        st.markdown("**수직성분**")
        st.latex(r"F_v = \gamma \cdot V \quad \text{(SciPy quad 적분)}")
        st.markdown("$V$ = 수문 위 물의 체적")
        
        st.markdown("**합력**")
        st.latex(r"F = \sqrt{F_h^2 + F_v^2}, \quad \theta = \arctan\left(\frac{F_v}{F_h}\right)")

    with c2:
        st.markdown("""<div class="result-card"><div class="result-title">▸ 형상별 단면 특성 (SymPy 유도)</div></div>""", unsafe_allow_html=True)
        st.markdown("**직사각형**")
        st.latex(r"A = b \cdot H, \quad h_c = h_w + \frac{H}{2}, \quad I_G = \frac{bH^3}{12}")
        
        st.markdown("**삼각형**")
        st.latex(r"A = \frac{bH}{2}, \quad h_c = h_w + \frac{H}{3}, \quad I_G = \frac{bH^3}{36}")
        
        st.markdown("**사다리꼴**")
        st.latex(r"A = \frac{(b+B)H}{2}, \quad I_G = \frac{H^3(b^2+4bB+B^2)}{36(b+B)}")
        
        st.markdown("**원형**")
        st.latex(r"A = \pi R^2, \quad h_c = h_w, \quad I_G = \frac{\pi R^4}{4}")
        
        st.markdown("**반원형**")
        st.latex(r"A = \frac{\pi R^2}{2}, \quad h_c = h_w + \frac{4R}{3\pi}, \quad I_G = \left(\frac{\pi}{8} - \frac{8}{9\pi}\right)R^4")
        
        st.markdown("**포물선**")
        st.latex(r"A_{proj} = 2w \cdot H, \quad F = \sqrt{F_h^2 + F_v^2}")

        st.markdown("---")
        st.markdown("""<div class="result-card"><div class="result-title">▸ 댐 설계 체크리스트</div></div>""", unsafe_allow_html=True)
        st.markdown("""
- ✅ 압력중심 $y_{cp}$는 항상 도심 $h_c$ 아래
- ✅ 수위 상승 시 편심거리 $e$ 감소 (안정↑)
- ✅ 곡면 수문: $F_v$ = 수문 위 물 무게
- ✅ 힌지 위치: 압력중심 근방 권장
- ✅ 전도 안전율 = 저항모멘트/전도모멘트 ≥ 1.5
- ⚠️ 동수압(지진·파압) 별도 고려 필요
- ⚠️ 기초 지반 반력 및 양압력 검토 필수
- ⚠️ 수문 자중 및 마찰력 별도 계산
""")

    st.markdown("""
    <div class="info-box" style="display:flex;gap:2rem;flex-wrap:wrap;justify-content:center;padding:.7rem 1.4rem;margin-top:.8rem">
        <span><b style="color:#00c8ff">SymPy</b> — 기호 계산</span>
        <span><b style="color:#00ff9f">SciPy</b> — 수치 적분</span>
        <span><b style="color:#ff6b35">Matplotlib 3D</b> — 시각화</span>
        <span><b style="color:#00c8ff">Multi API</b> — LLM</span>
        <span><b style="color:#00ff9f">Streamlit</b> — UI</span>
    </div>
    """, unsafe_allow_html=True)