import io
import re
import streamlit as st
import os
from datetime import date
from typing import TypedDict
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()  # Carga las variables del archivo .env

# =============================================
# ESTADO COMPARTIDO DEL GRAFO (LangGraph)
# =============================================
class MarketingProposalState(TypedDict):
    """Estado compartido que fluye entre todos los agentes."""
    api_key: str
    client_name: str
    client_company: str
    client_industry: str
    client_need: str
    target_audience: str
    service_type: str
    service_details: str
    budget_range: str
    timeline: str
    sender_company: str
    sender_name: str
    sender_role: str
    sender_contact: str
    today: str
    # Outputs de los agentes (inicialmente vacíos "")
    market_analysis: str
    brand_strategy: str
    proposal_draft: str
    final_proposal: str

    # ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def _llm(api_key: str, max_tokens: int = 1500) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=max_tokens,
    )

# =============================================
# GENERADOR DE PDF
# =============================================
def _inline_md(text: str) -> str:
    """Convierte Markdown inline (**bold**, *italic*) a tags ReportLab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text

def generate_pdf(proposal_text: str, client_name: str) -> bytes:
    """Genera un PDF profesional a partir del texto Markdown de la propuesta."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25 * mm, leftMargin=25 * mm,
        topMargin=28 * mm,   bottomMargin=28 * mm,
        title=f"Propuesta Comercial — {client_name}",
    )

    base = getSampleStyleSheet()
    COLOR_DARK  = HexColor("#0f3460")
    COLOR_MID   = HexColor("#16213e")
    COLOR_LIGHT = HexColor("#1a2f4b")
    COLOR_RULE  = HexColor("#0f3460")

    sH1    = ParagraphStyle("sH1", parent=base["Heading1"],
                            fontSize=18, textColor=COLOR_DARK,
                            spaceBefore=20, spaceAfter=10, leading=22)
    sH2    = ParagraphStyle("sH2", parent=base["Heading2"],
                            fontSize=14, textColor=COLOR_MID,
                            spaceBefore=14, spaceAfter=7, leading=18)
    sH3    = ParagraphStyle("sH3", parent=base["Heading3"],
                            fontSize=11, textColor=COLOR_LIGHT,
                            spaceBefore=10, spaceAfter=5, leading=15)
    sBody  = ParagraphStyle("sBody", parent=base["Normal"],
                            fontSize=10, leading=16, spaceAfter=5)
    sBullet = ParagraphStyle("sBullet", parent=base["Normal"],
                             fontSize=10, leading=16, leftIndent=18,
                             firstLineIndent=0, spaceAfter=3)
    sNum   = ParagraphStyle("sNum", parent=sBullet, leftIndent=22)

    story = []
    for raw in proposal_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        if re.match(r"^[━\-=]{3,}$", line):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=COLOR_RULE, spaceAfter=6))
            continue
        text = _inline_md(line)
        if line.startswith("### "):
            story.append(Paragraph(_inline_md(line[4:]), sH3))
        elif line.startswith("## "):
            story.append(Paragraph(_inline_md(line[3:]), sH2))
        elif line.startswith("# "):
            story.append(Paragraph(_inline_md(line[2:]), sH1))
        elif re.match(r"^[-*] ", line):
            story.append(Paragraph("•  " + _inline_md(line[2:]), sBullet))
        elif re.match(r"^\d+\.\s", line):
            story.append(Paragraph(text, sNum))
        else:
            story.append(Paragraph(text, sBody))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ══════════════════════════════════════════════
# AGENTE 1 — Analista de Mercado y Cliente
# ══════════════════════════════════════════════
def agent_market_analyst(state: MarketingProposalState) -> dict:
    """Analiza al cliente, su mercado, pain points y oportunidades."""
    llm = _llm(state["api_key"])  # max_tokens=1500 por defecto
    messages = [
        SystemMessage(content="""Eres un analista senior de mercado y marketing B2B con 15 años de experiencia en Latinoamérica.
Tu misión es producir un diagnóstico estratégico que permita al siguiente agente construir una propuesta altamente personalizada.

Aplica el framework PEST + Pain-Gain-Fear:

**PEST del sector:**
- Político/Legal: regulaciones o tendencias que afectan al sector
- Económico: presupuesto típico de marketing en ese sector y ciclo de venta
- Social: comportamiento y expectativas del consumidor final del cliente
- Tecnológico: madurez digital del sector y herramientas que ya usan

**Perfil Pain-Gain-Fear del decisor:**
- Pain: 3 obstáculos de marketing que enfrenta HOY (específicos, no genéricos)
- Gain: 3 beneficios concretos que espera obtener del servicio
- Fear: 2 riesgos percibidos al contratar una agencia externa

**Oportunidad estratégica:**
- 1 insight diferenciador que la propuesta debe capitalizar
- Posicionamiento actual frente a 1-2 competidores o referentes del sector

Formato: subtítulos en negrita + viñetas. Máximo 500 palabras. Todo en español."""),
        HumanMessage(content=f"""Analiza este cliente potencial de marketing usando PEST + Pain-Gain-Fear:

━━━ DATOS DEL CLIENTE ━━━
Empresa: {state['client_company']}
Sector / industria: {state['client_industry'] or 'No especificado — infiere del contexto'}
Contacto / decisor: {state['client_name']}
Necesidad de marketing declarada: {state['client_need'] or 'No especificada'}
Público objetivo del cliente: {state['target_audience'] or 'No especificado'}
Servicio de marketing de interés: {state['service_type']}

Genera el análisis completo. Sé específico para ESTE cliente, no genérico."""),
    ]
    response = llm.invoke(messages)
    return {"market_analysis": response.content}  # Solo escribe su clave

# ══════════════════════════════════════════════
# AGENTE 2 — Estratega de Marca y Marketing
# ══════════════════════════════════════════════
def agent_brand_strategist(state: MarketingProposalState) -> dict:
    """Define mensajes clave, propuesta de valor y estrategia persuasiva."""
    llm = _llm(state["api_key"])  # max_tokens=1500 por defecto
    messages = [
        SystemMessage(content="""Eres un estratega de marca y marketing B2B con 12 años de experiencia en agencias digitales de Latinoamérica.
Tu misión es construir la arquitectura estratégica que el redactor usará para escribir la propuesta.

Aplica el Messaging Hierarchy Canvas:

**Core Message (1 frase):**
- La promesa principal del servicio para ESTE cliente específico
- En términos de resultado de negocio, no de característica del servicio

**3 Proof Points (evidencias que sostienen el Core Message):**
- Proof 1: Dato o tendencia del sector que valida la urgencia
- Proof 2: Metodología o expertise de la agencia que diferencia
- Proof 3: Garantía o mecanismo de rendición de cuentas (KPIs)

**KPIs de Éxito (3-5, de negocio):**
- Métricas que el cliente valorará como evidencia de resultados
- Priorizar: conversión, costo por lead, ROI, tasa de retención (según servicio)
- Evitar métricas de vanidad (likes, seguidores) a menos que sean el núcleo del servicio

**Manejo de Objeciones (top 3 del sector):**
- Objeción → Contraargumento con dato o lógica persuasiva

**Tono Recomendado:**
- Describir el registro comunicacional ideal para el decisor identificado

Formato: subtítulos en negrita + viñetas. Máximo 520 palabras. Todo en español."""),
        HumanMessage(content=f"""Con base en este análisis de mercado y cliente:

━━━ ANÁLISIS DE MERCADO (PEST + Pain-Gain-Fear) ━━━
{state['market_analysis']}

━━━ DATOS DEL SERVICIO ━━━
Servicio: {state['service_type']}
Descripción: {state['service_details'] or 'A detallar en kick-off'}
Inversión: {state['budget_range'] or 'A convenir'}
Duración: {state['timeline'] or 'A convenir'}
Agencia: {state['sender_company'] or 'Nuestra agencia'}

Define la estrategia de marca y marketing aplicando el Messaging Hierarchy Canvas.
El Core Message debe ser específico para {state['client_company']}, no genérico."""),
    ]
    response = llm.invoke(messages)
    return {"brand_strategy": response.content}

# ══════════════════════════════════════════════
# AGENTE 3 — Redactor de Propuesta Comercial
# ══════════════════════════════════════════════
def agent_proposal_writer(state: MarketingProposalState) -> dict:
    """Redacta la propuesta completa usando el framework StoryBrand adaptado."""
    llm = _llm(state["api_key"], max_tokens=3500)  # Más tokens para propuesta completa
    messages = [
        SystemMessage(content="""Eres un redactor comercial senior especializado en propuestas de marketing B2B para empresas latinoamericanas.
Tu misión es integrar el análisis y la estrategia en una narrativa persuasiva de 11 secciones que motive al decisor a contratar.

Aplica el framework StoryBrand adaptado:
1. Personaje → El cliente como protagonista con su situación actual
2. Problema → El obstáculo de marketing específico que enfrenta
3. Guía → La agencia como experto que ya resolvió esto antes
4. Plan → El servicio en fases claras y ejecutables
5. CTA → Próximos pasos concretos con verbo imperativo
6. Éxito → Resultados esperados con KPIs de negocio medibles

Reglas de redacción:
- El nombre del cliente aparece al menos 3 veces integrado naturalmente
- Los beneficios son tangibles y cuantificables (evitar "mejoraremos su presencia")
- Cada sección prepara lógicamente la siguiente
- Tono consultivo: enfocado en el cliente, no en la agencia
- Mínimo 850 palabras en el cuerpo de la propuesta
- Usa Markdown limpio: ## para secciones, **bold** para énfasis, - para listas
- Todo en español."""),
        HumanMessage(content=f"""Redacta una propuesta comercial de marketing completa usando:

━━━ ANÁLISIS DE MERCADO Y CLIENTE ━━━
{state['market_analysis']}

━━━ ESTRATEGIA DE MARCA Y MARKETING ━━━
{state['brand_strategy']}

━━━ DATOS ESPECÍFICOS ━━━
- Empresa cliente: {state['client_company']}
- Contacto: {state['client_name']}
- Servicio de marketing: {state['service_type']}
- Descripción: {state['service_details'] or 'A detallar en reunión de kick-off'}
- Inversión: {state['budget_range'] or 'A convenir'}
- Duración: {state['timeline'] or 'A convenir'}
- Remitente: {state['sender_name'] or ''}{(', ' + state['sender_role']) if state['sender_role'] else ''} — {state['sender_company'] or ''}
- Contacto remitente: {state['sender_contact'] or ''}
- Fecha: {state['today']}

La propuesta DEBE incluir TODAS estas secciones en orden:
1. Encabezado formal (remitente, destinatario, fecha, N° propuesta)
2. Resumen Ejecutivo
3. Entendimiento del Cliente y su Mercado
4. Solución de Marketing Propuesta
5. Por Qué Esta Estrategia Funciona Para {state['client_company']}
6. Metodología y Plan de Acción (fases y actividades)
7. Entregables y Resultados Esperados (con KPIs)
8. Cronograma de Implementación
9. Inversión y Condiciones de Pago
10. Próximos Pasos (llamada a la acción)
11. Cierre y Firma"""),
    ]
    response = llm.invoke(messages)
    return {"proposal_draft": response.content}

# ══════════════════════════════════════════════
# AGENTE 4 — Editor Senior y Revisor de Calidad
# ══════════════════════════════════════════════
def agent_editor(state: MarketingProposalState) -> dict:
    """Revisa y pule la propuesta aplicando la rúbrica PACT."""
    llm = _llm(state["api_key"], max_tokens=4096)  # Máximo para devolver propuesta completa
    messages = [
        SystemMessage(content="""Eres un editor senior de comunicación comercial con 18 años de experiencia en marketing B2B latinoamericano.
Tu misión es elevar el borrador a una propuesta lista para enviar, aplicando la rúbrica PACT:

**P — Personalización:**
¿Parece escrita para ESTE cliente específico? El sector, el pain point y el nombre deben estar integrados naturalmente.

**A — Accionabilidad:**
¿El cliente sabe exactamente qué hacer después de leerla? Los Próximos Pasos deben tener verbo imperativo y un plazo sugerido.

**C — Claridad:**
¿Cada sección puede leerse en 60 segundos? Elimina frases vagas, jerga innecesaria y superlativos vacíos.

**T — Tono:**
¿Es consultivo (orientado al cliente) o vendedor (orientado a la agencia)? Ajusta hacia consultivo.

Checklist final antes de entregar:
- Las 11 secciones están presentes y en orden
- Los KPIs son de negocio (no solo de vanidad)
- La inversión tiene condiciones de pago explícitas
- El cronograma usa fases relativas (Semana 1, Mes 2...)
- El Markdown es consistente: ## secciones, **bold** énfasis, - listas

REGLA CRÍTICA: Devuelve la propuesta COMPLETA y mejorada. NUNCA recortes contenido."""),
        HumanMessage(content=f"""Revisa y mejora esta propuesta de marketing aplicando la rúbrica PACT:

{state['proposal_draft']}

━━━ CRITERIOS DE REVISIÓN ━━━
1. **Personalización:** ¿{state['client_company']} y su sector están integrados naturalmente en cada sección?
2. **Beneficios:** ¿Son concretos, medibles y relevantes para {state['client_industry'] or 'su sector'}?
3. **Estrategia integrada:** ¿Los mensajes clave del Estratega están bien reflejados en la propuesta?
4. **Tono consultivo:** ¿El lenguaje está orientado al cliente, no a la agencia?
5. **CTA activo:** ¿"Próximos Pasos" tiene un verbo imperativo y un plazo sugerido?
6. **Markdown limpio:** ¿El formato es consistente y profesional?

Entrega la versión final COMPLETA, lista para enviar a {state['client_name']} de {state['client_company']}."""),
    ]
    response = llm.invoke(messages)
    return {"final_proposal": response.content}

# ══════════════════════════════════════════════
# CONSTRUCCIÓN DEL GRAFO LANGGRAPH
# ══════════════════════════════════════════════
def build_graph():
    """Construye y compila el grafo de agentes con LangGraph."""
    graph = StateGraph(MarketingProposalState)

    # Registrar los 4 nodos (agentes)
    graph.add_node("analyst",    agent_market_analyst)
    graph.add_node("strategist", agent_brand_strategist)
    graph.add_node("writer",     agent_proposal_writer)
    graph.add_node("editor",     agent_editor)

    # Definir el nodo de entrada
    graph.set_entry_point("analyst")

    # Definir el flujo: analyst → strategist → writer → editor → FIN
    graph.add_edge("analyst",    "strategist")
    graph.add_edge("strategist", "writer")
    graph.add_edge("writer",     "editor")
    graph.add_edge("editor",     END)

    return graph.compile()  # Congela y compila el grafo


# Constantes para la UI de Streamlit
AGENT_LABELS = {
    "analyst":    "🔍 Agente 1: Analista de Mercado",
    "strategist": "📊 Agente 2: Estratega de Marca",
    "writer":     "✍️ Agente 3: Redactor Comercial",
    "editor":     "✅ Agente 4: Editor Senior",
}
AGENT_DESCRIPTIONS = {
    "analyst":    "Analiza al cliente, su mercado, pain points y oportunidades.",
    "strategist": "Define mensajes clave, propuesta de valor y estrategia persuasiva.",
    "writer":     "Redacta la propuesta completa con todas sus secciones formales.",
    "editor":     "Revisa, mejora y pule la propuesta para máximo impacto.",
}
OUTPUT_KEYS = {
    "analyst":    "market_analysis",
    "strategist": "brand_strategy",
    "writer":     "proposal_draft",
    "editor":     "final_proposal",
}

# ──────────────────────────────────────────────
# Configuración de página (DEBE IR PRIMERO)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Equipo de Agentes de Marketing | LangGraph + Claude",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
        color: white; padding: 30px 40px; border-radius: 12px;
        text-align: center; margin-bottom: 30px;
    }
    .main-title h1 { color: white; margin: 0 0 10px 0; font-size: 2em; }
    .main-title p  { color: #a8d8ea; margin: 0; font-size: 1.05em; }
    .agent-card {
        background: #f8f9fa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    }
    .agent-card h4 { color: #0f3460; margin: 0 0 4px 0; font-size: 0.95em; }
    .agent-card p  { color: #666; margin: 0; font-size: 0.82em; }
    .proposal-box {
        background: #f9fafb; border: 1px solid #d0d7de;
        border-radius: 10px; padding: 30px 35px; line-height: 1.8; font-size: 1em;
    }
    .info-card {
        background: #e8f4f8; border-left: 5px solid #0f3460;
        border-radius: 6px; padding: 14px 20px; margin-bottom: 20px;
        font-size: 0.92em; color: #1a2f4b;
    }
    .success-banner {
        background: #d4edda; border-left: 5px solid #27ae60;
        border-radius: 6px; padding: 14px 20px; color: #155724;
        font-weight: 500; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
<div class="main-title">
    <h1>🎯 Equipo de Agentes de Marketing</h1>
    <p>4 agentes especializados orquestados con <strong>LangGraph</strong> + <strong>Claude de Anthropic</strong><br>
    trabajan en cadena para generar tu propuesta comercial de marketing de alta calidad.</p>
</div>
""", unsafe_allow_html=True)

# Barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input(
        "API Key de Anthropic",
        type="password",
        placeholder="sk-ant-...",
        help="Ingresa tu API Key o defínela en el archivo .env",
    )
    st.markdown("---")
    st.markdown("**🤖 Equipo de Agentes:**")
    for key, label in AGENT_LABELS.items():
        st.markdown(
            f'<div class="agent-card"><h4>{label}</h4><p>{AGENT_DESCRIPTIONS[key]}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("**Orquestación:** LangGraph  \n**Modelo:** claude-haiku-4-5-20251001  \n**Flujo:** Analyst → Strategist → Writer → Editor")
    st.caption("2026 - Ing. Julian Andres Quimbayo Castro")

    st.subheader("📋 Información para la Propuesta de Marketing")
st.markdown(
    '<div class="info-card">Completa los datos del cliente y el servicio. '
    "El equipo de agentes usará esta información para generar una propuesta estratégica y personalizada.</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Datos del Cliente")
    client_name     = st.text_input("Nombre del contacto *", placeholder="Ej: María García")
    client_company  = st.text_input("Nombre de la empresa *", placeholder="Ej: Tech Solutions S.A.S.")
    client_industry = st.text_input("Sector o industria", placeholder="Ej: Tecnología, Salud, Retail...")
    target_audience = st.text_input("Público objetivo del cliente",
                                    placeholder="Ej: Mujeres de 25-40 años, profesionales urbanas")
    client_need     = st.text_area("Problema o necesidad de marketing",
                                   placeholder="Ej: Poca visibilidad online, redes sociales inactivas...",
                                   height=100)

with col2:
    st.markdown("#### 🎯 Servicio de Marketing")
    service_type = st.selectbox(
        "Tipo de servicio *",
        options=[
            "Selecciona un servicio...",
            "Gestión de Redes Sociales",
            "Estrategia de Marketing Digital",
            "Branding e Identidad de Marca",
            "Campaña Publicitaria (Meta / Google Ads)",
            "Posicionamiento SEO / SEM",
            "Marketing de Contenidos",
            "Email Marketing y Automatización",
            "Diseño Web y Landing Pages",
            "Consultoría de Marketing",
            "Plan de Marketing Integral",
        ],
    )
    service_details = st.text_area("Descripción del servicio",
                                   placeholder="Ej: Gestión mensual de Instagram y Facebook...",
                                   height=100)
    budget_range = st.text_input("Inversión mensual / total",
                                  placeholder="Ej: $2.500.000 COP / mes")
    timeline     = st.text_input("Duración del proyecto",
                                  placeholder="Ej: Contrato inicial de 3 meses")

st.markdown("#### 🏢 Tu Agencia / Empresa (quien envía la propuesta)")
col3, col4 = st.columns(2)
with col3:
    sender_company = st.text_input("Nombre de tu agencia", placeholder="Ej: CreativeHub Colombia")
    sender_name    = st.text_input("Tu nombre completo",   placeholder="Ej: Carlos Rodríguez")
with col4:
    sender_role    = st.text_input("Tu cargo",             placeholder="Ej: Director de Cuentas")
    sender_contact = st.text_input("Correo / Teléfono",    placeholder="Ej: carlos@creativehub.co")

st.markdown("---")

generate_btn = st.button("🚀 Activar Equipo de Agentes", type="primary", use_container_width=True)

if generate_btn:
    # Validar campos obligatorios
    missing = []
    if not client_name:    missing.append("Nombre del contacto")
    if not client_company: missing.append("Nombre de la empresa")
    if service_type == "Selecciona un servicio...": missing.append("Tipo de servicio")

    if missing:
        st.error(f"⚠️ Completa los campos obligatorios (*): {', '.join(missing)}")
    else:
        api_key = api_key_input.strip() or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.error("⚠️ Ingresa tu API Key en la barra lateral o defínela en el archivo `.env`.")
        else:
            # Construir el estado inicial del grafo
            initial_state: MarketingProposalState = {
                "api_key":        api_key,
                "client_name":    client_name,
                "client_company": client_company,
                "client_industry": client_industry,
                "client_need":    client_need,
                "target_audience": target_audience,
                "service_type":   service_type,
                "service_details": service_details,
                "budget_range":   budget_range,
                "timeline":       timeline,
                "sender_company": sender_company,
                "sender_name":    sender_name,
                "sender_role":    sender_role,
                "sender_contact": sender_contact,
                "today":          date.today().strftime("%d de %B de %Y"),
                "market_analysis": "",
                "brand_strategy":  "",
                "proposal_draft":  "",
                "final_proposal":  "",
            }

            graph = build_graph()
            final_proposal = ""

            try:
                # Ejecutar el grafo y mostrar el progreso en tiempo real
                with st.status("🤖 Equipo de agentes de marketing trabajando...", expanded=True) as main_status:
                    for step in graph.stream(initial_state):
                        for node_name, node_output in step.items():
                            if node_name not in AGENT_LABELS:
                                continue
                            content = node_output.get(OUTPUT_KEYS[node_name], "")
                            st.markdown(f"**{AGENT_LABELS[node_name]}** — ✅ Completado")
                            with st.expander(f"Ver output → {AGENT_LABELS[node_name]}", expanded=False):
                                st.markdown(content)
                            if node_name == "editor":
                                final_proposal = content

                    main_status.update(label="✅ ¡El equipo de agentes completó la propuesta!", state="complete")

                if final_proposal:
                    st.session_state["proposal"]        = final_proposal
                    st.session_state["proposal_client"] = client_company

            except Exception as e:
                err = str(e).lower()
                if "authentication" in err or "api_key" in err or "invalid" in err:
                    st.error("❌ API Key inválida. Verifica en console.anthropic.com.")
                elif "rate" in err or "quota" in err:
                    st.error("❌ Límite de uso alcanzado. Espera unos minutos e intenta de nuevo.")
                else:
                    st.error(f"❌ Error inesperado: {e}")



if "proposal" in st.session_state:
    st.markdown("---")
    st.markdown(
        '<div class="success-banner">✅ ¡Propuesta generada por el equipo de agentes! Revísala, edítala o descárgala.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📄 Vista Formateada", "📝 Texto Plano (copiar / editar)"])

    with tab1:
        st.markdown('<div class="proposal-box">', unsafe_allow_html=True)
        st.markdown(st.session_state["proposal"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.text_area(
            "Propuesta en texto plano:",
            value=st.session_state["proposal"],
            height=600,
            label_visibility="collapsed",
        )

    filename = (
        "propuesta_marketing_"
        + st.session_state.get("proposal_client", "cliente").replace(" ", "_").lower()
        + ".md"
    )
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇️ Descargar Propuesta (.md)",
            data=st.session_state["proposal"].encode("utf-8"),
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )
    with col_dl2:
        pdf_bytes = generate_pdf(
            st.session_state["proposal"],
            st.session_state.get("proposal_client", "cliente"),
        )
        st.download_button(
            label="⬇️ Descargar Propuesta (.pdf)",
            data=pdf_bytes,
            file_name=filename.replace(".md", ".pdf"),
            mime="application/pdf",
            use_container_width=True,
        )