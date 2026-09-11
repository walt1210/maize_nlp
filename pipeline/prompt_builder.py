"""
Builds the system prompt and Chain-of-Thought user prompt sent to Gemini.
No programmatic XAI text description is generated here — per the project's
Design Decision favoring multimodal grounding, Gemini is shown all three
images directly (original photo, segmentation boundary, XAI heatmap) and
reasons from them, rather than consuming a text proxy.
"""
import config

SYSTEM_PROMPT = """\
You are an expert plant pathologist and agricultural extension officer
advising Filipino maize farmers. You have deep knowledge of Maize Streak
Virus (MSV) and Maize Lethal Necrosis (MLN) as they affect Zea mays L.
in the Philippine context.

Your role is to convert computer vision diagnostic outputs into clear,
actionable, culturally appropriate guidance that a smallholder farmer
or agriculture student can immediately act on.

RULES:
1. Base all advice ONLY on the retrieved agricultural knowledge provided.
   Never generate management protocols from memory alone.
2. Always reference specific morphological markers visible in the attached
   images when writing the justification.
3. Severity Stage 3 (>60%) requires immediate rouging instructions.
4. NEVER state specific pesticide dosages, application rates, product
   quantities, or mixing ratios (e.g. "500g per 100kg seed", "2L per
   hectare"). This system is for educational and field-support purposes
   only and does not prescribe regulated agricultural interventions.
   When chemical control is relevant, name only the general product
   CATEGORY (e.g. "a DA-registered systemic insecticide seed treatment")
   and explicitly instruct the farmer to consult a licensed agriculturist,
   DA extension officer, or the product's own label for the correct
   dosage and application rate before use.
5. Tagalog translations must preserve clinical safety — never simplify
   a safety instruction to the point of ambiguity.
6. If a low-confidence notice is present in the diagnostic inputs, open
   the justification by noting the classification is uncertain and
   recommend the farmer confirm with a DA extension officer before
   acting on chemical control recommendations specifically.
7. Respond ONLY in valid JSON. No preamble, no markdown backticks.
"""

# Used for /chat, NOT /diagnose — shares the same persona/expertise
# framing as SYSTEM_PROMPT above, but deliberately WITHOUT rule 7
# ("respond only in valid JSON"). Reusing SYSTEM_PROMPT for chat caused
# a real bug: that system-level JSON instruction conflicted with the
# per-message "respond in plain text" instruction in
# conversation_manager.py's chat prompt, and Gemini would inconsistently
# honor the (wrong, JSON-demanding) system instruction instead —
# producing raw/truncated JSON leaking into what should be a natural
# conversational reply.
CHAT_SYSTEM_PROMPT = """\
You are an expert plant pathologist and agricultural extension officer
advising Filipino maize farmers. You have deep knowledge of Maize Streak
Virus (MSV) and Maize Lethal Necrosis (MLN) as they affect Zea mays L.
in the Philippine context.

You are having a follow-up conversation with a farmer about a diagnosis
they already received. Answer their questions clearly and conversationally.

RULES:
1. Base all advice ONLY on the diagnosis context and retrieved agricultural
   knowledge already provided. Never invent management protocols from memory.
2. NEVER state specific pesticide dosages, application rates, product
   quantities, or mixing ratios. This system is for educational and
   field-support purposes only and does not prescribe regulated
   agricultural interventions. When chemical control is relevant, name
   only the general product CATEGORY and explicitly instruct the farmer
   to consult a licensed agriculturist, DA extension officer, or the
   product's own label for the correct dosage before use.
3. Tagalog responses must preserve clinical safety — never simplify a
   safety instruction to the point of ambiguity.
4. Stay strictly within the scope of this diagnosis and maize streak
   diseases (MSV, MLN).
5. Respond with PLAIN CONVERSATIONAL TEXT ONLY — never JSON, never
   markdown code blocks, never a structured object. Just write like
   you're talking directly to the farmer.
"""

_COT_TEMPLATE = """\
DIAGNOSTIC INPUTS (from MAIze computer vision model):
- Disease classification : {classification}  (confidence: {confidence_pct})
- Infected tissue area   : {severity_pct:.1f}%
- Severity grade         : {grade_label}  (Grade {cimmyt_grade})
- Monitoring stage       : Stage {monitoring_stage} / 3
{low_confidence_notice}

[ATTACHED IMAGE 1 - Original leaf photograph]
[ATTACHED IMAGE 2 - Symptom boundary overlay: a crisp green contour from
                    the model's segmentation head, showing the precise
                    pixel-level extent of visible symptoms on the leaf]
[ATTACHED IMAGE 3 - {xai_method_name} diagnostic attention overlay:
                    amber/red regions = high model attention,
                    blue regions = low model attention. This is a
                    DIFFERENT thing from Image 2 — it shows which regions
                    drove the model's classification decision, not the
                    exact symptom boundary]

RETRIEVED EXPERT KNOWLEDGE (from verified agricultural sources):
{rag_context}

CHAIN-OF-THOUGHT REASONING - think step by step before responding:
Step 1: Looking at all three attached images, what specific visual markers
         are present on the leaf? Does the segmentation boundary (Image 2)
         match where the {xai_method_name} attention (Image 3) is
         concentrated, or do they diverge? What pattern do they form together?
Step 2: Why does this pattern indicate {classification} at this severity?
         What distinguishes it from other conditions (nutrient deficiency,
         other diseases)?
Step 3: Given Grade {cimmyt_grade} and Stage {monitoring_stage}, what is
         the urgency level? What happens if no action is taken?
Step 4: What are the most critical immediate actions for Stage {monitoring_stage}?
Step 5: What spread prevention measures are most important in the
         Philippine field context?

Now generate your response as a JSON object with this exact structure:
{{
  "justification": "2-3 sentences explaining WHY this classification was made, referencing the specific visual evidence you see in the attached images.",
  "xai_explanation": "1-2 sentences describing what the segmentation boundary and the {xai_method_name} heatmap each show, and what their agreement or divergence means diagnostically.",
  "key_fact": "One critical fact the farmer must know.",
  "immediate_actions": ["action1", "action2", "action3"],
  "management": ["step1", "step2", "step3", "step4"],
  "spread": "How this disease spreads (vector, mechanism, distance).",
  "distances": "Specific recommended distances for rouging/isolation.",
  "prevention": ["measure1", "measure2", "measure3"],
  "precautions": ["precaution1", "precaution2"],
  "detection": "How to monitor for progression or new infections.",
  "control": {{
    "chemical": "General DA-registered product CATEGORY only (e.g. 'a systemic insecticide seed treatment'). Do NOT state a specific dosage, application rate, or mixing ratio -- instead explicitly direct the farmer to consult a licensed agriculturist, DA extension officer, or the product label for the correct amount to use.",
    "biological": "Biological control options if available.",
    "cultural": "Cultural practices (crop rotation, planting schedule, etc.)"
  }},
  "protocol_title": "MSV/MLN Rapid Response Protocol",
  "protocol_steps": ["step1", "step2", "step3", "step4"]{tagalog_field}
}}
"""

# Appended into the schema only when include_tagalog=True. Previously
# this block was baked permanently into _COT_TEMPLATE, so Gemini generated
# a full parallel Tagalog translation of every field on EVERY /diagnose
# call regardless of the include_tagalog flag — the exact per-call output
# token cost the project-state notes describe as already fixed. It
# wasn't: build_cot_prompt() didn't even accept an include_tagalog
# parameter until this change, so the flag never reached the template.
_TAGALOG_SCHEMA_FIELD = """,
  "tagalog": {
    "justification": "...",
    "key_fact": "...",
    "protocol_steps": ["...", "...", "...", "..."],
    "immediate_actions": ["...", "...", "..."]
  }"""


def build_low_confidence_notice(confidence: float) -> str:
    if confidence < config.LOW_CONFIDENCE_THRESHOLD:
        return (
            f"- \u26a0 LOW CONFIDENCE ({confidence:.1%}): classification is "
            f"uncertain. Guidance must open with a caveat recommending the "
            f"farmer confirm with a DA extension officer before acting, "
            f"especially on chemical control."
        )
    return ""


def build_cot_prompt(
    classification: str,
    confidence: float,
    severity_pct: float,
    cimmyt_grade: int,
    monitoring_stage: int,
    grade_label: str,
    rag_context: str,
    include_tagalog: bool = False,
) -> str:
    return _COT_TEMPLATE.format(
        classification=classification,
        confidence_pct=f"{confidence:.1%}",
        severity_pct=severity_pct,
        grade_label=grade_label,
        cimmyt_grade=cimmyt_grade,
        monitoring_stage=monitoring_stage,
        low_confidence_notice=build_low_confidence_notice(confidence),
        rag_context=rag_context or "No specific retrieved context available.",
        xai_method_name=config.XAI_METHOD_DISPLAY_NAME,
        tagalog_field=_TAGALOG_SCHEMA_FIELD if include_tagalog else "",
    )
