"""
Pre-generated static guidance, used when Gemini is unreachable or its
output fails validation. Keyed by (classification, monitoring_stage).
Content here is intentionally generic and conservative — it is NOT
RAG-grounded or per-leaf specific, since it has to work with zero
network access. Review with a licensed agriculturist before shipping;
this is a reasonable starting draft, not a substitute for that review.
"""

OFFLINE_GUIDANCE = {
    ("HEALTHY", 0): {
        "justification": "No disease markers were detected on this leaf image.",
        "key_fact": "Regular scouting helps catch infections early, when they are easiest to manage.",
        "immediate_actions": ["No treatment needed at this time.", "Continue routine field monitoring."],
        "management": ["Maintain normal crop care and irrigation schedule."],
        "spread": "Not applicable — no infection detected.",
        "distances": "Not applicable.",
        "prevention": [
            "Use certified, disease-free seed for future planting.",
            "Rotate maize with non-host crops where possible.",
            "Scout weekly for early streak or mottling symptoms.",
        ],
        "precautions": ["Wash hands and tools after handling any plants suspected of disease."],
        "detection": "Inspect leaves weekly for chlorotic streaking (MSV) or mottling/necrosis (MLN).",
        "control": {"chemical": "Not applicable.", "biological": "Not applicable.", "cultural": "Not applicable."},
        "protocol_title": "Routine Monitoring",
        "protocol_steps": [
            "Continue weekly visual inspection.",
            "Record any new symptoms with photos.",
            "Re-scan with the MAIze app if symptoms appear.",
        ],
        "tagalog": {
            "justification": "Walang natukoy na senyales ng sakit sa larawan ng dahong ito.",
            "key_fact": "Ang regular na pagsusuri ay tumutulong makita agad ang impeksyon habang madali pang gamutin.",
            "protocol_steps": [
                "Ipagpatuloy ang lingguhang pagsusuri.",
                "Itala ang anumang bagong senyales kasama ang larawan.",
                "Mag-scan muli gamit ang MAIze app kung may lumitaw na senyales.",
            ],
            "immediate_actions": ["Walang kailangang gamutin sa ngayon.", "Ipagpatuloy ang regular na pagbabantay."],
        },
    },
    ("MSV", 1): {
        "justification": "Trace-level chlorotic streaking consistent with early-stage Maize Streak Virus was detected.",
        "key_fact": "MSV is spread by leafhoppers (Cicadulina spp.) — controlling the vector limits further spread.",
        "immediate_actions": [
            "Mark and monitor the affected plant closely.",
            "Scout neighboring plants for early streak symptoms.",
            "Begin leafhopper vector monitoring.",
        ],
        "management": [
            "Isolate monitoring focus on the affected plant and its immediate neighbors.",
            "Avoid working wet foliage of infected plants before healthy ones (mechanical spread risk).",
            "Continue weekly re-assessment of severity.",
        ],
        "spread": "MSV spreads via leafhopper feeding, not by direct plant-to-plant contact or wind.",
        "distances": "No rouging required at trace level; continue close monitoring instead.",
        "prevention": [
            "Plant resistant/tolerant maize varieties where available.",
            "Time planting to avoid peak leafhopper activity in your area.",
            "Remove grassy weeds nearby that host leafhoppers.",
        ],
        "precautions": ["Wash hands and tools after handling symptomatic plants."],
        "detection": "Re-inspect weekly; escalate to management protocol if streaking spreads beyond 10% of leaf area.",
        "control": {
            "chemical": "Consult your local DA extension officer for currently registered insecticides targeting leafhoppers.",
            "biological": "Encourage natural leafhopper predators (spiders, predatory bugs) by minimizing broad-spectrum spraying.",
            "cultural": "Maintain field sanitation; remove volunteer maize and grassy weeds.",
        },
        "protocol_title": "MSV Early-Stage Monitoring Protocol",
        "protocol_steps": [
            "Mark affected plant(s) for weekly tracking.",
            "Scout a 5-meter radius for additional cases.",
            "Begin vector (leafhopper) monitoring.",
            "Re-scan with the MAIze app in 7 days.",
        ],
        "tagalog": {
            "justification": "Natukoy ang unang antas ng streaking na kaugnay ng maagang Maize Streak Virus.",
            "key_fact": "Ang MSV ay kumakalat sa pamamagitan ng leafhopper — ang pagkontrol dito ay tumutulong pigilan ang paglaganap.",
            "protocol_steps": [
                "Markahan ang apektadong halaman para sa lingguhang pagsubaybay.",
                "Suriin ang 5-metrong paligid para sa karagdagang kaso.",
                "Simulan ang pagmomonitor ng leafhopper.",
                "Mag-scan muli gamit ang MAIze app pagkalipas ng 7 araw.",
            ],
            "immediate_actions": [
                "Markahan at bantayan nang mabuti ang apektadong halaman.",
                "Suriin ang kalapit na halaman para sa maagang senyales.",
                "Simulan ang pagmomonitor ng vector.",
            ],
        },
    },
    ("MSV", 2): {
        "justification": "Light-to-moderate chlorotic streaking consistent with progressing Maize Streak Virus was detected.",
        "key_fact": "Once streaking exceeds 25% of leaf area, yield impact becomes significant if untreated.",
        "immediate_actions": [
            "Apply the MSV management protocol immediately.",
            "Begin active leafhopper vector control.",
            "Increase scouting frequency to twice weekly.",
        ],
        "management": [
            "Apply vector control measures per DA-registered product guidance.",
            "Continue close monitoring of severity progression.",
            "Prepare for possible rouging if severity worsens.",
        ],
        "spread": "MSV spreads via leafhopper feeding; higher plant density and nearby infected volunteers increase risk.",
        "distances": "Monitor a 10-meter radius around confirmed cases; isolate severely affected plants if progression continues.",
        "prevention": [
            "Plant resistant/tolerant varieties in future seasons.",
            "Adjust planting schedule to avoid peak vector activity.",
            "Remove alternate host weeds near the field.",
        ],
        "precautions": ["Disinfect tools between handling infected and healthy plants."],
        "detection": "Re-scan every 3-4 days at this stage to track progression toward Stage 3.",
        "control": {
            "chemical": "Consult your local DA extension officer for currently registered leafhopper insecticides and application rates.",
            "biological": "Support natural predators; avoid unnecessary broad-spectrum insecticide use.",
            "cultural": "Remove nearby volunteer maize and grassy weeds that host leafhoppers.",
        },
        "protocol_title": "MSV Moderate-Stage Rapid Response Protocol",
        "protocol_steps": [
            "Apply vector control per DA-registered guidance.",
            "Increase scouting to twice weekly.",
            "Monitor 10-meter radius for spread.",
            "Prepare rouging plan in case of progression.",
        ],
        "tagalog": {
            "justification": "Natukoy ang katamtamang streaking na kaugnay ng umuusbong na Maize Streak Virus.",
            "key_fact": "Kapag lumagpas sa 25% ang apektadong lugar ng dahon, malaki na ang epekto sa ani kung hindi gagamutin.",
            "protocol_steps": [
                "Ilapat kaagad ang pagkontrol sa vector.",
                "Dagdagan ang pagsusuri sa dalawang beses kada linggo.",
                "Bantayan ang 10-metrong paligid para sa paglaganap.",
                "Maghanda ng plano sa pag-alis ng malalang apektadong halaman.",
            ],
            "immediate_actions": [
                "Ilapat kaagad ang MSV management protocol.",
                "Simulan ang aktibong pagkontrol sa leafhopper.",
                "Dagdagan ang dalas ng pagsusuri.",
            ],
        },
    },
    ("MSV", 3): {
        "justification": "Severe chlorotic streaking consistent with advanced Maize Streak Virus infection was detected.",
        "key_fact": "At this severity, rouging (removing infected plants) is critical to protect the rest of the field.",
        "immediate_actions": [
            "Rogue (remove and destroy) severely infected plants immediately.",
            "Apply vector control to the surrounding area without delay.",
            "Isolate the affected zone from further field activity until controlled.",
        ],
        "management": [
            "Remove and destroy infected plant material away from the field (burn or bury deep).",
            "Apply insecticide targeting leafhoppers per DA-registered guidance.",
            "Monitor daily until spread is confirmed contained.",
        ],
        "spread": "MSV spreads via leafhopper feeding; severe infections signal an active, established vector population.",
        "distances": "Rogue infected plants and monitor a minimum 15-meter radius daily for new cases.",
        "prevention": [
            "Switch to resistant/tolerant varieties for the next planting cycle.",
            "Review and adjust planting schedule relative to local vector activity peaks.",
            "Eliminate weed hosts field-wide, not just locally.",
        ],
        "precautions": [
            "Wear gloves when handling infected material.",
            "Disinfect all tools and footwear before moving to unaffected field sections.",
        ],
        "detection": "Daily monitoring is required until new infections stop appearing.",
        "control": {
            "chemical": "Consult your local DA extension officer immediately for registered insecticide options and correct dosing.",
            "biological": "Biological control alone is insufficient at this severity; combine with chemical/cultural measures.",
            "cultural": "Rogue and destroy infected plants; do not compost infected material within the field.",
        },
        "protocol_title": "MSV Severe-Stage Rapid Response Protocol",
        "protocol_steps": [
            "Rogue and destroy severely infected plants today.",
            "Apply vector control immediately.",
            "Monitor daily for 15-meter radius spread.",
            "Consult DA extension officer for confirmation and follow-up.",
        ],
        "tagalog": {
            "justification": "Natukoy ang malalang streaking na kaugnay ng malalim nang impeksyon ng Maize Streak Virus.",
            "key_fact": "Sa antas na ito, kailangang alisin (rouging) ang mga malalang apektadong halaman para maprotektahan ang buong bukid.",
            "protocol_steps": [
                "Alisin at sirain ang malalang apektadong halaman ngayon din.",
                "Ilapat kaagad ang pagkontrol sa vector.",
                "Araw-araw na bantayan ang 15-metrong paligid.",
                "Kumonsulta sa DA extension officer para sa kumpirmasyon at follow-up.",
            ],
            "immediate_actions": [
                "Alisin at sirain kaagad ang malalang apektadong halaman.",
                "Ilapat kaagad ang pagkontrol sa vector sa paligid.",
                "Ihiwalay ang apektadong bahagi hanggang makontrol.",
            ],
        },
    },
    ("MLN", 1): {
        "justification": "Fine chlorotic streaks/mottling consistent with early-stage Maize Lethal Necrosis was detected.",
        "key_fact": "MLN results from co-infection of two viruses (MCMV + a potyvirus) — controlling both insect vectors and mechanical spread matters.",
        "immediate_actions": [
            "Mark and monitor the affected plant closely.",
            "Avoid working infected plants before healthy ones with the same tools.",
            "Scout neighboring plants for early mottling.",
        ],
        "management": [
            "Continue close weekly monitoring of severity progression.",
            "Disinfect tools between plants.",
            "Prepare for management protocol escalation if severity increases.",
        ],
        "spread": "MLN spreads via insect vectors (thrips, aphids, beetles) and mechanically via contaminated tools, hands, and seed.",
        "distances": "No rouging required at trace level; focus on tool/hand sanitation instead.",
        "prevention": [
            "Use certified, MLN-tested seed.",
            "Control insect vectors (thrips, aphids) proactively.",
            "Avoid moving between fields without disinfecting tools and footwear.",
        ],
        "precautions": ["Disinfect hands and tools after handling any symptomatic plant."],
        "detection": "Re-inspect weekly; escalate if mottling spreads beyond 25% of leaf area.",
        "control": {
            "chemical": "Consult your local DA extension officer for registered insecticides targeting thrips/aphids.",
            "biological": "Encourage natural predators of aphids and thrips where feasible.",
            "cultural": "Disinfect tools between plants; avoid working wet foliage of infected plants first.",
        },
        "protocol_title": "MLN Early-Stage Monitoring Protocol",
        "protocol_steps": [
            "Mark affected plant(s) for weekly tracking.",
            "Disinfect tools after each use on this plant.",
            "Begin vector (thrips/aphid) monitoring.",
            "Re-scan with the MAIze app in 7 days.",
        ],
        "tagalog": {
            "justification": "Natukoy ang maagang chlorotic mottling na kaugnay ng unang antas ng Maize Lethal Necrosis.",
            "key_fact": "Ang MLN ay resulta ng dalawang virus nang sabay — kailangang kontrolin ang insect vector at mekanikal na paglaganap.",
            "protocol_steps": [
                "Markahan ang apektadong halaman para sa lingguhang pagsubaybay.",
                "I-disinfect ang mga kagamitan pagkatapos gamitin sa halamang ito.",
                "Simulan ang pagmomonitor ng thrips/aphid.",
                "Mag-scan muli gamit ang MAIze app pagkalipas ng 7 araw.",
            ],
            "immediate_actions": [
                "Markahan at bantayan nang mabuti ang apektadong halaman.",
                "Iwasang gamitin ang parehong kagamitan sa malusog na halaman nang hindi nadidisinfect.",
                "Suriin ang kalapit na halaman para sa maagang senyales.",
            ],
        },
    },
    ("MLN", 2): {
        "justification": "Chlorotic mottling and mosaic patterning consistent with progressing Maize Lethal Necrosis was detected.",
        "key_fact": "MLN progresses faster than MSV once established — prompt action at this stage significantly reduces total crop loss.",
        "immediate_actions": [
            "Apply the MLN management protocol immediately.",
            "Begin active vector control (thrips/aphids/beetles).",
            "Increase scouting frequency to twice weekly.",
        ],
        "management": [
            "Apply vector control measures per DA-registered product guidance.",
            "Strictly disinfect tools between plants and rows.",
            "Prepare for possible rouging if severity worsens toward necrosis.",
        ],
        "spread": "MLN spreads via insect vectors and mechanically; denser plantings and shared tools accelerate spread.",
        "distances": "Monitor a 10-meter radius around confirmed cases; isolate progressing plants if symptoms worsen.",
        "prevention": [
            "Use certified, MLN-tested seed in future plantings.",
            "Maintain strict tool and equipment sanitation across the field.",
            "Control insect vector populations proactively.",
        ],
        "precautions": ["Disinfect tools and hands between every plant in the affected zone."],
        "detection": "Re-scan every 3-4 days at this stage to track progression toward necrosis (Stage 3).",
        "control": {
            "chemical": "Consult your local DA extension officer for currently registered insecticide options for thrips/aphids/beetles.",
            "biological": "Support natural predator populations; avoid unnecessary broad-spectrum spraying.",
            "cultural": "Strict tool sanitation; avoid working infected rows before healthy ones.",
        },
        "protocol_title": "MLN Moderate-Stage Rapid Response Protocol",
        "protocol_steps": [
            "Apply vector control per DA-registered guidance.",
            "Enforce strict tool sanitation across the field.",
            "Increase scouting to twice weekly.",
            "Prepare rouging plan in case of progression to necrosis.",
        ],
        "tagalog": {
            "justification": "Natukoy ang lumalalang chlorotic mottling na kaugnay ng umuusbong na Maize Lethal Necrosis.",
            "key_fact": "Mas mabilis kumalat ang MLN kumpara sa MSV kapag naitatag na — mahalaga ang agarang aksyon sa antas na ito.",
            "protocol_steps": [
                "Ilapat kaagad ang pagkontrol sa vector.",
                "Ipatupad ang mahigpit na disinfection ng kagamitan.",
                "Dagdagan ang pagsusuri sa dalawang beses kada linggo.",
                "Maghanda ng plano sa pag-alis kung lumala tungo sa necrosis.",
            ],
            "immediate_actions": [
                "Ilapat kaagad ang MLN management protocol.",
                "Simulan ang aktibong pagkontrol sa vector.",
                "Dagdagan ang dalas ng pagsusuri.",
            ],
        },
    },
    ("MLN", 3): {
        "justification": "Excessive mottling, necrosis, or dead heart consistent with severe Maize Lethal Necrosis was detected.",
        "key_fact": "At this severity, MLN is often fatal to the plant — rouging and strict sanitation are critical to protect the rest of the field.",
        "immediate_actions": [
            "Rogue (remove and destroy) severely infected/necrotic plants immediately.",
            "Apply vector control to the surrounding area without delay.",
            "Enforce strict tool and equipment sanitation field-wide.",
        ],
        "management": [
            "Remove and destroy infected plant material away from the field (burn or bury deep).",
            "Apply insecticide targeting vectors per DA-registered guidance.",
            "Monitor daily until spread is confirmed contained.",
        ],
        "spread": "MLN spreads via insect vectors and mechanically; severe infection signals both an active vector population and possible tool-borne spread.",
        "distances": "Rogue infected plants and monitor a minimum 15-meter radius daily for new cases.",
        "prevention": [
            "Switch to MLN-tested, certified seed for the next planting cycle.",
            "Implement field-wide tool sanitation protocols going forward.",
            "Coordinate vector control with neighboring farms where possible.",
        ],
        "precautions": [
            "Wear gloves when handling infected material.",
            "Disinfect all tools and footwear before moving to unaffected field sections.",
        ],
        "detection": "Daily monitoring is required until new infections stop appearing.",
        "control": {
            "chemical": "Consult your local DA extension officer immediately for registered insecticide options and correct dosing.",
            "biological": "Biological control alone is insufficient at this severity; combine with chemical/cultural measures.",
            "cultural": "Rogue and destroy infected plants; do not compost infected material within the field.",
        },
        "protocol_title": "MLN Severe-Stage Rapid Response Protocol",
        "protocol_steps": [
            "Rogue and destroy severely infected plants today.",
            "Apply vector control immediately.",
            "Monitor daily for 15-meter radius spread.",
            "Consult DA extension officer for confirmation and follow-up.",
        ],
        "tagalog": {
            "justification": "Natukoy ang malalang mottling o necrosis na kaugnay ng malubhang Maize Lethal Necrosis.",
            "key_fact": "Sa antas na ito, madalas nang nakamamatay ang MLN sa halaman — kritikal ang pag-alis at mahigpit na sanitation.",
            "protocol_steps": [
                "Alisin at sirain ang malalang apektadong halaman ngayon din.",
                "Ilapat kaagad ang pagkontrol sa vector.",
                "Araw-araw na bantayan ang 15-metrong paligid.",
                "Kumonsulta sa DA extension officer para sa kumpirmasyon at follow-up.",
            ],
            "immediate_actions": [
                "Alisin at sirain kaagad ang malalang apektado o patay nang halaman.",
                "Ilapat kaagad ang pagkontrol sa vector sa paligid.",
                "Ipatupad ang mahigpit na disinfection ng kagamitan.",
            ],
        },
    },
}

# HEALTHY only ever maps to stage 0 (see severity_to_stage) — the entries
# below exist purely so an unexpected (HEALTHY, 1/2/3) lookup never KeyErrors.
OFFLINE_GUIDANCE[("HEALTHY", 1)] = OFFLINE_GUIDANCE[("HEALTHY", 0)]
OFFLINE_GUIDANCE[("HEALTHY", 2)] = OFFLINE_GUIDANCE[("HEALTHY", 0)]
OFFLINE_GUIDANCE[("HEALTHY", 3)] = OFFLINE_GUIDANCE[("HEALTHY", 0)]

# severity_to_stage() returns stage 0 whenever severity_pct < 1.0% —
# including for a genuinely positive MSV/MLN classification at trace
# level, not just for HEALTHY. Falling back to the HEALTHY ("no disease")
# message in that case would be actively misleading, so these reuse the
# stage-1 (trace-level) guidance, which is the closest accurate match.
OFFLINE_GUIDANCE[("MSV", 0)] = OFFLINE_GUIDANCE[("MSV", 1)]
OFFLINE_GUIDANCE[("MLN", 0)] = OFFLINE_GUIDANCE[("MLN", 1)]
