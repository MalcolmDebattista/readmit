"""Section 3 of the notebook: the knowledge base (ICD-9 chapters, Strack groups, medication classes), SHACL and SPARQL.

Everything from RD to shacl_violations is copied verbatim from the notebook; add_codes and competency_tables wrap
the notebook's loop and CQ1-CQ3 queries as functions.
"""
import pandas as pd
import pyshacl
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace
from rdflib.namespace import SKOS

RD = Namespace("https://ari3217.example/readmit#")

# ICD-9-CM chapters (CDC/NCHS tabular list): numeric range → chapter
CHAPTERS = [(1, 139, "infectious", "Infectious and parasitic diseases"), (140, 239, "neoplasms", "Neoplasms"),
            (240, 279, "endocrine", "Endocrine, nutritional and metabolic diseases, and immunity disorders"),
            (280, 289, "blood", "Diseases of the blood and blood-forming organs"), (290, 319, "mental", "Mental disorders"),
            (320, 389, "nervous", "Diseases of the nervous system and sense organs"),
            (390, 459, "circulatory", "Diseases of the circulatory system"), (460, 519, "respiratory", "Diseases of the respiratory system"),
            (520, 579, "digestive", "Diseases of the digestive system"), (580, 629, "genitourinary", "Diseases of the genitourinary system"),
            (630, 679, "pregnancy", "Complications of pregnancy, childbirth and the puerperium"),
            (680, 709, "skin", "Diseases of the skin and subcutaneous tissue"),
            (710, 739, "musculoskeletal", "Diseases of the musculoskeletal system and connective tissue"),
            (740, 759, "congenital", "Congenital anomalies"), (760, 779, "perinatal", "Certain conditions originating in the perinatal period"),
            (780, 799, "symptoms", "Symptoms, signs and ill-defined conditions"), (800, 999, "injury", "Injury and poisoning")]
CHAPTER_LABEL = {c: l for _, _, c, l in CHAPTERS} | {"V": "Factors influencing health status (V codes)",
                                                     "E": "External causes of injury (E codes)"}
# the nine analytic groups of Strack et al. (2014, Table 2): symptom codes 785–788 are filed with their organ system
STRACK = [("circulatory", [(390, 459), (785, 785)]), ("respiratory", [(460, 519), (786, 786)]),
          ("digestive", [(520, 579), (787, 787)]), ("injury", [(800, 999)]), ("musculoskeletal", [(710, 739)]),
          ("genitourinary", [(580, 629), (788, 788)]), ("neoplasms", [(140, 239)])]
DM_COMPLICATION = {"0": "none stated", "1": "ketoacidosis", "2": "hyperosmolarity", "3": "other coma", "4": "renal",
                   "5": "ophthalmic", "6": "neurological", "7": "peripheral circulatory", "8": "other specified",
                   "9": "unspecified"}
# labels for the most frequent 3-digit categories, for people and for the call-script context (not used by the model)
CATEGORY_LABEL = {"428": "heart failure", "414": "other chronic ischaemic heart disease", "786": "respiratory and chest symptoms",
                  "410": "acute myocardial infarction", "486": "pneumonia, organism unspecified", "427": "cardiac dysrhythmias",
                  "491": "chronic bronchitis", "715": "osteoarthrosis", "682": "other cellulitis and abscess",
                  "434": "occlusion of cerebral arteries", "780": "general symptoms", "996": "complications of specified procedures",
                  "276": "fluid, electrolyte and acid-base disorders", "038": "septicaemia", "250": "diabetes mellitus",
                  "599": "other disorders of the urinary tract", "584": "acute kidney failure", "401": "essential hypertension",
                  "403": "hypertensive chronic kidney disease", "585": "chronic kidney disease"}
# medication columns → therapeutic class (combinations belong to both); two columns we cannot place with confidence
MED_CLASS = {"metformin": ["biguanide"], "repaglinide": ["meglitinide"], "nateglinide": ["meglitinide"],
             "chlorpropamide": ["sulfonylurea"], "glimepiride": ["sulfonylurea"], "acetohexamide": ["sulfonylurea"],
             "glipizide": ["sulfonylurea"], "glyburide": ["sulfonylurea"], "tolbutamide": ["sulfonylurea"],
             "tolazamide": ["sulfonylurea"], "pioglitazone": ["thiazolidinedione"], "rosiglitazone": ["thiazolidinedione"],
             "troglitazone": ["thiazolidinedione"], "acarbose": ["alpha_glucosidase_inhibitor"],
             "miglitol": ["alpha_glucosidase_inhibitor"], "insulin": ["insulin"],
             "glyburide-metformin": ["sulfonylurea", "biguanide"], "glipizide-metformin": ["sulfonylurea", "biguanide"],
             "glimepiride-pioglitazone": ["sulfonylurea", "thiazolidinedione"],
             "metformin-rosiglitazone": ["biguanide", "thiazolidinedione"],
             "metformin-pioglitazone": ["biguanide", "thiazolidinedione"],
             "examide": [], "citoglipton": []}
MEDS = list(MED_CLASS)

def norm_icd9(code: str):
    """Undo the numeric mangling upstream: 38 → 038, 250.1 → 250.10, 250 → 250.00. '?' → None."""
    if code in ("?", ""):
        return None
    if code[0] in "VE":
        return code
    head, _, tail = code.partition(".")
    head = head.zfill(3)
    if head == "250":
        return f"250.{(tail or '0').ljust(2, '0')}"
    return head + (f".{tail}" if tail else "")

def chapter_of(code: str) -> str:
    if code[0] in "VE":
        return code[0]
    n = int(code[:3])
    return next(c for lo, hi, c, _ in CHAPTERS if lo <= n <= hi)

def group_of(code: str) -> str:
    if code.startswith("250"):
        return "diabetes"
    if code[0] in "VE":
        return "other"
    n = int(code[:3])
    return next((gname for gname, ranges in STRACK if any(lo <= n <= hi for lo, hi in ranges)), "other")

def build_kb(codes, notation=lambda c: c) -> Graph:
    kb = Graph(); kb.bind("rd", RD); kb.bind("skos", SKOS)
    for ch, label in CHAPTER_LABEL.items():
        kb.add((RD[f"chapter/{ch}"], RDF.type, RD.Chapter)); kb.add((RD[f"chapter/{ch}"], RDFS.label, Literal(label)))
    for gname in [s[0] for s in STRACK] + ["diabetes", "other"]:
        kb.add((RD[f"group/{gname}"], RDF.type, RD.AnalyticGroup))
    for code in codes:
        c, cat = RD[f"code/{code}"], code[:4] if code[0] == "E" else code[:3]
        kb.add((c, RDF.type, RD.Code)); kb.add((c, RD.notation, Literal(notation(code))))
        kb.add((c, SKOS.broader, RD[f"category/{cat}"])); kb.add((RD[f"category/{cat}"], RDF.type, RD.Category))
        kb.add((RD[f"category/{cat}"], SKOS.broader, RD[f"chapter/{chapter_of(code)}"]))
        if cat in CATEGORY_LABEL:
            kb.add((RD[f"category/{cat}"], RDFS.label, Literal(CATEGORY_LABEL[cat])))
        kb.add((c, RD.inGroup, RD[f"group/{group_of(code)}"]))
        if code.startswith("250."):
            kb.add((c, RDF.type, RD.DiabetesCode))
            kb.add((c, RD.complication, Literal(DM_COMPLICATION[code[4]])))
            kb.add((c, RD.type1, Literal(code[5] in "13", datatype=XSD.boolean)))
            kb.add((c, RD.uncontrolled, Literal(code[5] in "23", datatype=XSD.boolean)))
    for med, classes in MED_CLASS.items():
        kb.add((RD[f"med/{med}"], RDF.type, RD.Medication)); kb.add((RD[f"med/{med}"], RDFS.label, Literal(med)))
        for cl in classes:
            kb.add((RD[f"med/{med}"], RD.memberOf, RD[f"class/{cl}"])); kb.add((RD[f"class/{cl}"], RDF.type, RD.MedicationClass))
    return kb

SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> . @prefix rd: <https://ari3217.example/readmit#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> . @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
rd:CodeShape a sh:NodeShape ; sh:targetClass rd:Code ;
  sh:property [ sh:path rd:notation ; sh:minCount 1 ; sh:maxCount 1 ;
                sh:pattern "^([0-9]{3}(\\\\.[0-9]{1,2})?|V[0-9]{2}(\\\\.[0-9]{1,2})?|E[0-9]{3}(\\\\.[0-9])?)$" ;
                sh:message "not a well-formed ICD-9-CM code" ] ;
  sh:property [ sh:path skos:broader ; sh:class rd:Category ; sh:minCount 1 ; sh:maxCount 1 ; sh:message "code has no category" ] ;
  sh:property [ sh:path rd:inGroup ; sh:class rd:AnalyticGroup ; sh:minCount 1 ; sh:maxCount 1 ; sh:message "code has no analytic group" ] .
rd:CategoryShape a sh:NodeShape ; sh:targetClass rd:Category ;
  sh:property [ sh:path skos:broader ; sh:class rd:Chapter ; sh:minCount 1 ; sh:maxCount 1 ; sh:message "category outside every chapter" ] .
rd:DiabetesShape a sh:NodeShape ; sh:targetClass rd:DiabetesCode ;
  sh:property [ sh:path rd:complication ; sh:minCount 1 ; sh:maxCount 1 ; sh:message "diabetes code without a 4th digit" ] ;
  sh:property [ sh:path rd:type1 ; sh:datatype xsd:boolean ; sh:minCount 1 ; sh:maxCount 1 ; sh:message "diabetes type not coded" ] .
rd:MedicationShape a sh:NodeShape ; sh:targetClass rd:Medication ;
  sh:property [ sh:path rd:memberOf ; sh:class rd:MedicationClass ; sh:minCount 1 ; sh:message "medication has no therapeutic class" ] .
"""
SH = Namespace("http://www.w3.org/ns/shacl#")
SHAPES_G = Graph().parse(data=SHAPES, format="turtle")

def shacl_violations(kb: Graph) -> pd.Series:
    _, report, _ = pyshacl.validate(kb, shacl_graph=SHAPES_G)
    msgs = [str(report.value(r, SH.resultMessage)) for r in report.subjects(RDF.type, SH.ValidationResult)]
    return pd.Series(msgs, dtype=object).value_counts()


def add_codes(df: pd.DataFrame) -> pd.DataFrame:
    """code_1..3: the CSV's diag_1..3 normalised (38 -> 038, 250.1 -> 250.10); '?' -> None (Trap 6)."""
    df = df.copy()
    for i in (1, 2, 3):
        df[f"code_{i}"] = df[f"diag_{i}"].map(norm_icd9)
    return df


def all_codes(df: pd.DataFrame) -> list:
    return sorted(set(pd.concat([df.code_1, df.code_2, df.code_3]).dropna()))


CQ1 = """SELECT ?code ?chapter ?group WHERE { ?c a rd:Code ; rd:notation ?code ; skos:broader/skos:broader ?ch ; rd:inGroup ?g .
         BIND(STRAFTER(STR(?ch), "chapter/") AS ?chapter) BIND(STRAFTER(STR(?g), "group/") AS ?group) }"""
CQ2 = """SELECT ?code ?complication ?type1 ?uncontrolled WHERE { ?c a rd:DiabetesCode ; rd:notation ?code ;
         rd:complication ?complication ; rd:type1 ?type1 ; rd:uncontrolled ?uncontrolled }"""
CQ3 = """SELECT ?med ?cls WHERE { ?m a rd:Medication ; rdfs:label ?med . OPTIONAL { ?m rd:memberOf ?k .
         BIND(STRAFTER(STR(?k), "class/") AS ?cls) } }"""


def competency_tables(kb: Graph):
    """CQ1-CQ3 answered with SPARQL; the answers become the lookup tables the features use."""
    def q(s):
        return pd.DataFrame([[str(v) if v is not None else None for v in row]
                             for row in kb.query(s, initNs={"rd": RD, "skos": SKOS, "rdfs": RDFS})])
    cq1 = q(CQ1).set_axis(["code", "chapter", "group"], axis=1).set_index("code")
    cq2 = q(CQ2).set_axis(["code", "complication", "type1", "uncontrolled"], axis=1).set_index("code")
    cq3 = q(CQ3).set_axis(["med", "cls"], axis=1)
    return cq1, cq2, cq3
