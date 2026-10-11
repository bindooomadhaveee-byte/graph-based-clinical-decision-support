"""
Graph-Based Clinical Decision Support (base project)
=====================================================
A small knowledge graph (built with NetworkX) links:

    Symptom --INDICATES--> Disease --CONFIRMED_BY--> Test
                              |
                              +--TREATED_BY--> Drug --INTERACTS_WITH--> Drug
                                                 |
                                                 +--CONTRAINDICATED_IN--> Condition

Given a patient (symptoms, conditions, allergies, current meds) the system:
  1. Ranks likely diseases by graph-weighted symptom matching
  2. Suggests confirmatory tests
  3. Recommends treatments, filtering out unsafe drugs
     (allergy, contraindication, drug-drug interaction)

DISCLAIMER: Educational demo only. The medical data here is simplified
and NOT for real clinical use.

Install:  pip install networkx matplotlib
Run:      python graph_cds.py            (CLI demo)
          python graph_cds.py --plot     (also draws the graph)
"""

from dataclasses import dataclass, field
import sys
import networkx as nx


# ----------------------------------------------------------------------
# 1. Patient model
# ----------------------------------------------------------------------
@dataclass
class Patient:
    name: str
    age: int
    symptoms: set = field(default_factory=set)         # present symptoms
    absent_symptoms: set = field(default_factory=set)  # explicitly denied
    conditions: set = field(default_factory=set)       # e.g. kidney_disease
    allergies: set = field(default_factory=set)        # drug classes, e.g. penicillin
    medications: set = field(default_factory=set)      # current drugs


# ----------------------------------------------------------------------
# 2. Knowledge graph
# ----------------------------------------------------------------------
def build_knowledge_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # --- Symptom -> Disease (weight = how strongly symptom indicates disease)
    symptom_disease = {
        "Influenza": {"fever": 0.8, "cough": 0.6, "body_ache": 0.8,
                      "fatigue": 0.6, "headache": 0.5, "sore_throat": 0.3},
        "Common Cold": {"runny_nose": 0.9, "sore_throat": 0.6, "cough": 0.5,
                        "headache": 0.2},
        "Pneumonia": {"fever": 0.7, "cough": 0.8, "shortness_of_breath": 0.9,
                      "chest_pain": 0.6, "fatigue": 0.4},
        "Strep Throat": {"sore_throat": 0.9, "fever": 0.6, "headache": 0.4,
                         "nausea": 0.2},
        "Type 2 Diabetes": {"frequent_urination": 0.9, "excessive_thirst": 0.9,
                            "blurred_vision": 0.6, "fatigue": 0.5},
    }
    for disease, symptoms in symptom_disease.items():
        G.add_node(disease, type="disease")
        for s, w in symptoms.items():
            G.add_node(s, type="symptom")
            G.add_edge(s, disease, relation="INDICATES", weight=w)

    # --- Disease -> Test
    disease_tests = {
        "Influenza": ["Rapid influenza PCR"],
        "Common Cold": [],
        "Pneumonia": ["Chest X-ray", "Complete blood count"],
        "Strep Throat": ["Rapid strep test", "Throat culture"],
        "Type 2 Diabetes": ["Fasting blood glucose", "HbA1c"],
    }
    for disease, tests in disease_tests.items():
        for t in tests:
            G.add_node(t, type="test")
            G.add_edge(disease, t, relation="CONFIRMED_BY")

    # --- Drugs (with class, used for allergy matching)
    drugs = {
        "Oseltamivir": "antiviral",
        "Paracetamol": "analgesic",
        "Ibuprofen": "nsaid",
        "Aspirin": "nsaid",
        "Amoxicillin": "penicillin",
        "Azithromycin": "macrolide",
        "Metformin": "biguanide",
        "Warfarin": "anticoagulant",
    }
    for d, cls in drugs.items():
        G.add_node(d, type="drug", drug_class=cls)

    # --- Disease -> Drug (line: 1 = first line, 2 = alternative)
    disease_drugs = {
        "Influenza": [("Oseltamivir", 1), ("Paracetamol", 1), ("Ibuprofen", 2)],
        "Common Cold": [("Paracetamol", 1), ("Ibuprofen", 2)],
        "Pneumonia": [("Amoxicillin", 1), ("Azithromycin", 2),
                      ("Paracetamol", 1)],
        "Strep Throat": [("Amoxicillin", 1), ("Azithromycin", 2),
                         ("Paracetamol", 1)],
        "Type 2 Diabetes": [("Metformin", 1)],
    }
    for disease, lst in disease_drugs.items():
        for drug, line in lst:
            G.add_edge(disease, drug, relation="TREATED_BY", line=line)

    # --- Drug <-> Drug interactions (undirected -> add both directions)
    interactions = [
        ("Warfarin", "Ibuprofen", "major", "High bleeding risk"),
        ("Warfarin", "Aspirin", "major", "High bleeding risk"),
        ("Warfarin", "Azithromycin", "moderate", "May raise INR"),
        ("Warfarin", "Paracetamol", "minor", "Regular high doses may raise INR"),
        ("Ibuprofen", "Aspirin", "moderate", "Reduced aspirin effect, GI risk"),
    ]
    for a, b, sev, note in interactions:
        G.add_edge(a, b, relation="INTERACTS_WITH", severity=sev, note=note)
        G.add_edge(b, a, relation="INTERACTS_WITH", severity=sev, note=note)

    # --- Drug -> Condition contraindications
    contraindications = [
        ("Ibuprofen", "kidney_disease"),
        ("Ibuprofen", "peptic_ulcer"),
        ("Aspirin", "peptic_ulcer"),
        ("Metformin", "kidney_disease"),
        ("Paracetamol", "liver_disease"),
    ]
    for drug, cond in contraindications:
        G.add_node(cond, type="condition")
        G.add_edge(drug, cond, relation="CONTRAINDICATED_IN")

    return G


# ----------------------------------------------------------------------
# 3. Decision-support engine
# ----------------------------------------------------------------------
class ClinicalDecisionSupport:
    def __init__(self, graph: nx.DiGraph):
        self.G = graph

    # ---- helpers
    def _nodes_of(self, node_type):
        return [n for n, d in self.G.nodes(data=True) if d.get("type") == node_type]

    # ---- 1. Diagnosis
    def diagnose(self, patient: Patient, top_k: int = 3):
        """
        Score each disease:
            score = matched_weight / total_weight_of_disease
                    - 0.5 * (weight of symptoms the patient explicitly denies) / total
        Returns a list of (disease, score, matched_symptoms) sorted desc.
        """
        results = []
        for disease in self._nodes_of("disease"):
            total = matched = denied = 0.0
            matched_symptoms = []
            for symptom, _, data in self.G.in_edges(disease, data=True):
                if data.get("relation") != "INDICATES":
                    continue
                w = data["weight"]
                total += w
                if symptom in patient.symptoms:
                    matched += w
                    matched_symptoms.append(symptom)
                elif symptom in patient.absent_symptoms:
                    denied += w
            if matched == 0 or total == 0:
                continue
            score = max(0.0, (matched - 0.5 * denied) / total)
            results.append((disease, round(score, 3), matched_symptoms))
        results.sort(key=lambda r: r[1], reverse=True)
        return results[:top_k]

    # ---- 2. Tests
    def recommend_tests(self, disease: str):
        return [t for _, t, d in self.G.out_edges(disease, data=True)
                if d.get("relation") == "CONFIRMED_BY"]

    # ---- 3. Treatment with safety checks
    def recommend_treatment(self, disease: str, patient: Patient):
        safe, excluded = [], []
        for _, drug, data in self.G.out_edges(disease, data=True):
            if data.get("relation") != "TREATED_BY":
                continue
            reasons = self._safety_issues(drug, patient)
            blocking = [r for r in reasons if r["blocking"]]
            entry = {"drug": drug, "line": data["line"],
                     "warnings": [r["text"] for r in reasons if not r["blocking"]]}
            if blocking:
                entry["reasons"] = [r["text"] for r in blocking]
                excluded.append(entry)
            else:
                safe.append(entry)
        safe.sort(key=lambda e: (e["line"], len(e["warnings"])))
        return safe, excluded

    def _safety_issues(self, drug: str, patient: Patient):
        issues = []
        drug_class = self.G.nodes[drug].get("drug_class")

        # Allergy: matches the drug class or the drug name itself
        if drug_class in patient.allergies or drug.lower() in patient.allergies:
            issues.append({"blocking": True,
                           "text": f"Allergy to {drug_class}"})

        # Contraindication via graph edges
        for _, cond, d in self.G.out_edges(drug, data=True):
            if d.get("relation") == "CONTRAINDICATED_IN" and cond in patient.conditions:
                issues.append({"blocking": True,
                               "text": f"Contraindicated in {cond}"})

        # Drug-drug interactions with current medications
        for med in patient.medications:
            if self.G.has_edge(drug, med):
                e = self.G.edges[drug, med]
                if e.get("relation") == "INTERACTS_WITH":
                    sev = e["severity"]
                    issues.append({
                        "blocking": sev == "major",
                        "text": f"{sev} interaction with {med}: {e['note']}"})
        return issues

    # ---- Full report
    def report(self, patient: Patient):
        print("=" * 66)
        print(f"Patient: {patient.name}, age {patient.age}")
        print(f"Symptoms:    {', '.join(sorted(patient.symptoms)) or '-'}")
        print(f"Conditions:  {', '.join(sorted(patient.conditions)) or '-'}")
        print(f"Allergies:   {', '.join(sorted(patient.allergies)) or '-'}")
        print(f"Medications: {', '.join(sorted(patient.medications)) or '-'}")
        print("=" * 66)

        diagnoses = self.diagnose(patient)
        if not diagnoses:
            print("No matching disease found in the knowledge graph.")
            return

        for rank, (disease, score, matched) in enumerate(diagnoses, 1):
            print(f"\n#{rank} {disease}  (confidence {score:.0%})")
            print(f"   Matched symptoms: {', '.join(matched)}")

            tests = self.recommend_tests(disease)
            print(f"   Suggested tests : {', '.join(tests) if tests else 'none needed'}")

            safe, excluded = self.recommend_treatment(disease, patient)
            print("   Safe treatments :")
            if not safe:
                print("      (none - refer to physician)")
            for e in safe:
                tag = "1st line" if e["line"] == 1 else "alternative"
                print(f"      + {e['drug']} [{tag}]")
                for w in e["warnings"]:
                    print(f"          ! {w}")
            if excluded:
                print("   Excluded        :")
                for e in excluded:
                    print(f"      - {e['drug']}: {'; '.join(e['reasons'])}")
        print("\nNote: decision support only - clinician judgment required.\n")


# ----------------------------------------------------------------------
# 4. Optional visualization
# ----------------------------------------------------------------------
def plot_graph(G: nx.DiGraph):
    import matplotlib.pyplot as plt

    colors = {"symptom": "#f4a261", "disease": "#e76f51", "test": "#2a9d8f",
              "drug": "#457b9d", "condition": "#9b5de5"}
    pos = nx.spring_layout(G, seed=7, k=0.9)
    node_colors = [colors.get(G.nodes[n].get("type"), "gray") for n in G]
    plt.figure(figsize=(14, 10))
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=900,
                     font_size=7, arrows=True, edge_color="#bbbbbb")
    for t, c in colors.items():
        plt.scatter([], [], c=c, label=t)
    plt.legend(loc="upper left")
    plt.title("Clinical Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 5. Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    graph = build_knowledge_graph()
    cds = ClinicalDecisionSupport(graph)

    print(f"Knowledge graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges\n")

    # Case 1: flu-like illness, on warfarin, with kidney disease
    p1 = Patient(
        name="Patient A", age=58,
        symptoms={"fever", "cough", "body_ache", "fatigue", "headache"},
        absent_symptoms={"runny_nose"},
        conditions={"kidney_disease"},
        medications={"Warfarin"},
    )
    cds.report(p1)

    # Case 2: throat infection, penicillin allergy
    p2 = Patient(
        name="Patient B", age=24,
        symptoms={"sore_throat", "fever", "headache"},
        allergies={"penicillin"},
    )
    cds.report(p2)

    # Case 3: possible diabetes
    p3 = Patient(
        name="Patient C", age=47,
        symptoms={"frequent_urination", "excessive_thirst", "fatigue"},
        conditions={"kidney_disease"},
    )
    cds.report(p3)

    if "--plot" in sys.argv:
        plot_graph(graph)
