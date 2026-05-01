import re
import unicodedata
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


COMPETENCES_DICT = {
    "langages_programmation": [
        "python", "java", "javascript", "c", "c++", "c#", "r",
        "scala", "kotlin", "swift", "php", "ruby", "go", "rust",
        "matlab", "typescript",
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "random forest", "svm", "xgboost", "neural network", "tensorflow",
        "keras", "pytorch", "scikit-learn", "opencv", "transformers",
        "bert", "regression", "classification", "clustering",
        "reinforcement learning", "data science", "intelligence artificielle",
    ],
    "data_engineering": [
        "hadoop", "spark", "kafka", "airflow", "etl", "data warehouse",
        "big data", "hive", "databricks", "elasticsearch", "data pipeline",
        "data lake", "dbt", "talend",
    ],
    "bases_de_donnees": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
        "sqlite", "nosql", "firebase", "cassandra", "mariadb",
    ],
    "developpement_web": [
        "html", "css", "react", "angular", "vue", "nodejs", "django",
        "flask", "fastapi", "spring", "laravel", "rest api", "graphql",
        "bootstrap", "tailwind", "jquery", "webpack", "sass",
        "frontend", "backend", "fullstack",
    ],
    "devops_cloud": [
        "docker", "kubernetes", "aws", "azure", "gcp", "git", "github",
        "gitlab", "ci/cd", "jenkins", "linux", "bash", "terraform",
        "ansible", "nginx", "apache",
    ],
    "data_visualisation": [
        "tableau", "power bi", "looker", "matplotlib", "seaborn",
        "plotly", "google data studio",
    ],
    "outils_bureautiques": [
        "word", "powerpoint", "latex", "notion", "jira", "trello", "confluence",
    ],
}


DESCRIPTIONS_DOMAINES = {
    "Data Science & IA": """
        Je suis data scientist specialise en machine learning et deep learning.
        J'utilise Python pour construire des modeles predictifs avec TensorFlow,
        PyTorch, Keras et Scikit-learn.
    """,
    "Developpement Web": """
        Je suis developpeur web fullstack specialise en HTML, CSS et JavaScript.
        Je maitrise React, Angular, Vue, NodeJS, Django, Flask et FastAPI.
    """,
    "Data Engineering": """
        Je suis ingenieur de donnees specialise dans les pipelines de donnees.
        Je travaille avec Hadoop, Spark, Kafka, Airflow, ETL et data warehouse.
    """,
    "DevOps & Cloud": """
        Je suis ingenieur DevOps specialise en cloud et infrastructure.
        Je maitrise Docker, Kubernetes, AWS, Azure, GCP, Terraform et Ansible.
    """,
    "Cybersecurite": """
        Je suis expert en cybersecurite et securite informatique.
        Je realise des tests de penetration, audits et analyses de securite.
    """,
    "Developpement Mobile": """
        Je suis developpeur mobile specialise Android et iOS.
        Je developpe avec Flutter, React Native, Kotlin, Java et Swift.
    """,
    "Genie Logiciel": """
        Je suis ingenieur logiciel specialise en architecture et qualite logicielle.
        Je travaille avec design patterns, SOLID, tests et CI/CD.
    """,
}


@lru_cache(maxsize=1)
def _load_nlp_components():
    import spacy

    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        # Fallback: keep the app functional even when the French model is not installed.
        nlp = spacy.blank("fr")
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return nlp, sbert


def _clean_text_for_spacy(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _extract_competences(text_for_spacy: str, nlp) -> dict:
    if "competence_ruler" in nlp.pipe_names:
        nlp.remove_pipe("competence_ruler")

    if "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", name="competence_ruler", before="ner")
    else:
        ruler = nlp.add_pipe("entity_ruler", name="competence_ruler")
    patterns = []
    for categorie, competences in COMPETENCES_DICT.items():
        for comp in competences:
            patterns.append({"label": categorie.upper(), "pattern": comp})
    ruler.add_patterns(patterns)

    doc = nlp(text_for_spacy)
    competences_trouvees = {}
    for ent in doc.ents:
        categorie = ent.label_.lower()
        competence = ent.text.strip()
        if categorie in COMPETENCES_DICT:
            competences_trouvees.setdefault(categorie, [])
            if competence not in competences_trouvees[categorie]:
                competences_trouvees[categorie].append(competence)

    return competences_trouvees


def _identify_domain(competences_trouvees: dict, sbert):
    texte_competences = " ".join([
        comp
        for comps in competences_trouvees.values()
        for comp in comps
    ])

    if not texte_competences:
        return "Genie Logiciel", []

    vecteur_cv = sbert.encode([texte_competences])
    domaines = list(DESCRIPTIONS_DOMAINES.keys())
    descriptions = list(DESCRIPTIONS_DOMAINES.values())
    vecteurs_domaines = sbert.encode(descriptions)

    similarities = cosine_similarity(vecteur_cv, vecteurs_domaines)[0]
    scores = sorted(zip(domaines, similarities), key=lambda x: x[1], reverse=True)
    return scores[0][0], scores


def _normaliser(text: str) -> str:
    text = re.sub(r"\(cid:[^)]+\)", " ", text)
    text = re.sub(r"[`´''ʼ\u2018\u2019]", "", text)
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8").lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _detecter_niveau(raw_text: str):
    text_normalise = _normaliser(raw_text)

    annee_detectee = None
    patterns_annee = [
        r"(\d+)\s*eme\s*annee", r"(\d+)\s*ere\s*annee", r"(\d+)\s*ieme\s*annee",
        r"(\d+)\s*e\s*annee", r"(\d+)`eme\s*annee", r"(\d+)\s*`eme\s*annee",
        r"annee\s*(\d+)", r"(\d+)\s*year", r"(\d+)th\s*year", r"(\d+)rd\s*year",
        r"(\d+)nd\s*year", r"(\d+)st\s*year", r"year\s*(\d+)",
    ]
    for pattern in patterns_annee:
        match = re.search(pattern, text_normalise)
        if match:
            annee_detectee = match.group(1)
            if annee_detectee in ["1", "2", "3", "4", "5"]:
                break
            annee_detectee = None

    formations = {
        "Doctorat": ["doctorat", "phd", "ph.d", "docteur", "these"],
        "Master": ["master", "mastere", "m2", "m1"],
        "Ingenieur": ["ingenieur", "engineer", "cycle ingenieur", "ecole ingenieur", "bac+5"],
        "Licence": ["licence", "license", "bachelor", "bac+3", "l3"],
        "BTS / DUT": ["bts", "dut", "bac+2", "technicien superieur"],
        "Baccalaureat": ["baccalaureat", "bac", "terminal", "lycee"],
    }
    ordre = ["Doctorat", "Master", "Ingenieur", "Licence", "BTS / DUT", "Baccalaureat"]

    niveaux_trouves = {}
    for niveau, keywords in formations.items():
        mots = [kw for kw in keywords if kw in text_normalise]
        if mots:
            niveaux_trouves[niveau] = mots

    niveau_detecte = "Non detecte"
    for niveau in ordre:
        if niveau in niveaux_trouves:
            niveau_detecte = niveau
            break

    suffixes = {"1": "ere", "2": "eme", "3": "eme", "4": "eme", "5": "eme"}
    if annee_detectee and niveau_detecte != "Non detecte":
        formation = f"{annee_detectee}{suffixes.get(annee_detectee, 'eme')} annee {niveau_detecte}"
    elif niveau_detecte != "Non detecte":
        formation = niveau_detecte
    else:
        formation = "Non detecte"

    return formation, annee_detectee


def run_nlp_pipeline(cv_text: str) -> dict:
    """Global NLP function: CV text -> structured profile."""
    nlp, sbert = _load_nlp_components()

    text_for_spacy = _clean_text_for_spacy(cv_text)
    competences = _extract_competences(text_for_spacy, nlp)
    domaine, scores = _identify_domain(competences, sbert)
    niveau, annee = _detecter_niveau(cv_text)

    tech_stack = []
    for comps in competences.values():
        for comp in comps:
            if comp not in tech_stack:
                tech_stack.append(comp)
    if not tech_stack:
        tech_stack = ["python", "sql", "git"]

    experience_years = int(annee) if annee else 2
    description = (
        f"Profil local via spaCy/SBERT. Domaine detecte: {domaine}. "
        f"Niveau: {niveau}. Competences cles: {', '.join(tech_stack[:6])}."
    )

    return {
        "position": domaine,
        "description": description,
        "experience_years": experience_years,
        "tech_stack": tech_stack[:12],
        "competences": competences,
        "domaine": domaine,
        "niveau": niveau,
        "scores_domaines": [{"domaine": d, "score": float(s)} for d, s in scores],
    }
