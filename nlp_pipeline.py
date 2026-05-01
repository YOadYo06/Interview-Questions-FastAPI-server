import os
import re
import unicodedata

import numpy as np
from pinecone import Pinecone


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


DESCRIPTIONS_DOMAINES_EN = {
    "Data Science & AI": """
        I am a data scientist specialized in machine learning and deep learning.
        I use Python to build predictive models with TensorFlow, PyTorch, Keras, and Scikit-learn.
    """,
    "Web Development": """
        I am a full-stack web developer specialized in HTML, CSS, and JavaScript.
        I work with React, Angular, Vue, NodeJS, Django, Flask, and FastAPI.
    """,
    "Data Engineering": """
        I am a data engineer focused on data pipelines and warehouses.
        I work with Hadoop, Spark, Kafka, Airflow, ETL, and data warehouses.
    """,
    "DevOps & Cloud": """
        I am a DevOps engineer focused on cloud and infrastructure.
        I work with Docker, Kubernetes, AWS, Azure, GCP, Terraform, and Ansible.
    """,
    "Cybersecurity": """
        I am a cybersecurity specialist focused on audits and security analysis.
        I perform penetration testing and security assessments.
    """,
    "Mobile Development": """
        I am a mobile developer focused on Android and iOS.
        I build apps with Flutter, React Native, Kotlin, Java, and Swift.
    """,
    "Software Engineering": """
        I am a software engineer focused on architecture and software quality.
        I work with design patterns, SOLID, testing, and CI/CD.
    """,
}


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_MODEL = os.getenv("PINECONE_MODEL", "llama-text-embed-v2")


def _embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    result = pc.inference.embed(
        model=PINECONE_MODEL,
        inputs=texts,
        parameters={"input_type": input_type},
    )
    return [item["values"] for item in result.data]


def _clean_text(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _extract_competences(text_for_match: str) -> dict:
    competences_trouvees: dict[str, list[str]] = {}
    for categorie, competences in COMPETENCES_DICT.items():
        for comp in competences:
            pattern = r"\b" + re.escape(comp.lower()) + r"\b"
            if re.search(pattern, text_for_match):
                competences_trouvees.setdefault(categorie, [])
                if comp not in competences_trouvees[categorie]:
                    competences_trouvees[categorie].append(comp)
    return competences_trouvees


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _identify_domain(competences_trouvees: dict):
    texte_competences = " ".join([
        comp
        for comps in competences_trouvees.values()
        for comp in comps
    ])

    if not texte_competences:
        return "Genie Logiciel", []

    domaines = list(DESCRIPTIONS_DOMAINES.keys())
    descriptions_fr = [DESCRIPTIONS_DOMAINES[d] for d in domaines]
    descriptions_en = [
        DESCRIPTIONS_DOMAINES_EN[_map_domain_to_english(d)] for d in domaines
    ]

    vecteur_cv = _embed_texts([texte_competences], input_type="query")[0]
    vecteurs_fr = _embed_texts(descriptions_fr, input_type="passage")
    vecteurs_en = _embed_texts(descriptions_en, input_type="passage")

    similarities = []
    for vec_fr, vec_en in zip(vecteurs_fr, vecteurs_en):
        score_fr = _cosine_similarity(vecteur_cv, vec_fr)
        score_en = _cosine_similarity(vecteur_cv, vec_en)
        similarities.append(max(score_fr, score_en))

    scores = sorted(zip(domaines, similarities), key=lambda x: x[1], reverse=True)
    return scores[0][0], scores


def _map_domain_to_english(domaine_fr: str) -> str:
    mapping = {
        "Data Science & IA": "Data Science & AI",
        "Developpement Web": "Web Development",
        "Data Engineering": "Data Engineering",
        "DevOps & Cloud": "DevOps & Cloud",
        "Cybersecurite": "Cybersecurity",
        "Developpement Mobile": "Mobile Development",
        "Genie Logiciel": "Software Engineering",
    }
    return mapping.get(domaine_fr, "Software Engineering")


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
    text_for_match = _clean_text(cv_text)
    competences = _extract_competences(text_for_match)
    domaine, scores = _identify_domain(competences)
    domaine_en = _map_domain_to_english(domaine)
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
        f"Profil via Pinecone embeddings. Domaine detecte: {domaine}. "
        f"Niveau: {niveau}. Competences cles: {', '.join(tech_stack[:6])}."
    )
    description_en = (
        f"Profile via Pinecone embeddings. Detected domain: {domaine_en}. "
        f"Level: {niveau}. Key skills: {', '.join(tech_stack[:6])}."
    )

    scores_domaines_en = [
        {"domaine": _map_domain_to_english(d), "score": float(s)}
        for d, s in scores
    ]

    return {
        "position": domaine,
        "position_en": domaine_en,
        "description": description,
        "description_en": description_en,
        "experience_years": experience_years,
        "tech_stack": tech_stack[:12],
        "competences": competences,
        "domaine": domaine,
        "domaine_en": domaine_en,
        "niveau": niveau,
        "scores_domaines": [{"domaine": d, "score": float(s)} for d, s in scores],
        "scores_domaines_en": scores_domaines_en,
    }
