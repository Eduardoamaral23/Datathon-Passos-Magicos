from __future__ import annotations

from pathlib import Path
from typing import Any
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
MODEL_PATH = ROOT / "models" / "modelo_risco_defasagem.joblib"

INDICATOR_COLUMNS = ["IDA", "IEG", "IPS", "IPP", "IAA", "IPV", "IAN", "INDE"]
FEATURE_COLUMNS = [
    "IDA",
    "IEG",
    "IPS",
    "IPP",
    "IAA",
    "IPV",
    "Saude_Academica",
    "Bem_Estar_Psico",
    "Risco_Composto",
    "Gap_Expectativa_Realidade",
    "Coerencia_Autoavaliacao",
]


def build_feature_params(data: pd.DataFrame) -> dict[str, float]:
    return {
        "ida_q33": float(data["IDA"].quantile(0.33)),
        "ieg_q33": float(data["IEG"].quantile(0.33)),
    }


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return ascii_text.strip().upper()


def _standardize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    rename_map: dict[str, str] = {}
    current_inde_aliases = {
        2022: {"INDE 22", "INDE 2022"},
        2023: {"INDE 2023"},
        2024: {"INDE 2024"},
    }

    for col in df.columns:
        cleaned = str(col).strip()
        upper = _normalize_name(cleaned)

        if upper in current_inde_aliases[year]:
            rename_map[col] = "INDE"
        elif upper in {"MATEM", "MATEMATICA", "MATEMATICA 22"}:
            rename_map[col] = "Mat"
        elif upper in {"PORTUG", "PORTUGUES", "PORTUGUES 22"}:
            rename_map[col] = "Por"
        elif upper in {"INGLES", "INGLES 22"}:
            rename_map[col] = "Ing"
        elif upper.startswith("IDADE"):
            rename_map[col] = "Idade"
        elif upper in {"FASE IDEAL", "FASE_IDEAL"}:
            rename_map[col] = "Fase Ideal"
        elif upper.startswith("DEFAS"):
            rename_map[col] = "Defasagem"

    return df.rename(columns=rename_map)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Base nao encontrada em {path}. Coloque o arquivo Excel na pasta data/ antes de rodar."
        )

    frames = []
    for year in [2022, 2023, 2024]:
        sheet = f"PEDE{year}"
        frame = pd.read_excel(path, sheet_name=sheet)
        frame = _standardize_columns(frame, year)
        frame["Ano"] = year
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    for col in INDICATOR_COLUMNS + ["Mat", "Por", "Ing", "Idade"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def categorize_ian(value: float) -> str:
    if pd.isna(value):
        return "Sem informacao"
    if value < 3:
        return "Severamente defasado"
    if value < 5:
        return "Moderadamente defasado"
    if value < 7:
        return "Levemente defasado"
    return "Adequado"


def add_features(data: pd.DataFrame, params: dict[str, float] | None = None) -> pd.DataFrame:
    required = ["IDA", "IEG", "IPS", "IPP", "IAA", "IPV", "IAN"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    model_data = data.dropna(subset=required).copy()
    if params is None:
        params = build_feature_params(model_data)

    model_data["Categoria_IAN"] = model_data["IAN"].apply(categorize_ian)

    # IAN representa adequacao do nivel: quanto menor, maior a defasagem.
    # Ele define o alvo, mas nao entra como feature do modelo para evitar vazamento de informacao.
    model_data["Risco_Defasagem"] = (model_data["IAN"] < 7).astype(int)
    model_data["Saude_Academica"] = (model_data["IDA"] + model_data["IEG"]) / 2
    model_data["Bem_Estar_Psico"] = (model_data["IPS"] + model_data["IAA"]) / 2
    model_data["Risco_Composto"] = (
        (model_data["IDA"] < params["ida_q33"]).astype(int)
        + (model_data["IEG"] < params["ieg_q33"]).astype(int)
    )
    model_data["Gap_Expectativa_Realidade"] = model_data["IPV"] - model_data["IDA"]
    model_data["Coerencia_Autoavaliacao"] = (model_data["IAA"] - model_data["IDA"]).abs()
    return model_data


def train_models(model_data: pd.DataFrame) -> dict[str, Any]:
    X = model_data[FEATURE_COLUMNS]
    y = model_data["Risco_Defasagem"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "classification_report": classification_report(y_test, pred, zero_division=0),
        }

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best = results[best_name]

    return {
        "model": best["model"],
        "model_name": best_name,
        "features": FEATURE_COLUMNS,
        "metrics": {name: {k: v for k, v in result.items() if k != "model"} for name, result in results.items()},
        "target_definition": "Risco_Defasagem = 1 quando IAN < 7",
    }


def score_risk(bundle: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    featured = add_features(rows, bundle.get("feature_params"))
    return bundle["model"].predict_proba(featured[bundle["features"]])[:, 1]


def main() -> None:
    data = load_dataset()
    feature_params = build_feature_params(data.dropna(subset=["IDA", "IEG"]))
    model_data = add_features(data, feature_params)
    bundle = train_models(model_data)
    bundle["feature_params"] = feature_params

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"Registros carregados: {len(data)}")
    print(f"Registros usados na modelagem: {len(model_data)}")
    print(f"Modelo escolhido: {bundle['model_name']}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    for name, metrics in bundle["metrics"].items():
        print(f"\n{name}")
        print(f"Acuracia: {metrics['accuracy']:.3f}")
        print(f"Precisao: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
