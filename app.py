from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.datathon_pipeline import FEATURE_COLUMNS, MODEL_PATH, add_features


REQUIRED_COLUMNS = ["IDA", "IEG", "IPS", "IPP", "IAA", "IPV"]
OPTIONAL_COLUMNS = ["IAN"]


st.set_page_config(page_title="Risco de Defasagem", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.1rem;
    }
    .risk-label {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.92rem;
        margin-top: 4px;
    }
    .risk-low {
        background: #dcfce7;
        color: #166534;
    }
    .risk-medium {
        background: #fef3c7;
        color: #92400e;
    }
    .risk-high {
        background: #fee2e2;
        color: #991b1b;
    }
    .quick-read {
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        background: #eff6ff;
        border-radius: 6px;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def classify_risk(probability: float) -> tuple[str, str, str]:
    if probability >= 0.70:
        return "Alto risco", "risk-high", "Priorizar intervenção individual e reforço acadêmico."
    if probability >= 0.50:
        return "Risco moderado", "risk-medium", "Monitorar evolução e revisar plano de apoio."
    return "Baixo risco", "risk-low", "Manter acompanhamento regular e estratégias de engajamento."


def score_rows(bundle, rows: pd.DataFrame) -> pd.DataFrame:
    scoring_rows = rows.copy()
    if "IAN" not in scoring_rows.columns:
        scoring_rows["IAN"] = 7.0

    featured = add_features(scoring_rows, bundle.get("feature_params"))
    probabilities = bundle["model"].predict_proba(featured[FEATURE_COLUMNS])[:, 1]
    result = rows.loc[featured.index].copy()
    result["Probabilidade_Risco"] = probabilities
    result["Classificacao_Risco"] = [classify_risk(p)[0] for p in probabilities]
    if "IAN" in rows.columns:
        result["Categoria_IAN"] = featured["Categoria_IAN"]
    return result


def indicator_slider(label: str, value: float, help_text: str | None = None) -> float:
    return st.slider(label, 0.0, 10.0, value, 0.1, help=help_text)


bundle = load_model()

st.title("Risco de Defasagem Educacional")
st.caption("Associação Passos Mágicos | Datathon FIAP")

if bundle is None:
    st.warning(
        "Modelo treinado não encontrado. Execute `python src/datathon_pipeline.py` "
        "para gerar `models/modelo_risco_defasagem.joblib`."
    )
    st.stop()

metrics = bundle.get("metrics", {}).get(bundle.get("model_name", ""), {})

tab_individual, tab_lote = st.tabs(["Previsão individual", "Previsão por arquivo"])

with tab_individual:
    form_col, result_col = st.columns([1.2, 1], gap="large")

    with form_col:
        st.subheader("Dados do aluno")
        meta_cols = st.columns(4)
        with meta_cols[0]:
            ano = st.selectbox(
                "Ano avaliado",
                [2024, 2023, 2022],
                index=0,
                help="Ano em que os indicadores do aluno foram registrados.",
            )
        with meta_cols[1]:
            idade = st.number_input("Idade", min_value=6, max_value=25, value=12, step=1)
        with meta_cols[2]:
            fase = st.selectbox("Fase/Turma", ["0", "1", "2", "3", "4", "5", "6", "7", "8"], index=0)
        with meta_cols[3]:
            genero = st.selectbox("Gênero", ["Feminino", "Masculino", "Não informado"], index=0)

        meta_cols_2 = st.columns(3)
        with meta_cols_2[0]:
            ano_ingresso = st.number_input(
                "Ano de entrada",
                min_value=1990,
                max_value=2026,
                value=2022,
                step=1,
                help="Ano em que o aluno começou a participar do programa.",
            )
        with meta_cols_2[1]:
            instituicao = st.selectbox(
                "Instituição",
                ["Escola pública", "Escola privada", "Concluiu o 3º EM", "Não informado"],
                index=0,
            )
        with meta_cols_2[2]:
            pedra = st.selectbox("Pedra", ["Quartzo", "Ágata", "Ametista", "Topázio", "Não informado"], index=1)

        st.subheader("Indicadores")
        st.caption("Somente estes indicadores entram no modelo. O IAN foi usado apenas para criar o alvo de treino.")

        ind_cols = st.columns(4)
        with ind_cols[0]:
            iaa = indicator_slider("IAA", 8.8, "Autoavaliação do aluno")
            ida = indicator_slider("IDA", 6.8, "Desempenho acadêmico")
        with ind_cols[1]:
            ieg = indicator_slider("IEG", 8.9, "Engajamento nas atividades")
            ipv = indicator_slider("IPV", 7.8, "Ponto de virada")
        with ind_cols[2]:
            ips = indicator_slider("IPS", 6.9, "Aspectos psicossociais")
        with ind_cols[3]:
            ipp = indicator_slider("IPP", 0.0, "Aspectos psicopedagógicos")
            inde = indicator_slider("INDE", 7.4, "Índice geral, exibido apenas como contexto")

    row = pd.DataFrame(
        [
            {
                "Ano": ano,
                "Idade": idade,
                "Fase/Turma": fase,
                "Gênero": genero,
                "Ano ingresso": ano_ingresso,
                "Instituição": instituicao,
                "Pedra": pedra,
                "IAA": iaa,
                "IEG": ieg,
                "IPS": ips,
                "IPP": ipp,
                "IDA": ida,
                "IPV": ipv,
                "INDE": inde,
            }
        ]
    )

    prediction = score_rows(bundle, row).iloc[0]
    probability = float(prediction["Probabilidade_Risco"])
    level, risk_class, recommendation = classify_risk(probability)
    featured = add_features(row.assign(IAN=7.0), bundle.get("feature_params"))

    with result_col:
        st.subheader("Resultado")
        st.write("Probabilidade de risco no próximo ciclo")
        st.markdown(f"## {probability:.1%}")
        st.markdown(f'<span class="risk-label {risk_class}">{level}</span>', unsafe_allow_html=True)
        st.progress(min(max(probability, 0.0), 1.0))
        st.write(recommendation)

        st.subheader("Qualidade do modelo")
        metric_cols = st.columns(3)
        metric_cols[0].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        metric_cols[1].metric("Recall", f"{metrics.get('recall', 0):.3f}")
        metric_cols[2].metric("F1", f"{metrics.get('f1', 0):.3f}")

        st.subheader("Leitura rápida")
        st.markdown(
            f"""
            <div class="quick-read">
            O resultado deve ser usado como apoio à decisão, não como substituto da avaliação pedagógica.
            O modelo estima risco a partir dos demais indicadores, sem usar o IAN como entrada.
            Probabilidades altas indicam prioridade para revisão do plano de acompanhamento.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Ver features calculadas"):
            feature_view = featured[
                [
                    "Saude_Academica",
                    "Bem_Estar_Psico",
                    "Risco_Composto",
                    "Gap_Expectativa_Realidade",
                    "Coerencia_Autoavaliacao",
                    "Categoria_IAN",
                ]
            ].rename(
                columns={
                    "Saude_Academica": "Saúde acadêmica",
                    "Bem_Estar_Psico": "Bem-estar psicossocial",
                    "Risco_Composto": "Risco composto",
                    "Gap_Expectativa_Realidade": "Gap expectativa-realidade",
                    "Coerencia_Autoavaliacao": "Coerência da autoavaliação",
                    "Categoria_IAN": "Categoria IAN",
                }
            )
            st.dataframe(feature_view, hide_index=True, width="stretch")

with tab_lote:
    st.subheader("Previsão em lote")
    st.write(
        "Envie um arquivo `.csv` ou `.xlsx` com as colunas "
        "`IDA`, `IEG`, `IPS`, `IPP`, `IAA` e `IPV`. A coluna `IAN` é opcional para leitura da categoria."
    )

    uploaded = st.file_uploader("Arquivo de alunos", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                input_data = pd.read_csv(uploaded)
            else:
                input_data = pd.read_excel(uploaded)

            missing = [col for col in REQUIRED_COLUMNS if col not in input_data.columns]
            if missing:
                st.error(f"Colunas ausentes no arquivo: {', '.join(missing)}")
            else:
                scored = score_rows(bundle, input_data)
                st.success(f"{len(scored)} registros processados.")

                summary = (
                    scored["Classificacao_Risco"]
                    .value_counts()
                    .rename_axis("Classificacao")
                    .reset_index(name="Quantidade")
                )
                st.dataframe(summary, hide_index=True, width="stretch")

                display_cols = [
                    *REQUIRED_COLUMNS,
                    "Probabilidade_Risco",
                    "Classificacao_Risco",
                ]
                if "Categoria_IAN" in scored.columns:
                    display_cols.insert(6, "Categoria_IAN")
                st.dataframe(scored[display_cols], hide_index=True, width="stretch")

                csv = scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Baixar resultado em CSV",
                    data=csv,
                    file_name="previsao_risco_defasagem.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Não foi possível processar o arquivo: {exc}")
