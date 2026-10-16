"""Random Forest para data/processed/binario/*_tree.parquet.

Coloque este arquivo na pasta src/ do repositório arbovirus-brasil-eda.
Da raiz do repositório, execute: python src/05_random_forest_binario.py
Requer: numpy, pandas, pyarrow, scikit-learn, matplotlib.

O script presume y=1 para Dengue e y=0 para Descartado/Inconclusivo.
Esta codificação foi conferida em src/02_preprocess.py do repositório fornecido.
Por padrão, exclui as colunas hosp_* para avaliar a hipótese de triagem.
Use --com-hospitalizacao apenas para uma análise secundária do pipeline original.
Use --sem-contexto para medir o sinal restante sem mês e macrorregião.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_split(folder: Path, split: str) -> Path:
    aliases = {"train": ["train", "treino"],
               "val": ["val", "valid", "validation", "validacao"],
               "test": ["test", "teste"]}[split]
    files = sorted(folder.glob("*_tree.parquet"))
    found = [p for p in files if any(re.fullmatch(a, p.stem.split("_tree")[0], re.I)
                                   for a in aliases)]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Esperava um arquivo {split}_tree.parquet em {folder}. "
            f"Encontrei: {[p.name for p in files]}. Confira os nomes antes de continuar."
        )
    return found[0]


def read_sample(path: Path, limit: int, seed: int) -> pd.DataFrame:
    """Lê em lotes, sorteando linhas sem carregar o Parquet inteiro na RAM."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    if total == 0:
        raise ValueError(f"Arquivo vazio: {path}")
    prob = min(1.0, limit / total)
    rng = np.random.default_rng(seed)
    pieces = []
    for batch in pf.iter_batches(batch_size=50000):
        choose = rng.random(batch.num_rows) < prob
        if choose.any():
            pieces.append(batch.to_pandas().loc[choose])
    if not pieces:
        raise ValueError(f"A amostra ficou vazia: {path}; aumente o limite.")
    sample = pd.concat(pieces, ignore_index=True)
    print(f"{path.name}: {len(sample):,} de {total:,} linhas; fração ~{prob:.3%}")
    return sample


def check_and_prepare(parts: dict[str, pd.DataFrame], include_hospitalization: bool,
                      exclude_context: bool):
    original = [c for c in parts["train"].columns if c != "y"]
    hosp_cols = [c for c in original if c.lower().startswith("hosp_")]
    context_cols = [c for c in original if c in {"mes_sin", "mes_cos"} or c.startswith("reg_")]
    excluded = (set() if include_hospitalization else set(hosp_cols))
    if exclude_context:
        excluded.update(context_cols)
        print("Mês e macrorregião excluídos:", context_cols)
    reference = [c for c in original if c not in excluded]
    if hosp_cols:
        print("Hospitalização:", "incluída em análise secundária" if include_hospitalization
              else "excluída do modelo de triagem", hosp_cols)
    if not reference:
        raise ValueError("Nenhum preditor encontrado no treino.")
    forbidden = re.compile(
        r"CLASSI_FIN|ID_AGRAVO|RESUL_|_NS1|_PCR|SORO|EVOLUCAO|"
        r"ENCERRA|HOSPITALIZ|TPAUTOCTO|COUFINF|COPAISINF|COMUNINF|NU_ANO",
        re.I,
    )
    suspicious = [c for c in reference if forbidden.search(c)]
    if suspicious:
        raise ValueError(
            "Possíveis variáveis de vazamento/identificação ainda estão no arquivo tree: "
            f"{suspicious}. Discuta com o grupo e revise 02_preprocess.py."
        )

    x, y = {}, {}
    for name, frame in parts.items():
        if "y" not in frame:
            raise ValueError(f"Coluna y ausente no arquivo {name}.")
        features = [c for c in frame.columns if c != "y"]
        if set(features) != set(original):
            raise ValueError(f"Preditores de {name} diferem do treino. "
                             f"Só em {name}: {sorted(set(features)-set(original))[:10]}; "
                             f"só no treino: {sorted(set(original)-set(features))[:10]}")
        yy = pd.to_numeric(frame["y"], errors="raise")
        if yy.isna().any() or not set(yy.unique()) <= {0, 1} or yy.nunique() != 2:
            raise ValueError(f"y de {name} precisa conter ambas as classes 0 e 1: "
                             f"{frame['y'].value_counts(dropna=False).head().to_dict()}")
        xx = frame[reference].apply(pd.to_numeric, errors="raise")
        if xx.isna().any().any() or not np.isfinite(xx.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Preditores de {name} ainda têm ausentes/infinito. "
                             "Confira o pré-processamento antes de ajustar a floresta.")
        x[name] = xx.astype(np.float32)
        y[name] = yy.astype(np.int8).to_numpy()
        print(f"{name}: {len(yy):,} linhas, dengue={yy.mean():.2%}, "
              f"preditores={len(reference)}")
    return x, y, reference


def score_row(name: str, split: str, y_true, pred, prob) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    vn, fp, fn, vp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(modelo=name, conjunto=split, n=len(y_true),
                prevalencia_dengue=float(np.mean(y_true)),
                VN=int(vn), FP=int(fp), FN=int(fn), VP=int(vp),
                acuracia=accuracy_score(y_true, pred),
                acuracia_balanceada=balanced_accuracy_score(y_true, pred),
                precisao_dengue=precision_score(y_true, pred, zero_division=0),
                recall_dengue=recall_score(y_true, pred, zero_division=0),
                f1_dengue=f1_score(y_true, pred, zero_division=0),
                f1_macro=f1_score(y_true, pred, average="macro", zero_division=0),
                auc_roc=roc_auc_score(y_true, prob),
                average_precision=average_precision_score(y_true, prob))


def run(folder: Path, out: Path, train_max: int, val_max: int,
        test_max: int, seed: int, include_hospitalization: bool = False,
        exclude_context: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                                 RocCurveDisplay)

    paths = {s: find_split(folder, s) for s in ("train", "val", "test")}
    limits = dict(train=train_max, val=val_max, test=test_max)
    parts = {s: read_sample(paths[s], limits[s], seed + i)
             for i, s in enumerate(("train", "val", "test"))}
    x, y, features = check_and_prepare(parts, include_hospitalization, exclude_context)
    out.mkdir(parents=True, exist_ok=True)

    trials = []
    for leaf in (2, 10):
        model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=2,
        )
        model.fit(x["train"], y["train"])
        probability = model.predict_proba(x["val"])[:, list(model.classes_).index(1)]
        row = score_row(f"RF_folha_{leaf}", "validacao", y["val"],
                        (probability >= 0.5).astype(int), probability)
        trials.append(row)
        print(f"Validação: folha={leaf}, AUC={row['auc_roc']:.4f}")

    pd.DataFrame(trials).to_csv(out / "validacao_rf.csv", index=False)
    chosen = max(trials, key=lambda r: r["auc_roc"])
    leaf = int(chosen["modelo"].split("_")[-1])
    x_fit = pd.concat([x["train"], x["val"]], ignore_index=True)
    y_fit = np.concatenate([y["train"], y["val"]])
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf,
                                max_features="sqrt", class_weight="balanced",
                                random_state=seed, n_jobs=2)
    rf.fit(x_fit, y_fit)
    dummy = DummyClassifier(strategy="prior").fit(x_fit, y_fit)

    results = []
    fig, ax = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for name, model in (("Dummy", dummy), (f"RF_folha_{leaf}", rf)):
        prob = model.predict_proba(x["test"])[:, list(model.classes_).index(1)]
        pred = model.predict(x["test"])
        results.append(score_row(name, "teste", y["test"], pred, prob))
        RocCurveDisplay.from_predictions(y["test"], prob, name=name, ax=ax)
        PrecisionRecallDisplay.from_predictions(y["test"], prob, name=name, ax=ax_pr)
        if name != "Dummy":
            cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y["test"], pred, labels=[0, 1],
                display_labels=["Desc./inconc.", "Dengue"], ax=cm_ax,
            )
            cm_fig.tight_layout()
            cm_fig.savefig(out / "matriz_confusao_rf.png", dpi=170)
            plt.close(cm_fig)

    pd.DataFrame(results).to_csv(out / "metricas_teste.csv", index=False)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    fig.tight_layout(); fig.savefig(out / "curva_roc.png", dpi=170); plt.close(fig)
    fig_pr.tight_layout(); fig_pr.savefig(out / "curva_pr.png", dpi=170)
    plt.close(fig_pr)
    pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(25).to_csv(
        out / "importancia_atributos_rf.csv", header=["importancia_impureza"]
    )
    info = dict(cenario="binario", arquivos={s: str(p) for s, p in paths.items()},
                tamanhos_amostrados={s: len(parts[s]) for s in parts},
                seed=seed, folhas_candidatas=[2, 10], folha_escolhida=leaf,
                class_weight="balanced", hospitalizacao_incluida=include_hospitalization,
                mes_e_regiao_incluidos=not exclude_context,
                split_original="aleatorio estratificado 70/15/15 (src/02_preprocess.py)",
                criterio="AUC de validacao", threshold=0.5,
                nota="Importância por impureza é exploratória; pode favorecer atributos com muitas categorias.")
    (out / "protocolo_rf.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(f"Arquivos gerados em {out}")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=root / "data/processed/binario")
    parser.add_argument("--saida", type=Path, default=root / "reports/random_forest_binario")
    parser.add_argument("--train-max", type=int, default=100000)
    parser.add_argument("--val-max", type=int, default=30000)
    parser.add_argument("--test-max", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--com-hospitalizacao", action="store_true",
                        help="Inclui hosp_* somente como análise secundária; pode não estar disponível na triagem")
    parser.add_argument("--sem-contexto", action="store_true",
                        help="Exclui mes_sin, mes_cos e reg_* para verificar dependência de mês/região")
    args = parser.parse_args()
    if min(args.train_max, args.val_max, args.test_max) < 100:
        parser.error("Os limites de amostra devem ser pelo menos 100.")
    run(args.processed, args.saida, args.train_max, args.val_max, args.test_max,
        args.seed, args.com_hospitalizacao, args.sem_contexto)


if __name__ == "__main__":
    main()
