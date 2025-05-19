# src/analisar_dados.py
import pandas as pd

def resumo(df, nome):
    print(f"\n===== {nome} =====")
    print(f"Shape: {df.shape}")
    print("\nAmostra:")
    print(df.sample(min(5, len(df))))
    print("\n% de valores faltantes:")
    print(df.isnull().mean().sort_values(ascending=False).head(5))

if __name__ == "__main__":
    # Lê o Parquet final de features
    df_features = pd.read_parquet("dados/trusted/features.parquet")
    resumo(df_features, "Features (Parquet final)")

    # Lê os dados brutos para recuperar áreas
    df_applicants = pd.read_json("dados/raw/applicants.json", orient="index")
    df_vagas = pd.read_json("dados/raw/vagas.json", orient="index")

    # Extrai identificadores e áreas dos candidatos
    df_applicants["codigo_profissional"] = df_applicants["infos_basicas"].apply(
        lambda x: x.get("codigo_profissional") if isinstance(x, dict) else None
    )
    df_applicants["area_atuacao"] = df_applicants["informacoes_profissionais"].apply(
        lambda x: x.get("area_atuacao") if isinstance(x, dict) else None
    )

    # Extrai identificadores e áreas das vagas
    df_vagas["vaga_id"] = df_vagas.index.astype(str)
    df_vagas["area_vaga"] = df_vagas["perfil_vaga"].apply(
        lambda x: x.get("area_atuacao") if isinstance(x, dict) else None
    )

    # Merge com as features para verificar consistência da área
    df_merge = df_features.merge(
        df_applicants[["codigo_profissional", "area_atuacao"]],
        on="codigo_profissional", how="left"
    ).merge(
        df_vagas[["vaga_id", "area_vaga"]],
        on="vaga_id", how="left"
    )

    # Verifica inconsistências: area_match == 1 mas áreas não são literalmente iguais
    inconsistentes = df_merge[
        (df_merge["area_match"] == 1) & (df_merge["area_atuacao"] != df_merge["area_vaga"])
    ]
    print("\n🔍 Casos inconsistentes com area_match == 1 mas áreas diferentes:")
    print(inconsistentes[["codigo_profissional", "vaga_id", "area_atuacao", "area_vaga"]].sample(min(5, len(inconsistentes))))

    # Frequência geral das áreas
    print("\n📊 Distribuição das áreas dos candidatos:")
    print(df_merge["area_atuacao"].value_counts(dropna=False))

    print("\n📊 Distribuição das áreas das vagas:")
    print(df_merge["area_vaga"].value_counts(dropna=False))

    # Proporção real onde as áreas batem literalmente
    proporcao_igualdade = (df_merge["area_atuacao"] == df_merge["area_vaga"]).mean()
    print(f"\n✅ Proporção de pares onde áreas são literalmente iguais: {proporcao_igualdade:.2%}")

    # Validação da consistência da feature area_match
    area_match_real = (df_merge["area_match"] == (df_merge["area_atuacao"] == df_merge["area_vaga"])).mean()
    print(f"\n✅ area_match está correta em: {area_match_real:.2%} dos casos")

    # Distribuição da variável target
    print("\n🎯 Distribuição do target (match):")
    print(df_features['match'].value_counts(normalize=True))
