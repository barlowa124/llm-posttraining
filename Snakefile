PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",


rule data:
    output:
        "data/processed/sft.parquet",
        "data/processed/dpo.parquet",
        "data/processed/eval.parquet",
    shell:
        "{PP} {PY} -m posttrain.data data/processed"


rule sft:
    input:
        "data/processed/sft.parquet",
    output:
        "data/processed/sft.pt",
    shell:
        "{PP} {PY} -m posttrain.sft {input} {output}"


rule dpo:
    input:
        pairs="data/processed/dpo.parquet",
        sft=rules.sft.output,
    output:
        "data/processed/dpo.pt",
    shell:
        "{PP} {PY} -m posttrain.dpo {input.pairs} {input.sft} {output}"


rule evaluate:
    input:
        ev="data/processed/eval.parquet",
        sft=rules.sft.output,
        dpo=rules.dpo.output,
    output:
        "results/summary.json",
    shell:
        "{PP} {PY} -m posttrain.evaluate "
        "{input.ev} {input.sft} {input.dpo} {output}"
