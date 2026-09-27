PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",
        "results/rag_eval.json",
        "results/rag_repair.json",
        "results/rag_mix_repair.json",


rule data:
    output:
        "data/processed/sft.parquet",
        "data/processed/dpo.parquet",
        "data/processed/rl.parquet",
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


rule grpo:
    input:
        rl="data/processed/rl.parquet",
        init=rules.dpo.output,   # repair attempt on the collapsed checkpoint
        ref=rules.sft.output,    # KL measured against the frozen SFT policy
    output:
        "data/processed/grpo.pt",
    shell:
        "{PP} {PY} -m posttrain.grpo {input.rl} {input.init} {input.ref} {output}"


rule evaluate:
    input:
        ev="data/processed/eval.parquet",
        sft=rules.sft.output,
        dpo=rules.dpo.output,
        grpo=rules.grpo.output,
    output:
        "results/summary.json",
    shell:
        "{PP} {PY} -m posttrain.evaluate "
        "{input.ev} {input.sft} {input.dpo} {output} {input.grpo}"


rule eval_rag:
    input:
        ev="data/processed/eval.parquet",
        sft_df="data/processed/sft.parquet",
        grpo="data/processed/grpo_s105.pt",
    output:
        "results/rag_eval.json",
    shell:
        "{PP} {PY} -m posttrain.eval_rag "
        "{input.ev} {input.sft_df} {output}"


rule sft_rag:
    input:
        "data/processed/sft.parquet",
    output:
        parquet="data/processed/sft_rag.parquet",
        model="data/processed/sft_rag.pt",
    shell:
        "{PP} {PY} -m posttrain.sft_rag "
        "{input} {output.parquet} {output.model}"


rule rag_repair:
    input:
        ev="data/processed/eval.parquet",
        sft_df="data/processed/sft.parquet",
        model="data/processed/sft_rag.pt",
    output:
        "results/rag_repair.json",
    shell:
        "{PP} {PY} -m posttrain.rag_repair "
        "{input.ev} {input.sft_df} {input.model} {output}"


rule sft_mix:
    input:
        "data/processed/sft.parquet",
    output:
        parquet="data/processed/sft_mix.parquet",
        model="data/processed/sft_mix.pt",
    shell:
        "{PP} {PY} -m posttrain.sft_mix "
        "{input} {output.parquet} {output.model}"


rule rag_mix_repair:
    input:
        ev="data/processed/eval.parquet",
        sft_df="data/processed/sft.parquet",
        model="data/processed/sft_mix.pt",
    output:
        "results/rag_mix_repair.json",
    shell:
        "{PP} {PY} -m posttrain.rag_repair "
        "{input.ev} {input.sft_df} {input.model} {output}"
