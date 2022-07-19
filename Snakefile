import pathlib
from pipeline_config import *

from os.path import join
configfile: "config.yaml"
cfg = ParsedConfig(config)


# rule all:
#     input:
#         expand(join(cfg.ROOT, "results.tsv.gz"))


rule all:
    input:
        expand(str(cfg.ROOT) + '/' + str('{data_scenario}') + '/' + str('{gene_names}') + '/results.tsv.gz', gene_names=cfg.GENE_NAMES, data_scenario=cfg.DATA_SCENARIOS)


rule metrics:
    input: cfg.get_all_file_patterns("metrics")
    message: "Collect all integrated metrics"

all_metrics = rules.metrics.input

rule merge_metrics:
    input:
        tables = all_metrics,
        results = expand(str(cfg.ROOT) + '/' + str('{gene_names}') + '/results.tsv.gz', gene_names=cfg.GENE_NAMES),
        script = "scripts/merge_metrics.py"
    output:
    message: "Merge all metrics"
    params:
        cmd = f"conda run -n {cfg.py_env} python"
    shell: "{params.cmd} {input.script} -o {output} --root {cfg.ROOT}"

rule fit_model:
    input:
        script = "scripts/fit_model.py",
        queries = expand(str(cfg.ROOT) + '/' + '{gene_names}/queries.tsv', gene_names=cfg.GENE_NAMES)
    output:
        metrics = expand(str(cfg.ROOT) + '/' + str('{gene_names}/metrics.tsv'), gene_names=cfg.GENE_NAMES)

    message: "Merge all metrics"
    params:
        cmd = f"conda run -n {cfg.py_env} python",
        model = expand(str(cfg.ROOT) + '/' + str('{gene_names}/models'), gene_names=cfg.GENE_NAMES)   ,
    shell:
        """
        {params.cmd} {input.script} -i {input.queries} --out_model {params.model} --out_tsv {output.metrics}
        """

rule data_prepare:
    input:
        script = "scripts/data_prepare.py",
        # adata = expand(str('{data_scenario}'), data_scenario=cfg.DATA_SCENARIOS),
    output:
        queries = expand(str(cfg.ROOT) + '/{gene_names}/queries.tsv', gene_names=cfg.GENE_NAMES)
    message:
        """
        Preparing adata
        wildcards: {wildcards}
        parameters: {params}
        output: {output}
        """
    params:
        cmd       = f"conda run -n {cfg.py_env} python",
        tf_name = expand('{gene_names}', gene_names=cfg.GENE_NAMES),
        annotations = f"{cfg.ANNOTATIONS}"
    shell:
        """
        {params.cmd} {input.script} --annotations {params.annotations} --tf_name {params.tf_name} -o {output.queries}
        """
