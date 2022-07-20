import pathlib
from pipeline_config import *

from os.path import join

configfile: "config.yaml"
cfg = ParsedConfig(config)


rule all:
    input:
        files = expand(str(cfg.ROOT) + '/{experiment}/{gene_names}/metrics.tsv',
                       gene_names=cfg.GENE_NAMES, experiment=cfg.EXPERIMENT),
        script = "scripts/merge_metrics.py",
    output:
        results = str(cfg.ROOT) + '/results.tsv.gz'
    params:
        cmd = f"conda run -n {cfg.py_env} python",
    shell: "{params.cmd} {input.script} -i {input.files} -o {output.results}"

# rule metrics:
#     input: cfg.get_all_file_patterns("metrics")
#     message: "Collect all integrated metrics"

# all_metrics = rules.metrics.input

# rule merge_metrics:
#     input:
#         tables = all_metrics,
#         results = expand(str(cfg.ROOT) + '/' + str('{gene_names}') + '/results.tsv.gz', gene_names=cfg.GENE_NAMES),
#         script = "scripts/merge_metrics.py"
#     output:
#     message: "Merge all metrics"
#     params:
#         cmd = f"conda run -n {cfg.py_env} python"
#     shell: "{params.cmd} {input.script} -o {output} --root {cfg.ROOT}"

rule fit_model:
    input:
        script = "scripts/fit_model.py",
        queries = str(cfg.ROOT) + '/{experiment}/{gene_names}/queries.tsv'
    output:
        metrics = str(cfg.ROOT) + '/{experiment}/{gene_names}/metrics.tsv',
    message: "Merge all metrics"
    log:
        str(cfg.ROOT) + '/{experiment}/{gene_names}/fit_model.out.txt'
    params:
        cmd = f"conda run -n {cfg.py_env} python",
        model = str(cfg.ROOT) + '/{experiment}/{gene_names}/models',
        n_epochs = expand('{n_epochs}', n_epochs=cfg.HYPERPARAM['n_epochs']),
    shell:
        """
        {params.cmd} {input.script} -i {input.queries} --out_model {params.model} --out_tsv {output.metrics} \
            --n_epochs {params.n_epochs} 1> {log}
        """

rule data_prepare:
    input:
        script = "scripts/data_prepare.py",
    output:
        queries = str(cfg.ROOT) + '/{experiment}/{gene_names}/queries.tsv'
    message:
        """
        Preparing adata
        wildcards: {wildcards}
        parameters: {params}
        output: {output}
        """
    params:
        cmd       = f"conda run -n {cfg.py_env} python",
        # tf_name = expand('{gene_names}', gene_names=cfg.GENE_NAMES),
        # tf_name = lambda wildcards: 'wildcards.gene_names',
        tf_name = '{gene_names}'.split('//')[0],
        annot = f"{cfg.ANNOTATIONS}",
        n_sample = expand('{n_sample}', n_sample=cfg.HYPERPARAM['n_sample']),
    shell:
        """
        {params.cmd} {input.script} --annot {params.annot} --tf_name {params.tf_name} -o {output.queries} \
            --n_sample {params.n_sample}
        """

