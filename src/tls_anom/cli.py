import click, os
from tls_anom.appcontext import AppContext
from tls_anom.pipelines import extract, label, featurize, preprocess, train, predict, evaluate

@click.group()
def cli():
    """TLS Anomaly Detection – Unified CLI"""

@cli.command()
@click.option("--dataset", required=True, help="Path to zeek")
@click.option("--name", required=True, type=click.Choice(["normal","mix","botnet"]))
@click.option("--config", "config_path", default="config/default.yaml")
@click.option("--env", "overlay", default=None, help="Optional overlay config, e.g., config/dev.yaml")
@click.option("--stages", default=None, help="Comma-separated stages to run")

def run(dataset, name, config_path, overlay, stages):
    ctx = AppContext.from_yaml(config_path, overlay=overlay)
    logger = ctx.logger
    logger.info(f"Run: dataset={dataset}, name={name}")

    stage_list = stages.split(",") if stages else ctx.cfg["pipeline"]["stages"]

    base_prefix = os.path.join(ctx.cfg["paths"]["outputs_dir"], name)
    os.makedirs(base_prefix, exist_ok=True)

    processed_csv = os.path.join(ctx.cfg["paths"]["processed_dir"], f"{name}.csv")
    if "extract" in stage_list:
        extract.run(ctx, dataset, processed_csv)

    labeled_csv = os.path.join(ctx.cfg["paths"]["processed_dir"], f"{name}.labeled.csv")
    if "label" in stage_list:
        label.run(ctx, processed_csv, name, labeled_csv)

    feat_csv = os.path.join(ctx.cfg["paths"]["features_dir"], f"{name}.features.csv")
    if "featurize" in stage_list:
        featurize.run(ctx, labeled_csv, feat_csv)

    scaled_csv = os.path.join(ctx.cfg["paths"]["features_dir"], f"{name}.scaled.csv")
    if "preprocess" in stage_list:
        preprocess.run(ctx, feat_csv, scaled_csv)

    model_name = ctx.cfg["model"].get("train_on", "normal")

    model_path = os.path.join(
        ctx.cfg["paths"]["models_dir"],
        f"{model_name}.{ctx.cfg['model']['kind']}.joblib"
    )

    if "train" in stage_list:
        train.run(ctx, scaled_csv, model_path)

    pred_csv = os.path.join(ctx.cfg["paths"]["outputs_dir"], "predictions", f"{name}.pred.csv")
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    if "predict" in stage_list:
        predict.run(ctx, scaled_csv, model_path, pred_csv)

    if "evaluate" in stage_list:
        evaluate.run(ctx, scaled_csv, model_path, pred_csv)

    logger.info("Pipeline finished.")

def main():
    cli()
