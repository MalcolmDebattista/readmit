"""The MLflow model registry (Diary Entry 2, Session 06).

  readmit-registry register             register the run readmit-train just recorded: only if it passed the gate on the
                                        frozen holdout and came from committed code
  readmit-registry alias champion 2     point the champion alias at version 2 (a rollback is the same command)
  readmit-registry card 2 docs/model_card.md
  readmit-registry show                 versions, their holdout scores and aliases

The registered model is the whole bundle (LightGBM model, feature list, code tables, KB lookups), because serving
needs all of it to turn a raw discharge into the champion's features exactly as training did.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from readmit.config import load_config, setup_logging

logger = logging.getLogger(__name__)

CARD_LIMIT = 5000      # MLflow keeps a version's description in a varchar(5000) column
LAST_RUN = "models/last_run_id"

try:
    import mlflow.pyfunc

    class BundleModel(mlflow.pyfunc.PythonModel):
        """The bundle as an MLflow model: predict() takes raw discharge records, as the API does."""

        def load_context(self, context):
            from readmit.model import load_bundle
            self.bundle = load_bundle(Path(context.artifacts["bundle"]))

        def predict(self, context, model_input, params=None):
            from readmit.serve import score_records
            return score_records(model_input.to_dict("records"), self.bundle)
except ImportError:          # mlflow not installed: the registry is not used
    BundleModel = None


def load_registered_bundle(name, version):
    """Download a registered version's bundle file and load it (no need to unpickle the wrapper class)."""
    import mlflow
    from readmit.model import load_bundle
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{name}/{version}"))
    return load_bundle(next(local.rglob("bundle.joblib")))


def register(client, name, run_id=None):
    import mlflow
    if run_id is None:
        if not Path(LAST_RUN).exists():
            logger.error("No run recorded here yet: run readmit-train first, or pass --run-id")
            return 1
        run_id = Path(LAST_RUN).read_text().strip()
    tags = client.get_run(run_id).data.tags
    if tags.get("gate_passed") != "true":
        logger.error("Run %s did not pass the gate on the frozen holdout: tracked, but not registered", run_id)
        return 1
    if tags.get("git_dirty") != "false":
        logger.error("Run %s came from uncommitted changes: commit, train again, then register", run_id)
        return 1
    version = mlflow.register_model(tags["model_uri"], name)
    logger.info("Registered run %s as %s version %s", run_id, version.name, version.version)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Work with the readmission model registry.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register").add_argument("--run-id")
    alias = sub.add_parser("alias")
    alias.add_argument("alias")
    alias.add_argument("version")
    card = sub.add_parser("card")
    card.add_argument("version")
    card.add_argument("path")
    sub.add_parser("show")
    args = parser.parse_args(argv)

    import mlflow
    from mlflow import MlflowClient

    load_dotenv()
    setup_logging()
    config = load_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    name = config["mlflow"]["model_name"]

    if args.command == "register":
        sys.exit(register(client, name, args.run_id))
    if args.command == "alias":
        client.set_registered_model_alias(name, args.alias, args.version)
        logger.info("%s@%s -> version %s", name, args.alias, args.version)
    elif args.command == "card":
        text = Path(args.path).read_text(encoding="utf-8")
        if len(text) > CARD_LIMIT:
            logger.error("%s has %d characters; the registry keeps at most %d", args.path, len(text), CARD_LIMIT)
            sys.exit(1)
        client.update_model_version(name, args.version, description=text)
        logger.info("Model card attached to %s version %s", name, args.version)
    elif args.command == "show":
        if not client.search_registered_models(filter_string=f"name='{name}'"):
            logger.error("No model called %s is registered yet: run make register", name)
            sys.exit(1)
        aliases = {}
        for alias_name, version in client.get_registered_model(name).aliases.items():
            aliases.setdefault(version, []).append(alias_name)
        for v in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
            run = client.get_run(v.run_id)
            m, p = run.data.metrics, run.data.params
            print(f"version {v.version:>3}  run {v.run_id[:8]}  precision@capacity {m.get('holdout_precision_at_capacity', float('nan')):.3f}  "
                  f"AUPRC {m.get('holdout_auprc', float('nan')):.3f}  train_start_q {p.get('train_start_q'):<5}  "
                  f"{' '.join(aliases.get(v.version, []))}")
    sys.exit(0)


if __name__ == "__main__":
    main()
