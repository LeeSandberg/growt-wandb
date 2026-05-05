"""W&B callback for Growt structural auditing.

IMPORTANT — IP protection:
This callback extracts feature vectors and sends them to the Growt API.
ALL structural analysis happens server-side via the Growt API.
No engine code is included in this package.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, Optional

import torch
import wandb
from torch.utils.data import DataLoader

from growt_client import (
    AuditResult,
    GrowtClient,
    MetricsResult,
    format_audit_report,
    format_training_trajectory,
)
from growt_wandb.extractor import extract_features

logger = logging.getLogger("growt_wandb")


class GrowtWandbCallback:
    """Growt structural audit with automatic W&B logging.

    Runs transfer audit during and after training, logs all results
    to W&B (metrics, tables, alerts, artifacts), and optionally gates
    artifact promotion.

    Usage::

        from growt_wandb import GrowtWandbCallback

        callback = GrowtWandbCallback(
            api_key="growt_...",
            train_dataloader=train_loader,
        )
        # In your training loop:
        callback.on_train_end(model)

        # Or with PyTorch Lightning:
        # trainer = pl.Trainer(callbacks=[callback])

    Args:
        api_key: Growt API key. Required.
        api_url: Growt API base URL.
        train_dataloader: DataLoader for training data.
        deploy_dataloader: DataLoader for deployment data (defaults to train).
        layer_name: Dot-separated layer name for feature extraction.
        fail_on_red_flag: Raise error if diagnosis is RED_FLAG.
        max_samples: Maximum samples for feature extraction.
        alert_on_red_flag: Send W&B alert on RED_FLAG.
        log_artifact_metadata: Attach audit to W&B model artifacts.
        audit_every_n_epochs: Run periodic audits (0 = final only).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.transferoracle.ai",
        train_dataloader: Optional[DataLoader] = None,
        deploy_dataloader: Optional[DataLoader] = None,
        layer_name: Optional[str] = None,
        fail_on_red_flag: bool = True,
        max_samples: int = 5000,
        alert_on_red_flag: bool = True,
        log_artifact_metadata: bool = True,
        audit_every_n_epochs: int = 0,
    ) -> None:
        self._client = GrowtClient(api_url=api_url, api_key=api_key)
        self._train_dl = train_dataloader
        self._deploy_dl = deploy_dataloader
        self._layer_name = layer_name
        self._fail_on_red_flag = fail_on_red_flag
        self._max_samples = max_samples
        self._alert_on_red = alert_on_red_flag
        self._log_artifact = log_artifact_metadata
        self._audit_every_n = audit_every_n_epochs
        self._audit_history: list[tuple[int, AuditResult]] = []
        self._last_audit: Optional[AuditResult] = None
        self._last_metrics: Optional[MetricsResult] = None

    @property
    def last_audit(self) -> Optional[AuditResult]:
        return self._last_audit

    @property
    def last_metrics(self) -> Optional[MetricsResult]:
        return self._last_metrics

    @property
    def audit_history(self) -> list[tuple[int, AuditResult]]:
        return list(self._audit_history)

    # ------------------------------------------------------------------
    # Standalone API (for custom training loops)
    # ------------------------------------------------------------------

    def audit(
        self,
        model: torch.nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        deploy_dataloader: Optional[DataLoader] = None,
        epoch: int = 0,
    ) -> AuditResult:
        """Run a Growt audit and log results to W&B.

        Args:
            model: Trained PyTorch model.
            train_dataloader: Override training data.
            deploy_dataloader: Override deployment data.
            epoch: Current epoch (for logging).

        Returns:
            AuditResult with diagnosis.
        """
        audit, metrics = self._run_audit(
            model, train_dataloader or self._train_dl, deploy_dataloader or self._deploy_dl,
        )
        self._last_audit = audit
        self._last_metrics = metrics
        self._audit_history.append((epoch, audit))

        print(format_audit_report(audit, metrics, title="GROWT W&B AUDIT"))
        self._log_to_wandb(audit, metrics, epoch)

        if self._fail_on_red_flag and audit.diagnosis == "RED_FLAG":
            raise RuntimeError(
                f"[Growt] Model flagged as RED_FLAG — unsafe to deploy.\n{audit.report}"
            )

        return audit

    def on_train_end(self, model: torch.nn.Module, epoch: int = 0) -> AuditResult:
        """Convenience method — audit + log + alert."""
        return self.audit(model, epoch=epoch)

    def on_epoch_end(self, model: torch.nn.Module, epoch: int) -> Optional[AuditResult]:
        """Periodic audit if configured."""
        if self._audit_every_n <= 0:
            return None
        if epoch > 0 and epoch % self._audit_every_n == 0:
            logger.info("[Growt] Periodic audit at epoch %d...", epoch)
            return self.audit(model, epoch=epoch)
        return None

    # ------------------------------------------------------------------
    # Artifact integration
    # ------------------------------------------------------------------

    def create_audited_artifact(
        self,
        name: str,
        model_path: str,
        artifact_type: str = "model",
        extra_metadata: Optional[dict] = None,
    ) -> Any:
        """Create a W&B artifact with Growt audit metadata attached.

        Args:
            name: Artifact name (e.g. "resnet50-int8").
            model_path: Path to model file(s).
            artifact_type: W&B artifact type (default "model").
            extra_metadata: Additional metadata to attach.

        Returns:
            wandb.Artifact (logged to current run).
        """
        if self._last_audit is None:
            raise ValueError("[Growt] No audit results. Call audit() first.")

        audit = self._last_audit
        metadata = {
            "growt_diagnosis": audit.diagnosis,
            "growt_safe_to_deploy": audit.safe_to_deploy,
            "growt_transfer_oracle": audit.transfer_oracle,
            "growt_coverage_pct": audit.coverage_pct,
            "growt_classes_at_risk": audit.classes_at_risk,
            "growt_recommendations": audit.recommendations[:3],
        }
        if self._last_metrics:
            metadata["growt_sqnr_db"] = self._last_metrics.sqnr_db
            metadata["growt_cosine_mean"] = self._last_metrics.cosine_mean
        if extra_metadata:
            metadata.update(extra_metadata)

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            description=f"Growt audit: {audit.diagnosis}",
            metadata=metadata,
        )
        artifact.add_file(model_path)

        # Write audit report as artifact file
        audit_json = json.dumps(self._safe_audit_dict(audit), indent=2)
        with artifact.new_file("growt_audit.json") as f:
            f.write(audit_json)

        run = wandb.run
        if run:
            run.log_artifact(artifact)
            logger.info("[Growt] Artifact '%s' logged with audit metadata.", name)
        else:
            logger.warning("[Growt] No active W&B run — artifact not logged.")

        return artifact

    def link_to_registry(
        self,
        artifact: Any,
        registry_path: str = "wandb-registry-model/growt-audited",
    ) -> None:
        """Link an audited artifact to a W&B Model Registry collection.

        Only links if diagnosis is SAFE.
        """
        if self._last_audit and self._last_audit.diagnosis != "SAFE":
            logger.warning(
                "[Growt] Model diagnosis is %s — NOT linking to registry.",
                self._last_audit.diagnosis,
            )
            return

        run = wandb.run
        if run:
            run.link_artifact(artifact, target_path=registry_path)
            logger.info("[Growt] Artifact linked to registry: %s", registry_path)

    # ------------------------------------------------------------------
    # Core audit
    # ------------------------------------------------------------------

    def _run_audit(
        self,
        model: torch.nn.Module,
        train_dl: Optional[DataLoader],
        deploy_dl: Optional[DataLoader],
    ) -> tuple[AuditResult, Optional[MetricsResult]]:
        if train_dl is None:
            raise ValueError(
                "[Growt] No train_dataloader. Pass it to the constructor or audit()."
            )
        if deploy_dl is None:
            logger.info("[Growt] No deploy_dataloader — using train set as proxy.")
            deploy_dl = train_dl

        train_feats, train_labels = extract_features(
            model, train_dl,
            layer_name=self._layer_name, max_samples=self._max_samples,
        )
        deploy_feats, deploy_labels = extract_features(
            model, deploy_dl,
            layer_name=self._layer_name, max_samples=self._max_samples,
        )

        audit = self._client.audit_transfer(
            features_train=train_feats.tolist(),
            labels_train=train_labels.tolist(),
            features_deploy=deploy_feats.tolist(),
            labels_deploy=deploy_labels.tolist(),
        )

        metrics = None
        if len(train_feats) == len(deploy_feats):
            metrics = self._client.metrics_compare(
                features_reference=train_feats.tolist(),
                features_compare=deploy_feats.tolist(),
                labels_reference=train_labels.tolist(),
            )

        return audit, metrics

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    def _log_to_wandb(
        self, audit: AuditResult, metrics: Optional[MetricsResult], epoch: int,
    ) -> None:

        run = wandb.run
        if not run:
            logger.warning("[Growt] No active W&B run — skipping logging")
            return

        # Core metrics
        log_dict: dict[str, Any] = {
            "growt/diagnosis_safe": 1.0 if audit.diagnosis == "SAFE" else 0.0,
            "growt/transfer_oracle": audit.transfer_oracle or 0.0,
            "growt/coverage_pct": audit.coverage_pct or 0.0,
            "growt/n_flagged_samples": float(audit.n_flagged_samples),
            "growt/n_classes_at_risk": float(len(audit.classes_at_risk)),
        }
        if metrics:
            if metrics.sqnr_db is not None:
                log_dict["growt/sqnr_db"] = metrics.sqnr_db
            if metrics.cosine_mean is not None:
                log_dict["growt/cosine_mean"] = metrics.cosine_mean
            if metrics.rank_correlation is not None:
                log_dict["growt/rank_correlation"] = metrics.rank_correlation

        wandb.log(log_dict)

        # Summary metrics (best values)
        run.summary["growt_diagnosis"] = audit.diagnosis
        run.summary["growt_transfer_oracle"] = audit.transfer_oracle
        run.summary["growt_coverage_pct"] = audit.coverage_pct
        run.summary["growt_safe_to_deploy"] = audit.safe_to_deploy

        # Classes at risk table
        if audit.classes_at_risk:
            table = wandb.Table(
                columns=["class", "at_risk"],
                data=[[c, True] for c in audit.classes_at_risk],
            )
            wandb.log({"growt/classes_at_risk": table})

        # Audit report as HTML
        if audit.report:
            wandb.log({"growt/report": wandb.Html(f"<pre>{audit.report}</pre>", data_is_not_path=True)})

        # Alert on RED_FLAG
        if self._alert_on_red and audit.diagnosis == "RED_FLAG":
            wandb.alert(
                title="Growt: Model Structural Integrity Warning",
                text=(
                    f"Diagnosis: {audit.diagnosis}\n"
                    f"Coverage: {audit.coverage_pct:.1%}\n"
                    f"Classes at risk: {', '.join(audit.classes_at_risk[:5])}\n"
                    f"Run: {run.name}"
                ),
                level=wandb.AlertLevel.ERROR,
                wait_duration=timedelta(minutes=10),
            )

        # Trajectory (multi-epoch)
        if len(self._audit_history) > 1:
            traj_data = [
                [e, a.diagnosis, a.transfer_oracle or 0, a.coverage_pct or 0]
                for e, a in self._audit_history
            ]
            traj_table = wandb.Table(
                columns=["epoch", "diagnosis", "transfer_oracle", "coverage_pct"],
                data=traj_data,
            )
            wandb.log({"growt/audit_trajectory": traj_table})

    @staticmethod
    def _safe_audit_dict(audit: AuditResult) -> dict:
        """Return only safe-to-share fields (no raw API internals)."""
        return {
            "diagnosis": audit.diagnosis,
            "safe_to_deploy": audit.safe_to_deploy,
            "transfer_oracle": audit.transfer_oracle,
            "coverage_pct": audit.coverage_pct,
            "classes_at_risk": audit.classes_at_risk,
            "recommendations": audit.recommendations,
        }
