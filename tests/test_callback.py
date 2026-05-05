"""Unit tests for GrowtWandbCallback."""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from growt_wandb.callback import GrowtWandbCallback
from growt_wandb.extractor import extract_features, _resolve_layer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    def __init__(self, in_features: int = 10, hidden: int = 32, out_features: int = 5):
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden)
        self.classifier = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def dataloader():
    x = torch.randn(50, 10)
    y = torch.randint(0, 5, (50,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_basic_extraction(self, simple_model, dataloader):
        feats, labels = extract_features(simple_model, dataloader, layer_name="encoder")
        assert isinstance(feats, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert feats.shape == (50, 32)
        assert labels.shape == (50,)

    def test_max_samples(self, simple_model, dataloader):
        feats, _ = extract_features(simple_model, dataloader, layer_name="encoder", max_samples=10)
        assert feats.shape[0] == 10

    def test_auto_detect_layer(self, simple_model, dataloader):
        feats, _ = extract_features(simple_model, dataloader)
        assert feats.shape[1] == 32  # encoder output, not classifier

    def test_resolve_explicit_layer(self, simple_model):
        layer = _resolve_layer(simple_model, "encoder")
        assert layer is simple_model.encoder

    def test_resolve_penultimate(self, simple_model):
        layer = _resolve_layer(simple_model, None)
        assert layer is simple_model.encoder  # second-to-last child

    def test_3d_cls_token(self):
        """3D output [B, seq, D] should extract CLS token."""

        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 32)

            def forward(self, x):
                h = self.encoder(x)
                return h.unsqueeze(1).expand(-1, 8, -1)

        model = TransformerModel()
        dl = DataLoader(TensorDataset(torch.randn(8, 10), torch.zeros(8, dtype=torch.long)), batch_size=4)
        feats, _ = extract_features(model, dl, layer_name="encoder")
        assert feats.shape == (8, 32)


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------

class TestGrowtWandbCallback:

    @patch("growt_wandb.callback.GrowtClient")
    def test_callback_init(self, mock_client_cls):
        cb = GrowtWandbCallback(api_key="test_key")
        mock_client_cls.assert_called_once_with(
            api_url="https://api.transferoracle.ai", api_key="test_key",
        )
        assert cb.last_audit is None
        assert cb.last_metrics is None
        assert cb.audit_history == []

    @patch("growt_wandb.callback.GrowtClient")
    def test_audit_calls_api(self, mock_client_cls, simple_model, dataloader):
        client = MagicMock()
        client.audit_transfer.return_value = MagicMock(
            diagnosis="SAFE", safe_to_deploy=True, transfer_oracle=0.92,
            coverage_pct=0.85, classes_at_risk=[], recommendations=["Good"],
            n_flagged_samples=0, report="OK", raw={},
        )
        client.metrics_compare.return_value = MagicMock(
            sqnr_db=22.5, cosine_mean=0.98, rank_correlation=0.95,
        )
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            fail_on_red_flag=False, alert_on_red_flag=False,
        )

        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_wandb.run = None  # No active run
            audit = cb.audit(simple_model, epoch=1)

        assert client.audit_transfer.called
        assert audit.diagnosis == "SAFE"
        assert cb.last_audit.diagnosis == "SAFE"

    @patch("growt_wandb.callback.GrowtClient")
    def test_red_flag_raises(self, mock_client_cls, simple_model, dataloader):
        client = MagicMock()
        client.audit_transfer.return_value = MagicMock(
            diagnosis="RED_FLAG", safe_to_deploy=False, transfer_oracle=0.4,
            coverage_pct=0.2, classes_at_risk=["dog"],
            recommendations=["Retrain"], n_flagged_samples=10,
            report="Bad model", raw={},
        )
        client.metrics_compare.return_value = MagicMock(
            sqnr_db=5.0, cosine_mean=0.5, rank_correlation=0.3,
        )
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            fail_on_red_flag=True, alert_on_red_flag=False,
        )

        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_wandb.run = None
            with pytest.raises(RuntimeError, match="RED_FLAG"):
                cb.audit(simple_model)

    @patch("growt_wandb.callback.GrowtClient")
    def test_wandb_logging(self, mock_client_cls, simple_model, dataloader):
        """Verify metrics are logged to wandb correctly."""
        client = MagicMock()
        client.audit_transfer.return_value = MagicMock(
            diagnosis="SAFE", safe_to_deploy=True, transfer_oracle=0.92,
            coverage_pct=0.85, classes_at_risk=[], recommendations=[],
            n_flagged_samples=0, report="OK", raw={},
        )
        client.metrics_compare.return_value = MagicMock(
            sqnr_db=22.5, cosine_mean=0.98, rank_correlation=0.95,
        )
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            fail_on_red_flag=False, alert_on_red_flag=False,
        )

        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_run.summary = {}
            mock_wandb.run = mock_run
            mock_wandb.Table = MagicMock()
            mock_wandb.Html = MagicMock()

            cb.audit(simple_model, epoch=1)

            # Verify wandb.log was called with growt metrics
            mock_wandb.log.assert_called()
            call_args = mock_wandb.log.call_args_list[0][0][0]
            assert "growt/transfer_oracle" in call_args
            assert "growt/coverage_pct" in call_args
            assert "growt/sqnr_db" in call_args
            assert call_args["growt/transfer_oracle"] == 0.92

            # Verify summary
            assert mock_run.summary["growt_diagnosis"] == "SAFE"

    @patch("growt_wandb.callback.GrowtClient")
    def test_wandb_alert_on_red_flag(self, mock_client_cls, simple_model, dataloader):
        """RED_FLAG should trigger wandb.alert."""
        client = MagicMock()
        client.audit_transfer.return_value = MagicMock(
            diagnosis="RED_FLAG", safe_to_deploy=False, transfer_oracle=0.4,
            coverage_pct=0.2, classes_at_risk=["dog", "cat"],
            recommendations=[], n_flagged_samples=5, report="Bad", raw={},
        )
        client.metrics_compare.return_value = MagicMock(
            sqnr_db=5.0, cosine_mean=0.5, rank_correlation=0.3,
        )
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            fail_on_red_flag=False, alert_on_red_flag=True,
        )

        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_run.summary = {}
            mock_run.name = "test-run-1"
            mock_wandb.run = mock_run
            mock_wandb.AlertLevel.ERROR = "ERROR"
            mock_wandb.Table = MagicMock()
            mock_wandb.Html = MagicMock()

            cb.audit(simple_model, epoch=1)

            mock_wandb.alert.assert_called_once()
            alert_kwargs = mock_wandb.alert.call_args[1]
            assert "RED_FLAG" in alert_kwargs["text"]
            assert alert_kwargs["level"] == "ERROR"

    @patch("growt_wandb.callback.GrowtClient")
    def test_create_audited_artifact(self, mock_client_cls, simple_model, dataloader, tmp_path):
        """Test artifact creation with audit metadata."""
        client = MagicMock()
        client.audit_transfer.return_value = MagicMock(
            diagnosis="SAFE", safe_to_deploy=True, transfer_oracle=0.92,
            coverage_pct=0.85, classes_at_risk=[], recommendations=["Good"],
            n_flagged_samples=0, report="OK", raw={},
        )
        client.metrics_compare.return_value = MagicMock(
            sqnr_db=22.5, cosine_mean=0.98, rank_correlation=0.95,
        )
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            fail_on_red_flag=False, alert_on_red_flag=False,
        )

        # First run audit
        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_wandb.run = None
            cb.audit(simple_model)

        # Then create artifact
        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_artifact = MagicMock()
            mock_wandb.Artifact.return_value = mock_artifact
            mock_wandb.run = MagicMock()

            model_path = tmp_path / "model.pt"
            torch.save(simple_model.state_dict(), model_path)

            artifact = cb.create_audited_artifact("test-model", str(model_path))

            # Verify artifact was created with correct metadata
            create_kwargs = mock_wandb.Artifact.call_args[1]
            assert create_kwargs["metadata"]["growt_diagnosis"] == "SAFE"
            assert create_kwargs["metadata"]["growt_transfer_oracle"] == 0.92
            mock_artifact.add_file.assert_called_once_with(str(model_path))
            mock_wandb.run.log_artifact.assert_called_once_with(mock_artifact)

    @patch("growt_wandb.callback.GrowtClient")
    def test_link_to_registry_blocks_unsafe(self, mock_client_cls, simple_model, dataloader):
        """RED_FLAG models should NOT be linked to registry."""
        client = MagicMock()
        client.audit_transfer.return_value = MagicMock(
            diagnosis="RED_FLAG", safe_to_deploy=False, transfer_oracle=0.4,
            coverage_pct=0.2, classes_at_risk=["dog"],
            recommendations=[], n_flagged_samples=5, report="Bad", raw={},
        )
        client.metrics_compare.return_value = MagicMock(
            sqnr_db=5.0, cosine_mean=0.5, rank_correlation=0.3,
        )
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            fail_on_red_flag=False, alert_on_red_flag=False,
        )

        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_wandb.run = None
            cb.audit(simple_model)

        with patch("growt_wandb.callback.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            artifact = MagicMock()
            cb.link_to_registry(artifact)
            # Should NOT call link_artifact for RED_FLAG
            mock_wandb.run.link_artifact.assert_not_called()

    def test_no_train_dataloader_raises(self, simple_model):
        cb = GrowtWandbCallback(api_key="test")
        with pytest.raises(ValueError, match="No train_dataloader"):
            cb.audit(simple_model)

    @patch("growt_wandb.callback.GrowtClient")
    def test_periodic_audit_skips_non_matching_epochs(self, mock_client_cls, simple_model, dataloader):
        """on_epoch_end should only audit at configured intervals."""
        client = MagicMock()
        mock_client_cls.return_value = client

        cb = GrowtWandbCallback(
            api_key="test", train_dataloader=dataloader,
            audit_every_n_epochs=5, fail_on_red_flag=False,
        )

        result = cb.on_epoch_end(simple_model, epoch=3)
        assert result is None
        assert not client.audit_transfer.called

    @patch("growt_wandb.callback.GrowtClient")
    def test_safe_audit_dict_no_raw(self, mock_client_cls):
        """_safe_audit_dict should not include raw API response."""
        audit = MagicMock(
            diagnosis="SAFE", safe_to_deploy=True, transfer_oracle=0.9,
            coverage_pct=0.8, classes_at_risk=[], recommendations=["OK"],
            raw={"internal_secret": "should_not_appear"},
        )
        safe = GrowtWandbCallback._safe_audit_dict(audit)
        assert "raw" not in safe
        assert "internal_secret" not in str(safe)
        assert safe["diagnosis"] == "SAFE"
