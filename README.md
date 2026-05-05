# growt-wandb

Structural transfer audit with automatic Weights & Biases logging.

Know if your model will work in deployment. Every audit logs to your W&B dashboard.

## Install

```bash
pip install growt-wandb
```

## Quick Start — Custom Training Loop

```python
import wandb
from growt_wandb import GrowtWandbCallback

wandb.init(project="my-model")

callback = GrowtWandbCallback(
    api_key="growt_your-key",
    train_dataloader=train_loader,
    deploy_dataloader=val_loader,
)

# ... train your model ...

callback.on_train_end(model, epoch=num_epochs)
# Audit logged to W&B: metrics, tables, alerts
```

## Quick Start — With Artifact

```python
# After training + auditing:
artifact = callback.create_audited_artifact(
    name="resnet50-quantized",
    model_path="model.pt",
)

# Only link SAFE models to registry
callback.link_to_registry(artifact)
```

## What Gets Logged to W&B

| W&B Feature | What |
|-------------|------|
| Metrics | `growt/transfer_oracle`, `growt/coverage_pct`, `growt/sqnr_db`, `growt/cosine_mean` |
| Summary | `growt_diagnosis`, `growt_safe_to_deploy` |
| Tables | Classes at risk, audit trajectory |
| Alerts | Automatic alert on RED_FLAG diagnosis |
| Artifacts | Audit metadata + `growt_audit.json` attached to model artifacts |
| Registry | Only SAFE models linked to registry |

## What It Does

1. Extracts feature vectors from your model (penultimate layer)
2. Sends vectors to the [Growt API](https://transferoracle.ai) for structural analysis
3. Returns diagnosis: SAFE, RED_FLAG, BAD_MODEL, or UNDERTRAINED
4. Logs everything to W&B automatically
5. Sends alerts on failures
6. Attaches audit metadata to model artifacts

## Callback Options

```python
GrowtWandbCallback(
    api_key="growt_...",                     # Required
    api_url="https://api.transferoracle.ai", # API endpoint
    train_dataloader=train_loader,           # Required
    deploy_dataloader=val_loader,            # Falls back to train
    layer_name="encoder",                    # Auto-detected if omitted
    fail_on_red_flag=True,                   # Raise error on RED_FLAG
    alert_on_red_flag=True,                  # Send W&B alert
    log_artifact_metadata=True,              # Attach to artifacts
    max_samples=5000,                        # Samples for extraction
    audit_every_n_epochs=0,                  # 0 = final only
)
```

## Periodic Audits

```python
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader)
    callback.on_epoch_end(model, epoch=epoch)

# Final audit
callback.on_train_end(model, epoch=num_epochs)
```

## API Key

Get your free API key (1,000 credits/month) at [transferoracle.ai/growt/plugins](https://transferoracle.ai/growt/plugins).

## Other Growt Plugins

| Plugin | Platform | What |
|--------|----------|------|
| [growt-huggingface](https://github.com/LeeSandberg/growt-huggingface) | HuggingFace | TrainerCallback + Model Card audit badge |
| [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) | NVIDIA | ModelOpt quantization audit |
| [growt-vllm](https://github.com/LeeSandberg/growt-vllm) | NVIDIA/AMD | vLLM inference monitor |
| [growt-triton](https://github.com/LeeSandberg/growt-triton) | NVIDIA | Triton Inference Server monitor |
| [growt-nemo](https://github.com/LeeSandberg/growt-nemo) | NVIDIA | NeMo / PyTorch Lightning callback |
| [growt-quark](https://github.com/LeeSandberg/growt-quark) | AMD | Quark quantization audit |
| [mlflow-growt](https://github.com/LeeSandberg/mlflow-growt) | MLflow | Evaluator + Model Registry gate |
| [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) | NVIDIA | TensorRT engine validator |
| [growt-tao](https://github.com/LeeSandberg/growt-tao) | NVIDIA | TAO Toolkit pipeline |
| [growt-client](https://github.com/LeeSandberg/growt-client) | Core | Python client library |

## License

MPL-2.0
