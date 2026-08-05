import os
import torch
import onnxruntime as ort
from src.train_multimodal import NAADMultimodalEngine

def export_model(model, export_path):
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    model.eval()
    model.to("cpu")

    dummy_input_ids = torch.ones((1, 128), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)
    dummy_audio_feats = torch.randn((1, 32), dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_audio_feats),
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'audio_feats'],
        output_names=['logits', 'risk_score'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'audio_feats': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'risk_score': {0: 'batch_size'}
        }
    )
    print(f"✅ Exported ONNX model to {export_path}")
