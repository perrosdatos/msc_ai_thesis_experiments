import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../BenchMARL')))
import torch
import torch.nn as nn
import os
import subprocess
import datetime

def trace_model(model, input_tensor):
    trace = []
    x = input_tensor
    for name, layer in model.named_children():
        layer_name = layer.__class__.__name__
        in_shape = list(x.shape)
        
        # Count parameters
        params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        # Forward pass
        x = layer(x)
        out_shape = list(x.shape)
        
        # Get extra attributes
        kernel_size = getattr(layer, 'kernel_size', None)
        stride = getattr(layer, 'stride', None)
        padding = getattr(layer, 'padding', None)
        features = getattr(layer, 'out_features', getattr(layer, 'out_channels', None))
        
        trace.append({
            "name": layer_name,
            "in_shape": in_shape,
            "out_shape": out_shape,
            "params": params,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "features": features
        })
    return trace, x

def get_html_table(trace, title):
    html = f"""
    <div class="card mb-4 shadow border-secondary">
        <div class="card-header bg-dark text-white border-bottom border-secondary">
            <h5 class="mb-0">{title}</h5>
        </div>
        <div class="card-body p-0">
            <table class="table table-dark table-hover mb-0">
                <thead>
                    <tr>
                        <th>Layer Type</th>
                        <th>Kernel/Filter Info</th>
                        <th>Input Shape</th>
                        <th>Output Shape</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
    """
    for step in trace:
        info = ""
        if step['kernel_size'] is not None:
             info = f"k={step['kernel_size']}, s={step['stride']}, p={step['padding']}"
        elif step['features'] is not None:
             info = f"features={step['features']}"
        
        html += f"""
                    <tr>
                        <td><strong>{step['name']}</strong></td>
                        <td class="text-info">{info}</td>
                        <td><span class="badge bg-secondary">{'x'.join(map(str, step['in_shape']))}</span></td>
                        <td><span class="badge bg-success">{'x'.join(map(str, step['out_shape']))}</span></td>
                        <td class="text-warning">{step['params']:,}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return html

def main():
    # 1. Define the architecture matching `lux_sweep` CNN specs
    # cnn_num_cells: [32, 32, 32], kernel_size: 3, strides: 1, padding: 0
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, stride=1, padding=0),
        nn.Tanh(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
        nn.Tanh(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
        nn.Tanh(),
        nn.Flatten()
    )
    
    # Batch=1, Channels=14, Max_Units=1 is handled implicitly, we trace pure image resolution 24x24
    dummy_input = torch.zeros(1, 14, 24, 24)
    cnn_trace, cnn_out = trace_model(cnn, dummy_input)
    
    # 2. Define the MLP block
    # mlp_num_cells: [32], producing embedding bottleneck
    flatten_size = cnn_out.shape[1] # Expected 10368
    mlp = nn.Sequential(
        nn.Linear(flatten_size, 32),
        nn.Tanh()
    )
    
    mlp_trace, final_out = trace_model(mlp, cnn_out)
    
    # 3. Define the Policy Head Block
    # Discrete Policy outputs 5 probabilities (logits) corresponding to the 5 Lux discrete actions.
    policy = nn.Sequential(
        nn.Linear(32, 5) # Map embedding bottleneck directly to categorical choices
    )
    policy_trace, policy_out = trace_model(policy, final_out)
    
    # Compile Stats
    total_params_cnn = sum(layer['params'] for layer in cnn_trace)
    total_params_mlp = sum(layer['params'] for layer in mlp_trace)
    total_params_policy = sum(layer['params'] for layer in policy_trace)
    grand_total = total_params_cnn + total_params_mlp + total_params_policy
    
    # 3. Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lux AI S3 - MARL Internal CNN Architecture</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme: 'dark'}});</script>
    <style>
        body {{ background-color: #121212; color: #e0e0e0; }}
    </style>
</head>
<body class="py-5">
    <div class="container">
        <h1 class="display-6 mb-2 fw-bold text-primary">Model Architecture Radiography</h1>
        <p class="lead text-muted mb-5">Tracing the exact <code>MultiAgentConvNet</code> topology deployed inside BenchMARL mapping the dynamic `14-Channel` environment vector.</p>

        <div class="alert alert-info shadow-sm d-flex justify-content-between align-items-center mb-5">
            <div>
                <h4 class="alert-heading fw-bold">Network Summary</h4>
                <p class="mb-0 text-dark">Total Trainable Parameters across Shared Body (per agent evaluation path).</p>
            </div>
            <h2 class="mb-0 fw-bold badge bg-primary fs-3 shadow">{grand_total:,} params</h2>
        </div>
        
        {get_html_table(cnn_trace, "Convolutional Processor (Spatial Shrinkage)")}
        
        <div class="text-center my-4 text-muted">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" class="bi bi-arrow-down" viewBox="0 0 16 16">
              <path fill-rule="evenodd" d="M8 1a.5.5 0 0 1 .5.5v11.793l3.146-3.147a.5.5 0 0 1 .708.708l-4 4a.5.5 0 0 1-.708 0l-4-4a.5.5 0 0 1 .708-.708L7.5 13.293V1.5A.5.5 0 0 1 8 1z"/>
            </svg>
            <p class="mt-2 text-info fw-bold">Flattened Bottleneck Transition (Size: {flatten_size:,})</p>
        </div>
        
        {get_html_table(mlp_trace, "Multilayer Perceptron (Knowledge Embedding)")}
        
        <div class="text-center my-4 text-muted">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" class="bi bi-arrow-down" viewBox="0 0 16 16">
              <path fill-rule="evenodd" d="M8 1a.5.5 0 0 1 .5.5v11.793l3.146-3.147a.5.5 0 0 1 .708.708l-4 4a.5.5 0 0 1-.708 0l-4-4a.5.5 0 0 1 .708-.708L7.5 13.293V1.5A.5.5 0 0 1 8 1z"/>
            </svg>
            <p class="mt-2 text-warning fw-bold">Algorithm Branch Out (Size: 32)</p>
        </div>
        
        {get_html_table(policy_trace, "Policy Head (Categorical Architecture Output)")}
        
        <h3 class="mt-5 mb-4 border-bottom border-secondary pb-2">Topological Diagram: Body-to-Head Handshake</h3>
        <div class="card bg-dark border-secondary mb-5 shadow-lg">
            <div class="card-body text-center" style="background-color: #1a1a1a; overflow-y: auto;">
                <div class="mermaid">
                    graph TD
                    Env(("🌍 14-Channel Environment State Matrix<br/>(+ Fog-of-War Memory + Agent Trajectory)")) --> CNN["🧠 Shared CNN Brain<br/>(10,368 flattened nodes)"]
                    CNN -->|Linear Projection| Z["📉 Latent State Embedding (32 nodes)"]
                    
                    Z -->|Diverges Input| Actor["🕹️ Actor Head Policy<br/>(Multi-Agent Network)"]
                    Actor -->|Probabilities| Act(("⚡ 5 Discrete Actions<br/>(Up, Down, L, R, Center)"))
                    
                    Z -.->|Diverges Parallel Input| Critic["⚖️ Critic Head<br/>(Q-Value / State-Value estimator)"]
                    Critic -.-> Value(("💰 State Value<br/>(Scalar)"))
                    
                    style Env fill:#198754,stroke:#fff,stroke-width:2px,color:#fff
                    style CNN fill:#0d6efd,stroke:#fff,stroke-width:2px,color:#fff
                    style Z fill:#6f42c1,stroke:#fff,stroke-width:2px,color:#fff
                    style Actor fill:#fd7e14,stroke:#fff,stroke-width:2px,color:#fff
                    style Critic fill:#6c757d,stroke:#fff,stroke-width:2px,color:#fff
                    style Act fill:#dc3545,stroke:#fff,stroke-width:2px,color:#fff
                    style Value fill:#ffc107,stroke:#fff,stroke-width:2px,color:#000
                </div>
            </div>
        </div>
        
    </div>
</body>
</html>
"""

    out_dir = os.environ.get("REPORT_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "lux_cnn_architecture.html")
    else:
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        except Exception:
            git_hash = "unknown_commit"
            
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = f"html_reports/{git_hash}_{stamp}"
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "lux_cnn_architecture.html")
    
    with open(report_path, "w") as f:
        f.write(html_content)
        
    print(f"✅ Extracted Model Graph and calculated parameters.")
    print(f"✅ Generated Architecture Report: {report_path}")

if __name__ == "__main__":
    main()
