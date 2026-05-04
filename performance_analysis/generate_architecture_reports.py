import os
import torch
import torch.nn as nn

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

def generate_report(algo_name, trace_html, total_params, flatten_size, mermaid_graph, description):
    html_content = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lux AI S3 - {algo_name.upper()} Architecture (V2)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme: 'dark'}});</script>
    <style>
        body {{ background-color: #121212; color: #e0e0e0; }}
    </style>
</head>
<body class="py-5">
    <div class="container">
        <h1 class="display-6 mb-2 fw-bold text-primary">{algo_name.upper()} Architecture Radiography</h1>
        <p class="lead text-muted mb-5">{description}</p>

        <div class="alert alert-info shadow-sm d-flex justify-content-between align-items-center mb-5">
            <div>
                <h4 class="alert-heading fw-bold">Network Summary (Shared CNN Backbone)</h4>
                <p class="mb-0 text-dark">Total Trainable Parameters across Shared Body (per agent evaluation path).</p>
            </div>
            <h2 class="mb-0 fw-bold badge bg-primary fs-3 shadow">{total_params:,} params</h2>
        </div>
        
        {trace_html}
        
        <h3 class="mt-5 mb-4 border-bottom border-secondary pb-2">Topological Diagram: Algorithm-Specific Execution</h3>
        <div class="card bg-dark border-secondary mb-5 shadow-lg">
            <div class="card-body text-center" style="background-color: #1a1a1a; overflow-y: auto;">
                <div class="mermaid">
{mermaid_graph}
                </div>
            </div>
        </div>
        
    </div>
</body>
</html>
"""
    return html_content

def main():
    print("Tracing Base Convolutional Pipeline...")
    
    cnn = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten()
    )
    dummy_input = torch.zeros(1, 16, 24, 24)
    cnn_trace, cnn_out = trace_model(cnn, dummy_input)
    flatten_size = cnn_out.shape[1] 
    
    mlp = nn.Sequential(
        nn.Linear(flatten_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU()
    )
    mlp_trace, final_out = trace_model(mlp, cnn_out)
    
    policy = nn.Sequential(
        nn.Linear(128, 5) 
    )
    policy_trace, _ = trace_model(policy, final_out)
    
    total_params = sum(l['params'] for l in cnn_trace) + sum(l['params'] for l in mlp_trace) + sum(l['params'] for l in policy_trace)
    
    trace_html = get_html_table(cnn_trace, "Convolutional Processor (Spatial Shrinkage)")
    trace_html += f"""
        <div class="text-center my-4 text-muted">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" class="bi bi-arrow-down" viewBox="0 0 16 16">
              <path fill-rule="evenodd" d="M8 1a.5.5 0 0 1 .5.5v11.793l3.146-3.147a.5.5 0 0 1 .708.708l-4 4a.5.5 0 0 1-.708 0l-4-4a.5.5 0 0 1 .708-.708L7.5 13.293V1.5A.5.5 0 0 1 8 1z"/>
            </svg>
            <p class="mt-2 text-info fw-bold">Flattened Bottleneck Transition (Size: {flatten_size:,})</p>
        </div>
    """
    trace_html += get_html_table(mlp_trace, "Multilayer Perceptron (Knowledge Embedding)")
    trace_html += f"""
        <div class="text-center my-4 text-muted">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" class="bi bi-arrow-down" viewBox="0 0 16 16">
              <path fill-rule="evenodd" d="M8 1a.5.5 0 0 1 .5.5v11.793l3.146-3.147a.5.5 0 0 1-.708.708l-4 4a.5.5 0 0 1-.708 0l-4-4a.5.5 0 0 1 .708-.708L7.5 13.293V1.5A.5.5 0 0 1 8 1z"/>
            </svg>
            <p class="mt-2 text-warning fw-bold">Algorithm Branch Out (Size: 128)</p>
        </div>
    """
    trace_html += get_html_table(policy_trace, "Policy Head (Categorical Architecture Output)")
    
    # -------------------------------------------------------------------------
    # MAPPO Mermaid
    # -------------------------------------------------------------------------
    mappo_mermaid = """                    graph TD
                    Env(("🌍 16-Channel Environment State Matrix<br/>(+ Fog-of-War Memory + Agent Trajectory)")) --> CNN["🧠 Shared CNN Body<br/>(18,432 flattened nodes)"]
                    CNN -->|Linear Projection| Z["📉 Latent State Embedding (128 nodes)"]
                    
                    Z -->|Decentralized Execution| Actor["🕹️ Actor Policy Network<br/>(Categorical Distribution)"]
                    Actor -->|Probabilities| Act(("⚡ 5 Discrete Actions<br/>(Up, Down, L, R, Center)"))
                    
                    Z -.->|Centralized Training| Critic["⚖️ Centralized Critic Network<br/>(Evaluates Global State)"]
                    Critic -.-> Value(("💰 State Value V(s)<br/>(Scalar)"))
                    
                    style Env fill:#198754,stroke:#fff,stroke-width:2px,color:#fff
                    style CNN fill:#0d6efd,stroke:#fff,stroke-width:2px,color:#fff
                    style Z fill:#6f42c1,stroke:#fff,stroke-width:2px,color:#fff
                    style Actor fill:#fd7e14,stroke:#fff,stroke-width:2px,color:#fff
                    style Critic fill:#6c757d,stroke:#fff,stroke-width:2px,color:#fff
                    style Act fill:#dc3545,stroke:#fff,stroke-width:2px,color:#fff
                    style Value fill:#ffc107,stroke:#fff,stroke-width:2px,color:#000
    """
    mappo_desc = "On-Policy Actor-Critic algorithm. Uses a centralized critic to evaluate the global state V(s) during training to stabilize gradients, while execution relies strictly on the decentralized actor's categorical distribution probabilities."
    
    # -------------------------------------------------------------------------
    # MASAC Mermaid
    # -------------------------------------------------------------------------
    masac_mermaid = """                    graph TD
                    Env(("🌍 16-Channel Environment State Matrix<br/>(+ Fog-of-War Memory + Agent Trajectory)")) --> CNN["🧠 Shared CNN Body<br/>(18,432 flattened nodes)"]
                    CNN -->|Linear Projection| Z["📉 Latent State Embedding (128 nodes)"]
                    
                    Z -->|Decentralized Execution| Actor["🕹️ Soft Actor Policy Network<br/>(Max Entropy Objective)"]
                    Actor -->|Probabilities| Act(("⚡ 5 Discrete Actions<br/>(Up, Down, L, R, Center)"))
                    
                    Z -.->|Centralized Training| Critic["⚖️ Centralized Soft Critic Network<br/>(Evaluates State-Action Pairs)"]
                    Critic -.-> Value(("💰 Q-Value Q(s,a)<br/>(Continuous Tensor)"))
                    
                    style Env fill:#198754,stroke:#fff,stroke-width:2px,color:#fff
                    style CNN fill:#0d6efd,stroke:#fff,stroke-width:2px,color:#fff
                    style Z fill:#6f42c1,stroke:#fff,stroke-width:2px,color:#fff
                    style Actor fill:#fd7e14,stroke:#fff,stroke-width:2px,color:#fff
                    style Critic fill:#6c757d,stroke:#fff,stroke-width:2px,color:#fff
                    style Act fill:#dc3545,stroke:#fff,stroke-width:2px,color:#fff
                    style Value fill:#ffc107,stroke:#fff,stroke-width:2px,color:#000
    """
    masac_desc = "Off-Policy Soft Actor-Critic algorithm. Implements a max-entropy objective to encourage robust exploration. The critic evaluates state-action pairs Q(s,a) rather than purely state values."
    
    # -------------------------------------------------------------------------
    # QMIX Mermaid
    # -------------------------------------------------------------------------
    qmix_mermaid = """                    graph TD
                    Env(("🌍 16-Channel Environment State Matrix<br/>(+ Fog-of-War Memory + Agent Trajectory)")) --> CNN["🧠 Local CNN Body (Per Agent)<br/>(18,432 flattened nodes)"]
                    CNN -->|Linear Projection| Z["📉 Latent State Embedding (128 nodes)"]
                    
                    Z -->|Decentralized Execution| QNet["🧠 Individual Q-Network<br/>(Evaluates Local Actions)"]
                    QNet -->|Epsilon-Greedy| Act(("⚡ 5 Discrete Actions<br/>(Up, Down, L, R, Center)"))
                    QNet -->|Local Q-Value Q_a| Mixer["🔀 QMIX Hypernetwork Mixer<br/>(Enforces Monotonicity)"]
                    
                    GlobalState(("🌐 Global State Matrix")) -.->|Conditions| Hyper["🎛️ Hypernetworks<br/>(Generate Weights for Mixer)"]
                    Hyper -.->|Absolute Weights| Mixer
                    
                    Mixer --> QTot(("💰 Joint Q-Total<br/>(Global Scalar)"))
                    
                    style Env fill:#198754,stroke:#fff,stroke-width:2px,color:#fff
                    style GlobalState fill:#198754,stroke:#fff,stroke-width:2px,color:#fff
                    style CNN fill:#0d6efd,stroke:#fff,stroke-width:2px,color:#fff
                    style Z fill:#6f42c1,stroke:#fff,stroke-width:2px,color:#fff
                    style QNet fill:#fd7e14,stroke:#fff,stroke-width:2px,color:#fff
                    style Mixer fill:#e83e8c,stroke:#fff,stroke-width:2px,color:#fff
                    style Hyper fill:#6c757d,stroke:#fff,stroke-width:2px,color:#fff
                    style Act fill:#dc3545,stroke:#fff,stroke-width:2px,color:#fff
                    style QTot fill:#ffc107,stroke:#fff,stroke-width:2px,color:#000
    """
    qmix_desc = "Off-Policy purely Value-Based algorithm without an explicit Actor network. Relies on individual Q-networks per agent and an Epsilon-Greedy execution policy. Gradients flow through a Monotonic Hypernetwork Mixer that combines individual agent Q-values into a global Q-Total using the global state."
    
    # -------------------------------------------------------------------------
    # Generate Files
    # -------------------------------------------------------------------------
    out_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/architecture_reports"
    os.makedirs(out_dir, exist_ok=True)
    
    files = {
        "mappo": generate_report("MAPPO", trace_html, total_params, flatten_size, mappo_mermaid, mappo_desc),
        "masac": generate_report("MASAC", trace_html, total_params, flatten_size, masac_mermaid, masac_desc),
        "qmix": generate_report("QMIX", trace_html, total_params, flatten_size, qmix_mermaid, qmix_desc)
    }
    
    for algo, content in files.items():
        report_path = os.path.join(out_dir, f"{algo}_architecture.html")
        with open(report_path, "w") as f:
            f.write(content)
        print(f"✅ Generated Architecture Report: {report_path}")

if __name__ == "__main__":
    main()
