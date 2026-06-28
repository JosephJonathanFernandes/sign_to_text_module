import json

with open('dep_graph.json', 'r') as f:
    deps = json.load(f)

# Group files into subgraphs
groups = {
    "Root": ["main.py", "config.py"],
    "Inference": ["webcam.py", "ensemble.py", "onnx_inference.py", "onnx_ensemble.py", "onnx_ensemble_integration.py", "sentence_builder.py", "temporal_postprocessor.py", "nlp_postprocessor.py", "hand_selector.py", "pseudo_buffer.py"],
    "Training": ["train.py", "model.py", "spatial_gnn.py", "adapter_training.py", "adapter_model.py"],
    "Preprocessing": ["preprocess.py", "dataset.py", "augmentations.py", "merge_augmentations.py", "collect_data.py"],
    "Utilities": ["pipeline_logger.py", "profiling.py", "quantization_utils.py"],
    "Scripts": [],
    "Experimental": [],
    "Core/UI (New)": ["src/core/camera_manager.py", "src/core/inference_engine.py", "src/core/landmark_processor.py", "src/core/motion_tracker.py", "src/ui/renderer.py"]
}

# Assign any unmapped files
mapped = set(f for g in groups.values() for f in g)
for f in deps.keys():
    if f not in mapped:
        if f.startswith('scripts/'):
            groups["Scripts"].append(f)
        elif f.startswith('experimental/'):
            groups["Experimental"].append(f)
        elif f.startswith('Paper/'):
            pass # ignore paper
        elif f.startswith('tools/'):
            groups["Scripts"].append(f)
        else:
            groups["Utilities"].append(f)

mermaid = ["```mermaid", "graph TD", "    %% File Clusters"]

# Define node IDs
def get_node_id(filename):
    return filename.replace('/', '_').replace('.', '_')

for group_name, files in groups.items():
    if not files: continue
    mermaid.append(f"    subgraph {group_name}")
    for f in files:
        if f in deps:
            # Quote the label to avoid syntax issues with special chars
            mermaid.append(f"        {get_node_id(f)}[\"{f}\"]")
    mermaid.append("    end")
    
mermaid.append("")
mermaid.append("    %% Dependencies")

drawn_edges = set()
for source, targets in deps.items():
    if source.startswith('Paper/'): continue
    for target in targets:
        # Avoid drawing edges to self or missing files
        if target in deps and target != source:
            edge = f"    {get_node_id(source)} --> {get_node_id(target)}"
            if edge not in drawn_edges:
                mermaid.append(edge)
                drawn_edges.add(edge)

mermaid.append("```")

# Write to artifact
artifact_path = r"C:\Users\Joseph\.gemini\antigravity-ide\brain\1bfc2803-c415-4680-af77-36a82996deab\dependency_tree.md"
with open(artifact_path, 'w', encoding='utf-8') as f:
    f.write("# Visual Dependency Tree\n\n")
    f.write("Generated from `dep_graph.json` to visualize who imports whom.\n\n")
    f.write("\n".join(mermaid))
    
print("Generated dependency_tree.md")
