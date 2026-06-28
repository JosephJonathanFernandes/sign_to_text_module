import ast
import os
import json
from pathlib import Path

def analyze_repo(root_path):
    root = Path(root_path).resolve()
    
    dependencies = {}
    
    # Files to exclude from analysis
    exclude_dirs = {'.git', 'venv', '__pycache__', 'pseudo_data', 'adapter_weights', 'logs', 'Dataset'}
    
    python_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Modify dirnames in-place to skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            if f.endswith('.py'):
                python_files.append(Path(os.path.abspath(dirpath)) / f)
                
    # Map module name to file path
    module_to_file = {}
    for filepath in python_files:
        rel_path = filepath.relative_to(root)
        module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
        module_to_file[module_name] = str(rel_path).replace(os.sep, '/')
        module_to_file[filepath.stem] = str(rel_path).replace(os.sep, '/') # Local import fallback
        
    for filepath in python_files:
        rel_path = str(filepath.relative_to(root)).replace(os.sep, '/')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            continue
            
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    
        # Filter for local imports only
        local_imports = []
        for imp in imports:
            base_module = imp.split('.')[0]
            if base_module in module_to_file:
                mapped = module_to_file[base_module]
                if mapped != rel_path:
                    local_imports.append(mapped)
            elif imp in module_to_file:
                mapped = module_to_file[imp]
                if mapped != rel_path:
                    local_imports.append(mapped)
                    
        dependencies[rel_path] = list(set(local_imports))
        
    return dependencies

if __name__ == "__main__":
    deps = analyze_repo(".")
    with open('dep_graph.json', 'w') as f:
        json.dump(deps, f, indent=2)
    print("Done")
