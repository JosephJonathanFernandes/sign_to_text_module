import os
import glob
import ast
import importlib.util

def check_imports():
    errors = 0
    local_roots = ['src', 'scripts', 'tools', 'config', 'main', 'model', 'train', 'webcam', 'pseudo_utilities']
    
    for file in glob.glob("**/*.py", recursive=True):
        if "venv" in file or "__pycache__" in file:
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except SyntaxError as e:
            print(f"Syntax error in {file}: {e}")
            errors += 1
            continue
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split('.')[0]
                    if base in local_roots:
                        # Attempt to resolve
                        try:
                            if importlib.util.find_spec(alias.name) is None:
                                print(f"{file}:{node.lineno} - Cannot resolve: import {alias.name}")
                                errors += 1
                        except ModuleNotFoundError:
                            print(f"{file}:{node.lineno} - Cannot resolve: import {alias.name}")
                            errors += 1
                            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # ignore relative imports (level > 0)
                    if node.level > 0:
                        # try to resolve relative import path based on file location
                        # too complex, just skip or rely on runtime
                        continue
                    
                    base = node.module.split('.')[0]
                    if base in local_roots:
                        try:
                            if importlib.util.find_spec(node.module) is None:
                                print(f"{file}:{node.lineno} - Cannot resolve: from {node.module} import ...")
                                errors += 1
                        except ModuleNotFoundError:
                            print(f"{file}:{node.lineno} - Cannot resolve: from {node.module} import ...")
                            errors += 1
                            
    print(f"Total import resolution errors: {errors}")

if __name__ == "__main__":
    check_imports()
