import os

def setup_results_directory(base_dir='results'):
    subdirs = ['datasets', 'models', 'kernels', 'plots', 'circuits']
    paths = {}
    paths['base'] = base_dir
    for sub in subdirs:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
        paths[sub] = path
    return paths
