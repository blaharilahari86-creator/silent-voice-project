import os
import glob


def resolve_csv(arg_path, env_var, default_path, keywords=None):
    if arg_path:
        return arg_path
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
    if os.path.exists(default_path):
        return default_path
    candidates = glob.glob(os.path.join('dataset', '**', '*.csv'), recursive=True)
    if keywords:
        for c in candidates:
            low = c.lower()
            if any(k in low for k in keywords):
                return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError('No dataset CSV found')
