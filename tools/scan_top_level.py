import ast, glob, os, json
py_files = sorted(glob.glob('**/*.py', recursive=True))
results = []
for f in py_files:
    try:
        src = open(f, encoding='utf-8', errors='ignore').read()
        tree = ast.parse(src, filename=f)
    except Exception as e:
        results.append((f, 'PARSE_ERROR', str(e)))
        continue
    top_exec = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.Expr):
                val = node.value
                if isinstance(val, ast.Constant) or (hasattr(ast, 'Str') and isinstance(val, ast.Str)):
                    continue
                top_exec.append((node.lineno, f'Expr: {type(val).__name__}'))
            continue
        if isinstance(node, ast.If):
            try:
                test_src = ast.unparse(node.test) if hasattr(ast, 'unparse') else ''
            except Exception:
                test_src = ''
            if "__name__" in test_src and "__main__" in test_src:
                continue
            top_exec.append((node.lineno, f'IfStmt: {test_src}'))
            continue
        top_exec.append((node.lineno, type(node).__name__))
    if top_exec:
        results.append((f, 'TOP_LEVEL_EXEC', top_exec))
    else:
        results.append((f, 'CLEAN', None))

count_exec = sum(1 for r in results if r[1]=='TOP_LEVEL_EXEC')
print(json.dumps({'total_files': len(py_files), 'files_with_top_level_exec': count_exec, 'results': results}, default=str, indent=2))
