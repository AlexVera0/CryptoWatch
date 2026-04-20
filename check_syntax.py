import ast, sys
from pathlib import Path

errors = []
root = Path(r'x:\Antigravity Project\CryptoWatch')
files = list(root.rglob('*.py'))
for f in sorted(files):
    try:
        ast.parse(f.read_text(encoding='utf-8'))
    except SyntaxError as e:
        errors.append(f'{f.name}: line {e.lineno} - {e.msg}')

if errors:
    for e in errors:
        print('FAIL:', e)
    sys.exit(1)
else:
    print(f'[OK] All {len(files)} Python files passed syntax check')
