#!/usr/bin/env bash
# 릴리즈용 버전 올리기. 세 곳을 한 번에 맞춘다.
# 사용법: ./scripts/bump-version.sh 0.2.0
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "사용법: $0 <버전>  (예: $0 0.2.0)" >&2
  exit 1
fi
V="$1"
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "버전 형식이 이상해요: $V"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python3 - "$V" << 'EOF'
import json, re, sys
v = sys.argv[1]

p = "src-tauri/tauri.conf.json"
c = json.load(open(p, encoding="utf-8"))
c["version"] = v
json.dump(c, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(p, "a", encoding="utf-8").write("\n")

p = "frontend/package.json"
c = json.load(open(p, encoding="utf-8"))
c["version"] = v
json.dump(c, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(p, "a", encoding="utf-8").write("\n")
print(f"tauri.conf.json, package.json → {v}")
EOF

# Cargo.toml의 [package] version (의존성 줄은 건드리지 않음)
python3 - "$V" << 'EOF'
import re, sys
v = sys.argv[1]
p = "src-tauri/Cargo.toml"
s = open(p, encoding="utf-8").read()
s2, n = re.subn(r'(?m)^version = "[^"]+"', f'version = "{v}"', s, count=1)
assert n == 1
open(p, "w", encoding="utf-8").write(s2)
print(f"Cargo.toml → {v}")
EOF

echo "다음: git add -A && git commit -m 'v$V' && git tag v$V && git push origin main v$V"
