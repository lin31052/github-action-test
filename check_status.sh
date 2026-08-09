#!/bin/sh
# ============================================
# GitHub Actions 状态/结果查询脚本（无需认证）
# 用途：查看云端 task 的整体状态，判断是成功/失败、失败在哪个步骤。
# 用法：./check_status.sh  <owner>/<repo>
#   例：./check_status.sh lin31052/github-action-test
# 注意：
#   - 无认证只能看"整体结论(成功/失败)+失败步骤名"，看不到具体报错文本。
#   - 要看具体报错 → 需要用户登录 GitHub 看 Actions 日志，把日志发给助手。
# ============================================
REPO="${1:-lin31052/github-action-test}"
UA="User-Agent: curl"

echo "=========================================="
echo "最新 workflow runs 状态  ($REPO)"
echo "=========================================="
timeout 20 curl -s "https://api.github.com/repos/$REPO/actions/runs" -H "$UA" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
runs = d.get('workflow_runs', [])
if not runs:
    print('(无可见 run 或需认证)')
for r in runs[:6]:
    mark = '✅' if r.get('conclusion')=='success' else ('❌' if r.get('conclusion')=='failure' else '⏳')
    print(f\"  {mark} #{r['run_number']} 提交={r['head_sha'][:7]} 状态={r['status']} 结论={r.get('conclusion')} {r['display_title'][:35]}\")
"

echo ""
echo "=========================================="
echo "最新 run 的失败步骤(若有)"
echo "=========================================="
LATEST=$(timeout 20 curl -s "https://api.github.com/repos/$REPO/actions/runs" -H "$UA" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['workflow_runs'][0]['id'])" 2>/dev/null || echo "")
if [ -n "$LATEST" ]; then
  timeout 20 curl -s "https://api.github.com/repos/$REPO/actions/runs/$LATEST/jobs" -H "$UA" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for j in d.get('jobs', []):
    for s in j.get('steps', []):
        c = s.get('conclusion','')
        if c in ('failure','cancelled'):
            print(f\"  ❌ job[{j.get('name')}] -> 步骤: {s['name']}\")
"
fi

echo ""
echo "=========================================="
echo "仓库最新提交(看有没有 bot 回写结果)"
echo "=========================================="
timeout 20 curl -s "https://api.github.com/repos/$REPO/commits?per_page=3" -H "$UA" 2>/dev/null | python3 -c "
import sys, json
for c in json.load(sys.stdin):
    msg = c['commit']['message'].split('\n')[0]
    print(f\"  {c['sha'][:7]} {msg[:50]}\")
"
echo "=========================================="
