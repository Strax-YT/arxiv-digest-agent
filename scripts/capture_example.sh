#!/usr/bin/env bash
# Capture a REAL run (real arXiv, real LLM) into examples/real_run.md.
#
#   ./scripts/capture_example.sh 2401.12345
#   ./scripts/capture_example.sh "recent work on KV-cache compression for LLMs"
#
# Run this once before submitting so the README shows your own live output
# rather than the synthetic offline demo.
set -euo pipefail

QUERY="${1:?usage: capture_example.sh "<arxiv id or topic>"}"
OUT="examples/real_run.md"
mkdir -p examples

{
  echo "# Example run (live arXiv + live LLM)"
  echo
  echo "Captured: $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo
  echo '```'
  echo "$ python -m arxiv_agent brief \"$QUERY\" --no-chat"
} > "$OUT"

python -m arxiv_agent brief "$QUERY" --no-chat 2>&1 | tee -a "$OUT"

SESSION=$(python -m arxiv_agent sessions | awk 'NR==2 {print $1}')
echo "" >> "$OUT"

for Q in \
  "What is the core method in one paragraph?" \
  "What datasets and baselines were used?" \
  "What was the carbon cost of training the models?"
do
  echo "$ python -m arxiv_agent ask $SESSION \"$Q\"" >> "$OUT"
  python -m arxiv_agent ask "$SESSION" "$Q" 2>&1 | tee -a "$OUT"
  echo "" >> "$OUT"
done

echo '```' >> "$OUT"
echo "Wrote $OUT"
