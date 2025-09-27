// Per-question reason analysis for a run JSONL vs dataset JSONL.
// Usage: node scripts/analyze_per_question.js <run.jsonl> <dataset.jsonl>

const fs = require('fs');

function readJsonl(p) {
  const txt = fs.readFileSync(p, 'utf8');
  return txt
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      try {
        return JSON.parse(l);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function normalizeText(s) {
  if (!s) return '';
  let t = String(s).toLowerCase();
  t = t.normalize('NFKC');
  t = t.replace(/[\p{P}\p{S}]/gu, ' ');
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

function emF1(pred, gold) {
  const p = normalizeText(pred);
  const g = normalizeText(gold);
  const em = p === g && g !== '' ? 1.0 : 0.0;
  const pt = p ? p.split(' ') : [];
  const gt = g ? g.split(' ') : [];
  if (pt.length === 0 && gt.length === 0) return { em, f1: 1.0 };
  if (pt.length === 0 || gt.length === 0) return { em, f1: 0.0 };
  const pc = new Map();
  for (const w of pt) pc.set(w, (pc.get(w) || 0) + 1);
  const gc = new Map();
  for (const w of gt) gc.set(w, (gc.get(w) || 0) + 1);
  let common = 0;
  for (const [w, c] of pc.entries()) if (gc.has(w)) common += Math.min(c, gc.get(w));
  if (!common) return { em, f1: 0.0 };
  const precision = common / pt.length;
  const recall = common / gt.length;
  const f1 = (2 * precision * recall) / (precision + recall);
  return { em, f1 };
}

function isUnanswerable(gold) {
  const g = (gold || '').trim().toLowerCase();
  return g === '' || g === 'invalid question' || g === 'n/a' || g === 'unknown';
}

function stripCitations(s) {
  return String(s || '').replace(/\s*\[CIT:[^\]]+\]\s*/g, ' ').trim();
}

function isIDK(ans) {
  const core = stripCitations(ans)
    .toLowerCase()
    .replace(/[\p{P}\p{S}]/gu, '')
    .replace(/\s+/g, '');
  return core === 'idontknow';
}

function classifyReason(q, gold, ans, em, f1, abstain, overlap) {
  const ql = (q || '').toLowerCase();
  const gl = (gold || '').toLowerCase();
  const al = (ans || '').toLowerCase();
  const unans = isUnanswerable(gold);

  // Unanswerable cases
  if (unans) {
    return abstain ? 'Correct abstain (unanswerable)' : 'Hallucination on unanswerable';
  }

  // Answerable cases
  if (abstain) {
    if (/per game|average|ex-dividend|last month|q1|q2|q3|q4/.test(ql))
      return 'Abstained on answerable (likely missing unit/time anchors)';
    if (/which|who|what/.test(ql)) return 'Abstained on answerable (entity anchor missing)';
    return 'Abstained on answerable (insufficient context)';
  }

  if (f1 === 0) {
    if (/oscar|academy award|animated feature|visual effects/.test(ql))
      return 'Wrong answer: award year/category mismatch';
    if (/grand slam|u\.s\. open|us open|australian open|wimbledon|french open/.test(ql))
      return 'Wrong answer: tournament/event mismatch';
    if (/per game|average|ex-dividend|last month|q1|q2|q3|q4|stock/.test(ql))
      return 'Wrong answer: unit/timeframe/definition mismatch';
    return overlap >= 0.5
      ? 'Wrong answer: faithful to wrong context'
      : 'Wrong answer: retrieval off-topic';
  }

  if (em === 0 && f1 > 0) {
    if (/countries|list|name the|which of/.test(ql)) return 'Partial list / incomplete coverage (format mismatch)';
    return 'Formatting/verbosity mismatch (extra words or citations)';
  }

  if (em === 1) return 'Exact match';

  return 'Unclassified';
}

function main() {
  const [runPath, dsPath] = process.argv.slice(2);
  if (!runPath || !dsPath) {
    console.error('Usage: node scripts/analyze_per_question.js <run.jsonl> <dataset.jsonl>');
    process.exit(1);
  }
  const run = readJsonl(runPath).filter((r) => r.final_answer !== undefined);
  const ds = readJsonl(dsPath).filter((d) => d && d.id && d.question);
  const goldById = new Map(ds.map((d) => [d.id, d.gold || '']));
  const qById = new Map(ds.map((d) => [d.id, d.question]));

  const out = [];
  for (const r of run) {
    const id = r.qid || r.id;
    const question = qById.get(id) || '';
    const gold = goldById.get(id) || '';
    const ans = r.final_answer || '';
    const abstain = isIDK(ans);
    const { em, f1 } = emF1(ans, gold);
    const overlap = typeof r.final_o === 'number' ? r.final_o : 0;
    const reason = classifyReason(question, gold, ans, em, f1, abstain, overlap);
    out.push({ id, em: +em.toFixed(3), f1: +f1.toFixed(3), overlap: +overlap.toFixed(3), abstain, reason, question, gold, answer: ans });
  }

  // Print a compact summary plus first few detailed rows
  const buckets = out.reduce((acc, x) => {
    acc[x.reason] = (acc[x.reason] || 0) + 1;
    return acc;
  }, {});
  console.log('Reason buckets:');
  console.log(buckets);
  console.log('\nSample per-question analysis (first 8):');
  for (const row of out.slice(0, 8)) {
    console.log('- id:', row.id);
    console.log('  em:', row.em, 'f1:', row.f1, 'overlap:', row.overlap, 'abstain:', row.abstain);
    console.log('  reason:', row.reason);
    console.log('  Q:', row.question);
    console.log('  Gold:', row.gold);
    console.log('  Ans:', row.answer);
  }
}

main();
