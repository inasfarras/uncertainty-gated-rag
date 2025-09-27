// Compare two runs (old vs new) per question id.
// Usage: node scripts/compare_runs.js <old_run.jsonl> <new_run.jsonl> <dataset.jsonl>

const fs = require('fs');

function readJsonl(p) {
  const txt = fs.readFileSync(p, 'utf8');
  return txt
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      try { return JSON.parse(l); } catch { return null; }
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

function main() {
  const [oldPath, newPath, dsPath] = process.argv.slice(2);
  if (!oldPath || !newPath || !dsPath) {
    console.error('Usage: node scripts/compare_runs.js <old_run.jsonl> <new_run.jsonl> <dataset.jsonl>');
    process.exit(1);
  }
  const oldRun = readJsonl(oldPath).filter((r) => r.final_answer !== undefined);
  const newRun = readJsonl(newPath).filter((r) => r.final_answer !== undefined);
  const ds = readJsonl(dsPath).filter((d) => d && d.id && d.question);
  const goldById = new Map(ds.map((d) => [d.id, d.gold || '']));
  const qById = new Map(ds.map((d) => [d.id, d.question]));

  const oldById = new Map(oldRun.map((r) => [r.qid || r.id, r]));
  const newById = new Map(newRun.map((r) => [r.qid || r.id, r]));
  const ids = [...new Set([...oldById.keys(), ...newById.keys()])];

  let improvedIdkToAns = 0;
  let fixedWrongToCorrectOrAbstain = 0;
  let newHallucinations = 0;
  const changes = [];

  for (const id of ids) {
    const o = oldById.get(id);
    const n = newById.get(id);
    if (!o || !n) continue;
    const q = qById.get(id) || '';
    const g = goldById.get(id) || '';
    const oAns = o.final_answer || '';
    const nAns = n.final_answer || '';
    const oIdk = isIDK(oAns);
    const nIdk = isIDK(nAns);
    const oEF = emF1(oAns, g);
    const nEF = emF1(nAns, g);
    const oWrong = !oIdk && oEF.f1 === 0.0 && g.trim() !== '';
    const nWrong = !nIdk && nEF.f1 === 0.0 && g.trim() !== '';
    const unans = (g.trim().toLowerCase() === '' || g.trim().toLowerCase() === 'invalid question' || g.trim().toLowerCase() === 'n/a' || g.trim().toLowerCase() === 'unknown');

    if (oIdk && !nIdk && nEF.f1 > 0) improvedIdkToAns += 1;
    if (oWrong && (!nWrong || nIdk)) fixedWrongToCorrectOrAbstain += 1;
    if (unans && !nIdk) newHallucinations += 1;

    if (oIdk !== nIdk || oEF.f1 !== nEF.f1) {
      changes.push({ id, q, gold: g, old: { idk: oIdk, em: oEF.em, f1: oEF.f1, ans: oAns }, new: { idk: nIdk, em: nEF.em, f1: nEF.f1, ans: nAns } });
    }
  }

  console.log('Improvements:');
  console.log('  IDK -> Answer with F1>0:', improvedIdkToAns);
  console.log('  Wrong -> Correct or Abstain:', fixedWrongToCorrectOrAbstain);
  console.log('Regressions:');
  console.log('  New hallucinations on unanswerable:', newHallucinations);
  console.log('\nSample changes (first 10):');
  for (const c of changes.slice(0, 10)) {
    console.log('- id:', c.id);
    console.log('  Q:', c.q);
    console.log('  Gold:', c.gold);
    console.log('  Old:', c.old);
    console.log('  New:', c.new);
  }
}

main();
