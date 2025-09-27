// Analyze a run JSONL against a dataset JSONL by id, computing EM/F1 and categories.
// Usage: node scripts/analyze_run.js logs/1758298113_agent.jsonl data/crag_questions.jsonl

const fs = require('fs');
const path = require('path');

function readJsonl(filepath) {
  const txt = fs.readFileSync(filepath, 'utf8');
  return txt
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .map((l) => {
      try {
        return JSON.parse(l);
      } catch (e) {
        return null;
      }
    })
    .filter(Boolean);
}

function normalizeText(s) {
  if (!s) return '';
  let t = String(s).toLowerCase();
  // Remove punctuation
  t = t.replace(/[\p{P}\p{S}]/gu, ' ');
  // Remove articles
  t = t.replace(/\b(a|an|the)\b/gi, ' ');
  // Normalize whitespace
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

function tokens(s) {
  const t = normalizeText(s);
  return t.length ? t.split(' ') : [];
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
  for (const [w, c] of pc.entries()) {
    if (gc.has(w)) common += Math.min(c, gc.get(w));
  }
  if (common === 0) return { em, f1: 0.0 };
  const precision = common / pt.length;
  const recall = common / gt.length;
  const f1 = (2 * precision * recall) / (precision + recall);
  return { em, f1 };
}

function isUnanswerable(gold) {
  if (!gold) return true;
  const t = String(gold).trim().toLowerCase();
  return t === 'invalid question' || t === 'n/a' || t === 'unknown' || t === '';
}

function isIDK(answer) {
  const t = normalizeText(answer);
  return t === "i don't know".replace(/['\s]/g, '') || t === 'i dont know' || t === 'i dontknow' || t === 'idontknow';
}

function main() {
  const [runPath, datasetPath] = process.argv.slice(2);
  if (!runPath || !datasetPath) {
    console.error('Usage: node scripts/analyze_run.js <run.jsonl> <dataset.jsonl>');
    process.exit(1);
  }
  const run = readJsonl(runPath).filter((r) => r && r.qid && r.final_answer !== undefined);
  const ds = readJsonl(datasetPath).filter((d) => d && d.id && d.question);
  const goldById = new Map(ds.map((d) => [d.id, d.gold ?? '']))
  const qById = new Map(ds.map((d) => [d.id, d.question]));

  const rows = [];
  for (const r of run) {
    const id = r.qid || r.id;
    if (!goldById.has(id)) continue;
    const gold = goldById.get(id) || '';
    const ans = r.final_answer || '';
    const { em, f1 } = emF1(ans, gold);
    const unans = isUnanswerable(gold);
    const abstain = normalizeText(ans).replace(/\s+/g, '') === 'idontknow';
    rows.push({ id, question: qById.get(id) || '', gold, answer: ans, em, f1, unans, abstain });
  }

  const n = rows.length;
  const emAvg = rows.reduce((a, x) => a + x.em, 0) / (n || 1);
  const f1Avg = rows.reduce((a, x) => a + x.f1, 0) / (n || 1);
  const abstainRate = rows.reduce((a, x) => a + (x.abstain ? 1 : 0), 0) / (n || 1);
  const abstainCorrect = rows.reduce((a, x) => a + ((x.unans && x.abstain) ? 1 : 0), 0) / (n || 1);
  const hallucUnans = rows.reduce((a, x) => a + ((x.unans && !x.abstain) ? 1 : 0), 0) / (n || 1);

  console.log('Joined items:', n);
  console.log('EM:', emAvg.toFixed(3), 'F1:', f1Avg.toFixed(3), 'AbstainRate:', abstainRate.toFixed(3));
  console.log('AbstainAccuracy:', abstainCorrect.toFixed(3), 'HallucinationRate:', hallucUnans.toFixed(3));

  // Category breakdown
  const answ = rows.filter(x => !x.unans);
  const unans = rows.filter(x => x.unans);
  const correctByEM = answ.filter(x => x.em === 1).length;
  const partial = answ.filter(x => x.em === 0 && x.f1 > 0 && !x.abstain).length;
  const wrong = answ.filter(x => x.f1 === 0 && !x.abstain).length;
  const abstainedOnAnswerable = answ.filter(x => x.abstain).length;
  const abstainedOnUnanswerable = unans.filter(x => x.abstain).length;
  const hallucinatedOnUnanswerable = unans.filter(x => !x.abstain).length;
  console.log(`\nBreakdown (answerable=${answ.length}, unanswerable=${unans.length}):`);
  console.log('  EM-correct:', correctByEM);
  console.log('  Partial (0<F1<1):', partial);
  console.log('  Wrong (F1==0):', wrong);
  console.log('  Abstained on answerable:', abstainedOnAnswerable);
  console.log('  Abstained on unanswerable:', abstainedOnUnanswerable);
  console.log('  Hallucinated on unanswerable:', hallucinatedOnUnanswerable);

  // Show mismatches where F1==0 (worst cases)
  const worst = rows.filter((x) => x.f1 === 0 && !x.unans).slice(0, 5);
  if (worst.length) {
    console.log('\nWorst mismatches (F1==0; first 5):');
    for (const w of worst) {
      console.log('- id:', w.id);
      console.log('  Q:', w.question);
      console.log('  Gold:', w.gold);
      console.log('  Ans :', w.answer);
    }
  }

  // Strong partials: highest F1 among answerable, excluding abstains
  const strongPartials = rows
    .filter((x) => !x.unans && !x.abstain && x.em === 0 && x.f1 > 0)
    .sort((a, b) => b.f1 - a.f1)
    .slice(0, 5);
  if (strongPartials.length) {
    console.log('\nStrong partial matches (top 5 by F1):');
    for (const s of strongPartials) {
      console.log(`- id: ${s.id}  (F1=${s.f1.toFixed(3)})`);
      console.log('  Q:', s.question);
      console.log('  Gold:', s.gold);
      console.log('  Ans :', s.answer);
    }
  }

  // Show abstain on answerable questions (missed opportunities)
  const missed = rows.filter((x) => x.abstain && !x.unans).slice(0, 5);
  if (missed.length) {
    console.log('\nAbstained on answerable (first 5):');
    for (const m of missed) {
      console.log('- id:', m.id);
      console.log('  Q:', m.question);
      console.log('  Gold:', m.gold);
    }
  }

  // Show non-abstain on unanswerable (hallucinations)
  const halluc = rows.filter((x) => !x.abstain && x.unans).slice(0, 5);
  if (halluc.length) {
    console.log('\nHallucinated on unanswerable (first 5):');
    for (const h of halluc) {
      console.log('- id:', h.id);
      console.log('  Q:', h.question);
      console.log('  Gold:', h.gold);
      console.log('  Ans :', h.answer);
    }
  }
}

main();
