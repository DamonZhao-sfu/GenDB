/**
 * Pluggable text → vector embedding for the Experience Graph's vector-search
 * access pattern.
 *
 * Default: a deterministic, offline, dependency-free hashing embedding (bag of
 * hashed tokens, L2-normalized). It captures lexical overlap of SQL / strategy
 * text well enough for cosine ranking, and — being deterministic — makes
 * retrieval reproducible in tests.
 *
 * Phase (d) can swap `embed` for a real embedding model behind this exact
 * signature (Float32Array of length EMBED_DIM); nothing else changes.
 */

export const EMBED_DIM = 256;

/** FNV-1a 32-bit hash of a string. */
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Embed text into a unit-length Float32Array(EMBED_DIM). Deterministic, offline. */
export function embed(text) {
  const v = new Float32Array(EMBED_DIM);
  if (!text) return v;
  const tokens = String(text).toLowerCase().match(/[a-z_][a-z0-9_]{1,}/g);
  if (!tokens) return v;
  for (const t of tokens) {
    v[fnv1a(t) % EMBED_DIM] += 1;
  }
  let norm = 0;
  for (let i = 0; i < EMBED_DIM; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < EMBED_DIM; i++) v[i] /= norm;
  return v;
}
