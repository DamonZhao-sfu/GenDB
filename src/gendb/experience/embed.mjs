/**
 * Pluggable text → vector embedding for the Experience Graph's vector-search
 * access pattern.
 *
 * Phase (a/b) default: a deterministic, offline, dependency-free hashing
 * embedding (bag of hashed tokens, L2-normalized). It captures lexical overlap
 * of SQL / strategy text well enough for cosine ranking, and — being
 * deterministic — makes retrieval reproducible in tests.
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

/**
 * Embed text into a unit-length Float32Array(EMBED_DIM).
 * Deterministic and offline.
 */
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

/** Cosine similarity of two equal-length Float32Arrays (both assumed L2-normalized). */
export function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/** Encode a Float32Array as a Buffer for BLOB storage. */
export function toBlob(vec) {
  if (!vec) return null;
  const f = vec instanceof Float32Array ? vec : Float32Array.from(vec);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}

/** Decode a BLOB (Buffer/Uint8Array) back into a Float32Array. */
export function fromBlob(blob) {
  if (!blob) return null;
  const buf = Buffer.isBuffer(blob) ? blob : Buffer.from(blob);
  // Copy to guarantee 4-byte alignment for Float32Array.
  const copy = Buffer.from(buf);
  return new Float32Array(copy.buffer, copy.byteOffset, Math.floor(copy.byteLength / 4));
}
