/** Stable identity helpers for the Experience Graph (pure). */

export function taskId(benchmark, queryId, scaleFactor) {
  return `${benchmark}__${queryId}__sf${scaleFactor}`;
}
export function sessionId(runId, queryId) {
  return `${runId}__${queryId}`;
}
export function nodeId(sessionId_, iteration) {
  return `${sessionId_}__iter_${iteration}`;
}
