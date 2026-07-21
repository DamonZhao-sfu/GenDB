/**
 * Experience Graph — SQLite backend (async facade).
 *
 * Wraps the synchronous node:sqlite implementation (../store.mjs, ../query.mjs)
 * in the same async interface the Postgres backend exposes, so both are
 * interchangeable behind index.mjs. Vector search uses sqlite-vec
 * (exhaustive SIMD KNN) with a JS-cosine fallback.
 */

import {
  openStore, closeStore, upsertTask, openSession, recordNode, recordPrompt,
  closeSession, backpropReward,
} from "../store.mjs";
import * as query from "../query.mjs";

export function createSqliteBackend(opts = {}) {
  const store = openStore(opts.dir);
  const w = (fn) => async (...args) => fn(store, ...args);
  return {
    backend: "sqlite",
    vec: store.vec,
    // capture (write)
    upsertTask: w(upsertTask),
    openSession: w(openSession),
    recordNode: w(recordNode),
    recordPrompt: w(recordPrompt),
    closeSession: w(closeSession),
    backpropReward: w(backpropReward),
    // graph traverse
    getNode: w(query.getNode),
    getAncestors: w(query.getAncestors),
    getDescendants: w(query.getDescendants),
    getChildren: w(query.getChildren),
    getSiblings: w(query.getSiblings),
    getSessionTree: w(query.getSessionTree),
    // relation join
    getBestNodeForTask: w(query.getBestNodeForTask),
    getWinningPrompt: w(query.getWinningPrompt),
    getTaskLeaderboard: w(query.getTaskLeaderboard),
    // vector search
    searchSimilarTasks: w(query.searchSimilarTasks),
    searchSimilarStrategies: w(query.searchSimilarStrategies),
    // lifecycle
    close: async () => closeStore(store),
  };
}
