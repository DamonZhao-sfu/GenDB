/*
 * arrow_scaffold.h — Arrow C Data Interface output layer for GenDB.
 *
 * Ported verbatim from the Jailbreak project
 * (jailbreak-agentic/tools/arrow_scaffold.h). Dependency-free: provides
 * ArrowSchema/ArrowArray + a builder API (ab_create / ab_append_* / ab_finalize)
 * so a GENERATED ingest reader only writes decode logic — all buffer packing,
 * validity bitmaps, release callbacks, and children[] arrays are handled here.
 *
 * GenDB usage: the generated ingest.cpp decodes the source (Parquet columns via
 * Arrow C++, or .tbl text) and appends each value with ab_append_* into a
 * builder, then ab_finalize() to obtain an in-memory Arrow struct array
 * (C Data Interface). gendb_arrow_storage.h then persists that array to an
 * Arrow IPC/Feather file for later zero-copy, mmap'd reads by query binaries.
 *
 * Append fn <-> Arrow format: ab_append_bool "b" | int8 "c" | int16 "s" |
 *   int32 "i"/"tdD"(date32) | int64 "l"/"tsu:"(ts µs) | float32 "f" |
 *   float64 "g"/decimal-as-double | string "u"/"U"/"z"/"Z".
 */
/*
 * arrow_scaffold.h — Arrow C Data Interface output layer for jailbreak-agentic.
 *
 * Provides a builder API so generated readers only write decode logic.
 * All ArrowSchema/ArrowArray memory management, buffer packing, validity
 * bitmaps, release callbacks, and children[] pointer arrays are handled here.
 *
 * Usage pattern in a generated reader:
 *
 *   #include "arrow_scaffold.h"
 *
 *   extern "C" int db_to_arrow(
 *       const char* file_path, const char* col_spec,
 *       ArrowSchema* out_schema, ArrowArray* out_array,
 *       char* errmsg, int errmsg_len)
 *   {
 *       // 1. Parse col_spec → names[], arrow_fmts[], n_cols
 *       // 2. Create builder
 *       ArrowBuilder* ab = ab_create(n_cols, names, arrow_fmts);
 *       // 3. Decode file; for each row, for each column:
 *       //      ab_append_int32(ab, ci, val, is_null);   // "i"
 *       //      ab_append_int64(ab, ci, val, is_null);   // "l", "tsu:"
 *       //      ab_append_float64(ab, ci, val, is_null); // "g"
 *       //      ab_append_string(ab, ci, ptr, len, is_null); // "u"
 *       //      ... (see full API below)
 *       // 4. Finalize
 *       return ab_finalize(ab, out_schema, out_array, errmsg, errmsg_len);
 *       // On error before finalize: ab_destroy(ab); return -1;
 *   }
 *
 * Append function ↔ Arrow format mapping:
 *   ab_append_bool    → "b"
 *   ab_append_int8    → "c" (uint8: cast to int8_t)
 *   ab_append_int16   → "s"
 *   ab_append_int32   → "i", "tdD" (date32), "I" (uint32 cast)
 *   ab_append_int64   → "l", "tsu:" (timestamp µs), "L" (uint64 cast)
 *   ab_append_float32 → "f"
 *   ab_append_float64 → "g", "d:M,D" (decimal as double)
 *   ab_append_string  → "u", "U", "z", "Z"
 */

#pragma once
#ifndef ARROW_SCAFFOLD_H
#define ARROW_SCAFFOLD_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

// ── Arrow C Data Interface structs ─────────────────────────────────────────────
// https://arrow.apache.org/docs/format/CDataInterface.html
// Copy these exactly — do NOT add, remove, or reorder fields.
#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

struct ArrowSchema {
    const char*   format;
    const char*   name;
    const char*   metadata;
    int64_t       flags;
    int64_t       n_children;
    ArrowSchema** children;
    ArrowSchema*  dictionary;
    void (*release)(ArrowSchema*);
    void*         private_data;
};

struct ArrowArray {
    int64_t       length;
    int64_t       null_count;
    int64_t       offset;      // REQUIRED — always init to 0
    int64_t       n_buffers;
    int64_t       n_children;
    const void**  buffers;
    ArrowArray**  children;
    ArrowArray*   dictionary;
    void (*release)(ArrowArray*);
    void*         private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

// ── Internal per-column buffer ─────────────────────────────────────────────────
struct _AbColBuf {
    std::string  fmt;
    std::string  name;
    bool         is_str;   // "u","U","z","Z"
    bool         is_bool;  // "b"

    std::vector<uint8_t> fixed;    // fixed-width raw bytes (or packed bits for bool)
    std::vector<int32_t> offsets;  // string offsets: n_rows+1 entries, starts with {0}
    std::vector<uint8_t> chars;    // string UTF-8 bytes
    std::vector<uint8_t> validity; // LSB-first packed bitmap: bit i=1 means row i valid
    int64_t null_count;
    int64_t n_rows;

    const void* buf_ptrs[3];  // pointed at fixed/offsets/chars for finalize
    ArrowArray  arr;
    ArrowSchema sch;
};

// ── BatchData: single heap object owning all column data ───────────────────────
struct _AbBatchData {
    std::vector<_AbColBuf> cols;
    int64_t       n_rows  = 0;
    int           n_cols  = 0;
    ArrowArray**  arr_ch  = nullptr;  // freed by release_top_arr
    ArrowSchema** sch_ch  = nullptr;  // freed by release_top_sch (via private_data copy)
};

// ── Builder wrapper (freed by ab_finalize or ab_destroy) ──────────────────────
struct ArrowBuilder {
    _AbBatchData* bd;
};

// ── Release callbacks ──────────────────────────────────────────────────────────
// Children: trivial — just null the release pointer.
static inline void _ab_release_child_arr(ArrowArray* a)  { a->release = nullptr; }
static inline void _ab_release_child_sch(ArrowSchema* s) { s->release = nullptr; }

// Top-level array: calls child releases, frees arr_ch, then deletes BatchData.
// (sch_ch is a separate heap alloc freed by release_top_sch via its private_data.)
static inline void _ab_release_top_arr(ArrowArray* a) {
    if (!a->release) return;
    auto* bd = static_cast<_AbBatchData*>(a->private_data);
    for (int i = 0; i < bd->n_cols; ++i) {
        if (bd->cols[i].arr.release) bd->cols[i].arr.release(&bd->cols[i].arr);
        if (bd->cols[i].sch.release) bd->cols[i].sch.release(&bd->cols[i].sch);
    }
    delete[] bd->arr_ch;
    delete bd;  // destructors free all std::vector/std::string members
    a->release = nullptr;
}

// Top-level schema: frees only the sch_ch pointer array.
// (BatchData is owned by the array's release callback.)
static inline void _ab_release_top_sch(ArrowSchema* s) {
    if (!s->release) return;
    auto** ch = static_cast<ArrowSchema**>(s->private_data);
    delete[] ch;
    s->release = nullptr;
}

// ── Validity bitmap helper ─────────────────────────────────────────────────────
static inline void _ab_set_valid(std::vector<uint8_t>& bm, int64_t row, bool valid) {
    int idx = (int)(row >> 3);
    if (idx >= (int)bm.size()) bm.resize(idx + 1, 0);
    if (valid) bm[idx] |=  (1u << (row & 7));
    else       bm[idx] &= ~(1u << (row & 7));
}

// ── Builder creation ──────────────────────────────────────────────────────────
// n_cols    : number of columns
// names     : column names (Arrow schema name field per column)
// formats   : Arrow format strings per column (e.g. "i", "g", "u", "tdD")
static inline ArrowBuilder* ab_create(int n_cols,
                                       const char* const* names,
                                       const char* const* formats)
{
    auto* b = new ArrowBuilder();
    b->bd = new _AbBatchData();
    b->bd->n_cols = n_cols;
    b->bd->cols.resize(n_cols);
    for (int i = 0; i < n_cols; ++i) {
        auto& c      = b->bd->cols[i];
        c.name       = names[i]   ? names[i]   : "";
        c.fmt        = formats[i] ? formats[i] : "i";
        c.null_count = 0;
        c.n_rows     = 0;
        c.is_str     = (c.fmt == "u" || c.fmt == "U" ||
                        c.fmt == "z" || c.fmt == "Z");
        c.is_bool    = (c.fmt == "b");
        if (c.is_str) c.offsets.push_back(0);  // string offsets start at 0
        memset(c.buf_ptrs, 0, sizeof(c.buf_ptrs));
    }
    return b;
}

// ── Internal: record validity for the next row ────────────────────────────────
static inline void _ab_track(_AbColBuf& c, bool is_null) {
    _ab_set_valid(c.validity, c.n_rows, !is_null);
    if (is_null) ++c.null_count;
}

// ── Append functions ──────────────────────────────────────────────────────────

// bool — Arrow format "b" — packed bits, 1 bit per value, LSB first
static inline void ab_append_bool(ArrowBuilder* b, int col, bool val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    int byte_idx = (int)(c.n_rows >> 3);
    if (byte_idx >= (int)c.fixed.size()) c.fixed.push_back(0);
    if (!is_null && val) c.fixed[byte_idx] |= (1u << (c.n_rows & 7));
    ++c.n_rows;
}

// int8 — Arrow format "c"
static inline void ab_append_int8(ArrowBuilder* b, int col, int8_t val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    uint8_t v = is_null ? 0 : (uint8_t)val;
    c.fixed.push_back(v);
    ++c.n_rows;
}

// int16 — Arrow format "s"
static inline void ab_append_int16(ArrowBuilder* b, int col, int16_t val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    int16_t v = is_null ? 0 : val;
    c.fixed.insert(c.fixed.end(), (uint8_t*)&v, (uint8_t*)&v + 2);
    ++c.n_rows;
}

// int32 — Arrow format "i" (also used for date32 "tdD": pass days-since-Unix-epoch)
static inline void ab_append_int32(ArrowBuilder* b, int col, int32_t val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    int32_t v = is_null ? 0 : val;
    c.fixed.insert(c.fixed.end(), (uint8_t*)&v, (uint8_t*)&v + 4);
    ++c.n_rows;
}

// int64 — Arrow format "l" (also used for timestamp "tsu:": pass µs-since-Unix-epoch)
static inline void ab_append_int64(ArrowBuilder* b, int col, int64_t val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    int64_t v = is_null ? 0 : val;
    c.fixed.insert(c.fixed.end(), (uint8_t*)&v, (uint8_t*)&v + 8);
    ++c.n_rows;
}

// float32 — Arrow format "f"
static inline void ab_append_float32(ArrowBuilder* b, int col, float val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    float v = is_null ? 0.0f : val;
    c.fixed.insert(c.fixed.end(), (uint8_t*)&v, (uint8_t*)&v + 4);
    ++c.n_rows;
}

// float64 — Arrow format "g" (also used for numeric/decimal)
static inline void ab_append_float64(ArrowBuilder* b, int col, double val, bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    double v = is_null ? 0.0 : val;
    c.fixed.insert(c.fixed.end(), (uint8_t*)&v, (uint8_t*)&v + 8);
    ++c.n_rows;
}

// string — Arrow format "u"/"U"/"z"/"Z"
// data: UTF-8 bytes (not NUL-terminated); len: byte count (0 for empty, -1 = strlen)
static inline void ab_append_string(ArrowBuilder* b, int col,
                                     const char* data, int32_t len,
                                     bool is_null) {
    auto& c = b->bd->cols[col];
    _ab_track(c, is_null);
    if (!is_null && data) {
        if (len < 0) len = (int32_t)strlen(data);
        if (len > 0)
            c.chars.insert(c.chars.end(), (uint8_t*)data, (uint8_t*)data + len);
    }
    c.offsets.push_back((int32_t)c.chars.size());
    ++c.n_rows;
}

// ── Finalize: populate ArrowSchema / ArrowArray ────────────────────────────────
// Moves all data into out_schema/out_array for PyArrow import.
// After this call, the ArrowBuilder is freed — do NOT call ab_destroy().
// On success returns 0; on error returns -1 and writes to errmsg.
static inline int ab_finalize(ArrowBuilder* b,
                               ArrowSchema*  out_schema,
                               ArrowArray*   out_array,
                               char*         errmsg,
                               int           errmsg_len)
{
    _AbBatchData* bd = b->bd;
    b->bd = nullptr;
    delete b;  // free the thin wrapper; bd now owned by us until transferred below

    int     ncols  = bd->n_cols;
    int64_t n_rows = (ncols > 0) ? bd->cols[0].n_rows : 0;
    bd->n_rows = n_rows;

    // ── Per-column schema + array ──────────────────────────────────────────────
    for (int ci = 0; ci < ncols; ++ci) {
        _AbColBuf& col = bd->cols[ci];

        // Schema
        col.sch.format       = col.fmt.c_str();
        col.sch.name         = col.name.c_str();
        col.sch.metadata     = nullptr;
        col.sch.flags        = 0;
        col.sch.n_children   = 0;
        col.sch.children     = nullptr;
        col.sch.dictionary   = nullptr;
        col.sch.release      = _ab_release_child_sch;
        col.sch.private_data = nullptr;

        // Array
        col.arr.length       = n_rows;
        col.arr.null_count   = col.null_count;
        col.arr.offset       = 0;
        col.arr.n_children   = 0;
        col.arr.children     = nullptr;
        col.arr.dictionary   = nullptr;
        col.arr.release      = _ab_release_child_arr;
        col.arr.private_data = nullptr;

        // Validity bitmap: only needed if there are nulls
        const void* vbm = nullptr;
        if (col.null_count > 0) {
            int need = (int)((n_rows + 7) / 8);
            col.validity.resize(need, 0);
            vbm = col.validity.data();
        }

        if (!col.is_str) {
            // Fixed-width (and bool packed bits)
            col.buf_ptrs[0]  = vbm;
            col.buf_ptrs[1]  = col.fixed.data();
            col.arr.n_buffers = 2;
            col.arr.buffers   = col.buf_ptrs;
        } else {
            // String: ensure chars buffer is non-null even when all strings empty
            if (col.chars.empty()) col.chars.push_back(0);
            col.buf_ptrs[0]   = vbm;
            col.buf_ptrs[1]   = col.offsets.data();
            col.buf_ptrs[2]   = col.chars.data();
            col.arr.n_buffers = 3;
            col.arr.buffers   = col.buf_ptrs;
        }
    }

    // ── Children pointer arrays ────────────────────────────────────────────────
    bd->arr_ch = new ArrowArray*[ncols];
    bd->sch_ch = new ArrowSchema*[ncols];
    for (int ci = 0; ci < ncols; ++ci) {
        bd->arr_ch[ci] = &bd->cols[ci].arr;
        bd->sch_ch[ci] = &bd->cols[ci].sch;
    }

    // ── Top-level struct array ─────────────────────────────────────────────────
    // n_buffers=1 with a null validity bitmap is the correct layout for a struct.
    static const void* _ab_top_bufs[1] = {nullptr};
    out_array->length       = n_rows;
    out_array->null_count   = 0;
    out_array->offset       = 0;
    out_array->n_buffers    = 1;
    out_array->buffers      = _ab_top_bufs;
    out_array->n_children   = ncols;
    out_array->children     = bd->arr_ch;
    out_array->dictionary   = nullptr;
    out_array->release      = _ab_release_top_arr;
    out_array->private_data = bd;  // BatchData transferred here

    // ── Top-level struct schema ────────────────────────────────────────────────
    static const char _ab_fmt[]  = "+s";
    static const char _ab_name[] = "";
    out_schema->format       = _ab_fmt;
    out_schema->name         = _ab_name;
    out_schema->metadata     = nullptr;
    out_schema->flags        = 0;
    out_schema->n_children   = ncols;
    out_schema->children     = bd->sch_ch;
    out_schema->dictionary   = nullptr;
    out_schema->release      = _ab_release_top_sch;
    out_schema->private_data = bd->sch_ch;  // pointer value copy; freed independently

    (void)errmsg; (void)errmsg_len;
    return 0;
}

// ── Cleanup on error (use instead of ab_finalize when aborting) ───────────────
static inline void ab_destroy(ArrowBuilder* b) {
    if (!b) return;
    delete b->bd;
    delete b;
}

#endif  // ARROW_SCAFFOLD_H
