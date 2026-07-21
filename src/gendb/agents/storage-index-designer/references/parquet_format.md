# Parquet Storage Format Reference

Knowledge base for the Storage/Index Designer's **Parquet Architect** step. It contains
the file layout, physical/logical type rules, encodings, null handling, the exact Apache
Arrow C++ read patterns, and hard-won gotchas needed to produce a **correct** Parquet
ingest reader. GenDB is **Arrow-native**: the reader decodes each column and appends values
via `arrow_scaffold.h` (`ab_append_*`), then persists one **Arrow IPC/Feather** file per
table via `gendb_arrow_storage.h`. Query binaries mmap those files for zero-copy reads.

The Architect step reads this file, inspects the actual Parquet metadata, and emits a
`parquet_spec.json` (contract at the end of this file). A mechanical validator
(`tools/validate-parquet-spec.mjs`) then checks that spec before any C++ is written. The
single most common failure — a column whose written row count disagrees with its siblings
— comes from mishandling **dictionary encoding** or **multiple row groups / chunks**. Both
are covered in detail below.

---

## 1. File structure (top-down)

```
PAR1                              ← 4-byte magic at the START of the file
  Row Group 0
    Column Chunk (col A)
      [Dictionary Page]           ← optional; present when the column is dict-encoded
      Data Page 0 / Data Page 1 / …
    Column Chunk (col B)
      …
  Row Group 1
    …
  Row Group N
FileMetaData (Thrift-serialized)  ← the "footer": schema, num_rows, row groups, stats
4-byte footer length (little-endian)
PAR1                              ← 4-byte magic at the END of the file
```

Key facts:
- **A Parquet file is column-major within each row group, but has MULTIPLE row groups.**
  A logical column's values are scattered across one column chunk **per row group**. You
  MUST read and concatenate every row group to get all rows. Reading only row group 0 (or,
  in Arrow, only `chunk(0)`) is the #1 corruption bug.
- The **footer `FileMetaData`** is the source of truth: `num_rows` (total across all row
  groups), the schema (column names + types), and per-column-chunk statistics
  (min/max/null_count/distinct_count). Read `num_rows` from here and assert every column
  you write emits exactly that many values.
- **Pages** are the unit of encoding + compression. A column chunk may start with one
  **dictionary page** (the unique values, PLAIN-encoded) followed by **data pages** whose
  values are dictionary indices (RLE/bit-packed). Each page is independently compressed.

> You will almost never hand-parse this structure. Use Apache Arrow C++ (Section 6), which
> handles pages, decompression, and level decoding for you — but you MUST still handle
> multi-chunk `ChunkedArray`s and `DictionaryArray`s correctly (Sections 5 & 6).

---

## 2. Physical types

Every column has exactly one **physical type** (the on-disk primitive):

| Physical type          | Bytes            | Notes |
|------------------------|------------------|-------|
| `BOOLEAN`              | 1 bit            | bit-packed |
| `INT32`                | 4                | also carries DATE, DECIMAL(p≤9), INT(8/16/32) |
| `INT64`                | 8                | also carries TIMESTAMP, TIME(micros), DECIMAL(p≤18), INT64 |
| `INT96`                | 12 (**legacy**)  | ONLY legacy nanosecond timestamps — see §4 |
| `FLOAT`                | 4                | IEEE-754 single |
| `DOUBLE`               | 8                | IEEE-754 double |
| `BYTE_ARRAY`           | var              | STRING/UTF8, JSON, BSON, DECIMAL (big-endian), variable binary |
| `FIXED_LEN_BYTE_ARRAY` | fixed N          | DECIMAL (big-endian), UUID(16), FLOAT16(2), fixed binary |

The physical type alone is not enough — the **logical type** (§3) tells you how to
interpret the bytes.

---

## 3. Logical types (a.k.a. LogicalType / legacy ConvertedType)

The logical type sits on top of a physical type and defines the semantics. Map each SQL
schema column to `(physical_type, logical_type)` and pick the GenDB target type + decode.

| Logical type            | Physical carrier(s)                         | Decode → GenDB |
|-------------------------|---------------------------------------------|----------------|
| `STRING` / `UTF8`       | `BYTE_ARRAY`                                | UTF-8 bytes → GenDB string / `char` (for CHAR(1)) |
| `ENUM`                  | `BYTE_ARRAY`                                | like STRING |
| `DECIMAL(p,s)`          | `INT32` (p≤9), `INT64` (p≤18), `FIXED_LEN_BYTE_ARRAY`, `BYTE_ARRAY` | **unscaled integer**; real = unscaled × 10⁻ˢ. See §4. |
| `DATE`                  | `INT32`                                     | **days since 1970-01-01** (== Arrow date32 == GenDB `days_since_epoch_1970`). No shift needed. |
| `TIME(unit)`            | `INT32` (millis) / `INT64` (micros/nanos)   | elapsed time of day in `unit` |
| `TIMESTAMP(unit,utc)`   | `INT64`                                     | count of `unit` (ms/us/ns) since epoch; `isAdjustedToUTC` matters for tz |
| `TIMESTAMP` (legacy)    | `INT96`                                     | **legacy**: 12 bytes = int64 nanos-of-day + int32 Julian day. See §4. |
| `INT(bits, signed)`     | `INT32` / `INT64`                           | narrow to the declared width/signedness |
| `UINT_8/16/32/64`       | `INT32` / `INT64`                           | unsigned interpretation |
| `FLOAT16`               | `FIXED_LEN_BYTE_ARRAY(2)`                    | half-precision |
| `JSON` / `BSON`         | `BYTE_ARRAY`                                | treat as bytes/string |
| `UUID`                  | `FIXED_LEN_BYTE_ARRAY(16)`                  | 16 raw bytes |
| `LIST` / `MAP` / struct | nested groups                               | **not present in TPC-H**; flat-schema readers should reject/ignore |
| (none)                  | any primitive                               | use the physical type directly (e.g. raw INT64) |

---

## 4. Type decode details that bite

### DECIMAL
- Stored as an **unscaled two's-complement integer**. Real value = `unscaled × 10^(-scale)`.
- `INT32`/`INT64` carriers: read the little-endian integer directly, then apply scale.
- `FIXED_LEN_BYTE_ARRAY` / `BYTE_ARRAY` carriers: the bytes are **big-endian, signed
  two's complement**. Sign-extend from the top byte, assemble, then apply scale.
- GenDB target: usually `double` (`value = unscaled / pow(10, scale)`) or a scaled int64 if
  the storage design chose fixed-point. TPC-H money/quantity columns (e.g.
  `l_extendedprice`, `l_discount`, `l_tax`, `l_quantity`) are DECIMAL — get the scale right
  or every SUM is wrong.

### DATE
- `INT32` = **days since 1970-01-01**. This is identical to Arrow `date32` and to GenDB's
  `days_since_epoch_1970`. **No epoch shift** (unlike PostgreSQL heap, which uses 2000-01-01
  and needs +10957). TPC-H `l_shipdate`, `l_commitdate`, `l_receiptdate`, `o_orderdate`.

### TIMESTAMP
- Modern: `INT64` count of ms/us/ns since 1970-01-01. Convert to your storage unit; watch
  the declared `unit`.
- **Legacy `INT96`**: 12 bytes = `int64 nanoseconds_within_day` (first 8, little-endian) +
  `int32 julian_day` (last 4). Convert Julian day → days since epoch (`julian_day -
  2440588`) then combine. Some writers (old Spark/Impala) still emit this. Arrow surfaces it
  as a timestamp — prefer letting Arrow decode it.

### BOOLEAN
- Bit-packed 1 bit/value. Arrow returns a `BooleanArray`; read with `Value(i)`.

---

## 5. Encodings (how values are packed inside a page)

| Encoding                    | Where used | Decode approach |
|-----------------------------|-----------|-----------------|
| `PLAIN`                     | any        | raw fixed-width values back-to-back; BYTE_ARRAY = 4-byte little-endian length prefix + bytes |
| `RLE_DICTIONARY`            | data pages of dict-encoded columns (modern) | values are **indices** into the dictionary page; RLE/bit-packed hybrid |
| `PLAIN_DICTIONARY`          | same, **legacy** name (dictionary page PLAIN, indices in data pages) | identical intent to `RLE_DICTIONARY` |
| `RLE`                       | boolean data, definition/repetition levels | run-length + bit-packing hybrid |
| `BIT_PACKED` (deprecated)   | old level encoding | bit-packed |
| `DELTA_BINARY_PACKED`       | INT32/INT64 | delta + bit-pack; decode sequentially |
| `DELTA_LENGTH_BYTE_ARRAY`   | BYTE_ARRAY | delta-encoded lengths + concatenated bytes |
| `DELTA_BYTE_ARRAY`          | BYTE_ARRAY | incremental (prefix-shared) strings |
| `BYTE_STREAM_SPLIT`         | FLOAT/DOUBLE/fixed | byte-plane split; reassemble per value |

> With Arrow C++ you do NOT decode these by hand — Arrow decodes any encoding into a
> materialized typed array. **BUT**: a **dictionary-encoded** column may be surfaced to you
> as an `arrow::DictionaryArray` (indices + a dictionary), not the decoded values. If you
> write the index buffer or the dictionary length instead of one decoded value per row, the
> column's row count / values are wrong. See §6.3 — this is the `l_returnflag` bug.

---

## 6. Reading Parquet with Apache Arrow C++ (the practical path)

Link: `Makefile` MUST use `pkg-config --cflags --libs arrow parquet`.
Includes: `<arrow/io/file.h>`, `<parquet/arrow/reader.h>`, `<arrow/table.h>`,
`<arrow/array.h>`, `<arrow/compute/api.h>` (for Cast).

### 6.1 Open + read a whole table (small/medium tables)
```cpp
std::shared_ptr<arrow::io::ReadableFile> infile;
PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(path));
std::unique_ptr<parquet::arrow::FileReader> reader;
PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
std::shared_ptr<arrow::Table> table;
PARQUET_THROW_NOT_OK(reader->ReadTable(&table));   // reads ALL row groups
const int64_t num_rows = table->num_rows();        // == footer num_rows
```
Map columns BY NAME: `int idx = table->schema()->GetFieldIndex("l_shipdate");`
(never assume Parquet column order matches the SQL schema).

### 6.2 Multi-chunk ChunkedArray — YOU MUST ITERATE ALL CHUNKS
`table->column(idx)` is a `arrow::ChunkedArray` with **one or more chunks** (Arrow may
split by row group or by size). Reading only `chunk(0)` truncates the column.
```cpp
auto col = table->column(idx);                     // ChunkedArray
int64_t written = 0;
for (int c = 0; c < col->num_chunks(); ++c) {
    auto arr = col->chunk(c);                      // one Array
    // ... append every element of arr ...
    written += arr->length();
}
// written MUST equal num_rows for EVERY column.
```

### 6.3 Dictionary-encoded columns — DECODE to logical values
Arrow may hand a dict-encoded column back as an `arrow::DictionaryArray`. Two correct ways:
```cpp
// Option A (simplest, robust): cast the whole chunk to its value type first.
#include <arrow/compute/api.h>
auto value_type = arrow::utf8();                   // or the column's real type
arrow::Datum decoded;
ARROW_ASSIGN_OR_RAISE(decoded, arrow::compute::Cast(arr, value_type));
auto values = std::static_pointer_cast<arrow::StringArray>(decoded.make_array());
for (int64_t i = 0; i < values->length(); ++i)
    write_char(values->IsNull(i) ? '\0' : values->GetString(i)[0]);

// Option B (manual): index into the dictionary.
auto dict_arr = std::static_pointer_cast<arrow::DictionaryArray>(arr);
auto indices  = std::static_pointer_cast<arrow::Int32Array>(dict_arr->indices());
auto dict     = std::static_pointer_cast<arrow::StringArray>(dict_arr->dictionary());
for (int64_t i = 0; i < indices->length(); ++i)
    write_char(indices->IsNull(i) ? '\0' : dict->GetString(indices->Value(i))[0]);
```
- **NEVER** write `dict->length()` (the number of distinct values) or `indices` as if they
  were the column — you must emit **one decoded value per row**, `arr->length()` values.
- To avoid the DictionaryArray path entirely, you may set
  `reader->set_use_threads(true)` and disable dictionary output via
  `parquet::ArrowReaderProperties` (`set_read_dictionary(idx, false)`), which makes Arrow
  materialize plain values. Either approach is fine — just be consistent.

### 6.4 Streaming large tables (bound memory)
For big fact tables (lineitem), don't materialize the whole table:
```cpp
std::shared_ptr<arrow::RecordBatchReader> rb_reader;
PARQUET_THROW_NOT_OK(reader->GetRecordBatchReader(&rb_reader));
std::shared_ptr<arrow::RecordBatch> batch;
while (true) {
    PARQUET_THROW_NOT_OK(rb_reader->ReadNext(&batch));
    if (!batch) break;                             // ← loop until null; do not stop early
    // append batch->column(idx) for each column
}
```
Every column accumulates across ALL batches; the totals must equal `num_rows`.

### 6.5 Typed array accessors
| Arrow array           | physical/logical                | accessor |
|-----------------------|---------------------------------|----------|
| `Int32Array`          | INT32 / DATE / INT(32)          | `Value(i)` |
| `Int64Array`          | INT64 / TIMESTAMP / TIME        | `Value(i)` |
| `DoubleArray`/`FloatArray` | DOUBLE/FLOAT               | `Value(i)` |
| `Date32Array`         | DATE                            | `Value(i)` (days since epoch) |
| `Decimal128Array`/`Decimal256Array` | DECIMAL           | `Value(i)` raw bytes / `FormatValue(i)`; apply scale |
| `StringArray`         | STRING/UTF8                     | `GetString(i)` / `GetView(i)` |
| `BooleanArray`        | BOOLEAN                         | `Value(i)` |
| `DictionaryArray`     | any dict-encoded                | decode first (§6.3) |

### 6.6 Nulls
- `arr->IsNull(i)` / `arr->IsValid(i)`; `arr->null_count()`.
- Arrow reconstructs nulls from Parquet **definition levels** for you. Decide a GenDB
  sentinel/validity scheme matching the delimited-text path and apply it consistently.
- A **nullable** column with 0 actual nulls still has `null_count()==0` — fine.

---

## 7. Compression
Pages may be `SNAPPY`, `GZIP`, `ZSTD`, `LZ4`, `LZ4_RAW`, `BROTLI`, `LZO`, or
`UNCOMPRESSED`. Arrow decompresses transparently as long as the codec is compiled in
(arrow-cpp from conda-forge includes snappy/zstd/gzip/lz4/brotli). If a codec is missing,
`ReadTable` throws — surface the error, don't silently produce partial data.

---

## 8. Reading Parquet metadata WITHOUT writing C++ (for the Architect step)
Use these to build the spec. Prefer DuckDB (usually installed alongside arrow); pyarrow is
the fallback.

```bash
# Schema: column name, physical + logical type
duckdb -json -c "SELECT name, type, converted_type, logical_type
                 FROM parquet_schema('lineitem.parquet')"
# Per-column-chunk encodings + compression + stats (reveals dictionary encoding!)
duckdb -json -c "SELECT path_in_schema, encodings, compression, num_values,
                        stats_min, stats_max, stats_null_count
                 FROM parquet_metadata('lineitem.parquet')"
# Total rows and row-group count
duckdb -json -c "SELECT num_rows FROM parquet_file_metadata('lineitem.parquet')"
```
```python
# pyarrow fallback
import pyarrow.parquet as pq
pf = pq.ParquetFile('lineitem.parquet')
print(pf.metadata.num_rows, pf.metadata.num_row_groups)
print(pf.schema_arrow)                       # Arrow types
rg0 = pf.metadata.row_group(0)
for i in range(rg0.num_columns):
    c = rg0.column(i)
    print(c.path_in_schema, c.encodings, c.compression)   # 'RLE_DICTIONARY' => dict-encoded
```
A column is **dictionary-encoded** if its `encodings` include `PLAIN_DICTIONARY` or
`RLE_DICTIONARY` (with `PLAIN`/`RLE` for the dictionary page + levels). Flag those.

---

## 9. Known gotchas (hard rules — a violation silently corrupts storage)

1. **Every column of a table MUST emit exactly `footer.num_rows` values.** Add an
   ingest-time assertion; if any column differs, print `column / expected / actual` and
   `exit(1)`. Column row-count mismatch = the query binaries fail with
   "column row count mismatch".
2. **Dictionary columns:** decode to logical values, one per row (§6.3). Never write the
   dictionary size or index count. Low-cardinality string columns (TPC-H `l_returnflag`,
   `l_linestatus`, `l_shipinstruct`, `l_shipmode`, `o_orderstatus`, `n_name`, `r_name`)
   are almost always dictionary-encoded — treat them as the highest-risk columns.
3. **Multiple row groups / chunks:** iterate ALL chunks (`ChunkedArray`) or loop
   `ReadNext` until null. Never `chunk(0)`-only or stop at the first batch.
4. **Map columns BY NAME**, never by position.
5. **DECIMAL scale**: real = unscaled × 10⁻ˢ; big-endian two's complement for byte-array
   carriers. Wrong scale → wrong SUM/AVG.
6. **DATE** = days since 1970-01-01, no shift.
7. **INT96** legacy timestamps need special decode — prefer Arrow's decoding.
8. **Compression** must be handled (Arrow does it) — a missing codec throws, don't swallow.
9. **Nested types** (LIST/MAP/struct) are NOT in TPC-H; a flat reader should reject them
   loudly rather than mis-read.
10. **Nulls** come from definition levels; use `IsNull(i)`. Don't assume non-null.
11. **Preserve row order** within and across chunks/row groups (append in read order) so
    sort orders and positional joins in the storage design stay valid.

---

## 10. The `parquet_spec.json` contract (Architect output → validator input)

The Architect step writes ONE JSON file describing how to decode every column of every
table. `tools/validate-parquet-spec.mjs` checks it before any C++ is generated.

```json
{
  "source_format": "parquet",
  "file_layout": "one_file_per_table",
  "tables": {
    "lineitem": {
      "parquet_file": "lineitem.parquet",
      "num_rows_source": "footer.num_rows",
      "num_row_groups": "multiple",
      "read_strategy": "record_batch_stream",
      "columns": {
        "l_returnflag": {
          "parquet_name": "l_returnflag",
          "physical_type": "BYTE_ARRAY",
          "logical_type": "STRING",
          "encoding": ["RLE_DICTIONARY", "PLAIN", "RLE"],
          "is_dictionary_encoded": true,
          "compression": "SNAPPY",
          "arrow_array_type": "DictionaryArray",
          "arrow_format": "u",
          "ab_append_fn": "ab_append_string",
          "nullable": false,
          "null_handling": "no_nulls",
          "decode_strategy": "Cast the DictionaryArray chunk to utf8 (or index dictionary()[indices()[i]]); ab_append_string(one char) per row across ALL chunks; emit exactly num_rows values.",
          "gotchas": ["low-cardinality string; dictionary-encoded; decode to logical values, not indices"]
        },
        "l_extendedprice": {
          "parquet_name": "l_extendedprice",
          "physical_type": "INT64",
          "logical_type": "DECIMAL",
          "decimal_precision": 15,
          "decimal_scale": 2,
          "encoding": ["PLAIN"],
          "is_dictionary_encoded": false,
          "compression": "SNAPPY",
          "arrow_array_type": "Decimal128Array",
          "arrow_format": "g",
          "ab_append_fn": "ab_append_float64",
          "nullable": false,
          "null_handling": "no_nulls",
          "decode_strategy": "read unscaled integer, value = unscaled / 10^scale (scale=2), ab_append_float64 per row across all chunks",
          "gotchas": ["DECIMAL scale=2 — dividing wrong scales every SUM"]
        },
        "l_shipdate": {
          "parquet_name": "l_shipdate",
          "physical_type": "INT32",
          "logical_type": "DATE",
          "encoding": ["PLAIN"],
          "is_dictionary_encoded": false,
          "compression": "SNAPPY",
          "arrow_array_type": "Date32Array",
          "arrow_format": "tdD",
          "ab_append_fn": "ab_append_int32",
          "nullable": false,
          "null_handling": "no_nulls",
          "decode_strategy": "Date32 value is days since 1970-01-01 — ab_append_int32 directly, no shift",
          "gotchas": []
        }
      }
    }
  }
}
```

### Required keys
- Top level: `source_format` (must be `"parquet"`), `file_layout`, `tables` (non-empty).
- Per table: `parquet_file`, `num_rows_source`, `columns` (non-empty).
- Per column: `parquet_name`, `physical_type`, `logical_type`, `encoding` (array),
  `is_dictionary_encoded` (bool), `arrow_array_type`, `arrow_format`, `ab_append_fn`,
  `nullable` (bool), `null_handling`, `decode_strategy` (non-empty).
- `arrow_format` is the Arrow C Data Interface format string of the OUTPUT column, and
  `ab_append_fn` is the matching `arrow_scaffold.h` builder call — together they drive the
  generated ingest, which appends decoded values and persists an Arrow IPC/Feather file
  via `gendb_arrow_storage.h`.
- When `logical_type == "DECIMAL"`: also `decimal_precision` and `decimal_scale`.
- When `is_dictionary_encoded == true`: `decode_strategy` MUST describe decoding to logical
  values (mention "decode"/"dictionary"/"cast"/"values") and MUST NOT describe writing
  indices/counts.

### Allowed value sets (kept in sync with the validator)
- `physical_type` ∈ {BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY,
  FIXED_LEN_BYTE_ARRAY}.
- `logical_type` ∈ {STRING, ENUM, DECIMAL, DATE, TIME, TIMESTAMP, INT, UINT, FLOAT16, JSON,
  BSON, UUID, NONE}.
- `encoding` entries ∈ {PLAIN, PLAIN_DICTIONARY, RLE_DICTIONARY, RLE, BIT_PACKED,
  DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT}.
- `null_handling` ∈ {no_nulls, validity_bitmap, definition_levels}.
- `arrow_format` ∈ {b, c, C, s, S, i, I, l, L, f, g, u, U, z, Z, tdD, `d:M,D` (decimal),
  `tsX:tz` (timestamp)} — the Arrow C Data Interface format string. NEVER prefix with `?`
  (nullable uses a validity bitmap, not a `?` prefix).
- `ab_append_fn` ∈ {ab_append_bool, ab_append_int8, ab_append_int16, ab_append_int32,
  ab_append_int64, ab_append_float32, ab_append_float64, ab_append_string} and MUST match
  `arrow_format`: b→bool; c/C→int8; s/S→int16; i/I/tdD→int32; l/L/tsX→int64; f→float32;
  g/d:→float64; u/U/z/Z→string.

### Physical ↔ logical consistency the validator enforces
- STRING/ENUM/JSON/BSON → BYTE_ARRAY.
- DATE → INT32. TIME → INT32 or INT64. TIMESTAMP → INT64 or INT96.
- DECIMAL → INT32, INT64, FIXED_LEN_BYTE_ARRAY, or BYTE_ARRAY (+ precision/scale present).
- UUID/FLOAT16 → FIXED_LEN_BYTE_ARRAY.
- `nullable == false` ⇒ `null_handling == "no_nulls"`; `nullable == true` ⇒
  `null_handling ∈ {validity_bitmap, definition_levels}`.

---

## 11. QA oracle (correctness check after ingest)
Ground truth comes from reading the SAME Parquet with DuckDB and diffing aggregate stats:
```bash
duckdb -json -c "SELECT count(*) AS row_count,
                        min(l_extendedprice) AS min, max(l_extendedprice) AS max,
                        sum(l_extendedprice) AS sum,
                        approx_count_distinct(l_returnflag) AS distinct_rf
                 FROM 'lineitem.parquet'"
```
Compare against the ingested GenDB storage (row_count exact; numeric min/max/sum within a
small relative tolerance; distinct counts for dict columns). A mismatch on a specific
column that persists across ingest retries indicates a **spec-level** error for that column
— revise `parquet_spec.json` for that column, don't just regenerate code.

## 12. High-risk dictionary-encoded columns (TPC-H & TPC-DS)
Everything above is benchmark-agnostic — the same type/encoding/Arrow rules apply to any
workload. The columns below are the ones most likely to trigger the row-count / value bug
because they are **low-cardinality strings that writers almost always dictionary-encode**.
Verify their `is_dictionary_encoded` flag and decode strategy explicitly (§6.3).

**TPC-H:** `l_returnflag`, `l_linestatus`, `l_shipinstruct`, `l_shipmode`, `o_orderstatus`,
`o_orderpriority`, `c_mktsegment`, `n_name`, `r_name`, `p_brand`, `p_mfgr`, `p_container`.

**TPC-DS** (dimension tables are full of low-cardinality strings and Y/N flags):
- `item`: `i_brand`, `i_category`, `i_class`, `i_color`, `i_size`, `i_units`, `i_container`, `i_manufact`
- `customer_demographics`: `cd_gender`, `cd_marital_status`, `cd_education_status`, `cd_credit_rating`
- `household_demographics`: `hd_buy_potential`
- `customer`: `c_preferred_cust_flag`, `c_salutation`, `c_birth_country`
- `customer_address`: `ca_state`, `ca_country`, `ca_city`, `ca_zip`, `ca_location_type`, `ca_gmt_offset`
- `store` / `web_site` / `call_center`: `s_state`, `s_country`, `s_city`, `s_zip`, company/name columns
- `date_dim`: `d_day_name`, `d_quarter_name`, `d_holiday`, `d_weekend`, `d_following_holiday`
- `time_dim`: `t_am_pm`, `t_shift`, `t_sub_shift`, `t_meal_time`
- `promotion`: `p_channel_dmail`, `p_channel_email`, `p_channel_catalog`, … (Y/N flags)
- `ship_mode`, `reason`, `income_band`, `warehouse`, `web_page`, `catalog_page`: various codes/flags

## 13. TPC-DS specifics (differences from TPC-H that affect the reader)
1. **Many columns are NULLABLE.** Unlike TPC-H (mostly NOT NULL), TPC-DS fact tables
   (`store_sales`, `catalog_sales`, `web_sales`, and their returns) have numerous nullable
   measure and foreign-key columns, and dimension tables have nullable attributes. So
   `nullable: true` with `null_handling: "validity_bitmap"` (Arrow) is the COMMON case, not
   the exception. Never blanket-set `null_handling: "no_nulls"` — check each column's
   `null_count` in the Parquet stats / Arrow `null_count()`. Skipping nulls or treating them
   as 0/"" corrupts aggregates and join cardinalities.
2. **Surrogate keys are integers.** All `*_sk` columns (e.g. `ss_item_sk`, `d_date_sk`,
   `ss_sold_date_sk`) are `INT32`/`INT64` identifiers (logical `INT`/`NONE`), often the
   join keys. `ss_sold_date_sk` etc. can be **NULL** in the fact tables — handle it.
3. **Dates:** `date_dim.d_date` is a `DATE` (`INT32` days since epoch, §4). Most date
   filtering, however, joins facts to `date_dim` via the integer `*_date_sk`, then filters
   on `d_year`/`d_moy`/`d_qoy` (plain integers). `time_dim.t_time` is an integer number of
   seconds since midnight (logical `NONE` on `INT32`), not a Parquet `TIME`.
4. **Decimals everywhere.** Money/measure columns (e.g. `ss_ext_sales_price`,
   `ss_net_profit`, `cs_wholesale_cost`) are `DECIMAL(7,2)`-style — get `decimal_scale`
   right for every one or every SUM/AVG is wrong.
5. **Scale:** TPC-DS has 24 tables and up to ~429 columns total; produce a spec entry for
   **every** column of every table you ingest, and rely on the validator's schema-coverage
   check to catch omissions.
