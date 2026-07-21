/*
 * gendb_arrow_storage.h — Arrow IPC/Feather persistence + zero-copy read for GenDB.
 *
 * Full Arrow-native storage: the generated ingest builds an in-memory Arrow struct
 * array via arrow_scaffold.h (Jailbreak's C Data Interface builder), then persists it
 * here as an Arrow IPC / Feather V2 file (one file per table). Query binaries mmap that
 * file and get zero-copy, SIMD-aligned Arrow arrays back — and can hand them to
 * GPU/cuDF via the C Data Interface with zero copy.
 *
 * Requires linking Arrow C++:  pkg-config --cflags --libs arrow
 * (GenDB already links this for reading Parquet.)
 *
 * NOTE: this header uses standard Arrow C++ APIs but must be compile-verified on a
 * machine with arrow-cpp installed (it is not compiled in the CI sandbox).
 *
 * Ingest side:
 *   ArrowSchema sch; ArrowArray arr;
 *   ab_finalize(builder, &sch, &arr, err, sizeof(err));   // from arrow_scaffold.h
 *   gendb::WriteFeather(&sch, &arr, "<gendb_dir>/lineitem.feather");
 *
 * Query side:
 *   auto table = gendb::OpenTableMmap("<gendb_dir>/lineitem.feather").ValueOrDie();
 *   auto col   = table->GetColumnByName("l_extendedprice");   // zero-copy ChunkedArray
 *   // iterate col->chunk(c) across ALL chunks (see arrow decode rules).
 */
#pragma once
#ifndef GENDB_ARROW_STORAGE_H
#define GENDB_ARROW_STORAGE_H

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/c/bridge.h>          // ImportRecordBatch / ExportArray (C Data Interface)
#include <arrow/io/file.h>          // FileOutputStream, MemoryMappedFile
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>

// arrow_scaffold.h and <arrow/c/abi.h> share the ARROW_C_DATA_INTERFACE include guard,
// so the ArrowSchema/ArrowArray structs are the same ABI type regardless of include order.

namespace gendb {

/**
 * Persist a finalized Arrow struct array (from ab_finalize) to an Arrow IPC/Feather
 * file. Consumes the C Data Interface array+schema (ImportRecordBatch releases them).
 * One file per table. Returns Status::OK on success.
 */
inline arrow::Status WriteFeather(struct ArrowSchema* schema,
                                  struct ArrowArray*  array,
                                  const std::string&  path) {
  ARROW_ASSIGN_OR_RAISE(auto batch, arrow::ImportRecordBatch(array, schema));
  ARROW_ASSIGN_OR_RAISE(auto out,   arrow::io::FileOutputStream::Open(path));
  // Default IPC options; storage-design encodings (e.g. dictionary) live in the
  // Arrow arrays themselves. LZ4/ZSTD IPC compression can be enabled via options.
  ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeFileWriter(out, batch->schema()));
  ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
  ARROW_RETURN_NOT_OK(writer->Close());
  return out->Close();
}

/** Convenience wrapper for generated code: returns 0 on success, -1 on error (prints). */
inline int gendb_write_feather(struct ArrowSchema* schema,
                               struct ArrowArray*  array,
                               const char*         path) {
  auto st = WriteFeather(schema, array, path);
  if (!st.ok()) {
    std::fprintf(stderr, "[gendb] WriteFeather(%s) failed: %s\n", path, st.ToString().c_str());
    return -1;
  }
  return 0;
}

/**
 * Open a Feather/IPC table with memory mapping (zero-copy). The returned Table's
 * buffers point into the mmap'd file — keep no assumptions about lifetime beyond the
 * Table. Reads all record batches (there is normally one per ingested table).
 */
inline arrow::Result<std::shared_ptr<arrow::Table>> OpenTableMmap(const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto mmap,
      arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ));
  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchFileReader::Open(mmap));
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  batches.reserve(reader->num_record_batches());
  for (int i = 0; i < reader->num_record_batches(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto b, reader->ReadRecordBatch(i));
    batches.push_back(std::move(b));
  }
  return arrow::Table::FromRecordBatches(std::move(batches));
}

/** Fetch a column by name as a zero-copy ChunkedArray (nullptr if absent). */
inline std::shared_ptr<arrow::ChunkedArray>
Column(const std::shared_ptr<arrow::Table>& table, const std::string& name) {
  return table ? table->GetColumnByName(name) : nullptr;
}

/**
 * Export a column (concatenated to a single Array) through the Arrow C Data Interface,
 * for zero-copy handoff to GPU frameworks (cuDF / Spark RAPIDS) or other consumers.
 * The caller owns out_schema/out_array and must call their release callbacks.
 */
inline arrow::Status ExportColumnCData(const std::shared_ptr<arrow::ChunkedArray>& col,
                                       struct ArrowSchema* out_schema,
                                       struct ArrowArray*  out_array) {
  ARROW_ASSIGN_OR_RAISE(auto combined, col->View(col->type()));  // no-op view; keep type
  ARROW_ASSIGN_OR_RAISE(auto one, arrow::Concatenate(combined->chunks()));
  ARROW_RETURN_NOT_OK(arrow::ExportType(*one->type(), out_schema));
  return arrow::ExportArray(*one, out_array);
}

}  // namespace gendb

#endif  // GENDB_ARROW_STORAGE_H
