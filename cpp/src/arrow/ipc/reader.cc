// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/ipc/reader.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>  // IWYU pragma: export

#include "arrow/array.h"
#include "arrow/buffer.h"
#include "arrow/io/interfaces.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/File_generated.h"  // IWYU pragma: export
#include "arrow/ipc/Message_generated.h"
#include "arrow/ipc/Schema_generated.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/message.h"
#include "arrow/ipc/metadata-internal.h"
#include "arrow/record_batch.h"
#include "arrow/sparse_tensor.h"
#include "arrow/status.h"
#include "arrow/tensor.h"
#include "arrow/type.h"
#include "arrow/util/logging.h"
#include "arrow/util/bit-util.h"
#include "arrow/visitor_inline.h"

using arrow::internal::checked_pointer_cast;

namespace arrow {

namespace flatbuf = org::apache::arrow::flatbuf;

namespace ipc {

using internal::FileBlock;
using internal::kArrowMagicBytes;

namespace {

Status InvalidMessageType(Message::Type expected, Message::Type actual) {
  return Status::IOError("Expected IPC message of type ", FormatMessageType(expected),
                         " got ", FormatMessageType(actual));
}

#define CHECK_MESSAGE_TYPE(expected, actual)           \
  do {                                                 \
    if ((actual) != (expected)) {                      \
      return InvalidMessageType((expected), (actual)); \
    }                                                  \
  } while (0)

#define CHECK_HAS_BODY(message)                                       \
  do {                                                                \
    if ((message).body() == nullptr) {                                \
      return Status::IOError("Expected body in IPC message of type ", \
                             FormatMessageType((message).type()));    \
    }                                                                 \
  } while (0)

#define CHECK_HAS_NO_BODY(message)                                      \
  do {                                                                  \
    if ((message).body_length() != 0) {                                 \
      return Status::IOError("Unexpected body in IPC message of type ", \
                             FormatMessageType((message).type()));      \
    }                                                                   \
  } while (0)

}  // namespace

// ----------------------------------------------------------------------
// Record batch read path

/// Accessor class for flatbuffers metadata
class IpcComponentSource {
 public:
  IpcComponentSource(const flatbuf::RecordBatch* metadata, io::RandomAccessFile* file)
      : metadata_(metadata), file_(file) {}

  Status GetBuffer(int buffer_index, std::shared_ptr<Buffer>* out, int64_t offset = 0, int64_t length = std::numeric_limits<int64_t>::max()) {
    const flatbuf::Buffer* buffer = metadata_->buffers()->Get(buffer_index);
    auto ret_offset = buffer->offset() + offset;
    auto ret_length = std::min(length, buffer->length() - offset);

    if (ret_length == 0) {
      *out = nullptr;
      return Status::OK();
    } else {
      DCHECK(BitUtil::IsMultipleOf8(buffer->offset()))
          << "Buffer " << buffer_index
          << " did not start on 8-byte aligned offset: " << buffer->offset();
      return file_->ReadAt(ret_offset, ret_length, out);
    }
  }

  Status GetFieldMetadata(int field_index, ArrayData* out) {
    auto nodes = metadata_->nodes();
    // pop off a field
    if (field_index >= static_cast<int>(nodes->size())) {
      return Status::Invalid("Ran out of field metadata, likely malformed");
    }
    const flatbuf::FieldNode* node = nodes->Get(field_index);

    out->length = node->length();
    out->null_count = node->null_count();
    out->offset = 0;
    return Status::OK();
  }

 private:
  const flatbuf::RecordBatch* metadata_;
  io::RandomAccessFile* file_;
};

/// Bookkeeping struct for loading array objects from their constituent pieces of raw data
///
/// The field_index and buffer_index are incremented in the ArrayLoader
/// based on how much of the batch is "consumed" (through nested data
/// reconstruction, for example)
struct ArrayLoaderContext {
  IpcComponentSource* source;
  int buffer_index;
  int field_index;
  int max_recursion_depth;
};

static Status LoadArray(const std::shared_ptr<DataType>& type,
                        ArrayLoaderContext* context, ArrayData* out, int64_t offset = 0, int64_t length = std::numeric_limits<int64_t>::max());

class ArrayLoader {
 public:
  ArrayLoader(const std::shared_ptr<DataType>& type, ArrayData* out,
              ArrayLoaderContext* context, int64_t offset = 0, int64_t length = std::numeric_limits<int64_t>::max())
      : type_(type), context_(context), out_(out), _offset(offset), _length(length) {
  }

  Status Load() {
    if (context_->max_recursion_depth <= 0) {
      return Status::Invalid("Max recursion depth reached");
    }

    out_->type = type_;

    RETURN_NOT_OK(VisitTypeInline(*type_, this));
    return Status::OK();
  }

  Status GetBuffer(int buffer_index, std::shared_ptr<Buffer>* out, int64_t offset, int64_t length) {
    return context_->source->GetBuffer(buffer_index, out, offset, length);
  }

  Status LoadCommon() {
    // This only contains the length and null count, which we need to figure
    // out what to do with the buffers. For example, if null_count == 0, then
    // we can skip that buffer without reading from shared memory
    RETURN_NOT_OK(context_->source->GetFieldMetadata(context_->field_index++, out_));

    // extract null_bitmap which is common to all arrays
    if (out_->null_count == 0) {
      out_->buffers[0] = nullptr;
    } else {
      int64_t null_offset = _offset / 8;
      int64_t null_length = _length == std::numeric_limits<int64_t>::max() ? _length : (_length + 7)/8;
      if (_offset % 8 != 0) {
          null_length += 1;
      }
      RETURN_NOT_OK(GetBuffer(context_->buffer_index, &out_->buffers[0], null_offset, null_length));
      if (out_->buffers[0] && _offset % 8 != 0) {
          std::shared_ptr<Buffer> null_buffer;
          RETURN_NOT_OK(out_->buffers[0]->Copy(0, out_->buffers[0]->size(), &null_buffer));
          arrow::internal::CopyBitmap(out_->buffers[0]->data(), _offset % 8, out_->buffers[0]->size() * 8 - _offset % 8, null_buffer->mutable_data(), 0);
          out_->buffers[0] = null_buffer;
      }
    }
    if (_offset != 0 || _length != std::numeric_limits<int64_t>::max()) {
        if (_length != std::numeric_limits<int64_t>::max()) {
            out_->length = std::min(_length, out_->length - _offset);
            if (out_->null_count != 0) {
                out_->null_count = kUnknownNullCount;
            }
        }
        else {
            out_->length -= _offset;
        }
    }
    context_->buffer_index++;
    return Status::OK();
  }

  template <typename TYPE>
  Status LoadPrimitive() {
    out_->buffers.resize(2);

    RETURN_NOT_OK(LoadCommon());
    if (out_->length > 0) {
      TYPE t;
      if (t.bit_width() == 1) {
        int64_t null_offset = _offset / 8;
        int64_t null_length = _length == std::numeric_limits<int64_t>::max() ? _length : (_length + 7)/8;
        if (_offset % 8 != 0) {
            null_length += 1;
        }
        RETURN_NOT_OK(GetBuffer(context_->buffer_index++, &out_->buffers[1], null_offset, null_length));
        auto buffer = out_->buffers[1];
        if (buffer && _offset % 8 != 0) {
            std::shared_ptr<Buffer> null_buffer;
            RETURN_NOT_OK(buffer->Copy(0, buffer->size(), &null_buffer));
            arrow::internal::CopyBitmap(buffer->data(), _offset % 8, buffer->size() * 8 - _offset % 8, null_buffer->mutable_data(), 0);
            out_->buffers[1] = null_buffer;
          }
        }
      else {
         auto offset = get_offset(t.bit_width());
         RETURN_NOT_OK(GetBuffer(context_->buffer_index++, &out_->buffers[1], offset.first, offset.second));
      }
    } else {
      context_->buffer_index++;
      out_->buffers[1].reset(new Buffer(nullptr, 0));
    }
    return Status::OK();
  }

  template <typename TYPE>
  Status LoadBinary() {
    out_->buffers.resize(3);

    RETURN_NOT_OK(LoadCommon());
    auto length = _length;
    if (length != std::numeric_limits<int64_t>::max()) {
        length += 1;
        length *= sizeof(int32_t);
    }

    RETURN_NOT_OK(GetBuffer(context_->buffer_index++, &out_->buffers[1], _offset * sizeof(int32_t), length));
    int64_t start_offset = 0, data_length = std::numeric_limits<int64_t>::max();
    if (_offset != 0) {
        auto offsets_buffer = out_->buffers[1];
        if (not offsets_buffer->is_mutable()) {
            RETURN_NOT_OK(offsets_buffer->Copy(0, offsets_buffer->size(), &offsets_buffer));
        }
        auto offsets = reinterpret_cast<int32_t*>(offsets_buffer->mutable_data());
        start_offset = *offsets;
        if (offsets_buffer->size() == length) {
            data_length = offsets[offsets_buffer->size() / sizeof(int32_t) - 1] - start_offset;
        }
        std::transform(offsets, offsets + offsets_buffer->size() / sizeof(int32_t), offsets, [start_offset](int32_t o) { return o - start_offset; });
        out_->buffers[1] = offsets_buffer;
    }
    else if (_length != std::numeric_limits<int64_t>::max()) {
        if (out_->buffers[1]->size() == length) {
            auto offsets = reinterpret_cast<const int32_t *>(out_->buffers[1]->data());
            data_length = offsets[out_->buffers[1]->size() / sizeof(int32_t) - 1];
        }
    }
    return GetBuffer(context_->buffer_index++, &out_->buffers[2], start_offset, data_length);
  }

  Status LoadChild(const Field& field, ArrayData* out) {
    ArrayLoader loader(field.type(), out, context_);
    --context_->max_recursion_depth;
    RETURN_NOT_OK(loader.Load());
    ++context_->max_recursion_depth;
    return Status::OK();
  }

  Status LoadChildren(std::vector<std::shared_ptr<Field>> child_fields) {
    // TODO only load the children you need if _offset or _length are set.
    out_->child_data.reserve(static_cast<int>(child_fields.size()));

    for (const auto& child_field : child_fields) {
      auto field_array = std::make_shared<ArrayData>();
      RETURN_NOT_OK(LoadChild(*child_field.get(), field_array.get()));
      out_->child_data.emplace_back(field_array);
    }
    return Status::OK();
  }

  Status Visit(const NullType& type) {
    out_->buffers.resize(1);
    RETURN_NOT_OK(LoadCommon());
    auto offset = get_offset(1);
    RETURN_NOT_OK(GetBuffer(context_->buffer_index++, &out_->buffers[0], offset.first, offset.second));
    return Status::OK();
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<FixedWidthType, T>::value &&
                              !std::is_base_of<FixedSizeBinaryType, T>::value &&
                              !std::is_base_of<DictionaryType, T>::value,
                          Status>::type
  Visit(const T& type) {
    return LoadPrimitive<T>();
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<BinaryType, T>::value, Status>::type Visit(
      const T& type) {
    return LoadBinary<T>();
  }

  Status Visit(const FixedSizeBinaryType& type) {
    out_->buffers.resize(2);
    RETURN_NOT_OK(LoadCommon());
    auto t = reinterpret_cast<FixedSizeBinaryType*>(type_.get());
    auto offset = get_offset(t->bit_width());

    return GetBuffer(context_->buffer_index++, &out_->buffers[1], offset.first, offset.second);
  }

  Status Visit(const ListType& type) {
    out_->buffers.resize(2);

    RETURN_NOT_OK(LoadCommon());
    auto offset = get_offset(sizeof(int32_t)*8);
    if (offset.second != std::numeric_limits<int64_t>::max()) {
        offset.second += sizeof(int32_t);
    }
    RETURN_NOT_OK(GetBuffer(context_->buffer_index++, &out_->buffers[1], offset.first, offset.second));

    const int num_children = type.num_children();
    if (num_children != 1) {
      return Status::Invalid("Wrong number of children: ", num_children);
    }

    return LoadChildren(type.children());
  }

  Status Visit(const StructType& type) {
    if (_offset != 0 || _length != std::numeric_limits<int64_t>::max()) {
      return Status::NotImplemented("Can't read slices for Struct type arrays");
    }
    out_->buffers.resize(1);
    RETURN_NOT_OK(LoadCommon());
    return LoadChildren(type.children());
  }

  Status Visit(const UnionType& type) {
    out_->buffers.resize(3);

    RETURN_NOT_OK(LoadCommon());
    if (out_->length > 0) {
      auto offset = get_offset(sizeof(uint8_t)*8);
      RETURN_NOT_OK(GetBuffer(context_->buffer_index, &out_->buffers[1], offset.first, offset.second));
      if (type.mode() == UnionMode::DENSE) {
        offset = get_offset(sizeof(int32_t)*8);
        RETURN_NOT_OK(GetBuffer(context_->buffer_index + 1, &out_->buffers[2], offset.first, offset.second));
      }
    }
    context_->buffer_index += type.mode() == UnionMode::DENSE ? 2 : 1;
    return LoadChildren(type.children());
  }

  Status Visit(const DictionaryType& type) {
    RETURN_NOT_OK(LoadArray(type.index_type(), context_, out_, _offset, _length));
    out_->type = type_;
    return Status::OK();
  }

  Status Visit(const ExtensionType& type) {
      if (_offset != 0 || _length != std::numeric_limits<int64_t>::max()) {
          return arrow::Status::Invalid("Can't load ExtensionType data with offset or length");
      }
    RETURN_NOT_OK(LoadArray(type.storage_type(), context_, out_, 0, std::numeric_limits<int64_t>::max()));
    out_->type = type_;
    return Status::OK();
  }

 private:
  std::pair<int64_t, int64_t> get_offset(int32_t bit_size) const {
      int64_t l;
      if (_length == std::numeric_limits<int64_t>::max()) {
          l = _length;
      }
      else {
          l = (_length * bit_size+7)/8;
      }
      return std::make_pair(_offset * bit_size / 8, l);
  }
  const std::shared_ptr<DataType> type_;
  ArrayLoaderContext* context_;

  // Used in visitor pattern
  ArrayData* out_;
  int64_t _offset;
  int64_t _length;
};

static Status LoadArray(const std::shared_ptr<DataType>& type,
                        ArrayLoaderContext* context, ArrayData* out, int64_t offset, int64_t length) {
  ArrayLoader loader(type, out, context, offset, length);
  return loader.Load();
}

Status ReadRecordBatch(const Buffer& metadata, const std::shared_ptr<Schema>& schema,
                       io::RandomAccessFile* file, std::shared_ptr<RecordBatch>* out, int64_t offset, int64_t length) {
  return ReadRecordBatch(metadata, schema, kMaxNestingDepth, file, out, offset, length);
}

Status ReadRecordBatch(const Message& message, const std::shared_ptr<Schema>& schema,
                       std::shared_ptr<RecordBatch>* out) {
  CHECK_MESSAGE_TYPE(message.type(), Message::RECORD_BATCH);
  CHECK_HAS_BODY(message);
  io::BufferReader reader(message.body());
  return ReadRecordBatch(*message.metadata(), schema, kMaxNestingDepth, &reader, out);
}

// ----------------------------------------------------------------------
// Array loading

static Status LoadRecordBatchFromSource(const std::shared_ptr<Schema>& schema,
                                        int64_t num_rows, int max_recursion_depth,
                                        IpcComponentSource* source,
                                        std::shared_ptr<RecordBatch>* out,
                                        int64_t offset = 0, int64_t length = std::numeric_limits<int64_t>::max()) {
  ArrayLoaderContext context;
  context.source = source;
  context.field_index = 0;
  context.buffer_index = 0;
  context.max_recursion_depth = max_recursion_depth;

  std::vector<std::shared_ptr<ArrayData>> arrays(schema->num_fields());
  auto expected_num_rows = std::min(num_rows, length);
  if (offset != 0) {
      expected_num_rows = std::min(expected_num_rows, num_rows - offset);
  }

  for (int i = 0; i < schema->num_fields(); ++i) {
    auto arr = std::make_shared<ArrayData>();
    RETURN_NOT_OK(LoadArray(schema->field(i)->type(), &context, arr.get(), offset, length));
    DCHECK_EQ(expected_num_rows, arr->length) << "Array length did not match record batch length";
    arrays[i] = std::move(arr);
  }

  *out = RecordBatch::Make(schema, std::min(num_rows, length), std::move(arrays));
  return Status::OK();
}

static inline Status ReadRecordBatch(const flatbuf::RecordBatch* metadata,
                                     const std::shared_ptr<Schema>& schema,
                                     int max_recursion_depth, io::RandomAccessFile* file,
                                     std::shared_ptr<RecordBatch>* out,
                                     int64_t offset = 0, int64_t length = std::numeric_limits<int64_t>::max()) {
  IpcComponentSource source(metadata, file);
  return LoadRecordBatchFromSource(schema, metadata->length(), max_recursion_depth,
                                   &source, out, offset, length);
}

Status ReadRecordBatch(const Buffer& metadata, const std::shared_ptr<Schema>& schema,
                       int max_recursion_depth, io::RandomAccessFile* file,
                       std::shared_ptr<RecordBatch>* out,
                       int64_t offset, int64_t length) {
  auto message = flatbuf::GetMessage(metadata.data());
  if (message->header_type() != flatbuf::MessageHeader_RecordBatch) {
    DCHECK_EQ(message->header_type(), flatbuf::MessageHeader_RecordBatch);
  }
  if (message->header() == nullptr) {
    return Status::IOError("Header-pointer of flatbuffer-encoded Message is null.");
  }
  auto batch = reinterpret_cast<const flatbuf::RecordBatch*>(message->header());
  return ReadRecordBatch(batch, schema, max_recursion_depth, file, out, offset, length);
}

Status ReadDictionary(const Buffer& metadata, const DictionaryTypeMap& dictionary_types,
                      io::RandomAccessFile* file, int64_t* dictionary_id,
                      std::shared_ptr<Array>* out) {
  auto message = flatbuf::GetMessage(metadata.data());
  auto dictionary_batch =
      reinterpret_cast<const flatbuf::DictionaryBatch*>(message->header());

  int64_t id = *dictionary_id = dictionary_batch->id();
  auto it = dictionary_types.find(id);
  if (it == dictionary_types.end()) {
    return Status::KeyError("Do not have type metadata for dictionary with id: ", id);
  }

  std::vector<std::shared_ptr<Field>> fields = {it->second};

  // We need a schema for the record batch
  auto dummy_schema = std::make_shared<Schema>(fields);

  // The dictionary is embedded in a record batch with a single column
  std::shared_ptr<RecordBatch> batch;
  auto batch_meta =
      reinterpret_cast<const flatbuf::RecordBatch*>(dictionary_batch->data());
  RETURN_NOT_OK(
      ReadRecordBatch(batch_meta, dummy_schema, kMaxNestingDepth, file, &batch));
  if (batch->num_columns() != 1) {
    return Status::Invalid("Dictionary record batch must only contain one field");
  }

  *out = batch->column(0);
  return Status::OK();
}

static Status ReadMessageAndValidate(MessageReader* reader, Message::Type expected_type,
                                     bool allow_null, std::unique_ptr<Message>* message) {
  RETURN_NOT_OK(reader->ReadNextMessage(message));

  if (!(*message) && !allow_null) {
    return Status::Invalid("Expected ", FormatMessageType(expected_type),
                           " message in stream, was null or length 0");
  }

  if ((*message) == nullptr) {
    // End of stream?
    return Status::OK();
  }

  CHECK_MESSAGE_TYPE((*message)->type(), expected_type);
  return Status::OK();
}

// ----------------------------------------------------------------------
// RecordBatchStreamReader implementation

static inline FileBlock FileBlockFromFlatbuffer(const flatbuf::Block* block) {
  return FileBlock{block->offset(), block->metaDataLength(), block->bodyLength()};
}

class RecordBatchStreamReader::RecordBatchStreamReaderImpl {
 public:
  RecordBatchStreamReaderImpl() {}
  ~RecordBatchStreamReaderImpl() {}

  Status Open(std::unique_ptr<MessageReader> message_reader) {
    message_reader_ = std::move(message_reader);
    return ReadSchema();
  }

  Status ReadNextDictionary() {
    std::unique_ptr<Message> message;
    RETURN_NOT_OK(ReadMessageAndValidate(message_reader_.get(), Message::DICTIONARY_BATCH,
                                         false, &message));
    if (message == nullptr) {
      // End of stream
      return Status::IOError(
          "End of IPC stream when attempting to read dictionary batch");
    }

    CHECK_HAS_BODY(*message);
    io::BufferReader reader(message->body());

    std::shared_ptr<Array> dictionary;
    int64_t id;
    RETURN_NOT_OK(ReadDictionary(*message->metadata(), dictionary_types_, &reader, &id,
                                 &dictionary));
    return dictionary_memo_.AddDictionary(id, dictionary);
  }

  Status ReadSchema() {
    std::unique_ptr<Message> message;
    RETURN_NOT_OK(
        ReadMessageAndValidate(message_reader_.get(), Message::SCHEMA, false, &message));
    if (message == nullptr) {
      // End of stream
      return Status::IOError("End of IPC stream when attempting to read schema");
    }

    CHECK_HAS_NO_BODY(*message);
    if (message->header() == nullptr) {
      return Status::IOError("Header-pointer of flatbuffer-encoded Message is null.");
    }
    RETURN_NOT_OK(internal::GetDictionaryTypes(message->header(), &dictionary_types_));

    // TODO(wesm): In future, we may want to reconcile the ids in the stream with
    // those found in the schema
    int num_dictionaries = static_cast<int>(dictionary_types_.size());
    for (int i = 0; i < num_dictionaries; ++i) {
      RETURN_NOT_OK(ReadNextDictionary());
    }

    return internal::GetSchema(message->header(), dictionary_memo_, &schema_);
  }

  Status ReadNext(std::shared_ptr<RecordBatch>* batch) {
    std::unique_ptr<Message> message;
    RETURN_NOT_OK(ReadMessageAndValidate(message_reader_.get(), Message::RECORD_BATCH,
                                         true, &message));
    if (message == nullptr) {
      // End of stream
      *batch = nullptr;
      return Status::OK();
    }

    CHECK_HAS_BODY(*message);
    io::BufferReader reader(message->body());
    return ReadRecordBatch(*message->metadata(), schema_, &reader, batch);
  }

  std::shared_ptr<Schema> schema() const { return schema_; }

 private:
  std::unique_ptr<MessageReader> message_reader_;

  // dictionary_id -> type
  DictionaryTypeMap dictionary_types_;
  DictionaryMemo dictionary_memo_;
  std::shared_ptr<Schema> schema_;
};

RecordBatchStreamReader::RecordBatchStreamReader() {
  impl_.reset(new RecordBatchStreamReaderImpl());
}

RecordBatchStreamReader::~RecordBatchStreamReader() {}

Status RecordBatchStreamReader::Open(std::unique_ptr<MessageReader> message_reader,
                                     std::shared_ptr<RecordBatchReader>* reader) {
  // Private ctor
  auto result = std::shared_ptr<RecordBatchStreamReader>(new RecordBatchStreamReader());
  RETURN_NOT_OK(result->impl_->Open(std::move(message_reader)));
  *reader = result;
  return Status::OK();
}

Status RecordBatchStreamReader::Open(std::unique_ptr<MessageReader> message_reader,
                                     std::unique_ptr<RecordBatchReader>* reader) {
  // Private ctor
  auto result = std::unique_ptr<RecordBatchStreamReader>(new RecordBatchStreamReader());
  RETURN_NOT_OK(result->impl_->Open(std::move(message_reader)));
  *reader = std::move(result);
  return Status::OK();
}

Status RecordBatchStreamReader::Open(io::InputStream* stream,
                                     std::shared_ptr<RecordBatchReader>* out) {
  return Open(MessageReader::Open(stream), out);
}

Status RecordBatchStreamReader::Open(const std::shared_ptr<io::InputStream>& stream,
                                     std::shared_ptr<RecordBatchReader>* out) {
  return Open(MessageReader::Open(stream), out);
}

std::shared_ptr<Schema> RecordBatchStreamReader::schema() const {
  return impl_->schema();
}

Status RecordBatchStreamReader::ReadNext(std::shared_ptr<RecordBatch>* batch) {
  return impl_->ReadNext(batch);
}

// ----------------------------------------------------------------------
// Reader implementation

class RecordBatchFileReader::RecordBatchFileReaderImpl {
 public:
  RecordBatchFileReaderImpl() : file_(NULLPTR), footer_offset_(0), footer_(NULLPTR) {
    dictionary_memo_ = std::make_shared<DictionaryMemo>();
  }

  Status ReadFooter() {
    int magic_size = static_cast<int>(strlen(kArrowMagicBytes));

    if (footer_offset_ <= magic_size * 2 + 4) {
      return Status::Invalid("File is too small: ", footer_offset_);
    }

    std::shared_ptr<Buffer> buffer;
    int file_end_size = static_cast<int>(magic_size + sizeof(int32_t));
    RETURN_NOT_OK(file_->ReadAt(footer_offset_ - file_end_size, file_end_size, &buffer));

    const int64_t expected_footer_size = magic_size + sizeof(int32_t);
    if (buffer->size() < expected_footer_size) {
      return Status::Invalid("Unable to read ", expected_footer_size, "from end of file");
    }

    if (memcmp(buffer->data() + sizeof(int32_t), kArrowMagicBytes, magic_size)) {
      return Status::Invalid("Not an Arrow file");
    }

    int32_t footer_length = *reinterpret_cast<const int32_t*>(buffer->data());

    if (footer_length <= 0 || footer_length + magic_size * 2 + 4 > footer_offset_) {
      return Status::Invalid("File is smaller than indicated metadata size");
    }

    // Now read the footer
    RETURN_NOT_OK(file_->ReadAt(footer_offset_ - footer_length - file_end_size,
                                footer_length, &footer_buffer_));

    // TODO(wesm): Verify the footer
    footer_ = flatbuf::GetFooter(footer_buffer_->data());

    return Status::OK();
  }

  int num_dictionaries() const { return footer_->dictionaries()->size(); }

  int num_record_batches() const { return footer_->recordBatches()->size(); }

  arrow::Status record_batch_num_rows(int64_t batch_index, int64_t* ret) {
      DCHECK_GE(batch_index, 0);
      DCHECK_LT(batch_index, num_record_batches());
      FileBlock block = record_batch(batch_index);

      DCHECK(BitUtil::IsMultipleOf8(block.offset));
      DCHECK(BitUtil::IsMultipleOf8(block.metadata_length));
      DCHECK(BitUtil::IsMultipleOf8(block.body_length));

      std::unique_ptr<Message> message;
      RETURN_NOT_OK(ReadMessage(block.offset, block.metadata_length, file_, &message));

      // TODO(wesm): this breaks integration tests, see ARROW-3256
      // DCHECK_EQ(message->body_length(), block.body_length);

        // io::BufferReader reader(message->body());
      auto& metadata = *message->metadata();
      //::arrow::ipc::ReadRecordBatch(, schema_, &reader, batch, offset, length);
      //ReadRecordBatch(metadata, schema, kMaxNestingDepth, file, out, offset, length);
      auto metadata_message = flatbuf::GetMessage(metadata.data());
      if (metadata_message->header_type() != flatbuf::MessageHeader_RecordBatch) {
          DCHECK_EQ(metadata_message->header_type(), flatbuf::MessageHeader_RecordBatch);
      }
      if (metadata_message->header() == nullptr) {
          return Status::IOError("Header-pointer of flatbuffer-encoded Message is null.");
      }
      auto batch = reinterpret_cast<const flatbuf::RecordBatch*>(metadata_message->header());
      *ret = batch->length();
      return arrow::Status::OK();

  }

  MetadataVersion version() const {
    return internal::GetMetadataVersion(footer_->version());
  }

  FileBlock record_batch(int i) const {
    return FileBlockFromFlatbuffer(footer_->recordBatches()->Get(i));
  }

  FileBlock dictionary(int i) const {
    return FileBlockFromFlatbuffer(footer_->dictionaries()->Get(i));
  }

  Status ReadRecordBatch(int i, std::shared_ptr<RecordBatch>* batch, int64_t offset = 0, int64_t length = std::numeric_limits<int64_t>::max()) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, num_record_batches());
    FileBlock block = record_batch(i);

    DCHECK(BitUtil::IsMultipleOf8(block.offset));
    DCHECK(BitUtil::IsMultipleOf8(block.metadata_length));
    DCHECK(BitUtil::IsMultipleOf8(block.body_length));

    std::unique_ptr<Message> message;
    RETURN_NOT_OK(ReadMessage(block.offset, block.metadata_length, file_, &message));

    // TODO(wesm): this breaks integration tests, see ARROW-3256
    // DCHECK_EQ(message->body_length(), block.body_length);

    // TODO adorr - create a lazy reader.
    io::BufferReader reader(message->body());
    return ::arrow::ipc::ReadRecordBatch(*message->metadata(), schema_, &reader, batch, offset, length);
  }

  Status ReadRecordBatchAsBatches(int i, std::shared_ptr<IRecordBatchFileReader> *reader,
                                                       int32_t max_batch_size,
                                                       std::shared_ptr<RecordBatchFileReaderImpl> impl);

  Status ReadSchema() {
    RETURN_NOT_OK(internal::GetDictionaryTypes(footer_->schema(), &dictionary_fields_));

    // Read all the dictionaries
    for (int i = 0; i < num_dictionaries(); ++i) {
      FileBlock block = dictionary(i);

      DCHECK(BitUtil::IsMultipleOf8(block.offset));
      DCHECK(BitUtil::IsMultipleOf8(block.metadata_length));
      DCHECK(BitUtil::IsMultipleOf8(block.body_length));

      std::unique_ptr<Message> message;
      RETURN_NOT_OK(ReadMessage(block.offset, block.metadata_length, file_, &message));

      // TODO(wesm): this breaks integration tests, see ARROW-3256
      // DCHECK_EQ(message->body_length(), block.body_length);

      io::BufferReader reader(message->body());

      std::shared_ptr<Array> dictionary;
      int64_t dictionary_id;
      RETURN_NOT_OK(ReadDictionary(*message->metadata(), dictionary_fields_, &reader,
                                   &dictionary_id, &dictionary));
      RETURN_NOT_OK(dictionary_memo_->AddDictionary(dictionary_id, dictionary));
    }

    // Get the schema
    return internal::GetSchema(footer_->schema(), *dictionary_memo_, &schema_);
  }

  bool CanReadRecordBatchAsBatches(std::string* message = nullptr) const {
    auto s = schema();
    for (int32_t i = 0; i < s->num_fields(); i++) {
        switch (s->field(i)->type()->id()) {
            case Type::BOOL:
            case Type::UINT8:
            case Type::INT8:
            case Type::UINT16:
            case Type::INT16:
            case Type::UINT32:
            case Type::INT32:
            case Type::UINT64:
            case Type::INT64:
            case Type::HALF_FLOAT:
            case Type::FLOAT:
            case Type::DOUBLE:
            case Type::STRING:
            case Type::BINARY:
            case Type::FIXED_SIZE_BINARY:
            case Type::DATE32:
            case Type::DATE64:
            case Type::TIMESTAMP:
            case Type::TIME32:
            case Type::TIME64:
            case Type::INTERVAL:
            case Type::DECIMAL:
            case Type::DICTIONARY:
                break;
            case Type::LIST:
            case Type::STRUCT:
            case Type::UNION:
            case Type::MAP:
            case Type::EXTENSION:
            case Type::NA:
                if (message) {
                    *message = "Cannot rerad slice for field " + s->field(i)->ToString();
                }
                return false;
        }
    }
    return true;
  }

  Status Open(const std::shared_ptr<io::RandomAccessFile>& file, int64_t footer_offset) {
    owned_file_ = file;
    return Open(file.get(), footer_offset);
  }

  Status Open(io::RandomAccessFile* file, int64_t footer_offset) {
    file_ = file;
    footer_offset_ = footer_offset;
    RETURN_NOT_OK(ReadFooter());
    return ReadSchema();
  }

  std::shared_ptr<Schema> schema() const { return schema_; }

 private:
  io::RandomAccessFile* file_;

  std::shared_ptr<io::RandomAccessFile> owned_file_;

  // The location where the Arrow file layout ends. May be the end of the file
  // or some other location if embedded in a larger file.
  int64_t footer_offset_;

  // Footer metadata
  std::shared_ptr<Buffer> footer_buffer_;
  const flatbuf::Footer* footer_;

  DictionaryTypeMap dictionary_fields_;
  std::shared_ptr<DictionaryMemo> dictionary_memo_;

  // Reconstructed schema, including any read dictionaries
  std::shared_ptr<Schema> schema_;
private:
    class RecordSubBatchReader : public IRecordBatchFileReader {
    public:
        friend class RecordBatchFileReader;
        RecordSubBatchReader(std::shared_ptr<RecordBatchFileReader::RecordBatchFileReaderImpl> impl, int64_t batch_index, int64_t max_sub_batch_size, int64_t batch_length) :
            impl_(std::move(impl)), _max_batch_size(max_sub_batch_size), _batch_index(batch_index), _batch_length(batch_length) {
        }

        int num_record_batches() const override {
            return (_batch_length + _max_batch_size - 1)/_max_batch_size;
        }

        std::shared_ptr<Schema> schema() const override {
            return impl_->schema();
        }

        Status ReadRecordBatch(int i, std::shared_ptr<RecordBatch>* batch) final {
            auto batch_size = _max_batch_size;
            if ((i + 1) * _max_batch_size > _batch_length) {
                batch_size = _batch_length % _max_batch_size;
            }
            return impl_->ReadRecordBatch(_batch_index, batch, i * _max_batch_size, batch_size);
        }

    private:
        std::shared_ptr<RecordBatchFileReader::RecordBatchFileReaderImpl> impl_;
        int64_t _max_batch_size;
        int64_t _batch_index;
        int64_t _batch_length;
    };

};

Status RecordBatchFileReader::RecordBatchFileReaderImpl::ReadRecordBatchAsBatches(int i, std::shared_ptr<IRecordBatchFileReader> *reader,
                                int32_t max_batch_size, std::shared_ptr<RecordBatchFileReaderImpl> impl) {
    std::string message;
    if (not CanReadRecordBatchAsBatches(&message)) {
        return arrow::Status::Invalid(message);
    }
    std::shared_ptr<arrow::RecordBatch> batch;
    int64_t len;
    ARROW_RETURN_NOT_OK(record_batch_num_rows(i, &len));
    *reader = std::make_shared<RecordSubBatchReader>(impl, i, max_batch_size, len);
    return arrow::Status::OK();
}


RecordBatchFileReader::RecordBatchFileReader() {
  impl_.reset(new RecordBatchFileReaderImpl());
}

RecordBatchFileReader::~RecordBatchFileReader() {}

Status RecordBatchFileReader::Open(io::RandomAccessFile* file,
                                   std::shared_ptr<RecordBatchFileReader>* reader) {
  int64_t footer_offset;
  RETURN_NOT_OK(file->GetSize(&footer_offset));
  return Open(file, footer_offset, reader);
}

Status RecordBatchFileReader::Open(io::RandomAccessFile* file, int64_t footer_offset,
                                   std::shared_ptr<RecordBatchFileReader>* reader) {
  *reader = std::shared_ptr<RecordBatchFileReader>(new RecordBatchFileReader());
  return (*reader)->impl_->Open(file, footer_offset);
}

Status RecordBatchFileReader::Open(const std::shared_ptr<io::RandomAccessFile>& file,
                                   std::shared_ptr<RecordBatchFileReader>* reader) {
  int64_t footer_offset;
  RETURN_NOT_OK(file->GetSize(&footer_offset));
  return Open(file, footer_offset, reader);
}

Status RecordBatchFileReader::Open(const std::shared_ptr<io::RandomAccessFile>& file,
                                   int64_t footer_offset,
                                   std::shared_ptr<RecordBatchFileReader>* reader) {
  *reader = std::shared_ptr<RecordBatchFileReader>(new RecordBatchFileReader());
  return (*reader)->impl_->Open(file, footer_offset);
}

std::shared_ptr<Schema> RecordBatchFileReader::schema() const { return impl_->schema(); }

int RecordBatchFileReader::num_record_batches() const {
  return impl_->num_record_batches();
}

MetadataVersion RecordBatchFileReader::version() const { return impl_->version(); }

Status RecordBatchFileReader::ReadRecordBatch(int i,
                                              std::shared_ptr<RecordBatch>* batch) {
  return impl_->ReadRecordBatch(i, batch);
}

Status RecordBatchFileReader::ReadRecordBatchAsBatches(int i, std::shared_ptr<IRecordBatchFileReader> *reader,
                                                       int32_t max_batch_size) {
    return impl_->ReadRecordBatchAsBatches(i, reader, max_batch_size, impl_);
}

bool RecordBatchFileReader::CanReadRecordBatchAsBatches() const {
    return impl_->CanReadRecordBatchAsBatches();
}

        static Status ReadContiguousPayload(io::InputStream* file,
                            std::unique_ptr<Message>* message) {
  RETURN_NOT_OK(ReadMessage(file, message));
  if (*message == nullptr) {
    return Status::Invalid("Unable to read metadata at offset");
  }
  return Status::OK();
}

Status ReadSchema(io::InputStream* stream, std::shared_ptr<Schema>* out) {
  std::shared_ptr<RecordBatchReader> reader;
  RETURN_NOT_OK(RecordBatchStreamReader::Open(stream, &reader));
  *out = reader->schema();
  return Status::OK();
}

Status ReadSchema(const Message& message, std::shared_ptr<Schema>* out) {
  std::shared_ptr<RecordBatchReader> reader;
  DictionaryMemo dictionary_memo;
  return internal::GetSchema(message.header(), dictionary_memo, &*out);
}

Status ReadRecordBatch(const std::shared_ptr<Schema>& schema, io::InputStream* file,
                       std::shared_ptr<RecordBatch>* out) {
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(ReadContiguousPayload(file, &message));
  io::BufferReader buffer_reader(message->body());
  return ReadRecordBatch(*message->metadata(), schema, kMaxNestingDepth, &buffer_reader,
                         out);
}

Status ReadTensor(io::InputStream* file, std::shared_ptr<Tensor>* out) {
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(ReadContiguousPayload(file, &message));
  return ReadTensor(*message, out);
}

Status ReadTensor(const Message& message, std::shared_ptr<Tensor>* out) {
  std::shared_ptr<DataType> type;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<std::string> dim_names;
  RETURN_NOT_OK(internal::GetTensorMetadata(*message.metadata(), &type, &shape, &strides,
                                            &dim_names));
  *out = std::make_shared<Tensor>(type, message.body(), shape, strides, dim_names);
  return Status::OK();
}

namespace {

Status ReadSparseCOOIndex(const flatbuf::SparseTensor* sparse_tensor, int64_t ndim,
                          int64_t non_zero_length, io::RandomAccessFile* file,
                          std::shared_ptr<SparseIndex>* out) {
  auto* sparse_index = sparse_tensor->sparseIndex_as_SparseTensorIndexCOO();
  auto* indices_buffer = sparse_index->indicesBuffer();
  std::shared_ptr<Buffer> indices_data;
  RETURN_NOT_OK(
      file->ReadAt(indices_buffer->offset(), indices_buffer->length(), &indices_data));
  std::vector<int64_t> shape({non_zero_length, ndim});
  const int64_t elsize = sizeof(int64_t);
  std::vector<int64_t> strides({elsize, elsize * non_zero_length});
  *out = std::make_shared<SparseCOOIndex>(
      std::make_shared<SparseCOOIndex::CoordsTensor>(indices_data, shape, strides));
  return Status::OK();
}

Status ReadSparseCSRIndex(const flatbuf::SparseTensor* sparse_tensor, int64_t ndim,
                          int64_t non_zero_length, io::RandomAccessFile* file,
                          std::shared_ptr<SparseIndex>* out) {
  auto* sparse_index = sparse_tensor->sparseIndex_as_SparseMatrixIndexCSR();

  auto* indptr_buffer = sparse_index->indptrBuffer();
  std::shared_ptr<Buffer> indptr_data;
  RETURN_NOT_OK(
      file->ReadAt(indptr_buffer->offset(), indptr_buffer->length(), &indptr_data));

  auto* indices_buffer = sparse_index->indicesBuffer();
  std::shared_ptr<Buffer> indices_data;
  RETURN_NOT_OK(
      file->ReadAt(indices_buffer->offset(), indices_buffer->length(), &indices_data));

  std::vector<int64_t> indptr_shape({ndim + 1});
  std::vector<int64_t> indices_shape({non_zero_length});
  *out = std::make_shared<SparseCSRIndex>(
      std::make_shared<SparseCSRIndex::IndexTensor>(indptr_data, indptr_shape),
      std::make_shared<SparseCSRIndex::IndexTensor>(indices_data, indices_shape));
  return Status::OK();
}

Status MakeSparseTensorWithSparseCOOIndex(
    const std::shared_ptr<DataType>& type, const std::vector<int64_t>& shape,
    const std::vector<std::string>& dim_names,
    const std::shared_ptr<SparseCOOIndex>& sparse_index, int64_t non_zero_length,
    const std::shared_ptr<Buffer>& data, std::shared_ptr<SparseTensor>* out) {
  *out = std::make_shared<SparseTensorImpl<SparseCOOIndex>>(sparse_index, type, data,
                                                            shape, dim_names);
  return Status::OK();
}

Status MakeSparseTensorWithSparseCSRIndex(
    const std::shared_ptr<DataType>& type, const std::vector<int64_t>& shape,
    const std::vector<std::string>& dim_names,
    const std::shared_ptr<SparseCSRIndex>& sparse_index, int64_t non_zero_length,
    const std::shared_ptr<Buffer>& data, std::shared_ptr<SparseTensor>* out) {
  *out = std::make_shared<SparseTensorImpl<SparseCSRIndex>>(sparse_index, type, data,
                                                            shape, dim_names);
  return Status::OK();
}

}  // namespace

Status ReadSparseTensor(const Buffer& metadata, io::RandomAccessFile* file,
                        std::shared_ptr<SparseTensor>* out) {
  std::shared_ptr<DataType> type;
  std::vector<int64_t> shape;
  std::vector<std::string> dim_names;
  int64_t non_zero_length;
  SparseTensorFormat::type sparse_tensor_format_id;

  RETURN_NOT_OK(internal::GetSparseTensorMetadata(
      metadata, &type, &shape, &dim_names, &non_zero_length, &sparse_tensor_format_id));

  auto message = flatbuf::GetMessage(metadata.data());
  auto sparse_tensor = reinterpret_cast<const flatbuf::SparseTensor*>(message->header());
  const flatbuf::Buffer* buffer = sparse_tensor->data();
  DCHECK(BitUtil::IsMultipleOf8(buffer->offset()))
      << "Buffer of sparse index data "
      << "did not start on 8-byte aligned offset: " << buffer->offset();

  std::shared_ptr<Buffer> data;
  RETURN_NOT_OK(file->ReadAt(buffer->offset(), buffer->length(), &data));

  std::shared_ptr<SparseIndex> sparse_index;
  switch (sparse_tensor_format_id) {
    case SparseTensorFormat::COO:
      RETURN_NOT_OK(ReadSparseCOOIndex(sparse_tensor, shape.size(), non_zero_length, file,
                                       &sparse_index));
      return MakeSparseTensorWithSparseCOOIndex(
          type, shape, dim_names, checked_pointer_cast<SparseCOOIndex>(sparse_index),
          non_zero_length, data, out);

    case SparseTensorFormat::CSR:
      RETURN_NOT_OK(ReadSparseCSRIndex(sparse_tensor, shape.size(), non_zero_length, file,
                                       &sparse_index));
      return MakeSparseTensorWithSparseCSRIndex(
          type, shape, dim_names, checked_pointer_cast<SparseCSRIndex>(sparse_index),
          non_zero_length, data, out);

    default:
      return Status::Invalid("Unsupported sparse index format");
  }
}

Status ReadSparseTensor(const Message& message, std::shared_ptr<SparseTensor>* out) {
  io::BufferReader buffer_reader(message.body());
  return ReadSparseTensor(*message.metadata(), &buffer_reader, out);
}

Status ReadSparseTensor(io::InputStream* file, std::shared_ptr<SparseTensor>* out) {
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(ReadContiguousPayload(file, &message));
  CHECK_MESSAGE_TYPE(message->type(), Message::SPARSE_TENSOR);
  CHECK_HAS_BODY(*message);
  io::BufferReader buffer_reader(message->body());
  return ReadSparseTensor(*message->metadata(), &buffer_reader, out);
}

}  // namespace ipc
}  // namespace arrow
