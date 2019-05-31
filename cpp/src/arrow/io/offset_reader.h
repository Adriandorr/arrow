//
// Created by adorr on 31/05/19.
//

#ifndef ARROW_OFFSET_READER_H
#define ARROW_OFFSET_READER_H

namespace arrow {
    namespace io {
        class OffsetRandomAccessFile : public RandomAccessFile {
        public:
            OffsetRandomAccessFile(RandomAccessFile* file, int64_t offset, int64_t size) : file_(file), offset_(offset), size_(size), pos_(0) {
            }

            Status GetSize(int64_t* size) final {
                *size = size_;
                return arrow::Status::OK();
            };

            Status ReadAt(int64_t position, int64_t nbytes, int64_t* bytes_read, void* out) final {
                nbytes = std::min<int64_t>(nbytes, size_ - position);
                auto status = file_->ReadAt(position + offset_, nbytes, bytes_read, out);
                if (status.ok()) {
                    pos_ = position + nbytes;
                }
                return status;
            }

            Status ReadAt(int64_t position, int64_t nbytes, std::shared_ptr<Buffer>* out) final {
                nbytes = std::min<int64_t>(nbytes, size_ - position);
                auto status = file_->ReadAt(position + offset_, std::min<int64_t>(nbytes, size_ - position), out);
                if (status.ok()) {
                    pos_ = position + nbytes;
                }
                return status;
            }

            Status Read(int64_t nbytes, int64_t* bytes_read, void* out) {
                return ReadAt(pos_, nbytes, bytes_read, out);
            }

            // Does not copy if not necessary
            Status Read(int64_t nbytes, std::shared_ptr<Buffer>* out) {
                return ReadAt(pos_, nbytes, out);

            }

            util::string_view Peek(int64_t nbytes) const final {
                return file_->Peek(nbytes);
            }

            /// \brief Return true if InputStream is capable of zero copy Buffer reads
            bool supports_zero_copy() const {
                return file_->supports_zero_copy();
            }

            Status Close() final { return file_->Close(); }

            Status Tell(int64_t* position) const final {
                *position = pos_;
                return Status::OK();
            }

            bool closed() const final {
                return file_->closed();
            }

            Status Seek(int64_t position) final {
                if (position < size_) {
                    pos_ = position;
                    return Status::OK();
                }
                return Status::Invalid("Can't seek past the end: " + std::to_string(position));
            }


        private:
            RandomAccessFile* file_;
            int64_t offset_;
            int64_t size_;
            int64_t pos_;

        };
    }
}

#endif //ARROW_OFFSET_READER_H
