/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

#include <string_view>

namespace facebook::velox::parquet {

// DeltaByteArrayDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/encoding.cc#L2758-L2889
class DeltaLengthByteArrayDecoder {
 public:
  explicit DeltaLengthByteArrayDecoder(const char* start) {
    lengthDecoder_ = std::make_unique<DeltaBpDecoder>(start);
    decodeLengths();
    bufferStart_ = lengthDecoder_->bufferStart();
  }

  std::string_view readString() {
    const int64_t length = bufferedLength_[lengthIdx_++];
    VELOX_CHECK_GE(length, 0, "negative string delta length");
    bufferStart_ += length;
    return std::string_view(bufferStart_ - length, length);
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    for (int32_t i = 0; i < numValues; ++i) {
      readString();
    }
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    int32_t toSkip;
    bool atEnd = false;
    const bool allowNulls = hasNulls && visitor.allowNulls();
    for (;;) {
      if (hasNulls && allowNulls && bits::isBitNull(nulls, current)) {
        toSkip = visitor.processNull(atEnd);
      } else {
        if (hasNulls && !allowNulls) {
          toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
          if (!Visitor::dense) {
            skip<false>(toSkip, current, nullptr);
          }
          if (atEnd) {
            return;
          }
        }
        toSkip = visitor.process(readString(), atEnd);
      }
      ++current;
      if (toSkip) {
        skip<hasNulls>(toSkip, current, nulls);
        current += toSkip;
      }
      if (atEnd) {
        return;
      }
    }
  }

 private:
  void decodeLengths() {
    int64_t numLength = lengthDecoder_->validValuesCount();
    bufferedLength_.resize(numLength);
    lengthDecoder_->readValues<uint32_t>(bufferedLength_.data(), numLength);

    lengthIdx_ = 0;
    numValidValues_ = numLength;
  }

  const char* bufferStart_;
  std::unique_ptr<DeltaBpDecoder> lengthDecoder_;
  int32_t numValidValues_{0};
  uint32_t lengthIdx_{0};
  std::vector<uint32_t> bufferedLength_;
};

// DeltaByteArrayDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/encoding.cc#L3301-L3545
//
// Template parameter IsFixedLen: when true, the decoder validates that every
// decoded value has exactly the expected fixed length (for FIXED_LEN_BYTE_ARRAY
// columns). When false, no length check is performed (for BYTE_ARRAY columns).
template <bool IsFixedLen = false>
class DeltaByteArrayDecoder {
 public:
  explicit DeltaByteArrayDecoder(const char* start, size_t fixedLength = 0) {
    if constexpr (IsFixedLen) {
      VELOX_CHECK_GT(fixedLength, 0);
    }
    prefixLenDecoder_ = std::make_unique<DeltaBpDecoder>(start);
    int64_t numPrefix = prefixLenDecoder_->validValuesCount();
    bufferedPrefixLength_.resize(numPrefix);
    prefixLenDecoder_->readValues<uint32_t>(
        bufferedPrefixLength_.data(), numPrefix);

    suffixDecoder_ = std::make_unique<DeltaLengthByteArrayDecoder>(
        prefixLenDecoder_->bufferStart());

    if constexpr (IsFixedLen) {
      lastValue_.resize(fixedLength);
    }
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    for (int32_t i = 0; i < numValues; ++i) {
      readString();
    }
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    auto func = [&]() { return readString(); };
    readWithVisitorImpl<hasNulls>(nulls, visitor, func);
  }

  // readWithVisitor for fixed-width integer types (FLBA decimal).
  // Converts each decoded string to int128_t before passing to the visitor.
  template <bool hasNulls, typename Visitor>
  void readWithVisitorFixedWidth(
      const uint64_t* nulls,
      Visitor visitor,
      size_t fixedLength) {
    auto func = [&]() { return readAsInt128(fixedLength); };
    readWithVisitorImpl<hasNulls>(nulls, visitor, func);
  }

  std::string_view readString() {
    auto suffix = suffixDecoder_->readString();
    size_t prefixLength = bufferedPrefixLength_[prefixLenOffset_];
    size_t suffixLength = suffix.size();
    size_t length = prefixLength + suffixLength;
    ++prefixLenOffset_;

    if constexpr (IsFixedLen) {
      VELOX_CHECK_EQ(
          length,
          lastValue_.size(),
          "decoded length {} does not match fixed length {} in DELTA_BYTE_ARRAY",
          length,
          lastValue_.size());
    } else {
      VELOX_CHECK_LE(
          prefixLength,
          lastValue_.size(),
          "prefix length {} too large in DELTA_BYTE_ARRAY",
          prefixLength);
      lastValue_.resize(length);
    }
    memcpy(lastValue_.data() + prefixLength, suffix.data(), suffixLength);
    return {lastValue_.data(), length};
  }

 private:
  // Reads a fixed-length byte array value and converts it to int128_t.
  // Used for FLBA-encoded decimals. The bytes are big-endian and sign-extended.
  int128_t readAsInt128(size_t fixedLength) {
    VELOX_CHECK_LE(
        fixedLength,
        sizeof(int128_t),
        "DELTA_BYTE_ARRAY: fixedLength {} exceeds int128_t size",
        fixedLength);
    auto sv = readString();
    const char* src = sv.data();
    // Sign-extend based on the first byte.
    int128_t result = static_cast<int8_t>(src[0]) >> 7;
    char* dst =
        reinterpret_cast<char*>(&result) + sizeof(int128_t) - fixedLength;
    ::memcpy(dst, src, fixedLength);
    return bits::builtin_bswap128(result);
  }

  // Shared visitor loop.
  template <bool hasNulls, typename Visitor, typename ReadFn>
  void readWithVisitorImpl(
      const uint64_t* nulls,
      Visitor visitor,
      ReadFn readFn) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    bool atEnd = false;
    const bool allowNulls = hasNulls && visitor.allowNulls();
    do {
      int32_t toSkip = 0;
      if (hasNulls && allowNulls && bits::isBitNull(nulls, current)) {
        toSkip = visitor.processNull(atEnd);
      } else {
        if (hasNulls && !allowNulls) {
          toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
          if (!Visitor::dense) {
            skip<false>(toSkip, current, nullptr);
          }
          if (atEnd) break;
        }
        toSkip = visitor.process(readFn(), atEnd);
      }
      ++current;
      if (toSkip) {
        skip<hasNulls>(toSkip, current, nulls);
        current += toSkip;
      }
    } while (!atEnd);
  }

  std::unique_ptr<DeltaBpDecoder> prefixLenDecoder_;
  std::unique_ptr<DeltaLengthByteArrayDecoder> suffixDecoder_;

  std::vector<char> lastValue_;
  uint32_t prefixLenOffset_{0};
  std::vector<uint32_t> bufferedPrefixLength_;
};

} // namespace facebook::velox::parquet
