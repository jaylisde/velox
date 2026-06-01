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
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Nulls.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/type/Filter.h"

namespace facebook::velox::parquet {

// DeltaBpDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-12.0.0/cpp/src/parquet/encoding.cc#LL2357C18-L2586C3
class DeltaBpDecoder {
 public:
  explicit DeltaBpDecoder(const char* start) : bufferStart_(start) {
    initHeader();
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  FOLLY_ALWAYS_INLINE void
  skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    for (int32_t i = 0; i < numValues; ++i) {
      readLong();
    }
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    if constexpr (
        Visitor::dense && !hasNulls && Visitor::FilterType::deterministic &&
        std::is_same_v<typename Visitor::HookType, dwio::common::NoHook> &&
        std::is_integral_v<typename Visitor::DataType>) {
      readWithVisitorDenseBatched(visitor);
      return;
    }
    if constexpr (
        !Visitor::dense && !hasNulls && Visitor::FilterType::deterministic &&
        std::is_same_v<typename Visitor::HookType, dwio::common::NoHook> &&
        std::is_integral_v<typename Visitor::DataType>) {
      readWithVisitorSparseBuffered(visitor);
      return;
    }
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

        // We are at a non-null value on a row to visit.
        toSkip = visitor.process(readLong(), atEnd);
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

  const char* bufferStart() {
    return bufferStart_;
  }

  int64_t validValuesCount() {
    return static_cast<int64_t>(totalValuesRemaining_);
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE void readValues(T* values, int32_t numValues) {
    VELOX_DCHECK_LE(numValues, totalValuesRemaining_);
    if constexpr (std::is_integral_v<T>) {
      decodeLongs(values, numValues);
    } else {
      for (auto i = 0; i < numValues; i++) {
        values[i] = T(readLong());
      }
    }
  }

 private:
  // Dense + integral + NoHook fast path: decode each chunk directly into
  // visitor's output buffer, then dispatch one processRun() per chunk.
  template <typename Visitor>
  void readWithVisitorDenseBatched(Visitor& visitor) {
    using DataType = typename Visitor::DataType;
    constexpr bool kHasFilter =
        !std::
            is_same_v<typename Visitor::FilterType, velox::common::AlwaysTrue>;
    constexpr int32_t kBatch = 1024;
    const int32_t total = visitor.numRows();
    DataType* output = visitor.rawValues(total);
    int32_t* filterHits = kHasFilter ? visitor.outputRows(total) : nullptr;
    int32_t numValues = 0;
    int32_t consumed = 0;
    while (consumed < total) {
      const int32_t n = std::min<int32_t>(kBatch, total - consumed);
      DataType* dst = output + numValues;
      decodeLongs(dst, n);
      visitor.template processRun<
          kHasFilter,
          /*hasHook=*/false,
          /*scatter=*/false>(
          dst,
          n,
          /*scatterRows=*/nullptr,
          filterHits,
          output,
          numValues);
      consumed += n;
    }
    visitor.setNumValues(numValues);
  }

  // Sparse + deterministic-filter + NoHook + integral DataType fast
  // path. DELTA's delta-chain forces every physical row to be decoded
  // even when the row set is sparse, so we batch the decode side into
  // a stack buffer and run visitor.process() scalar over the buffer
  // (buffer index instead of readLong() per row).
  template <typename Visitor>
  void readWithVisitorSparseBuffered(Visitor& visitor) {
    using DataType = typename Visitor::DataType;
    constexpr int32_t kBatch = 1024;
    DataType buf[kBatch];

    const auto* rows = visitor.rows();
    const int32_t numRows = visitor.numRows();
    int32_t rowIdx = 0;
    int32_t currentPhys = (numRows > 0) ? rows[0] : 0;
    bool atEnd = false;
    while (!atEnd && rowIdx < numRows) {
      const int32_t remaining = static_cast<int32_t>(totalValuesRemaining_);
      if (remaining <= 0) {
        return;
      }
      // Cap n to the visitor's residual physical span; decoding past it
      // would leave the decoder at the wrong physical position and
      // misalign the next readWithVisitor call.
      const int32_t lastRow = rows[numRows - 1];
      const int32_t maxSpan = lastRow - currentPhys + 1;
      const int32_t n = std::min<int32_t>({kBatch, remaining, maxSpan});
      decodeLongs(buf, n);
      const int32_t batchPhysStart = currentPhys;
      const int32_t batchPhysEnd = currentPhys + n; // exclusive
      currentPhys = batchPhysEnd;
      while (rowIdx < numRows && rows[rowIdx] < batchPhysEnd) {
        const int32_t i = rows[rowIdx] - batchPhysStart;
        visitor.process(buf[i], atEnd);
        ++rowIdx;
        if (atEnd) {
          return;
        }
      }
    }
  }

  // Inlined readLong() with state hoisted to locals; advances a running
  // bitOffset to avoid a multiply per row. Whole bit-aligned miniblocks
  // dispatch to decodeMiniBlockSimd(); the scalar tail handles partial
  // miniblocks, bit_width > 32, and the page's first (header) value.
  // DataType narrowing covers int32.
  template <typename DataType>
  void decodeLongs(DataType* out, int32_t n) {
    const char* bufStart = bufferStart_;
    uint64_t valsPerMiniBlk = valuesPerMiniBlock_;
    uint64_t miniBlockRemaining = valuesRemainingCurrentMiniBlock_;
    uint64_t totalRemaining = totalValuesRemaining_;
    int64_t lastValue = lastValue_;
    int64_t minDelta = minDelta_;
    uint64_t deltaBitWidth = deltaBitWidth_;
    uint64_t bitOffset = (valsPerMiniBlk - miniBlockRemaining) * deltaBitWidth;

    int32_t i = 0;
    while (i < n) {
      if (miniBlockRemaining == 0) {
        // Refill the miniblock. Two distinct cases:
        //   1. The very first value of the page lives in the page
        //      header (lastValue_), not in a packed miniblock. Emit
        //      it and initialize the first block if there is one.
        //   2. Subsequent miniblocks: advance to the next miniblock
        //      (or new block) but DO NOT consume any value. Leaves
        //      miniBlockRemaining = valsPerMiniBlk so the SIMD fast
        //      path below can swallow the whole miniblock.
        if (!firstBlockInitialized_) {
          bufferStart_ = bufStart;
          valuesRemainingCurrentMiniBlock_ = 0;
          totalValuesRemaining_ = totalRemaining;
          lastValue_ = lastValue;
          int64_t v = readLong();
          out[i] = static_cast<DataType>(v);
          bufStart = bufferStart_;
          miniBlockRemaining = valuesRemainingCurrentMiniBlock_;
          totalRemaining = totalValuesRemaining_;
          lastValue = lastValue_;
          minDelta = minDelta_;
          deltaBitWidth = deltaBitWidth_;
          bitOffset = (valsPerMiniBlk - miniBlockRemaining) * deltaBitWidth;
          ++i;
          continue;
        }
        bufferStart_ = bufStart;
        valuesRemainingCurrentMiniBlock_ = 0;
        totalValuesRemaining_ = totalRemaining;
        lastValue_ = lastValue;
        advanceMiniBlock();
        bufStart = bufferStart_;
        miniBlockRemaining = valuesRemainingCurrentMiniBlock_;
        minDelta = minDelta_;
        deltaBitWidth = deltaBitWidth_;
        bitOffset = 0;
      }

      // Whole-miniblock SIMD fast path. Eligibility:
      //   - We are at the miniblock start (bit-aligned, full count
      //     remaining).
      //   - The whole miniblock fits within the requested chunk and
      //     the remaining page values.
      //   - bit_width <= 32: an inline kernel materialized for each
      //     compile-time bit width handles the unpack. Widths 33..64
      //     fall through to the per-row scalar inner loop. Width 0
      //     is a constant-stride arithmetic sequence (no bit-extract
      //     needed).
      if (miniBlockRemaining == valsPerMiniBlk &&
          static_cast<uint64_t>(n - i) >= valsPerMiniBlk &&
          totalRemaining >= valsPerMiniBlk && deltaBitWidth <= 32) {
        const int32_t mbValues = static_cast<int32_t>(valsPerMiniBlk);
        DataType* dst = out + i;
        bool dispatched = true;
        switch (deltaBitWidth) {
          case 0:
            decodeMiniBlockConstantDelta(mbValues, minDelta, lastValue, dst);
            break;
#define VELOX_DELTA_BP_DISPATCH(bw)                    \
  case bw:                                             \
    decodeMiniBlockSimdImpl<DataType, bw>(             \
        bufStart, mbValues, minDelta, lastValue, dst); \
    break;
            VELOX_DELTA_BP_DISPATCH(1)
            VELOX_DELTA_BP_DISPATCH(2)
            VELOX_DELTA_BP_DISPATCH(3)
            VELOX_DELTA_BP_DISPATCH(4)
            VELOX_DELTA_BP_DISPATCH(5)
            VELOX_DELTA_BP_DISPATCH(6)
            VELOX_DELTA_BP_DISPATCH(7)
            VELOX_DELTA_BP_DISPATCH(8)
            VELOX_DELTA_BP_DISPATCH(9)
            VELOX_DELTA_BP_DISPATCH(10)
            VELOX_DELTA_BP_DISPATCH(11)
            VELOX_DELTA_BP_DISPATCH(12)
            VELOX_DELTA_BP_DISPATCH(13)
            VELOX_DELTA_BP_DISPATCH(14)
            VELOX_DELTA_BP_DISPATCH(15)
            VELOX_DELTA_BP_DISPATCH(16)
            VELOX_DELTA_BP_DISPATCH(17)
            VELOX_DELTA_BP_DISPATCH(18)
            VELOX_DELTA_BP_DISPATCH(19)
            VELOX_DELTA_BP_DISPATCH(20)
            VELOX_DELTA_BP_DISPATCH(21)
            VELOX_DELTA_BP_DISPATCH(22)
            VELOX_DELTA_BP_DISPATCH(23)
            VELOX_DELTA_BP_DISPATCH(24)
            VELOX_DELTA_BP_DISPATCH(25)
            VELOX_DELTA_BP_DISPATCH(26)
            VELOX_DELTA_BP_DISPATCH(27)
            VELOX_DELTA_BP_DISPATCH(28)
            VELOX_DELTA_BP_DISPATCH(29)
            VELOX_DELTA_BP_DISPATCH(30)
            VELOX_DELTA_BP_DISPATCH(31)
            VELOX_DELTA_BP_DISPATCH(32)
#undef VELOX_DELTA_BP_DISPATCH
          default:
            dispatched = false;
        }
        if (dispatched) {
          bufStart += bits::nbytes(deltaBitWidth * valsPerMiniBlk);
          const uint64_t consumed = valsPerMiniBlk;
          miniBlockRemaining = 0;
          totalRemaining -= consumed;
          i += static_cast<int32_t>(consumed);
          bitOffset = 0;
          continue;
        }
      }

      uint64_t value = 0;
      if (deltaBitWidth) {
        value = bits::detail::loadBits<uint64_t>(
            reinterpret_cast<const uint64_t*>(bufStart),
            bitOffset,
            deltaBitWidth);
        value &= (~0ULL >> (64 - deltaBitWidth));
      }
      uint64_t result = static_cast<uint64_t>(minDelta) + value +
          static_cast<uint64_t>(lastValue);
      lastValue = static_cast<int64_t>(result);
      out[i] = static_cast<DataType>(result);
      bitOffset += deltaBitWidth;
      --miniBlockRemaining;
      --totalRemaining;
      if (miniBlockRemaining == 0 || totalRemaining == 0) {
        bufStart += bits::nbytes(deltaBitWidth * valsPerMiniBlk);
        bitOffset = 0;
      }
      ++i;
    }

    bufferStart_ = bufStart;
    valuesRemainingCurrentMiniBlock_ = miniBlockRemaining;
    totalValuesRemaining_ = totalRemaining;
    lastValue_ = lastValue;
  }

  /// Decode one whole miniblock with prefix-sum fused into the
  /// bit-extract loop. All arithmetic is unsigned mod-2^64 (Parquet
  /// spec). Safety: the unpack reads up to 7 bytes past the last
  /// miniblock byte; PageReader::readBytes guarantees
  /// kPageReadPadding (8) trailing bytes, and intra-page overshoots
  /// fall into the next miniblock — both valid memory.
  template <typename DataType, uint8_t bitWidth>
  FOLLY_ALWAYS_INLINE void decodeMiniBlockSimdImpl(
      const char* src,
      int32_t numValues,
      int64_t minDelta,
      int64_t& lastValue,
      DataType* out) {
    static_assert(bitWidth >= 1 && bitWidth <= 32);
    constexpr uint64_t mask =
        (bitWidth == 32) ? 0xFFFFFFFFULL : ((1ULL << bitWidth) - 1);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    uint64_t cumulative = static_cast<uint64_t>(lastValue);
    const uint64_t step = static_cast<uint64_t>(minDelta);
    if constexpr (bitWidth <= 16) {
      // Process 4 values per iteration. A single unaligned 64-bit
      // load holds at least 4 values for any bitWidth in [1..16]
      // (4*bw <= 64). All 4 values come from one load + one shift.
      for (int32_t i = 0; i < numValues; i += 4) {
        const int32_t bitPos = i * bitWidth;
        const int32_t byteOff = bitPos >> 3;
        const int32_t bitInByte = bitPos & 7;
        const uint64_t word =
            *reinterpret_cast<const uint64_t*>(p + byteOff) >> bitInByte;
        cumulative += step + (word & mask);
        out[i + 0] = static_cast<DataType>(cumulative);
        cumulative += step + ((word >> bitWidth) & mask);
        out[i + 1] = static_cast<DataType>(cumulative);
        cumulative += step + ((word >> (2 * bitWidth)) & mask);
        out[i + 2] = static_cast<DataType>(cumulative);
        cumulative += step + ((word >> (3 * bitWidth)) & mask);
        out[i + 3] = static_cast<DataType>(cumulative);
      }
    } else {
      // bitWidth in [17..32]: 2 values per iteration. After shifting
      // by bitInByte (0..7), 2*bw + bitInByte can reach 71 bits, past
      // a u64 window. Use __uint128_t for the unaligned load — on
      // x86_64 this lowers to two u64 loads and a SHRD-style funnel
      // shift, with no branch on bitInByte. The trailing u64 read
      // (byteOff + 8) is always safe: intra-page it falls into the
      // next miniblock; at page end the kPageReadPadding (8) bytes
      // cover it.
      for (int32_t i = 0; i < numValues; i += 2) {
        const int32_t bitPos = i * bitWidth;
        const int32_t byteOff = bitPos >> 3;
        const int32_t bitInByte = bitPos & 7;
        const __uint128_t window =
            static_cast<__uint128_t>(
                *reinterpret_cast<const uint64_t*>(p + byteOff)) |
            (static_cast<__uint128_t>(
                 *reinterpret_cast<const uint64_t*>(p + byteOff + 8))
             << 64);
        const uint64_t word = static_cast<uint64_t>(window >> bitInByte);
        cumulative += step + (word & mask);
        out[i + 0] = static_cast<DataType>(cumulative);
        cumulative += step + ((word >> bitWidth) & mask);
        out[i + 1] = static_cast<DataType>(cumulative);
      }
    }
    lastValue = static_cast<int64_t>(cumulative);
  }

  // Specialization for bit_width == 0: every value is constant-stride
  // (lastValue + (i+1)*minDelta).
  template <typename DataType>
  FOLLY_ALWAYS_INLINE void decodeMiniBlockConstantDelta(
      int32_t numValues,
      int64_t minDelta,
      int64_t& lastValue,
      DataType* out) {
    uint64_t cumulative = static_cast<uint64_t>(lastValue);
    const uint64_t step = static_cast<uint64_t>(minDelta);
    for (int32_t i = 0; i < numValues; ++i) {
      cumulative += step;
      out[i] = static_cast<DataType>(cumulative);
    }
    lastValue = static_cast<int64_t>(cumulative);
  }

  bool getVlqInt(uint64_t& v);

  bool getZigZagVlqInt(int64_t& v);

  void initHeader();

  void initBlock();

  void initMiniBlock(int32_t bitWidth);

  // Advances to the next miniblock without consuming a value. Mirrors
  // the inner branch of readLong()'s `valuesRemainingCurrentMiniBlock_
  // == 0` case (post-firstBlockInitialized_) but stops short of
  // decoding the first value, so callers that decode a whole miniblock
  // in one shot (decodeLongs SIMD path) can pick it up cleanly.
  void advanceMiniBlock() {
    VELOX_DCHECK(firstBlockInitialized_);
    VELOX_DCHECK_EQ(valuesRemainingCurrentMiniBlock_, 0);
    ++miniBlockIdx_;
    if (miniBlockIdx_ < miniBlocksPerBlock_) {
      initMiniBlock(deltaBitWidths_[miniBlockIdx_]);
    } else {
      initBlock();
    }
  }

  FOLLY_ALWAYS_INLINE int64_t readLong() {
    int64_t value = 0;
    if (valuesRemainingCurrentMiniBlock_ == 0) {
      if (!firstBlockInitialized_) {
        value = lastValue_;
        // When block is uninitialized we have two different possibilities:
        // 1. totalValueCount_ == 1, which means that the page may have only
        // one value (encoded in the header), and we should not initialize
        // any block.
        // 2. totalValueCount_ != 1, which means we should initialize the
        // incoming block for subsequent reads.
        if (totalValueCount_ != 1) {
          initBlock();
        }
        totalValuesRemaining_--;
        return value;
      } else {
        ++miniBlockIdx_;
        if (miniBlockIdx_ < miniBlocksPerBlock_) {
          initMiniBlock(deltaBitWidths_[miniBlockIdx_]);
        } else {
          initBlock();
        }
      }
    }

    uint64_t consumedBits =
        (valuesPerMiniBlock_ - valuesRemainingCurrentMiniBlock_) *
        deltaBitWidth_;
    if (deltaBitWidth_) {
      value = bits::detail::loadBits<uint64_t>(
          reinterpret_cast<const uint64_t*>(bufferStart_),
          consumedBits,
          deltaBitWidth_);
      value &= (~0ULL >> (64 - deltaBitWidth_));
    }
    // Addition between minDelta_, packed int and lastValue_ should be treated
    // as unsigned addition. Overflow is as expected.
    value = static_cast<uint64_t>(minDelta_) + static_cast<uint64_t>(value) +
        static_cast<uint64_t>(lastValue_);
    lastValue_ = value;
    valuesRemainingCurrentMiniBlock_--;
    totalValuesRemaining_--;

    if (valuesRemainingCurrentMiniBlock_ == 0 || totalValuesRemaining_ == 0) {
      bufferStart_ += bits::nbytes(deltaBitWidth_ * valuesPerMiniBlock_);
    }
    return value;
  }

  static constexpr int kMaxDeltaBitWidth =
      static_cast<int>(sizeof(int64_t) * 8);

  const char* bufferStart_;

  uint64_t valuesPerBlock_;
  uint64_t miniBlocksPerBlock_;
  uint64_t valuesPerMiniBlock_;
  uint64_t totalValueCount_;

  uint64_t totalValuesRemaining_;
  // Remaining values in current mini block. If the current block is the last
  // mini block, valuesRemainingCurrentMiniBlock_ may greater than
  // totalValuesRemaining_.
  uint64_t valuesRemainingCurrentMiniBlock_;

  // If the page doesn't contain any block, `firstBlockInitialized_` will
  // always be false. Otherwise, it will be true when first block initialized.
  bool firstBlockInitialized_;
  int64_t minDelta_;
  uint64_t miniBlockIdx_;
  std::vector<uint8_t> deltaBitWidths_;
  uint64_t deltaBitWidth_;

  int64_t lastValue_;
};

} // namespace facebook::velox::parquet
