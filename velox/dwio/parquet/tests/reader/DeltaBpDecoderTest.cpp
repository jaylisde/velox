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

#include <folly/Varint.h>
#include <gtest/gtest.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Nulls.h"
#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace facebook::velox::parquet;

namespace {

// Minimal DELTA_BINARY_PACKED encoder for test data generation.
class DeltaEncoder {
 public:
  void encode(const int64_t* values, int32_t numValues) {
    buf_.clear();
    putVlq(kValuesPerBlock);
    putVlq(kMiniBlocksPerBlock);
    putVlq(numValues);
    putZigZag(values[0]);

    for (int32_t bs = 1; bs < numValues; bs += kValuesPerBlock) {
      int32_t be = std::min(bs + kValuesPerBlock, numValues);
      int32_t nd = be - bs;
      int64_t deltas[kValuesPerBlock];
      for (int32_t i = 0; i < nd; ++i) {
        deltas[i] = values[bs + i] - values[bs + i - 1];
      }
      int64_t minDelta = *std::min_element(deltas, deltas + nd);
      for (int32_t i = nd; i < kValuesPerBlock; ++i) {
        deltas[i] = minDelta;
      }
      putZigZag(minDelta);

      uint8_t bitWidths[kMiniBlocksPerBlock];
      for (int32_t mb = 0; mb < kMiniBlocksPerBlock; ++mb) {
        uint8_t maxBits = 0;
        for (int32_t j = 0; j < kValuesPerMiniBlock; ++j) {
          uint64_t adj = static_cast<uint64_t>(
              deltas[mb * kValuesPerMiniBlock + j] - minDelta);
          if (adj) {
            maxBits = std::max(
                maxBits, static_cast<uint8_t>(64 - __builtin_clzll(adj)));
          }
        }
        bitWidths[mb] = maxBits;
        buf_.push_back(maxBits);
      }

      for (int32_t mb = 0; mb < kMiniBlocksPerBlock; ++mb) {
        int32_t nbytes = (kValuesPerMiniBlock * bitWidths[mb] + 7) / 8;
        int32_t sp = buf_.size();
        buf_.resize(sp + nbytes, 0);
        for (int32_t j = 0; j < kValuesPerMiniBlock; ++j) {
          uint64_t val = static_cast<uint64_t>(
              deltas[mb * kValuesPerMiniBlock + j] - minDelta);
          int32_t bitOff = j * bitWidths[mb];
          *reinterpret_cast<uint64_t*>(&buf_[sp + bitOff / 8]) |=
              (val << (bitOff % 8));
        }
      }
    }
    // Safety padding for unaligned reads.
    buf_.resize(buf_.size() + 8, 0);
  }

  const char* data() const {
    return buf_.data();
  }

 private:
  static constexpr int32_t kValuesPerBlock = 128;
  static constexpr int32_t kMiniBlocksPerBlock = 4;
  static constexpr int32_t kValuesPerMiniBlock =
      kValuesPerBlock / kMiniBlocksPerBlock;

  void putVlq(uint64_t v) {
    while (v >= 0x80) {
      buf_.push_back(static_cast<char>(v | 0x80));
      v >>= 7;
    }
    buf_.push_back(static_cast<char>(v));
  }

  void putZigZag(int64_t v) {
    putVlq(static_cast<uint64_t>((v << 1) ^ (v >> 63)));
  }

  std::vector<char> buf_;
};

void verifyDecode(
    const std::vector<int64_t>& expected,
    const char* description) {
  DeltaEncoder encoder;
  encoder.encode(expected.data(), expected.size());

  std::vector<int64_t> output(expected.size());
  DeltaBpDecoder decoder(encoder.data());
  decoder.readValues(output.data(), expected.size());

  ASSERT_EQ(output, expected) << description;
}

} // namespace

TEST(DeltaBpDecoderTest, smallDeltas) {
  // Typical sorted ID column: small positive deltas, ~6 bits.
  std::vector<int64_t> values(10000);
  values[0] = 1000000;
  for (int i = 1; i < 10000; ++i) {
    values[i] = values[i - 1] + (i % 50) + 1;
  }
  verifyDecode(values, "small deltas ~6 bits");
}

TEST(DeltaBpDecoderTest, constantDelta) {
  // All deltas are the same → bitWidth = 0.
  std::vector<int64_t> values(1000);
  for (int i = 0; i < 1000; ++i) {
    values[i] = 100 + i * 7;
  }
  verifyDecode(values, "constant delta (bitWidth=0)");
}

TEST(DeltaBpDecoderTest, largeDeltas) {
  // Large random deltas, ~32 bits.
  std::mt19937 rng(42);
  std::vector<int64_t> values(5000);
  values[0] = 0;
  for (int i = 1; i < 5000; ++i) {
    values[i] = values[i - 1] + (rng() % 2000000000) + 1;
  }
  verifyDecode(values, "large deltas ~32 bits");
}

TEST(DeltaBpDecoderTest, maxBitWidth) {
  // Worst case: deltas requiring close to 64 bits.
  std::mt19937_64 rng(123);
  std::vector<int64_t> values(1000);
  values[0] = 0;
  for (int i = 1; i < 1000; ++i) {
    values[i] = values[i - 1] +
        static_cast<int64_t>(rng() % (INT64_MAX / 1000)) + 1;
  }
  verifyDecode(values, "max bitWidth (~60+ bits)");
}

TEST(DeltaBpDecoderTest, negativeDeltas) {
  // Decreasing values → negative deltas.
  std::vector<int64_t> values(1000);
  values[0] = 1000000;
  for (int i = 1; i < 1000; ++i) {
    values[i] = values[i - 1] - (i % 10) - 1;
  }
  verifyDecode(values, "negative deltas");
}

TEST(DeltaBpDecoderTest, singleValue) {
  std::vector<int64_t> values = {42};
  verifyDecode(values, "single value");
}

TEST(DeltaBpDecoderTest, partialBlock) {
  // 200 values = 1 full block (128) + partial block (72).
  std::vector<int64_t> values(200);
  values[0] = 0;
  for (int i = 1; i < 200; ++i) {
    values[i] = values[i - 1] + i;
  }
  verifyDecode(values, "partial block (200 values)");
}
