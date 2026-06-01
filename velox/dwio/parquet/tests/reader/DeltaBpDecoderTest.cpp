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

#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace facebook::velox::parquet::test {
namespace {

// Tiny DELTA_BINARY_PACKED encoder used to feed DeltaBpDecoder with byte
// streams whose contents we control exactly. Mirrors the reference encoder
// in the Parquet spec; not optimized.
class DeltaEncoder {
 public:
  // Encodes 'values' with one block of 'valuesPerBlock' values divided into
  // 'miniBlocksPerBlock' miniblocks. The encoder picks one bit width per
  // miniblock equal to the number of bits required to represent the largest
  // (delta - minDelta) within that miniblock. The first value of the page
  // is stored in the header. The result includes 'kPageReadPadding' trailing
  // zero bytes so that DeltaBpDecoder's readLong() loadBits never reads past
  // valid memory.
  std::vector<uint8_t> encode(
      const std::vector<int64_t>& values,
      int valuesPerBlock,
      int miniBlocksPerBlock) {
    VELOX_CHECK_GT(valuesPerBlock, 0);
    VELOX_CHECK_EQ(valuesPerBlock % 128, 0);
    VELOX_CHECK_GT(miniBlocksPerBlock, 0);
    const int valuesPerMiniBlock = valuesPerBlock / miniBlocksPerBlock;
    VELOX_CHECK_EQ(valuesPerMiniBlock % 32, 0);

    out_.clear();
    writeVlq(static_cast<uint64_t>(valuesPerBlock));
    writeVlq(static_cast<uint64_t>(miniBlocksPerBlock));
    writeVlq(static_cast<uint64_t>(values.size()));
    if (values.empty()) {
      writeZigZag(0);
      out_.insert(out_.end(), 8, 0); // padding
      return out_;
    }
    writeZigZag(values.front());

    int64_t prev = values.front();
    size_t pos = 1;
    while (pos < values.size()) {
      const int blockSize = std::min<int>(valuesPerBlock, values.size() - pos);

      std::vector<int64_t> deltas(valuesPerBlock, 0);
      int64_t minDelta = std::numeric_limits<int64_t>::max();
      for (int i = 0; i < blockSize; ++i) {
        deltas[i] = values[pos + i] - prev;
        prev = values[pos + i];
        minDelta = std::min(minDelta, deltas[i]);
      }
      // Per spec, missing trailing miniblocks may use any width; we use 0.
      // Compute per-miniblock bit width over the residual (delta - minDelta).
      std::vector<uint8_t> widths(miniBlocksPerBlock, 0);
      for (int m = 0; m < miniBlocksPerBlock; ++m) {
        const int begin = m * valuesPerMiniBlock;
        const int end = std::min(begin + valuesPerMiniBlock, blockSize);
        if (end <= begin) {
          continue;
        }
        uint64_t maxResidual = 0;
        for (int i = begin; i < end; ++i) {
          maxResidual = std::max<uint64_t>(maxResidual, deltas[i] - minDelta);
        }
        widths[m] = bitsRequired(maxResidual);
      }

      writeZigZag(minDelta);
      for (int m = 0; m < miniBlocksPerBlock; ++m) {
        out_.push_back(widths[m]);
      }

      // Pack each miniblock: write valuesPerMiniBlock residuals at the
      // miniblock's bit width even when the block has < valuesPerBlock real
      // values (trailing slots use zero residual, which is fine — the
      // decoder advances totalValuesRemaining_ separately and will not emit
      // them).
      for (int m = 0; m < miniBlocksPerBlock; ++m) {
        const int begin = m * valuesPerMiniBlock;
        const int end = std::min(begin + valuesPerMiniBlock, blockSize);
        std::vector<uint64_t> residuals(valuesPerMiniBlock, 0);
        for (int i = begin; i < end; ++i) {
          residuals[i - begin] = static_cast<uint64_t>(deltas[i] - minDelta);
        }
        writeBitPacked(residuals, widths[m]);
      }

      pos += blockSize;
    }
    out_.insert(out_.end(), 8, 0); // padding
    return out_;
  }

 private:
  static uint8_t bitsRequired(uint64_t v) {
    if (v == 0) {
      return 0;
    }
    uint8_t b = 0;
    while (v > 0) {
      ++b;
      v >>= 1;
    }
    return b;
  }

  void writeVlq(uint64_t v) {
    while (v >= 0x80) {
      out_.push_back(static_cast<uint8_t>(v | 0x80));
      v >>= 7;
    }
    out_.push_back(static_cast<uint8_t>(v));
  }

  void writeZigZag(int64_t v) {
    auto u = static_cast<uint64_t>((v << 1) ^ (v >> 63));
    writeVlq(u);
  }

  void writeBitPacked(const std::vector<uint64_t>& vals, uint8_t bitWidth) {
    if (bitWidth == 0) {
      return;
    }
    uint64_t buffer = 0;
    int filled = 0;
    for (uint64_t v : vals) {
      buffer |= (v & ((1ULL << bitWidth) - 1)) << filled;
      filled += bitWidth;
      while (filled >= 8) {
        out_.push_back(static_cast<uint8_t>(buffer & 0xff));
        buffer >>= 8;
        filled -= 8;
      }
    }
    if (filled > 0) {
      out_.push_back(static_cast<uint8_t>(buffer & 0xff));
    }
  }

  std::vector<uint8_t> out_;
};

// Decodes an entire page through the same path the SIMD-eligible production
// caller uses: readValues<int64_t>. Returns the produced values.
std::vector<int64_t> decodeAll(
    const std::vector<uint8_t>& bytes,
    size_t numValues) {
  DeltaBpDecoder decoder(reinterpret_cast<const char*>(bytes.data()));
  std::vector<int64_t> out(numValues);
  decoder.readValues<int64_t>(out.data(), static_cast<int32_t>(numValues));
  return out;
}

} // namespace

// Encodes 'numValues' values whose first value is 'first' and whose
// successive deltas are bounded by 'maxDelta' (so residuals fit in
// ceil(log2(maxDelta+1)) bits).
std::vector<int64_t>
makeAscendingValues(int numValues, int64_t first, int64_t maxDelta) {
  std::vector<int64_t> v;
  v.reserve(numValues);
  v.push_back(first);
  for (int i = 1; i < numValues; ++i) {
    const int64_t pseudo = static_cast<int64_t>((i + 7) * 1'000'003LL);
    v.push_back(v.back() + (pseudo % (maxDelta + 1)));
  }
  return v;
}

void runRoundtrip(
    const std::vector<int64_t>& values,
    int valuesPerBlock,
    int miniBlocksPerBlock) {
  DeltaEncoder enc;
  const auto bytes = enc.encode(values, valuesPerBlock, miniBlocksPerBlock);
  const auto decoded = decodeAll(bytes, values.size());
  EXPECT_EQ(decoded, values);
}

// Roundtrip test: encode then decode 32 sequential values whose deltas
// require exactly 10 bits. Targets the unpack9to15 PDEP path.
TEST(DeltaBpDecoderTest, bitWidth10SingleMiniBlock) {
  const auto values = makeAscendingValues(32, 1'000'000, 1023);
  runRoundtrip(values, /*valuesPerBlock=*/128, /*miniBlocksPerBlock=*/4);
}

// bit width 8 — targets unpack8 (8→32 cvt path).
TEST(DeltaBpDecoderTest, bitWidth8SingleMiniBlock) {
  const auto values = makeAscendingValues(32, 0, 255);
  runRoundtrip(values, 128, 4);
}

// bit width 16 — targets unpack16 (16→32 cvt path).
TEST(DeltaBpDecoderTest, bitWidth16SingleMiniBlock) {
  const auto values = makeAscendingValues(32, 0, 65'535);
  runRoundtrip(values, 128, 4);
}

// bit width 24 — targets unpack22to31 PDEP path.
TEST(DeltaBpDecoderTest, bitWidth24SingleMiniBlock) {
  const auto values = makeAscendingValues(32, 0, (1 << 24) - 1);
  runRoundtrip(values, 128, 4);
}

// bit width 32 — targets unpack32 (memcpy path).
TEST(DeltaBpDecoderTest, bitWidth32SingleMiniBlock) {
  const auto values = makeAscendingValues(32, 0, (1LL << 32) - 1);
  runRoundtrip(values, 128, 4);
}

// bit width 0 — every delta equals minDelta. Decoded values form a
// constant-stride arithmetic sequence; the bit-unpack body produces zero.
TEST(DeltaBpDecoderTest, bitWidth0ConstantDelta) {
  std::vector<int64_t> values{42};
  for (int i = 1; i < 128; ++i) {
    values.push_back(values.back() + 7);
  }
  runRoundtrip(values, 128, 4);
}

// Multi-block decode in a single readValues call. With valuesPerBlock=128
// and 384 total values, the decoder crosses two block boundaries
// (initBlock invoked twice during the call). Every miniblock and every
// block boundary must be handled without losing state.
TEST(DeltaBpDecoderTest, multiBlockSingleCall) {
  const auto values = makeAscendingValues(384, 0, 1023);
  runRoundtrip(values, 128, 4);
}

// Decode the page in two calls, with the split landing mid-miniblock. The
// SIMD fast path is only safe at miniblock-start (bitOffset == 0); the
// second call must fall back to the scalar inner loop for the partial
// miniblock and then resume the SIMD path on the next miniblock.
TEST(DeltaBpDecoderTest, splitInsideMiniBlock) {
  const auto values = makeAscendingValues(128, 1'000, 1023);
  DeltaEncoder enc;
  const auto bytes = enc.encode(values, 128, 4);
  DeltaBpDecoder decoder(reinterpret_cast<const char*>(bytes.data()));
  std::vector<int64_t> out(128);
  decoder.readValues<int64_t>(out.data(), 17); // mid-miniblock split
  decoder.readValues<int64_t>(out.data() + 17, 128 - 17);
  EXPECT_EQ(out, values);
}

// Negative minDelta — checks that the unsigned arithmetic on the
// hot-path correctly wraps when minDelta is sign-extended.
TEST(DeltaBpDecoderTest, negativeMinDelta) {
  std::vector<int64_t> values{1'000'000};
  // Deltas oscillate around -100, residuals (delta - minDelta) ∈ [0,200].
  for (int i = 1; i < 64; ++i) {
    values.push_back(values.back() - 100 + ((i * 17) % 201));
  }
  runRoundtrip(values, 128, 4);
}

// bit width > 32 — must take the scalar fallback because
// dwio::common::unpack<uint32_t> only handles up to 32 bits.
TEST(DeltaBpDecoderTest, bitWidthAbove32Fallback) {
  std::vector<int64_t> values{0};
  for (int i = 1; i < 32; ++i) {
    values.push_back(values.back() + (1LL << 33) + (i * 7));
  }
  runRoundtrip(values, 128, 4);
}

// int32_t output type — exercises the narrowing static_cast in
// readValues / decodeLongs.
TEST(DeltaBpDecoderTest, narrowingToInt32) {
  std::vector<int64_t> values64;
  values64.reserve(32);
  values64.push_back(100);
  for (int i = 1; i < 32; ++i) {
    values64.push_back(values64.back() + ((i * 11) & 0x3ff));
  }

  DeltaEncoder enc;
  const auto bytes = enc.encode(values64, 128, 4);
  DeltaBpDecoder decoder(reinterpret_cast<const char*>(bytes.data()));
  std::vector<int32_t> out(32);
  decoder.readValues<int32_t>(out.data(), 32);
  for (size_t i = 0; i < values64.size(); ++i) {
    EXPECT_EQ(out[i], static_cast<int32_t>(values64[i]));
  }
}

// Parameterized roundtrip across every supported SIMD-dispatched bit
// width (1..32). Each width drives one miniblock of 32 values whose
// residuals saturate the requested bit width — guaranteeing the
// encoder picks exactly that width — and verifies decoded output
// matches input byte-for-byte. Exercises:
//   - bit_width <= 16 (4-value/iter unaligned 64-bit load path)
//   - bit_width 17..32 (2-value/iter __uint128_t funnel-shift path)
//   - bit-offset misalignment within u64/u128 windows
class DeltaBpDecoderBitWidthTest : public ::testing::TestWithParam<int> {};

TEST_P(DeltaBpDecoderBitWidthTest, roundtrip32Values) {
  const int bitWidth = GetParam();
  // Saturate residual range for this bit width: max residual is
  // (1 << bitWidth) - 1. Use minDelta = 0 so encoded residual ==
  // delta, forcing the encoder to pick exactly 'bitWidth'.
  const int64_t maxDelta =
      (bitWidth == 32) ? 0xFFFFFFFFLL : ((1LL << bitWidth) - 1);
  std::vector<int64_t> values;
  values.reserve(32);
  // Start far from zero so int64 narrowing still works for bw=32.
  values.push_back(1'000'000);
  for (int i = 1; i < 32; ++i) {
    int64_t delta;
    if (i == 1) {
      // Ensure the encoder sees at least one value at the maximum,
      // so bitsRequired() picks 'bitWidth' — not a smaller width.
      delta = maxDelta;
    } else {
      const int64_t pseudo = static_cast<int64_t>((i + 7) * 1'000'003LL);
      delta = pseudo % (maxDelta + 1);
    }
    values.push_back(values.back() + delta);
  }

  DeltaEncoder enc;
  const auto bytes =
      enc.encode(values, /*valuesPerBlock=*/128, /*miniBlocksPerBlock=*/4);
  DeltaBpDecoder decoder(reinterpret_cast<const char*>(bytes.data()));
  std::vector<int64_t> out(values.size());
  decoder.readValues<int64_t>(out.data(), static_cast<int32_t>(out.size()));
  EXPECT_EQ(out, values) << "bit_width=" << bitWidth;
}

INSTANTIATE_TEST_SUITE_P(
    AllBitWidths,
    DeltaBpDecoderBitWidthTest,
    ::testing::Range(1, 33),
    [](const ::testing::TestParamInfo<int>& info) {
      return "bw" + std::to_string(info.param);
    });

} // namespace facebook::velox::parquet::test
