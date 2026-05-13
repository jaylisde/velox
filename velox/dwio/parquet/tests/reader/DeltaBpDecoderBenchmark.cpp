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

/// Micro-benchmark comparing bits::copyBits vs inline shift+mask for
/// bit-packed value extraction, the hot path in DeltaBpDecoder::readLong.
/// Uses a 3-bit start offset to simulate real DELTA_BINARY_PACKED layout
/// where miniblocks rarely start at byte boundaries.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/base/BitUtil.h"

#include <random>
#include <vector>

using namespace facebook::velox;

namespace {

constexpr int32_t kNumValues = 1000000;
// In real parquet, miniblocks start after variable-length headers so the
// packed data rarely begins at a byte boundary. This offset ensures all
// bit extractions are non-aligned.
constexpr int32_t kStartBitOffset = 3;

struct PackedData {
  std::vector<char> buf;
  int32_t bitWidth;
};

PackedData makePackedData(int32_t bitWidth) {
  int32_t totalBits = kStartBitOffset + kNumValues * bitWidth;
  int32_t numBytes = (totalBits + 7) / 8 + 8;
  std::vector<char> buf(numBytes, 0);
  std::mt19937_64 rng(42);
  uint64_t mask = bitWidth == 64 ? ~0ULL : (1ULL << bitWidth) - 1;
  for (int32_t i = 0; i < kNumValues; ++i) {
    uint64_t val = rng() & mask;
    int32_t bitOff = kStartBitOffset + i * bitWidth;
    *reinterpret_cast<uint64_t*>(&buf[bitOff / 8]) |= (val << (bitOff % 8));
  }
  return {std::move(buf), bitWidth};
}

PackedData g10, g16, g32, g64;

void benchCopyBits(const PackedData& pd) {
  const auto* src = reinterpret_cast<const uint64_t*>(pd.buf.data());
  int64_t sum = 0;
  for (int32_t i = 0; i < kNumValues; ++i) {
    int64_t value = 0;
    bits::copyBits(
        src,
        kStartBitOffset + static_cast<uint64_t>(i) * pd.bitWidth,
        reinterpret_cast<uint64_t*>(&value),
        0,
        pd.bitWidth);
    sum += value;
  }
  folly::doNotOptimizeAway(sum);
}

void benchInline(const PackedData& pd) {
  const char* data = pd.buf.data();
  uint64_t mask =
      pd.bitWidth == 64 ? ~0ULL : (1ULL << pd.bitWidth) - 1;
  int64_t sum = 0;
  for (int32_t i = 0; i < kNumValues; ++i) {
    uint64_t consumedBits =
        kStartBitOffset + static_cast<uint64_t>(i) * pd.bitWidth;
    auto byteOffset = consumedBits / 8;
    auto bitOffset = consumedBits & 7;
    auto word = folly::loadUnaligned<uint64_t>(data + byteOffset);
    int64_t value = static_cast<int64_t>((word >> bitOffset) & mask);
    if (bitOffset + pd.bitWidth > 64) {
      auto next =
          static_cast<uint64_t>(static_cast<uint8_t>(data[byteOffset + 8]));
      value |= static_cast<int64_t>((next << (64 - bitOffset)) & mask);
    }
    sum += value;
  }
  folly::doNotOptimizeAway(sum);
}

BENCHMARK(copyBits_10bit) { benchCopyBits(g10); }
BENCHMARK_RELATIVE(inline_10bit) { benchInline(g10); }
BENCHMARK_DRAW_LINE();
BENCHMARK(copyBits_16bit) { benchCopyBits(g16); }
BENCHMARK_RELATIVE(inline_16bit) { benchInline(g16); }
BENCHMARK_DRAW_LINE();
BENCHMARK(copyBits_32bit) { benchCopyBits(g32); }
BENCHMARK_RELATIVE(inline_32bit) { benchInline(g32); }
BENCHMARK_DRAW_LINE();
BENCHMARK(copyBits_64bit) { benchCopyBits(g64); }
BENCHMARK_RELATIVE(inline_64bit) { benchInline(g64); }

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  g10 = makePackedData(10);
  g16 = makePackedData(16);
  g32 = makePackedData(32);
  g64 = makePackedData(64);
  folly::runBenchmarks();
  return 0;
}
