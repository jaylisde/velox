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

#include <folly/Varint.h>

namespace facebook::velox::parquet {

bool DeltaBpDecoder::getVlqInt(uint64_t& v) {
  uint64_t tmp = 0;
  for (int i = 0; i < folly::kMaxVarintLength64; i++) {
    uint8_t byte = *(bufferStart_++);
    tmp |= static_cast<uint64_t>(byte & 0x7F) << (7 * i);
    if ((byte & 0x80) == 0) {
      v = tmp;
      return true;
    }
  }
  return false;
}

bool DeltaBpDecoder::getZigZagVlqInt(int64_t& v) {
  uint64_t u;
  if (!getVlqInt(u)) {
    return false;
  }
  v = (u >> 1) ^ (~(u & 1) + 1);
  return true;
}

void DeltaBpDecoder::initHeader() {
  if (!getVlqInt(valuesPerBlock_) || !getVlqInt(miniBlocksPerBlock_) ||
      !getVlqInt(totalValueCount_) || !getZigZagVlqInt(lastValue_)) {
    VELOX_FAIL("initHeader EOF");
  }

  VELOX_CHECK_GT(valuesPerBlock_, 0, "cannot have zero value per block");
  VELOX_CHECK_EQ(
      valuesPerBlock_ % 128,
      0,
      "the number of values in a block must be multiple of 128, but it's {}",
      valuesPerBlock_);
  VELOX_CHECK_GT(
      miniBlocksPerBlock_, 0, "cannot have zero miniblock per block");
  valuesPerMiniBlock_ = valuesPerBlock_ / miniBlocksPerBlock_;
  VELOX_CHECK_GT(
      valuesPerMiniBlock_, 0, "cannot have zero value per miniblock");
  VELOX_CHECK_EQ(
      valuesPerMiniBlock_ % 32,
      0,
      "the number of values in a miniblock must be multiple of 32, but it's {}",
      valuesPerMiniBlock_);

  totalValuesRemaining_ = totalValueCount_;
  deltaBitWidths_.resize(miniBlocksPerBlock_);
  firstBlockInitialized_ = false;
  valuesRemainingCurrentMiniBlock_ = 0;
}

void DeltaBpDecoder::initBlock() {
  VELOX_DCHECK_GT(totalValuesRemaining_, 0, "initBlock called at EOF");

  if (!getZigZagVlqInt(minDelta_)) {
    VELOX_FAIL("initBlock EOF");
  }

  // read the bitwidth of each miniblock
  for (uint32_t i = 0; i < miniBlocksPerBlock_; ++i) {
    deltaBitWidths_[i] = *(bufferStart_++);
    // Note that non-conformant bitwidth entries are allowed by the Parquet
    // spec for extraneous miniblocks in the last block (GH-14923), so we
    // check the bitwidths when actually using them (see initMiniBlock()).
  }

  miniBlockIdx_ = 0;
  firstBlockInitialized_ = true;
  initMiniBlock(deltaBitWidths_[0]);
}

void DeltaBpDecoder::initMiniBlock(int32_t bitWidth) {
  VELOX_DCHECK_LE(
      bitWidth,
      kMaxDeltaBitWidth,
      "delta bit width larger than integer bit width");
  deltaBitWidth_ = bitWidth;
  valuesRemainingCurrentMiniBlock_ = valuesPerMiniBlock_;
}

} // namespace facebook::velox::parquet
