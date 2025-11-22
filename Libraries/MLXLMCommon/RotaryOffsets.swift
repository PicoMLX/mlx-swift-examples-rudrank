import Foundation
import MLX

public enum RotaryOffsets {
    case uniform(Int)
    case perSequence([Int])
}

public func makeRotaryOffsets(cache: KVCache?, batchSize: Int) -> RotaryOffsets? {
    guard batchSize > 0, let cache else { return nil }

    if let batchCache = cache as? BatchKVCache {
        return .perSequence(extractOffsets(
            offsets: batchCache.offsets,
            expectedCount: batchSize,
            fallback: cache.offset
        ))
    }

    if let rotatingCache = cache as? BatchRotatingKVCache {
        return .perSequence(extractOffsets(
            offsets: rotatingCache.offsets,
            expectedCount: batchSize,
            fallback: cache.offset
        ))
    }

    return .uniform(cache.offset)
}

public func applyRotaryEmbedding(
    _ tensor: MLXArray,
    offsets: RotaryOffsets?,
    base: (MLXArray) -> MLXArray,
    withOffset: (MLXArray, Int) -> MLXArray
) -> MLXArray {
    guard let offsets else {
        return base(tensor)
    }

    switch offsets {
    case .uniform(let value):
        return withOffset(tensor, value)
    case .perSequence(var values):
        let batchSize = tensor.dim(0)
        if values.count != batchSize {
            values = normalizeOffsets(values, expectedCount: batchSize, fillValue: values.last ?? 0)
        }

        if values.isEmpty {
            return tensor
        }

        var slices: [MLXArray] = []
        slices.reserveCapacity(values.count)

        for (index, offset) in values.enumerated() {
            let slice = tensor[index ..< (index + 1), .ellipsis]
            slices.append(withOffset(slice, offset))
        }

        return MLX.concatenated(slices, axis: 0)
    }
}

private func extractOffsets(offsets: MLXArray, expectedCount: Int, fallback: Int) -> [Int] {
    if offsets.size == 0 {
        return Array(repeating: fallback, count: expectedCount)
    }

    let converted = offsets.asType(.int32).asArray(Int32.self).map(Int.init)
    return normalizeOffsets(converted, expectedCount: expectedCount, fillValue: fallback)
}

private func normalizeOffsets(_ offsets: [Int], expectedCount: Int, fillValue: Int) -> [Int] {
    guard expectedCount > 0 else { return [] }
    if offsets.count == expectedCount { return offsets }
    if offsets.count > expectedCount {
        return Array(offsets.prefix(expectedCount))
    }
    var result = offsets
    result.append(contentsOf: Array(repeating: fillValue, count: expectedCount - offsets.count))
    return result
}
