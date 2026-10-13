import CoreML
import Foundation
import os

/// Signposts and timings for Match: Instruments' os_signpost track, subsystem com.tonehub.ios,
/// category "match" (intervals "frontend", "inference", "render", "refine", "match").
let matchLog = Logger(subsystem: "com.tonehub.ios", category: "match")
let matchSignposter = OSSignposter(logger: matchLog)

/// ToneHub's tone encoder ("the ears") on Core ML: ToneEncoder.mlpackage, trained and exported by
/// the bench (bench/train_encoder.py, bench/export_coreml.py; Xcode compiles it to ToneEncoder.mlmodelc).
/// Checkpoint ckpt-019530 (bench best.pt); the export matches PyTorch within 2e-7. Feed it the
/// capture as is: source separation (Demucs) before the encoder made it worse on the bench
/// (2.11 → 2.61 dB), so there is none.
///
/// Input: a mono 48 kHz clip (0.33 to 10.9 s). `LogMelFrontEnd` turns it into the model's
/// (1, 1, 96, T) log-mel; the model returns
/// - `embedding`: 128-d, L2-normalised. Tones that sound alike are close (cosine), whatever notes
///   are played; `ToneEncoderRunner.toneDB(distance:)` maps the distance to the bench's tone score.
/// - `knobs`: 7 values in 0...1 for the bench's gray-box amp (`ModelKnobs`).
///
/// Threading: everything here blocks and allocates. Background threads only (never the main
/// thread, never the audio thread). Calls are serialised by a lock. Runs on the CPU in float32,
/// which is what the export was checked against (the Neural Engine would run it in float16).
final class ToneEncoderRunner: @unchecked Sendable {
    static let shared = ToneEncoderRunner()

    struct Output: Sendable {
        let embedding: [Float]
        let knobs: [Float]
    }

    private let lock = NSLock()
    private var model: MLModel?
    private var loadFailed = false
    /// Why the last load or run failed, for Match's diagnostics (`MatchIssue`). Nil after a run
    /// that worked.
    private var _lastError: String?
    var lastError: String? {
        lock.lock()
        defer { lock.unlock() }
        return _lastError
    }

    /// Bench calibration (bench/train_encoder.py `Head`, from best.pt): the bench's tone_db
    /// (multi-resolution STFT error, dB) ≈ slope * (1 - cos) + intercept. Exported to
    /// ToneHubTests/Fixtures/tone_encoder_reference.json ("calibration").
    static let calibrationSlope = 3.5309970378875732
    static let calibrationIntercept = -0.4979875385761261

    /// The estimated tone difference in dB for an embedding distance (1 - cosine), never below 0.
    static func toneDB(distance: Double) -> Double {
        max(0, calibratedDB(distance: distance))
    }

    /// The calibration line itself, below 0 for very close embeddings (1 - cos < ~0.14). For
    /// ranking candidates against each other (`ToneMatchScorer`): the clamped `toneDB` ties them
    /// all at 0 there.
    static func calibratedDB(distance: Double) -> Double {
        calibrationSlope * distance + calibrationIntercept
    }

    /// Loads the compiled model once. False when it is missing or fails to load (Match then keeps
    /// its sample settings). Background thread.
    func loadIfNeeded() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return loadLocked() != nil
    }

    private func loadLocked() -> MLModel? {
        if let model { return model }
        if loadFailed { return nil }
        let state = matchSignposter.beginInterval("load")
        defer { matchSignposter.endInterval("load", state) }
        guard let url = Bundle.main.url(forResource: "ToneEncoder", withExtension: "mlmodelc") else {
            matchLog.error("ToneEncoder.mlmodelc is not in the bundle")
            _lastError = "ToneEncoder.mlmodelc is not in the bundle"
            loadFailed = true
            return nil
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        do {
            let loaded = try MLModel(contentsOf: url, configuration: config)
            model = loaded
            return loaded
        } catch {
            matchLog.error("ToneEncoder failed to load: \(error.localizedDescription, privacy: .public)")
            _lastError = "ToneEncoder failed to load: \(error.localizedDescription)"
            loadFailed = true
            return nil
        }
    }

    /// Front end + model for one clip (mono, 48 kHz, `LogMelFrontEnd.minSamples...maxSamples`).
    /// Nil if the model is unavailable or the clip is out of range.
    func run(_ samples48k: [Float]) -> Output? {
        let clock = ContinuousClock()
        let frontStart = clock.now
        let frontState = matchSignposter.beginInterval("frontend")
        let mel = LogMelFrontEnd.logMel(samples48k)
        matchSignposter.endInterval("frontend", frontState)
        let frontTime = clock.now - frontStart

        lock.lock()
        defer { lock.unlock() }
        guard let mel else {
            _lastError = "front end refused the clip (\(samples48k.count) samples at 48 kHz)"
            return nil
        }
        guard let model = loadLocked() else { return nil }
        let inferStart = clock.now
        let inferState = matchSignposter.beginInterval("inference")
        defer { matchSignposter.endInterval("inference", inferState) }
        do {
            let input = try MLMultiArray(shape: [1, 1, NSNumber(value: LogMelFrontEnd.nMels), NSNumber(value: mel.frames)],
                                         dataType: .float32)
            let dst = input.dataPointer.bindMemory(to: Float.self, capacity: mel.values.count)
            mel.values.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: $0.count) }
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["logmel": input]))
            guard let e = out.featureValue(for: "embedding")?.multiArrayValue,
                  let k = out.featureValue(for: "knobs")?.multiArrayValue, e.count == 128, k.count == 7 else {
                _lastError = "ToneEncoder returned unexpected outputs"
                return nil
            }
            let result = Output(embedding: (0..<e.count).map { e[$0].floatValue },
                                knobs: (0..<k.count).map { k[$0].floatValue })
            let inferTime = clock.now - inferStart
            _lastError = nil
            matchLog.debug("encoder: \(mel.frames) frames, front end \(frontTime.inMilliseconds, format: .fixed(precision: 1)) ms, inference \(inferTime.inMilliseconds, format: .fixed(precision: 1)) ms")
            return result
        } catch {
            matchLog.error("ToneEncoder prediction failed: \(error.localizedDescription, privacy: .public)")
            _lastError = "ToneEncoder prediction failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// 1 - cosine of two L2-normalised embeddings (0 = same tone, up to 2).
    static func distance(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count else { return 2 }
        var dot = 0.0
        for i in a.indices { dot += Double(a[i]) * Double(b[i]) }
        return 1 - dot
    }
}

extension Duration {
    var inMilliseconds: Double {
        let c = components
        return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
    }
}
