import Accelerate
import Foundation

/// The tone encoder's front end, the Swift twin of `tonebench.encoder.LogMel` (bench/tonebench/
/// encoder.py; the export is described in bench/export_coreml.py). The Core ML model
/// (ToneEncoder.mlpackage) starts after this step, so it has to match PyTorch closely:
///
/// 1. mono 48 kHz (callers resample first, `ToneAudio.resample(_:from:)`), RMS-normalised
///    (`x / max(rms, 1e-6)`);
/// 2. STFT: n_fft 2048, hop 512, periodic Hann (`torch.hann_window`), centred with reflect padding of
///    1024 on both sides (`torch.stft(center=True)`), so T = 1 + samples / 512 frames;
/// 3. power spectrum |X|^2 (1025 bins);
/// 4. 96 triangular HTK-mel bands, 30 Hz to 20 kHz (`melMatrix`, built exactly like
///    `encoder.mel_matrix()`: float64 maths, stored as float32);
/// 5. `10 * log10(mel + 1e-10)`, minus the mean over all bands and frames, divided by 20.
///
/// Output layout is [band][frame] (row-major 96 x T), i.e. the C order of the model's (1, 1, 96, T)
/// input. Pure Accelerate and Foundation; no Core ML here, so the unit tests compile it alone
/// (ToneHubTests/ToneEncoderFrontEndTests.swift checks it against a PyTorch fixture).
///
/// Never call this on the audio thread: it allocates.
enum LogMelFrontEnd {
    static let sampleRate = 48_000.0
    static let nFFT = 2048
    static let hop = 512
    static let nMels = 96
    static let nBins = nFFT / 2 + 1
    static let fMin = 30.0
    static let fMax = 20_000.0
    /// The model's frame range (Core ML RangeDim 32...1024).
    static let minFrames = 32
    static let maxFrames = 1024

    /// Frames for a clip of `samples` (centred STFT): 1 + samples / hop.
    static func frameCount(samples: Int) -> Int { 1 + samples / hop }

    /// Longest clip the model takes: (maxFrames - 1) * hop + hop - 1 samples (about 10.9 s).
    static var maxSamples: Int { maxFrames * hop - 1 }
    /// Shortest clip the model takes (about 0.33 s); also > n_fft / 2, which reflect padding needs.
    static var minSamples: Int { (minFrames - 1) * hop }

    // MARK: Constants

    /// Periodic Hann: 0.5 - 0.5 cos(2 pi n / N), n = 0..<N (torch.hann_window(2048)).
    static let window: [Float] = (0..<nFFT).map { n in
        Float(0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(nFFT)))
    }

    /// (nMels x nBins) row-major, float32. Same construction as `encoder.mel_matrix()`:
    /// hz2mel(f) = 2595 log10(1 + f / 700); 98 points evenly spaced in mel (numpy linspace, last point
    /// exactly fMax); bin frequencies k * sr / n_fft; weight = max(0, min(rising, falling)).
    static let melMatrix: [Float] = {
        func hz2mel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func mel2hz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let m0 = hz2mel(fMin), m1 = hz2mel(fMax)
        let count = nMels + 2
        let step = (m1 - m0) / Double(count - 1)
        let points: [Double] = (0..<count).map { i in
            mel2hz(i == count - 1 ? m1 : m0 + Double(i) * step)
        }
        var fb = [Float](repeating: 0, count: nMels * nBins)
        for i in 0..<nMels {
            let lo = points[i], c = points[i + 1], hi = points[i + 2]
            for k in 0..<nBins {
                let f = Double(k) * sampleRate / Double(nFFT)
                let w = min((f - lo) / (c - lo), (hi - f) / (hi - c))
                if w > 0 { fb[i * nBins + k] = Float(w) }
            }
        }
        return fb
    }()

    // MARK: Front end

    /// The model input for a mono 48 kHz clip: (values [band][frame], frames). Nil when the clip
    /// is outside the model's range (shorter than `minSamples` or longer than `maxSamples`) or silent.
    static func logMel(_ samples: [Float]) -> (values: [Float], frames: Int)? {
        guard samples.count >= minSamples, samples.count <= maxSamples else { return nil }
        guard var db = melDB(samples) else { return nil }
        let count = db.values.count
        var mean: Float = 0
        vDSP_meanv(db.values, 1, &mean, vDSP_Length(count))
        var shift = -mean
        var scale: Float = 1.0 / 20.0
        // (x - mean) / 20
        vDSP_vsadd(db.values, 1, &shift, &db.values, 1, vDSP_Length(count))
        vDSP_vsmul(db.values, 1, &scale, &db.values, 1, vDSP_Length(count))
        return db
    }

    /// Steps 1 to 4 plus `10 log10(. + 1e-10)`, before the mean is taken out: [band][frame] dB of an
    /// RMS-normalised clip. Also used for the measured play-along distance (`ToneDistance.melDB`).
    static func melDB(_ samples: [Float]) -> (values: [Float], frames: Int)? {
        let n = samples.count
        guard n > nFFT / 2 else { return nil }

        // 1. RMS normalisation (float32 result, accumulated in double).
        var sumSquares = 0.0
        for s in samples { sumSquares += Double(s) * Double(s) }
        let rms = max(Float((sumSquares / Double(n)).squareRoot()), 1e-6)

        // 2. Reflect padding (numpy/torch "reflect": the edge sample is not repeated).
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        for j in 0..<padded.count {
            var idx = j - pad
            if idx < 0 { idx = -idx }
            if idx >= n { idx = 2 * (n - 1) - idx }
            padded[j] = samples[idx] / rms
        }

        let frames = frameCount(samples: n)
        let half = nFFT / 2
        guard let fft = vDSP.FFT(log2n: vDSP_Length(11), radix: .radix2, ofType: DSPSplitComplex.self) else { return nil }

        // Power, [bin][frame] so the mel product is one matrix multiply.
        var power = [Float](repeating: 0, count: nBins * frames)
        var frame = [Float](repeating: 0, count: nFFT)
        var real = [Float](repeating: 0, count: half)
        var imag = [Float](repeating: 0, count: half)
        padded.withUnsafeBufferPointer { src in
            window.withUnsafeBufferPointer { win in
                for f in 0..<frames {
                    vDSP_vmul(src.baseAddress! + f * hop, 1, win.baseAddress!, 1, &frame, 1, vDSP_Length(nFFT))
                    real.withUnsafeMutableBufferPointer { rp in
                        imag.withUnsafeMutableBufferPointer { ip in
                            var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                            frame.withUnsafeBufferPointer { fp in
                                fp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                                    vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                                }
                            }
                            fft.forward(input: split, output: &split)
                            // vDSP's real FFT returns 2x the DFT; DC in realp[0], Nyquist in imagp[0].
                            let dc = rp[0] * 0.5, nyquist = ip[0] * 0.5
                            power[f] = dc * dc
                            power[half * frames + f] = nyquist * nyquist
                            for k in 1..<half {
                                let re = rp[k] * 0.5, im = ip[k] * 0.5
                                power[k * frames + f] = re * re + im * im
                            }
                        }
                    }
                }
            }
        }

        // 4. Mel bands: (nMels x nBins) x (nBins x frames).
        var mel = [Float](repeating: 0, count: nMels * frames)
        vDSP_mmul(melMatrix, 1, power, 1, &mel, 1, vDSP_Length(nMels), vDSP_Length(frames), vDSP_Length(nBins))

        // 10 log10(x + 1e-10)
        var floor: Float = 1e-10
        vDSP_vsadd(mel, 1, &floor, &mel, 1, vDSP_Length(mel.count))
        var count = Int32(mel.count)
        mel.withUnsafeMutableBufferPointer { buf in
            vvlog10f(buf.baseAddress!, buf.baseAddress!, &count)
        }
        var ten: Float = 10
        vDSP_vsmul(mel, 1, &ten, &mel, 1, vDSP_Length(mel.count))
        return (mel, frames)
    }
}

/// Distances between two clips, on the front end's mel bands.
enum ToneDistance {
    /// Measured log-mel distance between two time-aligned mono 48 kHz clips, in dB: each clip
    /// RMS-normalised, 96 mel bands per frame (the encoder's front end), both floored 80 dB under
    /// their joint peak, mean absolute difference per frame, then the mean of the best 80 % of
    /// frames (so a few wrong notes or a misaligned stretch don't dominate). The bench's `mel_db`
    /// with `mrstft_trimmed_db`'s trimming. 0 = identical. Nil when either clip is too short.
    static func melDB(_ a: [Float], _ b: [Float], keep: Double = 0.8, rangeDB: Float = 80) -> Double? {
        let n = min(a.count, b.count)
        guard n > LogMelFrontEnd.nFFT,
              let la = LogMelFrontEnd.melDB(Array(a.prefix(n))),
              let lb = LogMelFrontEnd.melDB(Array(b.prefix(n))) else { return nil }
        return melDB(la, lb, keep: keep, rangeDB: rangeDB)
    }

    /// The same on two `LogMelFrontEnd.melDB` matrices (so a song's spectrum can be kept instead of
    /// its audio). Frames beyond the shorter one are ignored.
    static func melDB(_ la: (values: [Float], frames: Int), _ lb: (values: [Float], frames: Int),
                      keep: Double = 0.8, rangeDB: Float = 80) -> Double? {
        let frames = min(la.frames, lb.frames)
        let bands = LogMelFrontEnd.nMels
        guard frames > 0, la.values.count == bands * la.frames, lb.values.count == bands * lb.frames else { return nil }
        let floor = max(la.values.max() ?? 0, lb.values.max() ?? 0) - rangeDB
        var perFrame = [Double](repeating: 0, count: frames)
        for band in 0..<bands {
            for f in 0..<frames {
                let va = la.values[band * la.frames + f], vb = lb.values[band * lb.frames + f]
                perFrame[f] += Double(abs(max(va, floor) - max(vb, floor)))
            }
        }
        let sorted = perFrame.map { $0 / Double(bands) }.sorted()
        let kept = max(1, Int(keep * Double(sorted.count)))
        return sorted.prefix(kept).reduce(0, +) / Double(kept)
    }
}
