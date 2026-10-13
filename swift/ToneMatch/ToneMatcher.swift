import AVFoundation
import Foundation

// Match with ToneHub's own tone encoder (ToneEncoderRunner): settings predicted from the captured
// song, an optional short refinement on ToneHub's amp, and an honest 0-100 score. Everything here
// runs on a background task (ListenController's analysis task), never on the main or audio thread.
//
// What the bench found (bench/README.md, bench/results/overnight/tables.md): through a phone mic in
// a room the encoder clearly beats the old AFx-Rep search (real-recording closeness ~70 vs ~49); on
// clean studio recordings AFx-Rep search is slightly better; source separation doesn't help. Listen
// mode is the phone-mic case, so the encoder is what Match uses.

// MARK: - The model's knobs

/// The encoder's 7 knob estimates, 0...1, in `tonebench.amp.KNOBS` order. They are settings of the
/// bench's gray-box amp (bench/tonebench/amp.py), not of any app amp:
///
/// | knob      | bench amp                                                   |
/// |-----------|-------------------------------------------------------------|
/// | gain      | preamp drive into an asymmetric soft clip, +0 ... +48 dB     |
/// | bass      | low shelf 110 Hz, ±12 dB (after the preamp clip)            |
/// | mid       | peak 650 Hz, Q 0.7, ±12 dB                                  |
/// | treble    | high shelf 2.8 kHz, ±12 dB                                  |
/// | presence  | high shelf 4.5 kHz after the power-amp clip, ±8 dB          |
/// | cab_tone  | cab low-pass corner, 2.5 kHz (dark) ... 7.5 kHz (bright)    |
/// | cab_res   | cab low resonance at 110 Hz, 0 ... +9 dB                     |
///
/// How they reach the app (`MockToneSettings(model:)`), knob by knob:
/// - **gain → Gain**, dial = 10 x gain. The bench knob spans 48 dB of drive; ToneHub's Gain dial
///   spans 48 dB of input gain (-24 ... +24 dB, `AudioEngineController.inputGainKnob`), so the dial
///   moves the same number of dB. The absolute drive differs (a NAM capture has its own input
///   sensitivity) and on the user's amp it is only "how far round the gain knob", so it is a
///   starting point, not a calibration.
/// - **bass / mid / treble → Bass / Mid / Treble**, dial = 10 x knob. ToneHub's tone stack
///   (GTRToneStack.h) has the same ranges and nearly the same filters: low shelf 120 Hz vs 110,
///   peak 650 Hz Q 0.7 (identical), high shelf 3.2 kHz vs 2.8 kHz, all ±12 dB. One real difference:
///   the bench puts the stack between the preamp and power-amp clippers; ToneHub's sits after the
///   amp model. On the user's own amp, when its tone stack is in the catalog (gear.json `tone`),
///   `AmpTranslator` finds the knob positions whose curve on that amp's circuit best reproduces
///   this EQ (and maps the gain into the chosen channel's range); other amps fall back to the knobs
///   with those names (`setting(forAmpKnob:)`).
/// - **presence → Presence**, dial = 10 x presence, on amps that have a Presence knob. ToneHub's amp
///   has none, so there it is not used.
/// - **cab_tone, cab_res**: no amp control corresponds (they are the speaker cabinet). They are not
///   turned into knobs. The score's render uses them as a stand-in cab (`CabStandIn`), so the
///   score describes the amp settings through a cab like the one the model heard.
/// - Boost: the model says nothing about pedals, so no boost is suggested (pedals then follow the
///   existing rules from gain alone).
struct ModelKnobs: Codable, Hashable, Sendable {
    var gain: Double
    var bass: Double
    var mid: Double
    var treble: Double
    var presence: Double
    var cabTone: Double
    var cabRes: Double

    init?(_ v: [Float]) {
        guard v.count == 7, v.allSatisfy({ $0.isFinite }) else { return nil }
        let c = v.map { min(max(Double($0), 0), 1) }
        gain = c[0]; bass = c[1]; mid = c[2]; treble = c[3]; presence = c[4]; cabTone = c[5]; cabRes = c[6]
    }
}

extension MockToneSettings {
    /// Settings from the encoder's knobs (see `ModelKnobs` for the mapping and its limits).
    init(model k: ModelKnobs) {
        self.init(gain: 10 * k.gain, bass: 10 * k.bass, mid: 10 * k.mid, treble: 10 * k.treble,
                  presence: 10 * k.presence, boost: false)
    }

    /// Every dial on the half steps the result shows, so a score is for exactly the knobs on screen.
    var snapped: MockToneSettings {
        func s(_ v: Double) -> Double { (min(max(v, 0), 10) * 2).rounded() / 2 }
        return MockToneSettings(gain: s(gain), bass: s(bass), mid: s(mid), treble: s(treble),
                                presence: s(presence), boost: boost)
    }
}

// MARK: - Score

/// The match score: 100 x exp(-dB / 6), dB being a tone distance (6 dB ≈ 37). The bench's
/// "closeness" uses the same formula on its tone_db (≥ 75 close, 50-75 recognizable, < 50 off).
/// It is a relabelled spectral error, not a listening test. The card shows only a plain verdict
/// (`verdict`); the number lives in Show details (`detailText`).
///
/// Listen (mic only) scores are model estimates, and their scale is narrow: the calibration
/// can barely go under ~50, two unrelated tones read ~60, and a matcher that ignores the song
/// already reaches 54-75 on the bench (bugs-8 L2). So an estimate's verdict is relative: the same
/// scorer rates ToneHub's amp at noon (every knob 5, the target-ignoring baseline) for this very
/// take, and "Close match" needs the match to beat that by `closeMarginDB`. With no baseline an
/// estimate gets no verdict at all.
///
/// Near noon (bugs-9 L1): when the song's own tone sits near noon, noon is already close and no
/// match can beat it by a margin, however good. So when the baseline itself is in the bench's
/// "Close" band (distance ≤ `closeDB`, score ≥ 75), the estimate is also judged on its absolute
/// distance with the bench bands (≤ `closeDB` close, ≤ `nearDB` near), and the better of the two
/// verdicts is shown. A close baseline only ever lifts the verdict, never caps it.
struct ToneMatchScore: Codable, Hashable, Sendable {
    /// 0...100.
    let value: Int
    /// True: the model's estimate (listen mode). False: measured against the song (play-along).
    let isEstimate: Bool
    /// The tone distance behind `value` (dB). Nil on older saves.
    var db: Double? = nil
    /// The same distance for ToneHub's amp at noon on this take (the baseline). Nil when unknown.
    var baselineDB: Double? = nil
    /// The estimate came from an audio file (Match › Use a file), not the mic. Nil on mic matches
    /// and older saves.
    var fromFile: Bool? = nil

    /// Better than the noon baseline by this much: "Close match". By `nearMarginDB`: "Near".
    static let closeMarginDB = 1.5
    static let nearMarginDB = 0.5
    /// The bench bands as distances: score 75 and 50 (score = 100·e^(-dB/6)).
    static let closeDB = 6 * log(4.0 / 3.0)
    static let nearDB = 6 * log(2.0)

    enum Verdict: String, Sendable {
        case close = "Close match", near = "Near", rough = "Rough"

        /// The pill text, localized (the raw value stays English for logs).
        var title: String {
            switch self {
            case .close: return String(localized: "Close match", comment: "Match verdict: the tone is close to the song")
            case .near: return String(localized: "Near", comment: "Match verdict: the tone is fairly close to the song")
            case .rough: return String(localized: "Rough", comment: "Match verdict: the tone is not close to the song")
            }
        }

        /// Higher is better.
        var rank: Int {
            switch self {
            case .close: return 2
            case .near: return 1
            case .rough: return 0
            }
        }
    }

    /// The plain verdict for the card. Measured: the bench bands (≥ 75, 50-75, < 50). Estimated:
    /// against the baseline (see the type comment); nil without one, never a guess.
    var verdict: Verdict? {
        if !isEstimate {
            return value >= 75 ? .close : (value >= 50 ? .near : .rough)
        }
        guard let db, let baselineDB else { return nil }
        let gain = baselineDB - db
        let relative: Verdict = gain >= Self.closeMarginDB ? .close : (gain >= Self.nearMarginDB ? .near : .rough)
        guard baselineDB <= Self.closeDB else { return relative }
        // Noon is already close for this song: judge the match on its own distance too.
        let absolute: Verdict = db <= Self.closeDB ? .close : (db <= Self.nearDB ? .near : .rough)
        return relative.rank >= absolute.rank ? relative : absolute
    }

    /// Show details: "Score 82 · measured", or for an estimate a plain comparison with no number
    /// ("Estimated from your mic · clearly closer than a flat amp"). An estimate's number reads
    /// high even for unrelated tones (see the type comment), so a "96" would promise too much.
    var detailText: String {
        guard isEstimate else { return String(localized: "Score \(value) · measured") }
        let source = fromFile == true ? String(localized: "Estimated from the file") : String(localized: "Estimated from your mic")
        switch verdict {
        case .close: return String(localized: "\(source) · clearly closer than a flat amp")
        case .near: return String(localized: "\(source) · a little closer than a flat amp")
        case .rough: return String(localized: "\(source) · not much closer than a flat amp")
        case nil: return source
        }
    }

    init(value: Int, isEstimate: Bool) {
        self.value = value
        self.isEstimate = isEstimate
    }

    /// Nil when the distance isn't a real number (never a made-up score). An estimate never claims 100.
    init?(db: Double, isEstimate: Bool, baselineDB: Double? = nil, fromFile: Bool = false) {
        guard db.isFinite else { return nil }
        self.fromFile = fromFile ? true : nil
        let raw = Int((100 * exp(-max(0, db) / 6)).rounded())
        value = min(max(raw, 0), isEstimate ? 99 : 100)
        self.isEstimate = isEstimate
        self.db = db
        self.baselineDB = baselineDB.flatMap { $0.isFinite ? $0 : nil }
    }
}

// MARK: - Why there is no score

/// Why a listen match has no score, or a score but no verdict. Recorded on every path that drops
/// them, logged (match log) and shown as one plain line in the result's Show details.
enum MatchIssue: String, Sendable {
    /// The take was too quiet to match (`ListenLevels.isLoudEnough`); the encoder never ran.
    case tooQuiet
    /// Too little audio for the encoder (under its minimum length).
    case tooShort
    /// The analysis was stopped before it finished (the flow closed or restarted).
    case cancelled
    /// The tone encoder isn't bundled, didn't load, or failed on the clip.
    case encoderFailed
    /// The reference DI for scoring couldn't be made.
    case referenceFailed
    /// ToneHub's amp couldn't be rendered (or the render couldn't be encoded) to compare.
    case renderFailed
    /// A score, but no neutral (noon) baseline to judge it against: no verdict.
    case noBaseline

    /// One short line for everyone (Show details).
    var plain: String {
        switch self {
        case .tooQuiet: return String(localized: "No score: couldn't hear enough of the song")
        case .tooShort: return String(localized: "No score: the take was too short")
        case .cancelled: return String(localized: "No score: the match stopped before it finished")
        case .encoderFailed: return String(localized: "No score: the tone model didn't run")
        case .referenceFailed, .renderFailed: return String(localized: "No score: couldn't play the ToneHub amp to compare")
        case .noBaseline: return String(localized: "No verdict: couldn't compare it with a neutral tone")
        }
    }
}

/// Everything one listen match did, for the DEBUG / Labs "Debug" panel on the result and the
/// match log. Numbers only, never audio.
struct MatchTrace: Sendable {
    /// The encoder clip's length (the loudest stretch of the take, 48 kHz).
    var clipSeconds: Double?
    var encoderRan = false
    var encoderMs: Double?
    var knobs: ModelKnobs?
    var predicted: MockToneSettings?
    var refined: MockToneSettings?
    var renders = 0
    /// How the ToneHub-amp settings were searched ("greedy", "cma-es 6 gen"). Nil before a search.
    var search: String?
    /// Which built-in amp the match picked and what each candidate scored at the prediction
    /// ("Lead 100 · ladder 3.10, Lead 100 2.41 dB"). Nil before the pick.
    var amp: String?
    var distanceDB: Double?
    var baselineDB: Double?
    var score: ToneMatchScore?
    var matchMs: Double?
    var issue: MatchIssue?
    /// The technical reason behind `issue` (DEBUG / Labs only on screen; always logged).
    var technical: String?

    mutating func fail(_ issue: MatchIssue, _ technical: String?) {
        self.issue = issue
        self.technical = technical
        matchLog.error("match: \(issue.rawValue, privacy: .public)\(technical.map { " (\($0))" } ?? "", privacy: .public)")
    }
}

// MARK: - Result

/// What Match got from the encoder for one capture.
struct ToneMatchResult: Sendable {
    enum Mode: String, Codable, Sendable {
        /// Mic only (Listen): the song's sound, no DI. The score is the model's estimate.
        case listen
        /// The user's DI time-aligned with the song: the score is measured.
        case playAlong
        /// Clean audio from a file (Match › Use a file): the song's sound, no DI, no room. The
        /// score is the model's estimate; the search is CMA-ES from the encoder's prediction.
        case file
    }

    let mode: Mode
    let knobs: ModelKnobs
    /// The knob head's settings (the user's own amp: we can't render it, so no search).
    let predicted: MockToneSettings
    /// Settings for ToneHub's amp: the prediction, refined by a short search when there was time.
    let toneHub: MockToneSettings
    /// True when the search moved at least one ToneHub knob.
    let refined: Bool
    /// The score of `toneHub` (un-nudged); nil when it couldn't be computed.
    let score: ToneMatchScore?
    /// Re-scores other settings (after a nudge). Nil when no score can be computed.
    let scorer: ToneMatchScorer?
    /// The Listen take's analysed clip, held in memory for the result's "Song vs yours" A/B and
    /// wiped when the result closes (ListenController). Nil for play-along and once wiped.
    var heard: HeardClip? = nil
    /// Why `score` is nil, or why it has no verdict (`MatchIssue.noBaseline`). Nil when complete.
    var issue: MatchIssue? = nil
    /// The built-in amp `toneHub` is for (`ToneMatchScorer.pickAmp`): one of the cab-baked
    /// captures, or nil for the amp-only gain ladder (`ToneHubAmp.closest(toGain:)`, as before).
    /// Hear it, the verdict, "Try on the amp" and Save all use it.
    var builtInAmp: ToneHubAmp? = nil

    func settings(onToneHubAmp: Bool) -> MockToneSettings { onToneHubAmp ? toneHub : predicted }
}

// MARK: - Matcher

enum ToneMatcher {
    /// The listen clip the encoder reads: the loudest stretch of this length (where the band, and
    /// usually the guitar, is fully in), within the model's 10.9 s limit.
    static let clipSeconds = 10.0
    /// Wall-clock budget for one match, prediction + refinement + score (the match moment's knobs
    /// keep searching meanwhile).
    static let budget: Duration = .seconds(3)

    /// True when the encoder is bundled and loads (else Match keeps its sample settings).
    static var isAvailable: Bool { ToneEncoderRunner.shared.loadIfNeeded() }

    /// Copies the loudest `clipSeconds` of a capture and resamples it to 48 kHz mono. The copy is the
    /// caller's: ListenController hands it to the result as a `HeardClip` (memory only, wiped when
    /// the result closes) or wipes it (`wipe(_:)`). Nil when the clip is too short.
    static func listenClip(_ samples: UnsafeBufferPointer<Float>, sampleRate: Double) -> [Float]? {
        guard sampleRate > 0, let base = samples.baseAddress else { return nil }
        let count = samples.count
        let length = min(count, Int(clipSeconds * sampleRate))
        guard Double(length) / sampleRate * LogMelFrontEnd.sampleRate >= Double(LogMelFrontEnd.minSamples) else { return nil }
        // Loudest window, in quarter-second steps (prefix sums of squares).
        let step = max(1, Int(0.25 * sampleRate))
        var bestStart = 0
        if length < count {
            var prefix = [Double](repeating: 0, count: count / step + 2)
            var acc = 0.0
            var b = 0
            for i in 0..<count {
                if i % step == 0 { prefix[b] = acc; b += 1 }
                acc += Double(base[i]) * Double(base[i])
            }
            prefix[b] = acc
            let blocks = length / step
            var best = -1.0
            var start = 0
            while start + blocks < b {
                let e = prefix[start + blocks] - prefix[start]
                if e > best { best = e; bestStart = start * step }
                start += 1
            }
        }
        var window = Array(UnsafeBufferPointer(start: base + bestStart, count: min(length, count - bestStart)))
        guard var clip = ToneAudio.resample(window, from: sampleRate) else {
            wipe(&window)
            return nil
        }
        if abs(sampleRate - LogMelFrontEnd.sampleRate) >= 0.5 { wipe(&window) } // else `clip` is this buffer
        if clip.count > LogMelFrontEnd.maxSamples { clip.removeLast(clip.count - LogMelFrontEnd.maxSamples) }
        return clip.count >= LogMelFrontEnd.minSamples ? clip : nil
    }

    /// Zeroes a clip copy (listen audio never outlives the result that shows it).
    static func wipe(_ clip: inout [Float]) {
        clip.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }
        clip = []
    }

    /// Listen (mic only). Encoder on the clip → knobs → settings; then, inside the budget, the amp
    /// pick and a short search on ToneHub's amp by embedding distance between the song and our
    /// render of a real guitar DI (`ReferenceDI.guitar()`), and the estimated score from the same
    /// distance on the pluck synth DI (`ReferenceDI.samples()`). Nil when the model is unavailable.
    /// Background thread.
    static func matchListen(clip48k: [Float], hint: ToneHubAmp.MatchHint? = nil,
                            isCancelled: @escaping @Sendable () -> Bool = { false }) -> ToneMatchResult? {
        var trace = MatchTrace()
        return matchListen(clip48k: clip48k, hint: hint, isCancelled: isCancelled, trace: &trace)
    }

    /// `matchListen`, recording what happened (and why there is no score) in `trace`. `hint`: the
    /// song's researched rig (Original Rig), when known, for the built-in amp pre-pick.
    static func matchListen(clip48k: [Float], hint: ToneHubAmp.MatchHint? = nil,
                            isCancelled: @escaping @Sendable () -> Bool = { false },
                            trace: inout MatchTrace) -> ToneMatchResult? {
        let clock = ContinuousClock()
        let start = clock.now
        let deadline = start + budget
        let state = matchSignposter.beginInterval("match", "listen")
        defer {
            matchSignposter.endInterval("match", state)
            trace.matchMs = (clock.now - start).inMilliseconds
        }
        trace.clipSeconds = Double(clip48k.count) / LogMelFrontEnd.sampleRate
        let encoderStart = clock.now
        guard let target = ToneEncoderRunner.shared.run(clip48k), let knobs = ModelKnobs(target.knobs) else {
            trace.fail(.encoderFailed, ToneEncoderRunner.shared.lastError ?? "encoder returned no knobs")
            return nil
        }
        trace.encoderRan = true
        trace.encoderMs = (clock.now - encoderStart).inMilliseconds
        trace.knobs = knobs
        let predicted = MockToneSettings(model: knobs).snapped
        trace.predicted = predicted
        guard let reference = ReferenceDI.guitar(), let scoreReference = ReferenceDI.samples() else {
            trace.fail(.referenceFailed, "ReferenceDI returned nil (preview phrases or pluck synth)")
            return ToneMatchResult(mode: .listen, knobs: knobs, predicted: predicted, toneHub: predicted,
                                   refined: false, score: nil, scorer: nil, issue: .referenceFailed)
        }
        guard !isCancelled() else {
            trace.fail(.cancelled, "cancelled after the encoder")
            return nil
        }
        let scorer = ToneMatchScorer(listenTarget: target.embedding, reference: reference,
                                     scoreReference: scoreReference, knobs: knobs)
        return finish(mode: .listen, knobs: knobs, predicted: predicted, scorer: scorer, hint: hint, deadline: deadline,
                      isCancelled: isCancelled, trace: &trace)
    }

    /// Wall-clock budget for a file match: the user picked the part and waits on the match moment,
    /// and clean audio rewards a longer search (bench: pred_then_search ≈ 79 vs the model alone ≈ 76
    /// on clean recordings).
    static let fileBudget: Duration = .seconds(8)

    /// Clean audio from a file (decoded, 48 kHz mono, the loudest ≤ 10 s of the picked part).
    /// Encoder → knobs → settings, then CMA-ES on ToneHub's amp seeded at the encoder's guess
    /// (sigma 2 dial steps, a prior pulling toward the guess), scored like Listen by embedding
    /// distance of the reference DI rendered at each candidate. Nil when the model is unavailable.
    /// Background thread.
    static func matchFile(clip48k: [Float], hint: ToneHubAmp.MatchHint? = nil,
                          isCancelled: @escaping @Sendable () -> Bool = { false },
                          trace: inout MatchTrace) -> ToneMatchResult? {
        let clock = ContinuousClock()
        let start = clock.now
        let deadline = start + fileBudget
        let state = matchSignposter.beginInterval("match", "file")
        defer {
            matchSignposter.endInterval("match", state)
            trace.matchMs = (clock.now - start).inMilliseconds
        }
        trace.clipSeconds = Double(clip48k.count) / LogMelFrontEnd.sampleRate
        let encoderStart = clock.now
        guard let target = ToneEncoderRunner.shared.run(clip48k), let knobs = ModelKnobs(target.knobs) else {
            trace.fail(.encoderFailed, ToneEncoderRunner.shared.lastError ?? "encoder returned no knobs")
            return nil
        }
        trace.encoderRan = true
        trace.encoderMs = (clock.now - encoderStart).inMilliseconds
        trace.knobs = knobs
        let predicted = MockToneSettings(model: knobs).snapped
        trace.predicted = predicted
        guard let reference = ReferenceDI.samples() else {
            trace.fail(.referenceFailed, "ReferenceDI.samples() returned nil (pluck synth render)")
            return ToneMatchResult(mode: .file, knobs: knobs, predicted: predicted, toneHub: predicted,
                                   refined: false, score: nil, scorer: nil, issue: .referenceFailed)
        }
        guard !isCancelled() else {
            trace.fail(.cancelled, "cancelled after the encoder")
            return nil
        }
        let scorer = ToneMatchScorer(listenTarget: target.embedding, reference: reference, knobs: knobs, fromFile: true)
        return finish(mode: .file, knobs: knobs, predicted: predicted, scorer: scorer, hint: hint, deadline: deadline,
                      isCancelled: isCancelled, trace: &trace)
    }

    /// Play-along: the song clip and the user's DI from the same moment, both mono 48 kHz and
    /// time-aligned (the caller aligns them; the recorder's DI and a song capture of the same take).
    /// Encoder on the song → settings; the search and the score use the measured log-mel distance
    /// between the song and the user's DI rendered through ToneHub's amp (`ToneDistance`). Only the
    /// song's mel spectrum is kept, not its audio. Background thread.
    ///
    /// No capture path feeds this yet: Match's only capture today is Listen (the mic), and the
    /// recorder's play-along takes have no song audio next to the DI (Apple Music and YouTube audio
    /// never reach the app). It is here for when a song capture runs alongside the recorder.
    static func matchPlayAlong(song48k: [Float], di48k: [Float],
                               isCancelled: @escaping @Sendable () -> Bool = { false }) -> ToneMatchResult? {
        let clock = ContinuousClock()
        let deadline = clock.now + budget
        let state = matchSignposter.beginInterval("match", "play-along")
        defer { matchSignposter.endInterval("match", state) }
        // Up to 6 s: long enough for a stable score, short enough for several renders in the budget.
        let n = min(song48k.count, di48k.count, Int(6 * LogMelFrontEnd.sampleRate))
        guard n >= LogMelFrontEnd.minSamples else { return nil }
        let song = Array(song48k.prefix(n)), di = Array(di48k.prefix(n))
        guard let target = ToneEncoderRunner.shared.run(song), let knobs = ModelKnobs(target.knobs) else { return nil }
        let predicted = MockToneSettings(model: knobs).snapped
        guard let songMel = LogMelFrontEnd.melDB(song) else {
            return ToneMatchResult(mode: .playAlong, knobs: knobs, predicted: predicted, toneHub: predicted,
                                   refined: false, score: nil, scorer: nil)
        }
        let scorer = ToneMatchScorer(playAlongSongMel: songMel, di: di, knobs: knobs)
        var trace = MatchTrace()
        return finish(mode: .playAlong, knobs: knobs, predicted: predicted, scorer: scorer, hint: nil, deadline: deadline,
                      isCancelled: isCancelled, trace: &trace)
    }

    private static func finish(mode: ToneMatchResult.Mode, knobs: ModelKnobs, predicted: MockToneSettings,
                               scorer: ToneMatchScorer, hint: ToneHubAmp.MatchHint?, deadline: ContinuousClock.Instant,
                               isCancelled: @escaping @Sendable () -> Bool, trace: inout MatchTrace) -> ToneMatchResult {
        // Which built-in amp: the gain ladder and a couple of cab-baked captures picked by gain
        // style and family, each rendered once at the prediction (the ladder's render is the
        // search's first one, so the pick costs at most two extra renders).
        let pick = scorer.pickAmp(predicted: predicted,
                                  candidates: ToneHubAmp.matchCandidates(predictedGain: predicted.gain, hint: hint),
                                  preferred: hint?.familyAmp, deadline: deadline, isCancelled: isCancelled)
        trace.amp = pick.summary
        let refinedSearch: (settings: MockToneSettings, db: Double, renders: Int)?
        if mode == .file {
            let cma = scorer.searchCMA(from: predicted, deadline: deadline, isCancelled: isCancelled)
            refinedSearch = cma.map { ($0.settings, $0.db, $0.renders) }
            trace.search = cma.map { "cma-es \($0.generations) gen" } ?? "cma-es (no render)"
        } else {
            refinedSearch = scorer.refine(from: predicted, deadline: deadline, isCancelled: isCancelled)
            trace.search = "greedy"
        }
        let toneHub = refinedSearch?.settings ?? predicted
        trace.refined = refinedSearch?.settings
        trace.renders = refinedSearch?.renders ?? 0
        // The score's distance: the search's own when it searched on the scoring reference, else one
        // more render (Listen searches on the guitar reference, scores on the synth one).
        let scoredDB = refinedSearch.flatMap { scorer.scoreDB($0.settings, searched: $0.db) }
        trace.distanceDB = scoredDB
        // The baseline (noon) for a listen verdict: one more render, skipped when cancelled.
        let cancelledBeforeBaseline = isCancelled()
        let estimate = mode != .playAlong
        let baseline = estimate && !cancelledBeforeBaseline ? scorer.baselineDB() : nil
        trace.baselineDB = baseline
        let score = scoredDB.flatMap { ToneMatchScore(db: $0, isEstimate: estimate, baselineDB: baseline, fromFile: mode == .file) }
        trace.score = score
        var issue: MatchIssue?
        if refinedSearch == nil {
            issue = .renderFailed
            trace.fail(.renderFailed, scorer.lastError ?? "the first render of the predicted settings failed")
        } else if score == nil {
            issue = .renderFailed
            trace.fail(.renderFailed, "distance was not a finite number")
        } else if estimate, baseline == nil {
            issue = .noBaseline
            trace.fail(.noBaseline, cancelledBeforeBaseline ? "baseline skipped: cancelled"
                                                         : (scorer.lastError ?? "baseline (noon) render failed"))
        }
        if let score {
            matchLog.notice("match: score \(score.value) dB \(score.db ?? .nan, format: .fixed(precision: 2)) baseline \(baseline ?? .nan, format: .fixed(precision: 2)) verdict \(score.verdict?.rawValue ?? "none", privacy: .public)")
        }
        return ToneMatchResult(mode: mode, knobs: knobs, predicted: predicted, toneHub: toneHub,
                               refined: refinedSearch.map { $0.settings != predicted } ?? false,
                               score: score, scorer: score == nil ? nil : scorer, issue: issue,
                               builtInAmp: pick.amp)
    }
}

// MARK: - Scorer

/// Renders settings through ToneHub's amp (a private NAM model copy, the tone stack, and the
/// stand-in cab unless the capture has its own) and measures how far that is from the song. The
/// amp is the one `pickAmp` chose (nil: the gain ladder). Owns its model copies; one call at a time
/// (lock). Background threads only.
final class ToneMatchScorer: @unchecked Sendable {
    private enum Target {
        /// Song embedding; candidates render the reference DI (the pick and the search) and the score
        /// reference (the score, its baseline and re-scores; nil: the reference).
        case listen(embedding: [Float], reference: [Float], scoreReference: [Float]?)
        /// Song mel spectrum (dB, [band][frame]); candidates render the user's aligned DI.
        case playAlong(songMel: (values: [Float], frames: Int), di: [Float])
    }

    private let target: Target
    private let renderer: ToneHubRenderer
    private let lock = NSLock()
    private struct RenderKey: Hashable {
        let settings: MockToneSettings
        let amp: ToneHubAmp?
        var forScore = false
    }
    private var cache: [RenderKey: Double] = [:]
    /// How long the last three uncached renders (each with its encode or mel) took.
    private var recentRenderTimes: [Duration] = []
    /// A typical render's time: the fastest of the recent ones (the first render on an amp also
    /// loads its model).
    private var renderTime: Duration { recentRenderTimes.min() ?? .zero }
    /// The built-in amp searches and re-scores render on (`pickAmp`); nil: the gain ladder.
    private var amp: ToneHubAmp?
    /// Why the last render or encode failed (diagnostics).
    private var _lastError: String?
    var lastError: String? {
        lock.lock()
        defer { lock.unlock() }
        return _lastError
    }
    /// Re-score requests: only the newest one renders (a burst of nudges, last one wins).
    private let requestLock = NSLock()
    private var latestRequest = 0

    /// Scores say "estimated from the file" (Match › Use a file).
    let fromFile: Bool

    /// `scoreReference`: the DI the score, its baseline and re-scores render (nil: `reference`).
    /// Listen searches on a real guitar (`ReferenceDI.guitar()`) and scores on the pluck synth
    /// (`ReferenceDI.samples()`): the score's calibration and verdict margins were set on the synth,
    /// and a score taken on another DI than the one the knobs were tuned on doesn't flatter them.
    init(listenTarget embedding: [Float], reference: [Float], scoreReference: [Float]? = nil, knobs: ModelKnobs,
         fromFile: Bool = false) {
        self.fromFile = fromFile
        target = .listen(embedding: embedding, reference: reference, scoreReference: scoreReference)
        renderer = ToneHubRenderer(cab: CabStandIn(tone: knobs.cabTone, resonance: knobs.cabRes))
    }

    init(playAlongSongMel songMel: (values: [Float], frames: Int), di: [Float], knobs: ModelKnobs) {
        fromFile = false
        target = .playAlong(songMel: songMel, di: di)
        renderer = ToneHubRenderer(cab: CabStandIn(tone: knobs.cabTone, resonance: knobs.cabRes))
    }

    /// A reopened Recent Match: re-scores nudges only (no pick, no search), on the score
    /// reference and the amp the match picked. Nil when the embedding or the reference is missing.
    convenience init?(restoring embedding: [Float], knobs: ModelKnobs, fromFile: Bool, amp: ToneHubAmp?) {
        guard !embedding.isEmpty, let reference = ReferenceDI.samples() else { return nil }
        self.init(listenTarget: embedding, reference: reference, knobs: knobs, fromFile: fromFile)
        self.amp = amp
    }

    /// The song's embedding (Listen and file matches): 128 numbers describing the tone, never
    /// audio. Kept with a Recent Match so a reopened result can re-score. Nil for play-along.
    var songEmbedding: [Float]? {
        if case .listen(let embedding, _, _) = target { return embedding }
        return nil
    }

    /// True when the score renders another reference than the search (Listen). Caller holds the lock.
    private var hasScoreReference: Bool {
        if case .listen(_, _, .some) = target { return true }
        return false
    }

    var isEstimate: Bool {
        if case .listen = target { return true }
        return false
    }

    /// The settings everyone starts from: every dial at noon (a matcher that ignores the song).
    static let noon = MockToneSettings(gain: 5, bass: 5, mid: 5, treble: 5, presence: 5, boost: false)

    /// The distance for ToneHub's amp at noon: the baseline an estimate's verdict is relative to.
    /// Always the gain ladder (Rhythm at noon), whichever amp was picked: a matcher that ignores
    /// the song picks no amp either, so the pick has to earn its verdict like the knobs do.
    func baselineDB() -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return distanceDB(Self.noon, on: nil, forScore: true)
    }

    /// The score's distance (dB) for settings the search ended on, `searched` being the search's
    /// own distance for them: the same number when the score renders the search's reference, else
    /// one render of the score reference on the picked amp.
    func scoreDB(_ settings: MockToneSettings, searched: Double) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        guard hasScoreReference else { return searched }
        return distanceDB(settings.snapped, on: amp, forScore: true)
    }

    /// The built-in amp the scorer renders on (nil: the gain ladder).
    var builtInAmp: ToneHubAmp? {
        lock.lock()
        defer { lock.unlock() }
        return amp
    }

    /// What `pickAmp` chose: the amp (nil: the gain ladder) and, for the trace, each candidate's
    /// distance at the prediction.
    struct AmpPick: Sendable {
        let amp: ToneHubAmp?
        let tried: [(amp: ToneHubAmp?, db: Double)]

        var summary: String {
            let name = amp?.title ?? "ladder"
            let list = tried.map { "\($0.amp?.title ?? "ladder") \(String(format: "%.2f", $0.db))" }.joined(separator: ", ")
            return list.isEmpty ? name : "\(name) · \(list) dB"
        }
    }

    /// When the research's family amp is within this of the best candidate (dB at the
    /// prediction), it is kept: the documented rig breaks near-ties, the score decides the rest.
    static let familyMarginDB = 0.2

    /// Picks the built-in amp for this match and keeps it for every later render: the gain
    /// ladder (the amp-only captures through the stand-in cab, as before) or one of `candidates`
    /// (cab-baked captures, their own cab, no stand-in), whichever renders `predicted` closest to
    /// the song. `preferred` (the researched family's capture) wins unless another candidate is
    /// better by more than `familyMarginDB`. The ladder renders first (it is also the search's
    /// seed, cached); a candidate is skipped when it would run past the deadline or the match is
    /// cancelled, and a failed render just drops out. With no candidate rendered, the ladder.
    func pickAmp(predicted: MockToneSettings, candidates: [ToneHubAmp], preferred: ToneHubAmp?,
                 deadline: ContinuousClock.Instant, isCancelled: () -> Bool = { false }) -> AmpPick {
        lock.lock()
        defer { lock.unlock() }
        let clock = ContinuousClock()
        let state = matchSignposter.beginInterval("amp-pick")
        defer { matchSignposter.endInterval("amp-pick", state) }
        let seed = predicted.snapped
        let t0 = clock.now
        guard !candidates.isEmpty, let ladderDB = rankDB(seed, on: nil) else {
            amp = nil
            return AmpPick(amp: nil, tried: [])
        }
        let perCandidate = max(clock.now - t0, .milliseconds(1))
        var tried: [(amp: ToneHubAmp?, db: Double)] = [(nil, ladderDB)]
        for candidate in candidates where candidate.modelURL != nil {
            // Leave the search room for at least its own seed and one step.
            guard clock.now + perCandidate * 3 < deadline, !isCancelled() else { break }
            guard let db = rankDB(seed, on: candidate) else { continue }
            tried.append((candidate, db))
        }
        var best = tried.min { $0.db < $1.db }!
        if let preferred, let family = tried.first(where: { $0.amp == preferred }),
           family.db <= best.db + Self.familyMarginDB {
            best = family
        }
        amp = best.amp
        tried = tried.map { ($0.amp, max(0, $0.db)) }
        // Re-scores render on the pick or the ladder (the baseline) only: free the candidates
        // that lost (each NAM model is a few MB, and the scorer lives as long as the result).
        renderer.keepModels(Set(ToneHubAmp.gainLadder + [best.amp].compactMap { $0 }))
        matchLog.notice("amp pick: \(best.amp?.rawValue ?? "ladder", privacy: .public) of \(tried.count) in \((clock.now - t0).inMilliseconds, format: .fixed(precision: 0)) ms")
        return AmpPick(amp: best.amp, tried: tried)
    }

    /// Marks a new re-score request; pass it to `score(_:request:)`.
    func nextRequest() -> Int {
        requestLock.lock()
        defer { requestLock.unlock() }
        latestRequest += 1
        return latestRequest
    }

    private func isLatest(_ request: Int?) -> Bool {
        guard let request else { return true }
        requestLock.lock()
        defer { requestLock.unlock() }
        return request == latestRequest
    }

    /// The score for settings (snapped to the dial), or nil if a render fails or a newer request
    /// (`nextRequest`) superseded this one before it started rendering.
    func score(_ settings: MockToneSettings, request: Int? = nil) -> ToneMatchScore? {
        lock.lock()
        defer { lock.unlock() }
        guard isLatest(request) else { return nil }
        let baseline = isEstimate ? distanceDB(Self.noon, on: nil, forScore: true) : nil
        guard isLatest(request) else { return nil }
        return distanceDB(settings.snapped, on: amp, forScore: true).flatMap { ToneMatchScore(db: $0, isEstimate: isEstimate, baselineDB: baseline, fromFile: fromFile) }
    }

    /// The prior's weight in the CMA-ES cost (the bench's W_PRIOR): a candidate's mean squared
    /// step from the prediction (0...1 knob units, / 0.25) against its distance scaled to 1 at the
    /// prediction.
    static let cmaPriorWeight = 0.25
    /// A hard cap on CMA-ES generations (λ 8 each), whatever the clock says.
    static let cmaMaxGenerations = 60

    /// Predict-then-search (the bench's `pred_then_search`): CMA-ES over Gain, Bass, Mid and Treble
    /// (ToneHub's amp has no Presence) from `start`, sigma 0.2 (2 dial steps), λ 8, cost = distance
    /// / distance at `start` + `cmaPriorWeight` × prior toward `start`. Candidates are snapped to
    /// the dial's half steps (repeats are free, from the cache). Stops at convergence, the deadline
    /// (leaving room for one more render) or a cancel; a generation cut short is dropped. Returns
    /// the lowest-cost settings and their distance (dB); nil if even `start` can't be rendered.
    func searchCMA(from start: MockToneSettings, deadline: ContinuousClock.Instant,
                   isCancelled: () -> Bool = { false })
        -> (settings: MockToneSettings, db: Double, renders: Int, generations: Int)? {
        lock.lock()
        defer { lock.unlock() }
        let clock = ContinuousClock()
        let state = matchSignposter.beginInterval("cma-es")
        defer { matchSignposter.endInterval("cma-es", state) }
        let t0 = clock.now
        let seed = start.snapped
        guard let seedDB = distanceDB(seed) else { return nil }
        let perCandidate = clock.now - t0
        var renders = 1
        let center = [seed.gain, seed.bass, seed.mid, seed.treble].map { $0 / 10 }
        let scale = max(seedDB, 1e-3)
        func settings(_ x: [Double]) -> MockToneSettings {
            MockToneSettings(gain: 10 * x[0], bass: 10 * x[1], mid: 10 * x[2], treble: 10 * x[3],
                             presence: seed.presence, boost: seed.boost).snapped
        }
        var best = seed, bestDB = seedDB, bestCost = 1.0
        var es = CMAES(mean: center, sigma: 0.2, lambda: 8)
        var generations = 0
        // The deadline and the cancel are checked every generation too: once the candidates snap
        // to settings already rendered (cached), the per-candidate check below never runs, and a
        // search that hovers without converging would otherwise spin past the budget.
        run: while !es.hasConverged, generations < Self.cmaMaxGenerations, clock.now < deadline, !isCancelled() {
            let xs = es.ask()
            var costs: [Double] = []
            for x in xs {
                let s = settings(x)
                let seen = cache[RenderKey(settings: s, amp: amp)] != nil
                if !seen { guard clock.now + perCandidate < deadline, !isCancelled() else { break run } }
                guard let db = distanceDB(s) else {
                    costs.append(.infinity)
                    continue
                }
                if !seen { renders += 1 }
                let prior = zip(x, center).map { ($0 - $1) * ($0 - $1) }.reduce(0, +) / Double(x.count) / 0.25
                let cost = db / scale + Self.cmaPriorWeight * prior
                costs.append(cost)
                if cost < bestCost - 1e-6 {
                    best = s
                    bestDB = db
                    bestCost = cost
                }
            }
            es.tell(xs, costs)
            generations += 1
        }
        matchLog.debug("cma-es: \(generations) gen, \(renders) renders, \((clock.now - t0).inMilliseconds, format: .fixed(precision: 0)) ms, distance \(seedDB, format: .fixed(precision: 2)) → \(bestDB, format: .fixed(precision: 2)) dB")
        return (best, bestDB, renders, generations)
    }

    /// Greedy passes of ±1 dial steps on Gain, Treble, Bass and Mid around `start`, keeping a step
    /// whenever it lowers the distance, until a pass changes nothing (at most 4 passes) or the
    /// deadline. The steps correct for the encoder's knobs describing the bench amp, not ToneHub's
    /// captures. Measured on the app's amps (docs/encoder-run2.md, "Listen refine on the app's amps";
    /// `ToneMatchRefineEval`): closer to the true sound than the prediction in 79/120 cases (median
    /// -0.58 dB log-mel), most on the cab-baked captures, a wash on the gain ladder. It was worse
    /// every time Class A Chime was picked while Listen searched on the synth reference; on the
    /// guitar one (`ReferenceDI.guitar()`) it is neutral there ("Amp pick and refine"). Compares
    /// unclamped distances (`rankDB`). Returns the best settings and their distance (dB, clamped at
    /// 0); nil if even `start` can't be rendered.
    func refine(from start: MockToneSettings, deadline: ContinuousClock.Instant,
                isCancelled: () -> Bool = { false }) -> (settings: MockToneSettings, db: Double, renders: Int)? {
        lock.lock()
        defer { lock.unlock() }
        let clock = ContinuousClock()
        let state = matchSignposter.beginInterval("refine")
        defer { matchSignposter.endInterval("refine", state) }
        let t0 = clock.now
        var best = start.snapped
        guard var bestDB = rankDB(best, on: amp) else { return nil }
        // The seed is usually cached by the pick, so a render's time comes from the recent real ones
        // (`renderTime`). Each step needs room for itself and, when the score renders another
        // reference (Listen), for the score's own render after the search.
        let reserve = hasScoreReference ? 2 : 1
        var tried = 1
        let moves: [(WritableKeyPath<MockToneSettings, Double>, Double)] = [
            (\.gain, 1), (\.gain, -1), (\.treble, 1), (\.treble, -1), (\.bass, 1), (\.bass, -1), (\.mid, 1), (\.mid, -1),
        ]
        search: for _ in 0..<4 {
            var improved = false
            for (key, delta) in moves {
                guard clock.now + renderTime * reserve < deadline, !isCancelled() else { break search }
                var candidate = best
                candidate[keyPath: key] = min(max(candidate[keyPath: key] + delta, 0), 10)
                guard candidate != best else { continue }
                let seen = cache[RenderKey(settings: candidate, amp: amp)] != nil
                guard let db = rankDB(candidate, on: amp) else { continue }
                if !seen { tried += 1 }
                if db < bestDB - 1e-3 {
                    best = candidate
                    bestDB = db
                    improved = true
                }
            }
            if !improved { break }
        }
        matchLog.debug("refine: \(tried) renders, \((clock.now - t0).inMilliseconds, format: .fixed(precision: 0)) ms, \(self.renderTime.inMilliseconds, format: .fixed(precision: 0)) ms each, distance \(bestDB, format: .fixed(precision: 2)) dB")
        return (best, max(0, bestDB), tried)
    }

    #if DEBUG
    /// `ToneMatchRefineEval` only: render on this amp from now on, without a pick.
    func debugUse(amp: ToneHubAmp?) {
        lock.lock()
        defer { lock.unlock() }
        self.amp = amp
    }

    /// `ToneMatchRefineEval` only: the distance (dB) of settings on the current amp.
    func debugDistanceDB(_ s: MockToneSettings) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return distanceDB(s.snapped)
    }
    #endif

    /// The distance in dB on the picked amp. Caller holds the lock.
    private func distanceDB(_ s: MockToneSettings) -> Double? { distanceDB(s, on: amp) }

    /// The distance in dB: estimated (calibrated embedding distance) for listen, measured (log-mel)
    /// for play-along. `amp` nil: the gain ladder. Never below 0. Caller holds the lock.
    private func distanceDB(_ s: MockToneSettings, on amp: ToneHubAmp?, forScore: Bool = false) -> Double? {
        rankDB(s, on: amp, forScore: forScore).map { max(0, $0) }
    }

    /// `distanceDB` before the clamp at 0, for comparing candidates (the amp pick and the refine).
    /// A listen estimate's calibration line goes below 0 for close embeddings, and on a real-guitar
    /// reference many renders land there (docs/encoder-run2.md, "Amp pick and refine"): clamped,
    /// they all tie at 0, so the pick falls to the first candidate (the ladder) and the refine
    /// stops at the first step that reaches 0. Caller holds the lock.
    private func rankDB(_ s: MockToneSettings, on amp: ToneHubAmp?, forScore: Bool = false) -> Double? {
        let useScoreReference = forScore && hasScoreReference
        let key = RenderKey(settings: s, amp: amp, forScore: useScoreReference)
        if let cached = cache[key] { return cached }
        let renderStart = ContinuousClock.now
        defer { recentRenderTimes = Array((recentRenderTimes + [ContinuousClock.now - renderStart]).suffix(3)) }
        let db: Double?
        switch target {
        case .listen(let embedding, let searchReference, let scoreReference):
            let reference = useScoreReference ? (scoreReference ?? searchReference) : searchReference
            guard let rendered = renderer.render(reference, s, amp: amp) else {
                _lastError = renderer.lastError ?? "render failed"
                return nil
            }
            guard let out = ToneEncoderRunner.shared.run(rendered) else {
                _lastError = "encoder on the render: " + (ToneEncoderRunner.shared.lastError ?? "failed")
                return nil
            }
            db = ToneEncoderRunner.calibratedDB(distance: ToneEncoderRunner.distance(embedding, out.embedding))
        case .playAlong(let songMel, let di):
            guard let rendered = renderer.render(di, s, amp: amp), let mel = LogMelFrontEnd.melDB(rendered) else {
                _lastError = renderer.lastError ?? "mel of the render failed"
                return nil
            }
            db = ToneDistance.melDB(mel, songMel)
        }
        if let db { cache[key] = db }
        return db
    }
}

// MARK: - Built-in amp pre-pick

extension ToneHubAmp {
    /// What the song's researched rig (Original Rig) says about its amp, for the pre-pick. Sendable
    /// so it can cross into the analysis task.
    struct MatchHint: Sendable, Hashable {
        /// The cab-baked capture of the researched amp's family (`capture(forFamily:)`), if any.
        var familyAmp: ToneHubAmp?
        /// The researched amp's tone-stack family (its catalog profile), if known.
        var voice: ToneStackFamily?
        /// The researched gain style as a Gain dial position (0...10); nil when unknown.
        var styleGain: Double?
    }

    /// Where on the Gain dial each capture's own character sits (clean ... high gain), from its
    /// description: a heuristic for picking which captures to try, not a measurement. The scorer's
    /// renders decide among them.
    var styleGain: Double {
        switch self {
        case .cleanGlass, .texasClean: return 2
        case .classAChime: return 4
        case .britVintage: return 5
        case .britCrunch: return 5.5
        case .brit800: return 6
        case .citrusDrive: return 6.5
        case .lead100: return 8
        case .highGain: return 8.5
        case .modernHighGain: return 9
        }
    }

    /// The cab-baked capture for a researched amp family (`tone-research`'s `family` enum). Nil
    /// for families none of them stands in for ("ampeg", "other").
    static func capture(forFamily family: String) -> ToneHubAmp? {
        switch family {
        case "fender_tweed", "fender_blackface", "fender_hot_rod", "roland_jc": return .texasClean
        case "vox_ac30", "vox_ac15": return .classAChime
        case "marshall_plexi", "hiwatt": return .britVintage
        case "marshall_jcm800", "marshall_jcm900_dsl", "marshall_silver_jubilee": return .brit800
        case "orange": return .citrusDrive
        case "soldano_friedman": return .lead100
        case "mesa_rectifier", "mesa_mark", "peavey_5150": return .modernHighGain
        default: return nil
        }
    }

    /// The cab-baked captures a match tries besides the gain ladder (at most two, so the pick costs
    /// at most two extra renders): the researched family's capture when there is one, then the
    /// capture nearest the gain style (the research's when known, else the encoder's gain), among
    /// the researched voice's captures when one is close enough.
    static func matchCandidates(predictedGain: Double, hint: MatchHint?) -> [ToneHubAmp] {
        let gain = hint?.styleGain ?? predictedGain
        let baked = allCases.filter(\.includesCab)
        var picks: [ToneHubAmp] = []
        if let family = hint?.familyAmp { picks.append(family) }
        func nearest(_ pool: [ToneHubAmp]) -> [ToneHubAmp] {
            pool.filter { !picks.contains($0) }.sorted { abs($0.styleGain - gain) < abs($1.styleGain - gain) }
        }
        if let voice = hint?.voice,
           let sameVoice = nearest(baked.filter { $0.voice == voice }).first,
           abs(sameVoice.styleGain - gain) <= 2 {
            picks.append(sameVoice)
        }
        for amp in nearest(baked) where picks.count < 2 { picks.append(amp) }
        return Array(picks.prefix(2))
    }
}

extension ToneResearch {
    /// The rig's amp, for Match's built-in amp pre-pick. Nil when the research found nothing to go
    /// on (no amp and no gain style).
    var ampHint: ToneHubAmp.MatchHint? {
        let amp = mainAmp.flatMap { $0.confidence == .unknown ? nil : $0 }
        let style: Double? = switch gainStyle {
        case "clean": 2
        case "edge_of_breakup": 4
        case "crunch": 6
        case "high_gain": 8.5
        case "fuzz": 5.5
        default: nil
        }
        let hint = ToneHubAmp.MatchHint(familyAmp: amp.flatMap { ToneHubAmp.capture(forFamily: $0.family) },
                                        voice: amp == nil ? nil : ampProfile?.model.family,
                                        styleGain: style)
        return hint.familyAmp == nil && hint.voice == nil && hint.styleGain == nil ? nil : hint
    }

    /// The built-in amp a researched tone (no match, so nothing to render against) plays on: the
    /// family's cab-baked capture, else nil (the gain ladder).
    var toneHubAmp: ToneHubAmp? { ampHint?.familyAmp }
}

// MARK: - Offline render of ToneHub's amp

/// The cab the model heard, from its cab knobs, as the bench amp builds it (bench/tonebench/amp.py):
/// 75 Hz high-pass (2nd order), a 110 Hz resonance of 0...+9 dB (Q 1.4) and a 4th-order low-pass at
/// 2.5 kHz x 3^tone. For ToneHub's amp-only captures (`ToneHubAmp.includesCab` false): scoring and
/// Hear it render them through this, and the live path plays them through `neutral` as an IR
/// (unless the user loaded their own cab IR, which is not known here). Cab-baked captures skip it.
struct CabStandIn: Sendable {
    let tone: Double
    let resonance: Double

    /// Hear it's default (ResultPreviewRow) and the live path's cab for the bundled amps.
    static let neutral = CabStandIn(tone: 0.5, resonance: 0.4)

    func apply(_ x: inout [Float], sampleRate: Double) {
        var filters = [
            Biquad.highPass(75, q: 0.5.squareRoot(), fs: sampleRate),
            Biquad.peak(110, q: 1.4, db: 9 * resonance, fs: sampleRate),
        ]
        let corner = min(2500 * pow(3, tone), 0.45 * sampleRate)
        // 4th-order Butterworth = two biquads with Q 0.5412 and 1.3066.
        filters.append(Biquad.lowPass(corner, q: 0.541_196_1, fs: sampleRate))
        filters.append(Biquad.lowPass(corner, q: 1.306_563, fs: sampleRate))
        for var f in filters { f.process(&x) }
    }
}

/// RBJ cookbook biquad, direct form I in double. Offline only.
struct Biquad {
    var b0, b1, b2, a1, a2: Double
    private var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0

    init(b0: Double, b1: Double, b2: Double, a0: Double, a1: Double, a2: Double) {
        self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0; self.a1 = a1 / a0; self.a2 = a2 / a0
    }

    static func lowPass(_ f: Double, q: Double, fs: Double) -> Biquad {
        let w = 2 * Double.pi * f / fs, alpha = sin(w) / (2 * q), c = cos(w)
        return Biquad(b0: (1 - c) / 2, b1: 1 - c, b2: (1 - c) / 2, a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func highPass(_ f: Double, q: Double, fs: Double) -> Biquad {
        let w = 2 * Double.pi * f / fs, alpha = sin(w) / (2 * q), c = cos(w)
        return Biquad(b0: (1 + c) / 2, b1: -(1 + c), b2: (1 + c) / 2, a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func peak(_ f: Double, q: Double, db: Double, fs: Double) -> Biquad {
        let a = pow(10, db / 40), w = 2 * Double.pi * f / fs, alpha = sin(w) / (2 * q), c = cos(w)
        return Biquad(b0: 1 + alpha * a, b1: -2 * c, b2: 1 - alpha * a, a0: 1 + alpha / a, a1: -2 * c, a2: 1 - alpha / a)
    }

    mutating func process(_ x: inout [Float]) {
        for i in x.indices {
            let input = Double(x[i])
            let y = b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1; x1 = input; y2 = y1; y1 = y
            x[i] = Float(y)
        }
    }
}

/// ToneHub's amp offline, as "Try on the amp" sets it up (`ToneHubAmp.load`): the given built-in
/// capture (nil: the one the gain picks from the amp-only ladder), the Gain dial as input gain
/// (dial x 4.8 - 24 dB), the amp model, the Bass / Mid / Treble tone stack (±12 dB), then
/// `CabStandIn` unless the capture has its cab baked in. No effects. Private model copies, freed
/// here.
final class ToneHubRenderer: @unchecked Sendable {
    private let cab: CabStandIn
    private var models: [ToneHubAmp: OpaquePointer] = [:]
    private static let rate = LogMelFrontEnd.sampleRate
    /// Why the last render failed (diagnostics). Callers serialise renders.
    private(set) var lastError: String?

    init(cab: CabStandIn) { self.cab = cab }

    deinit { for model in models.values { gtr_nam_model_free(model) } }

    func render(_ di: [Float], _ s: MockToneSettings, amp pick: ToneHubAmp? = nil) -> [Float]? {
        guard !di.isEmpty else {
            lastError = "empty DI"
            return nil
        }
        let amp = ToneHubAmp.resolve(pick, gain: s.gain)
        guard let model = model(for: amp) else { return nil }
        let state = matchSignposter.beginInterval("render")
        defer { matchSignposter.endInterval("render", state) }
        let gain = Float(pow(10, (min(max(s.gain, 0), 10) * 4.8 - 24) / 20))
        let input = di.map { $0 * gain }
        var output = [Float](repeating: 0, count: input.count)
        let ok = input.withUnsafeBufferPointer { i in
            output.withUnsafeMutableBufferPointer { o in
                gtr_nam_model_render_offline(model, i.baseAddress!, o.baseAddress!, Int32(input.count))
            }
        }
        guard ok != 0 else {
            lastError = "NAM offline render failed (\(amp.rawValue))"
            return nil
        }
        func db(_ dial: Double) -> Float { Float((min(max(dial, 0), 10) - 5) / 5 * 12) }
        output.withUnsafeMutableBufferPointer {
            gtr_tone_stack_render_offline($0.baseAddress!, Int32($0.count), db(s.bass), db(s.mid), db(s.treble), Self.rate)
        }
        if !amp.includesCab { cab.apply(&output, sampleRate: Self.rate) }
        guard output.allSatisfy({ $0.isFinite }) else {
            lastError = "render produced non-finite samples"
            return nil
        }
        return output
    }

    /// Frees every cached model but `amps`. Callers serialise renders, as for `render`.
    func keepModels(_ amps: Set<ToneHubAmp>) {
        for (amp, model) in models where !amps.contains(amp) {
            gtr_nam_model_free(model)
            models[amp] = nil
        }
    }

    private func model(for amp: ToneHubAmp) -> OpaquePointer? {
        if let m = models[amp] { return m }
        guard let path = amp.modelURL?.path else {
            lastError = "amp model \(amp.rawValue) is not in the bundle"
            return nil
        }
        guard let m = path.withCString({ gtr_nam_model_load($0, Self.rate, 4096, nil, 0) }) else {
            lastError = "NAM model failed to load: \((path as NSString).lastPathComponent)"
            return nil
        }
        models[amp] = m
        return m
    }
}

// MARK: - Reference DI

/// A guitar-like DI for listen-mode scoring, made on the device by the practice pluck synth
/// (Karplus-Strong, our own code): 3 s of an E5 / A5 / G5 riff and a short single-note line, peak
/// -9 dBFS. Listen mode has no DI of the user's, so the estimate asks "does our amp, set like this,
/// make a guitar sound like the song?". The encoder was trained to ignore which notes are played,
/// but a synthetic pluck is not a real guitar: that is part of why the listen score is an estimate.
enum ReferenceDI {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cached: [Float]?
    nonisolated(unsafe) private static var cachedGuitar: [Float]?

    /// A real guitar DI for Listen's amp pick and refine: 1.5 s of the bundled "Chords" preview phrase then
    /// 1.5 s of the "Lead line" (3 s, like `samples()`), peak -9 dBFS. Falls back to the pluck
    /// synth when they can't be read. Why not the synth: its spectrum is not a guitar's, and the
    /// refine bends the knobs to make up for it. On the clean cab-baked captures that went wrong
    /// every time (Class A Chime: worse in 110 of 120 cases by +2.8 dB median, Texas Clean 79 of
    /// 120); on this reference the same refine is neutral on Class A Chime (42 better / 41 worse)
    /// and helps more often on every other amp (docs/encoder-run2.md, "Amp pick and refine").
    static func guitar() -> [Float]? {
        lock.lock()
        if let cachedGuitar {
            lock.unlock()
            return cachedGuitar
        }
        lock.unlock()
        let half = Int(1.5 * LogMelFrontEnd.sampleRate)
        guard let chords = PreviewPhrase.chords.samples(), let lead = PreviewPhrase.lead.samples(),
              chords.count >= half, lead.count >= half else { return samples() }
        var out = Array(chords.prefix(half)) + Array(lead.prefix(half))
        let peak = out.reduce(Float(0)) { max($0, abs($1)) }
        guard peak > 0, peak.isFinite else { return samples() }
        let scale = Float(pow(10, -9.0 / 20)) / peak
        out = out.map { $0 * scale }
        lock.lock()
        cachedGuitar = out
        lock.unlock()
        return out
    }

    static func samples() -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        if let cached { return cached }
        let rate = LogMelFrontEnd.sampleRate
        var events: [GTRSynthEvent] = []
        func chord(_ t: Double, _ root: Float, _ dur: Double) {
            for (i, interval) in [Float(0), 7, 12].enumerated() {
                events.append(GTRSynthEvent(time: t + Double(i) * 0.008, duration: dur, midi: root + interval,
                                            velocity: 0.8, voice: GTRSynthVoicePluck))
            }
        }
        chord(0.0, 40, 0.45)   // E5
        chord(0.5, 45, 0.45)   // A5
        chord(1.0, 43, 0.45)   // G5
        for (i, note) in [Float(52), 55, 57, 59, 62].enumerated() {
            events.append(GTRSynthEvent(time: 1.5 + Double(i) * 0.18, duration: 0.17, midi: note, velocity: 0.75,
                                        voice: GTRSynthVoicePluck))
        }
        chord(2.4, 40, 0.6)    // E5
        let frames = Int(3.0 * rate)
        var out = [Float](repeating: 0, count: frames)
        let peak = events.withUnsafeBufferPointer { ev in
            out.withUnsafeMutableBufferPointer { o in
                gtr_synth_render(ev.baseAddress!, Int32(ev.count), rate, o.baseAddress!, Int64(frames), 0, 7)
            }
        }
        guard peak > 0, peak.isFinite else { return nil }
        let scale = Float(pow(10, -9.0 / 20)) / peak
        out = out.map { $0 * scale }
        cached = out
        return out
    }
}

// MARK: - Audio helpers

enum ToneAudio {
    /// Mono Float32 at `rate` → 48 kHz (AVAudioConverter, max quality). The input as is at 48 kHz.
    static func resample(_ x: [Float], from rate: Double) -> [Float]? {
        let target = LogMelFrontEnd.sampleRate
        guard rate > 0, !x.isEmpty else { return nil }
        if abs(rate - target) < 0.5 { return x }
        guard let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: target, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let input = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(x.count)),
              let output = AVAudioPCMBuffer(pcmFormat: outFormat,
                                            frameCapacity: AVAudioFrameCount(Double(x.count) * target / rate) + 4096)
        else { return nil }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        input.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }
        var fed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return input
        }
        // The input buffer held a copy of the clip: zero it before it is freed.
        input.floatChannelData![0].update(repeating: 0, count: x.count)
        guard error == nil, let data = output.floatChannelData, output.frameLength > 0 else { return nil }
        let result = Array(UnsafeBufferPointer(start: data[0], count: Int(output.frameLength)))
        data[0].update(repeating: 0, count: Int(output.frameLength))
        return result
    }
}

#if DEBUG
// MARK: - Timing bench (DEBUG)

/// DEBUG only: launch with `-matchbench` to time the encoder path on this device or simulator. A
/// synthetic "song" (the reference DI through ToneHub's amp at known settings, 10 s) goes through
/// `matchListen` and `matchPlayAlong`; timings print to the console and the "match" log.
enum ToneMatchBench {
    static var isRequested: Bool { ProcessInfo.processInfo.arguments.contains("-matchbench") }

    static func runIfRequested() {
        guard isRequested else { return }
        Task.detached(priority: .userInitiated) { run() }
    }

    private static func run() {
        let clock = ContinuousClock()
        func ms(_ d: Duration) -> String { String(format: "%.1f ms", d.inMilliseconds) }
        var t = clock.now
        let loaded = ToneEncoderRunner.shared.loadIfNeeded()
        report("model load \(ms(clock.now - t)) (\(loaded ? "ok" : "FAILED"))")
        guard loaded, let reference = ReferenceDI.samples() else { return }
        let truth = MockToneSettings(gain: 6, bass: 6, mid: 4, treble: 7, presence: 5, boost: false)
        let renderer = ToneHubRenderer(cab: CabStandIn(tone: 0.4, resonance: 0.5))
        t = clock.now
        guard let song3 = renderer.render(reference, truth) else { return report("render failed") }
        report("NAM render 3 s \(ms(clock.now - t))")
        var song = [Float]()
        while song.count < 10 * 48_000 { song += song3 }
        song = Array(song.prefix(10 * 48_000))
        for _ in 0..<2 {
            t = clock.now
            _ = LogMelFrontEnd.logMel(song)
            let front = clock.now - t
            t = clock.now
            _ = ToneEncoderRunner.shared.run(song)
            report("10 s clip: front end \(ms(front)), front end + inference \(ms(clock.now - t))")
        }
        t = clock.now
        var trace = MatchTrace()
        let listen = ToneMatcher.matchListen(clip48k: song, trace: &trace)
        report("matchListen amp \(listen?.builtInAmp?.title ?? "ladder") (\(trace.amp ?? "no pick"))")
        report("matchListen total \(ms(clock.now - t)); knobs \(listen.map { "\($0.knobs)" } ?? "nil"); predicted \(listen.map { "\($0.predicted)" } ?? "-"); toneHub \(listen.map { "\($0.toneHub)" } ?? "-"); refined \(listen?.refined ?? false); score \(listen?.score.map { "\($0.detailText) \($0.verdict?.rawValue ?? "no verdict")" } ?? "none")")
        // The same take with a hint as research would give it (a Marshall JCM800 at crunch): times
        // the family path. A fixed test input, not a researched rig.
        t = clock.now
        var hinted = MatchTrace()
        let hint = ToneHubAmp.MatchHint(familyAmp: ToneHubAmp.capture(forFamily: "marshall_jcm800"), voice: .marshallTMB, styleGain: 6)
        let withHint = ToneMatcher.matchListen(clip48k: song, hint: hint, trace: &hinted)
        report("matchListen (hint) total \(ms(clock.now - t)); amp \(withHint?.builtInAmp?.title ?? "ladder") (\(hinted.amp ?? "no pick")); toneHub \(withHint.map { "\($0.toneHub)" } ?? "-"); score \(withHint?.score.map { "\($0.detailText) \($0.verdict?.rawValue ?? "no verdict")" } ?? "none")")
        var di = [Float]()
        while di.count < 10 * 48_000 { di += reference }
        di = Array(di.prefix(10 * 48_000))
        t = clock.now
        let play = ToneMatcher.matchPlayAlong(song48k: song, di48k: di)
        report("matchPlayAlong total \(ms(clock.now - t)); toneHub \(play.map { "\($0.toneHub)" } ?? "-"); refined \(play?.refined ?? false); score \(play?.score.map { "\($0.detailText) \($0.verdict?.rawValue ?? "no verdict")" } ?? "none")")
    }

    private static func report(_ line: String) {
        print("[matchbench] \(line)")
        matchLog.notice("[matchbench] \(line, privacy: .public)")
    }
}
#endif
