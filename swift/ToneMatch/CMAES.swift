import Foundation

/// A small CMA-ES (Hansen's (μ/μ_w, λ) evolution strategy with rank-one and rank-μ covariance
/// updates and cumulative step-size adaptation) for the clean-audio match, where the bench's
/// `pred_then_search` (bench/tonebench/matchers.py) beat the encoder alone: CMA-ES started at the
/// encoder's prediction with sigma 0.2, its prior pulling toward the prediction.
///
/// Box-bounded to [0, 1] by clipping the samples ("repair"); the update uses the clipped points,
/// as pycma's simple bound handling does for a few dimensions. Deterministic for a seed, so the
/// same file and window always land on the same settings. Pure maths, no allocation on a hot path
/// worth mentioning (n is 4).
struct CMAES {
    let n: Int
    let lambda: Int
    private let mu: Int
    private let weights: [Double]
    private let mueff: Double
    private let cc, cs, c1, cmu, damps, chiN: Double

    private(set) var mean: [Double]
    private(set) var sigma: Double
    private var pc: [Double]
    private var ps: [Double]
    private var C: [[Double]]
    private var B: [[Double]]
    private var D: [Double]
    private var generation = 0
    private var rng: SplitMix64

    init(mean: [Double], sigma: Double, lambda: Int? = nil, seed: UInt64 = 0x70E_4B0B) {
        let n = mean.count
        self.n = n
        let lam = max(4, lambda ?? (4 + Int(3 * log(Double(mean.count)))))
        self.lambda = lam
        let mu = lam / 2
        self.mu = mu
        let raw = (0..<mu).map { log(Double(lam + 1) / 2) - log(Double($0 + 1)) }
        let sum = raw.reduce(0, +)
        let weights = raw.map { $0 / sum }
        self.weights = weights
        let mueff = 1 / weights.map { $0 * $0 }.reduce(0, +)
        self.mueff = mueff
        let nd = Double(n)
        cc = (4 + mueff / nd) / (nd + 4 + 2 * mueff / nd)
        cs = (mueff + 2) / (nd + mueff + 5)
        c1 = 2 / ((nd + 1.3) * (nd + 1.3) + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((nd + 2) * (nd + 2) + mueff))
        damps = 1 + 2 * max(0, ((mueff - 1) / (nd + 1)).squareRoot() - 1) + cs
        chiN = nd.squareRoot() * (1 - 1 / (4 * nd) + 1 / (21 * nd * nd))
        self.mean = mean.map { min(max($0, 0), 1) }
        self.sigma = sigma
        pc = Array(repeating: 0, count: n)
        ps = Array(repeating: 0, count: n)
        let identity: [[Double]] = (0..<n).map { i in (0..<n).map { $0 == i ? 1 : 0 } }
        C = identity
        B = identity
        D = Array(repeating: 1, count: n)
        rng = SplitMix64(seed: seed)
    }

    /// True once the steps are far below a dial's half step (nothing left to find).
    var hasConverged: Bool { sigma * (D.max() ?? 1) < 0.01 }

    /// λ candidates, clipped to [0, 1].
    mutating func ask() -> [[Double]] {
        let steps: [[Double]] = (0..<lambda).map { _ in
            let z = (0..<n).map { _ in rng.nextGaussian() }
            let dz = (0..<n).map { D[$0] * z[$0] }
            return (0..<n).map { i in (0..<n).reduce(0) { $0 + B[i][$1] * dz[$1] } }
        }
        return steps.map { y in (0..<n).map { min(max(mean[$0] + sigma * y[$0], 0), 1) } }
    }

    /// Updates from the costs of `ask()`'s candidates (lower is better; non-finite is worst).
    mutating func tell(_ xs: [[Double]], _ costs: [Double]) {
        guard xs.count == lambda, costs.count == lambda else { return }
        generation += 1
        let order = (0..<lambda).sorted { (costs[$0].isFinite ? costs[$0] : .infinity) < (costs[$1].isFinite ? costs[$1] : .infinity) }
        let old = mean
        // The clipped points, as steps from the old mean in sigma units.
        let ys = order.prefix(mu).map { k in (0..<n).map { (xs[k][$0] - old[$0]) / sigma } }
        var yw = Array(repeating: 0.0, count: n)
        for (w, y) in zip(weights, ys) { for i in 0..<n { yw[i] += w * y[i] } }
        mean = (0..<n).map { min(max(old[$0] + sigma * yw[$0], 0), 1) }

        // C^(-1/2) · yw = B · D^-1 · Bᵀ · yw
        var bty = Array(repeating: 0.0, count: n)
        for j in 0..<n { for i in 0..<n { bty[j] += B[i][j] * yw[i] } }
        for j in 0..<n { bty[j] /= max(D[j], 1e-12) }
        var invSqrtY = Array(repeating: 0.0, count: n)
        for i in 0..<n { for j in 0..<n { invSqrtY[i] += B[i][j] * bty[j] } }

        let csFactor = (cs * (2 - cs) * mueff).squareRoot()
        for i in 0..<n { ps[i] = (1 - cs) * ps[i] + csFactor * invSqrtY[i] }
        let psNorm = ps.map { $0 * $0 }.reduce(0, +).squareRoot()
        let hsig = psNorm / (1 - pow(1 - cs, 2 * Double(generation))).squareRoot() / chiN < 1.4 + 2 / Double(n + 1)
        let ccFactor = (cc * (2 - cc) * mueff).squareRoot()
        for i in 0..<n { pc[i] = (1 - cc) * pc[i] + (hsig ? ccFactor * yw[i] : 0) }

        let deltaH = hsig ? 0 : cc * (2 - cc)
        for i in 0..<n {
            for j in 0...i {
                var rankMu = 0.0
                for (w, y) in zip(weights, ys) { rankMu += w * y[i] * y[j] }
                let v = (1 - c1 - cmu) * C[i][j] + c1 * (pc[i] * pc[j] + deltaH * C[i][j]) + cmu * rankMu
                C[i][j] = v
                C[j][i] = v
            }
        }
        sigma *= exp((cs / damps) * (psNorm / chiN - 1))
        sigma = min(max(sigma, 1e-6), 1)
        let (values, vectors) = Self.eigen(C)
        D = values.map { max($0, 1e-20).squareRoot() }
        B = vectors
    }

    /// Jacobi eigen-decomposition of a small symmetric matrix: eigenvalues and the eigenvectors as
    /// columns.
    static func eigen(_ m: [[Double]]) -> ([Double], [[Double]]) {
        let n = m.count
        var a = m
        var v = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }
        for _ in 0..<50 {
            var off = 0.0
            for i in 0..<n { for j in 0..<n where i != j { off += a[i][j] * a[i][j] } }
            if off < 1e-22 { break }
            for p in 0..<n {
                for q in (p + 1)..<n where abs(a[p][q]) > 1e-300 {
                    let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c
                    for k in 0..<n {
                        let akp = a[k][p], akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for k in 0..<n {
                        let apk = a[p][k], aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for k in 0..<n {
                        let vkp = v[k][p], vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }
        return ((0..<n).map { a[$0][$0] }, v)
    }
}

/// Small, fast, seedable generator (Steele et al.'s SplitMix64) with Box-Muller normals.
struct SplitMix64 {
    private var state: UInt64
    private var spare: Double?

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    mutating func nextUnit() -> Double { (Double(next() >> 11) + 0.5) / Double(1 << 53) }

    mutating func nextGaussian() -> Double {
        if let s = spare {
            spare = nil
            return s
        }
        let u1 = nextUnit(), u2 = nextUnit()
        let r = (-2 * log(u1)).squareRoot()
        spare = r * sin(2 * .pi * u2)
        return r * cos(2 * .pi * u2)
    }
}
