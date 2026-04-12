// apple_translate — persistent translation process using Apple Translation.framework.
//
// Keeps ONE process alive for the entire run. The TranslationSession is created
// once and reused across all pages, eliminating per-page startup overhead and
// avoiding daemon contention from multiple concurrent sessions.
//
// Protocol: newline-delimited JSON on stdin/stdout.
//   stdin : {"texts": ["...", "..."], "source": "en", "target": "vi"}\n
//   stdout: {"translations": ["...", "..."]}\n
//
// Compile:
//   swiftc -O apple_translate.swift -o .apple_translate
//
// Requires macOS 15+ with Translation language packs installed.

import Foundation
import Translation

private struct BatchInput: Decodable {
    let texts: [String]
    let source: String
    let target: String
}

private struct BatchOutput: Encodable {
    let translations: [String]
}

private var cachedSession: TranslationSession?
private var cachedSource = ""
private var cachedTarget = ""

@MainActor
private func handleRequest(_ input: BatchInput) async {
    // Reuse session when language pair is unchanged
    if cachedSession == nil || input.source != cachedSource || input.target != cachedTarget {
        cachedSession = TranslationSession(
            installedSource: Locale.Language(identifier: input.source),
            target: Locale.Language(identifier: input.target)
        )
        cachedSource = input.source
        cachedTarget = input.target
    }
    let session = cachedSession!

    // Build batch requests, preserving original indices via clientIdentifier
    var result = input.texts
    var requests: [TranslationSession.Request] = []
    for (i, text) in input.texts.enumerated() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { continue }
        requests.append(.init(sourceText: trimmed, clientIdentifier: "\(i)"))
    }

    if !requests.isEmpty {
        do {
            for try await response in session.translate(batch: requests) {
                if let idStr = response.clientIdentifier, let i = Int(idStr) {
                    result[i] = response.targetText
                }
            }
        } catch {
            FileHandle.standardError.write(Data("apple_translate error: \(error)\n".utf8))
        }
    }

    if let outData = try? JSONEncoder().encode(BatchOutput(translations: result)) {
        FileHandle.standardOutput.write(outData)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }

    // Ready for the next request
    scheduleRead()
}

func scheduleRead() {
    DispatchQueue.global(qos: .userInitiated).async {
        guard let line = readLine(strippingNewline: true), !line.isEmpty else {
            exit(0)  // EOF — Python closed stdin, clean shutdown
        }
        guard let data = line.data(using: .utf8),
              let input = try? JSONDecoder().decode(BatchInput.self, from: data) else {
            FileHandle.standardError.write(Data("apple_translate: invalid JSON, skipping\n".utf8))
            scheduleRead()
            return
        }
        Task { @MainActor in
            await handleRequest(input)
        }
    }
}

scheduleRead()
RunLoop.main.run()
