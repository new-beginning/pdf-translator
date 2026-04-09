// apple_translate — one-shot batch translation using Apple Translation.framework.
//
// Reads ONE JSON object from stdin, translates, writes ONE JSON object to stdout, exits.
// Python spawns a fresh process per page — startup is ~0.1s so overhead is negligible,
// and this avoids main-thread deadlock issues with the framework's async callbacks.
//
//   stdin : {"texts": ["...", "..."], "source": "en", "target": "vi"}
//   stdout: {"translations": ["...", "..."]}
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

let inputData = FileHandle.standardInput.readDataToEndOfFile()
guard let input = try? JSONDecoder().decode(BatchInput.self, from: inputData) else {
    FileHandle.standardError.write(Data("apple_translate: invalid input\n".utf8))
    exit(1)
}

var exitCode: Int32 = 0
let semaphore = DispatchSemaphore(value: 0)

Task {
    do {
        let session = TranslationSession(
            installedSource: Locale.Language(identifier: input.source),
            target: Locale.Language(identifier: input.target)
        )
        var result = input.texts
        for (i, text) in input.texts.enumerated() {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let response = try await session.translate(trimmed)
            result[i] = response.targetText
        }
        let outData = try JSONEncoder().encode(BatchOutput(translations: result))
        FileHandle.standardOutput.write(outData)
        FileHandle.standardOutput.write(Data("\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("apple_translate error: \(error)\n".utf8))
        exitCode = 1
    }
    semaphore.signal()
}

semaphore.wait()
exit(exitCode)
