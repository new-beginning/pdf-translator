// apple_translate — long-lived translation helper using Apple Foundation Models.
//
// Protocol: line-delimited JSON on stdin/stdout.
//   stdin  (one line per batch): {"texts": ["...", "..."], "target": "Vietnamese"}
//   stdout (one line per batch): {"translations": ["...", "..."]}
//
// Compile:
//   swiftc -O apple_translate.swift -o .apple_translate
//
// Requires macOS 26+ (FoundationModels framework).

import Foundation
import FoundationModels

private struct BatchInput: Decodable {
    let texts: [String]
    let target: String   // human-readable language name, e.g. "Vietnamese"
}

private struct BatchOutput: Encodable {
    let translations: [String]
}

let encoder = JSONEncoder()
let decoder = JSONDecoder()

// Create the session once — model is loaded on first respond() call.
let session = LanguageModelSession()

while let line = readLine(strippingNewline: true) {
    guard !line.isEmpty,
          let data = line.data(using: .utf8),
          let input = try? decoder.decode(BatchInput.self, from: data) else {
        FileHandle.standardError.write(Data("apple_translate: bad input line\n".utf8))
        continue
    }

    let semaphore = DispatchSemaphore(value: 0)
    var translations: [String] = Array(repeating: "", count: input.texts.count)

    Task {
        for (i, text) in input.texts.enumerated() {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let prompt = """
                Translate the following English text to \(input.target). \
                Reply with only the translation and nothing else:

                \(trimmed)
                """
            if let response = try? await session.respond(to: prompt) {
                translations[i] = response.content
            } else {
                translations[i] = trimmed   // fallback: return original
            }
        }
        semaphore.signal()
    }

    semaphore.wait()

    if let outData = try? encoder.encode(BatchOutput(translations: translations)),
       let outStr = String(data: outData, encoding: .utf8) {
        print(outStr)   // print() appends \n and flushes on line-buffered stdout
    }
}
