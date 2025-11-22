// Copyright © 2024 Apple Inc.

import AsyncAlgorithms
import MLX
import MLXLLM
import MLXLMCommon
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {
    @Environment(DeviceStat.self) private var deviceStat

    @State var llm = LLMEvaluator()

    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = displayStyle.markdown

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                #if os(iOS)
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle(isOn: $llm.includeWeatherTool) {
                            Text("Include tools")
                        }
                        Toggle(isOn: $llm.enableThinking) {
                            Text("Thinking")
                                .help(
                                    "Switches between thinking and non-thinking modes. Support: Qwen3")
                        }
                        Picker("Mode", selection: Bindable(llm).generationMode) {
                            ForEach(LLMEvaluator.GenerationMode.allCases) { mode in
                                Text(mode.title)
                                    .tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        HStack {
                            Spacer()
                            if llm.running {
                                ProgressView()
                                    .frame(maxHeight: 20)
                                Spacer()
                            }
                            Picker("", selection: $selectedDisplayStyle) {
                                ForEach(displayStyle.allCases, id: \.self) { option in
                                    Text(option.rawValue.capitalized)
                                        .tag(option)
                                }

                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 150)
                        }
                    }
                #else
                    HStack {
                        Toggle(isOn: $llm.includeWeatherTool) {
                            Text("Include tools")
                        }
                        .frame(maxWidth: 150, alignment: .leading)
                        Toggle(isOn: $llm.enableThinking) {
                            Text("Thinking")
                                .help(
                                    "Switches between thinking and non-thinking modes. Support: Qwen3")
                        }
                        .frame(maxWidth: 150, alignment: .leading)
                        Picker("Mode", selection: Bindable(llm).generationMode) {
                            ForEach(LLMEvaluator.GenerationMode.allCases) { mode in
                                Text(mode.title)
                                    .tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        #if os(visionOS)
                            .frame(maxWidth: 250)
                        #else
                            .frame(maxWidth: 250)
                        #endif
                        Spacer()
                        if llm.running {
                            ProgressView()
                                .frame(maxHeight: 20)
                            Spacer()
                        }
                        Picker("", selection: $selectedDisplayStyle) {
                            ForEach(displayStyle.allCases, id: \.self) { option in
                                Text(option.rawValue.capitalized)
                                    .tag(option)
                            }

                        }
                        .pickerStyle(.segmented)
                        #if os(visionOS)
                            .frame(maxWidth: 250)
                        #else
                            .frame(maxWidth: 150)
                        #endif
                    }
                #endif
            }

            // show the model output
            if llm.isBatchMode && !llm.batchResults.isEmpty {
                // Side-by-side batch results
                HStack(spacing: 0) {
                    ForEach(Array(llm.batchResults.enumerated()), id: \.offset) { idx, result in
                        VStack(alignment: .leading, spacing: 8) {
                            // Header
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Prompt \(idx + 1)")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(.blue)
                                Text(result.prompt)
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary)
                                    .lineLimit(2)
                                Divider()
                            }
                            .padding(.horizontal, 12)
                            .padding(.top, 8)

                            // Response in scrollable area
                            ScrollView(.vertical) {
                                VStack(alignment: .leading, spacing: 8) {
                                    if selectedDisplayStyle == .plain {
                                        Text(result.response)
                                            .font(.system(size: 11))
                                            .textSelection(.enabled)
                                    } else {
                                        Markdown(result.response)
                                            .markdownTextStyle(\.text) {
                                                FontSize(11)
                                            }
                                            .textSelection(.enabled)
                                    }

                                    // Footer
                                    let tpsText = result.tokensPerSecond.map { String(format: "%.2f tokens/s", $0) }
                                    let summary = [
                                        "\(result.tokenCount) tokens",
                                        result.finishReason,
                                        tpsText
                                    ].compactMap { $0 }.joined(separator: " • ")
                                    Text(summary)
                                        .font(.system(size: 9))
                                        .foregroundColor(.secondary)
                                        .padding(.top, 8)
                                }
                                .padding(.horizontal, 12)
                                .padding(.bottom, 8)
                            }
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color.gray.opacity(0.05))

                        if idx < llm.batchResults.count - 1 {
                            Divider()
                        }
                    }
                }
            } else {
                // Standard single output
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            if selectedDisplayStyle == .plain {
                                Text(llm.output)
                                    .textSelection(.enabled)
                            } else {
                                Markdown(llm.output)
                                    .textSelection(.enabled)
                            }
                        }
                        .onChange(of: llm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }

                        Spacer()
                            .frame(width: 1, height: 1)
                            .id("bottom")
                    }
                }
            }

            HStack {
                if llm.isBatchMode {
                    ZStack(alignment: .topLeading) {
                        if llm.prompt.isEmpty {
                            Text("Enter multiple prompts, one per line:\nHi\nWhat is quantum computing and why is it important?")
                                .foregroundColor(.gray.opacity(0.6))
                                .padding(.top, 8)
                                .padding(.leading, 5)
                        }
                        TextEditor(text: Bindable(llm).prompt)
                            .frame(height: 60)
                            .disabled(llm.running)
                            .opacity(llm.prompt.isEmpty ? 0.5 : 1)
                    }
                    #if os(visionOS)
                        .border(Color.gray, width: 1)
                    #else
                        .border(Color.gray.opacity(0.5), width: 1)
                    #endif
                } else {
                    TextField("prompt", text: Bindable(llm).prompt)
                        .onSubmit(generate)
                        .disabled(llm.running)
                        #if os(visionOS)
                            .textFieldStyle(.roundedBorder)
                        #endif
                }
                Button(llm.running ? "stop" : "generate", action: llm.running ? cancel : generate)
            }
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }

        }
        .task {
            do {
                // pre-load the weights on launch to speed up the first generation
                _ = try await llm.load()
            } catch {
                llm.output = "Failed: \(error)"
            }
        }
    }

    private func generate() {
        llm.generate()
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}

@Observable
@MainActor
class LLMEvaluator {

    var running = false

    enum GenerationMode: String, CaseIterable, Identifiable {
        case single
        case batch
        case continuous

        var id: Self { self }

        var title: String {
            switch self {
            case .single:
                return "Non-batch"
            case .batch:
                return "Batched"
            case .continuous:
                return "Continuous batch"
            }
        }
    }

    // Store batch results for side-by-side display
    struct BatchResult {
        let prompt: String
        let response: String
        let tokenCount: Int
        let finishReason: String
        let tokensPerSecond: Double?
    }
    var batchResults: [BatchResult] = []

    var generationMode: GenerationMode = .single

    var isBatchMode: Bool {
        generationMode != .single
    }

    var includeWeatherTool = false
    var enableThinking = false

    var prompt = ""
    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. Using Llama 3.2 3B to match Python's batch generation example.
    let modelConfiguration = LLMRegistry.llama3_2_3B_4bit

    /// parameters controlling the output
    let generateParameters = GenerateParameters(maxTokens: 500, temperature: 0.6)
    let updateInterval = Duration.seconds(0.25)
    let continuousStaggerSeconds: Double = 1.0

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    let currentWeatherTool = Tool<WeatherInput, WeatherOutput>(
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: [
            .required(
                "location", type: .string, description: "The city and state, e.g. San Francisco, CA"
            ),
            .optional(
                "unit",
                type: .string,
                description: "The unit of temperature",
                extraProperties: [
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                ]
            ),
        ]
    ) { input in
        let range = input.unit == "celsius" ? (min: -20.0, max: 40.0) : (min: 0, max: 100)
        let temperature = Double.random(in: range.min ... range.max)

        let conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy"].randomElement()!

        return WeatherOutput(temperature: temperature, conditions: conditions)
    }

    let addTool = Tool<AddInput, AddOutput>(
        name: "add_two_numbers",
        description: "Add two numbers together",
        parameters: [
            .required("first", type: .int, description: "The first number to add"),
            .required("second", type: .int, description: "The second number to add"),
        ]
    ) { input in
        AddOutput(result: input.first + input.second)
    }

    let timeTool = Tool<EmptyInput, TimeOutput>(
        name: "get_time",
        description: "Get the current time",
        parameters: []
    ) { _ in
        TimeOutput(time: Date.now.formatted())
    }

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            self.prompt = modelConfiguration.defaultPrompt
            self.modelInfo =
                "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func generate(prompt: String, toolResult: String? = nil) async {

        self.output = ""
        var chat: [Chat.Message] = [
            .system("You are a helpful assistant"),
            .user(prompt),
        ]

        if let toolResult {
            chat.append(.tool(toolResult))
        }

        let userInput = UserInput(
            chat: chat,
            tools: includeWeatherTool
                ? [currentWeatherTool.schema, addTool.schema, timeTool.schema] : nil,
            additionalContext: ["enable_thinking": enableThinking]
        )

        do {
            let modelContainer = try await load()

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) -> Void in
                let lmInput = try await context.processor.prepare(input: userInput)
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context)

                // generate and output in batches
                for await batch in stream._throttle(
                    for: updateInterval, reducing: Generation.collect)
                {
                    let output = batch.compactMap { $0.chunk }.joined(separator: "")
                    if !output.isEmpty {
                        Task { @MainActor [output] in
                            self.output += output
                        }
                    }

                    if let completion = batch.compactMap({ $0.info }).first {
                        Task { @MainActor in
                            self.stat = "\(completion.tokensPerSecond) tokens/s"
                        }
                    }

                    if let toolCall = batch.compactMap({ $0.toolCall }).first {
                        try await handleToolCall(toolCall, prompt: prompt)
                    }
                }
            }

        } catch {
            output = "Failed: \(error)"
        }
    }

    func generate() {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        if isBatchMode {
            batchResults = []
        }

        generationTask = Task {
            running = true
            defer { running = false }

            switch generationMode {
            case .single:
                await generate(prompt: currentPrompt)
            case .batch:
                await generateBatchPrompts(promptText: currentPrompt)
            case .continuous:
                await generateContinuousBatchPrompts(promptText: currentPrompt)
            }
        }
    }

    private func generateBatchPrompts(promptText: String) async {
        self.output = ""

        // Split prompts by newlines and filter empty ones
        let prompts = promptText
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !prompts.isEmpty else {
            self.output = "No prompts provided. Enter prompts separated by newlines."
            return
        }

        self.output = "Processing \(prompts.count) prompts in batch...\n\n"

        do {
            let modelContainer = try await load()

            // Seed for deterministic results
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) -> Void in
                // Create batch parameters with deterministic sampling
                var batchGenParams = generateParameters
                batchGenParams.temperature = 0.0  // Greedy/deterministic sampling

                let batchParams = BatchGenerateParameters(
                    maxTokens: generateParameters.maxTokens ?? 128,
                    completionBatchSize: 32,
                    prefillBatchSize: 8,
                    prefillStepSize: 2048,
                    generation: batchGenParams
                )
                 
                // Use streaming batch generation for multiple prompts
                // Apply chat template to each prompt
                let tokenized: [[Int]] = try prompts.map { prompt in
                    let messages = [[
                        "role": "user",
                        "content": prompt
                    ]]
                    return try context.tokenizer.applyChatTemplate(messages: messages)
                }
                let defaultMaxTokens = Array(
                    repeating: batchParams.defaultMaxTokens, count: tokenized.count)

                // Set up stop tokens (matches batchGenerate pattern)
                var stopTokens = Set<Int>()
                if let eos = context.tokenizer.eosTokenId {
                    stopTokens.insert(eos)
                }
                for token in context.configuration.extraEOSTokens {
                    if let id = context.tokenizer.convertTokenToId(token) {
                        stopTokens.insert(id)
                    }
                }

                var iterator = BatchTokenIterator(
                    model: context.model,
                    parameters: batchParams,
                    stopTokens: stopTokens,
                    unknownTokenId: context.tokenizer.unknownTokenId
                )

                let uids = iterator.insert(prompts: tokenized, maxTokens: defaultMaxTokens)

                // Initialize batch results with empty responses
                Task { @MainActor in
                    self.batchResults = prompts.enumerated().map { (idx, prompt) in
                        BatchResult(
                            prompt: prompt,
                            response: "",
                            tokenCount: 0,
                            finishReason: "generating",
                            tokensPerSecond: nil
                        )
                    }
                }

                var generatedTokens = [Int: [Int]]()
                var finishReasons = [Int: BatchGenerateResult.FinishReason]()

                // Stream tokens and update UI incrementally
                while let responses = iterator.next(), !responses.isEmpty {
                    for response in responses {
                        if response.finishReason != .stop {
                            generatedTokens[response.uid, default: []].append(response.token)
                        }
                        if let reason = response.finishReason {
                            finishReasons[response.uid] = reason
                        }
                    }

                    // Update UI with current state - create isolated copies for MainActor
                    let currentResults = uids.enumerated().map { (idx, uid) -> BatchResult in
                        let tokens = generatedTokens[uid] ?? []
                        let text = context.tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                        let finish = finishReasons[uid]?.rawValue ?? "generating"

                        return BatchResult(
                            prompt: prompts[idx],
                            response: text,
                            tokenCount: tokens.count,
                            finishReason: finish,
                            tokensPerSecond: nil
                        )
                    }
                    
                    Task { @MainActor [currentResults] in
                        self.batchResults = currentResults
                    }
                }

                Stream().synchronize()

                let stats = iterator.stats()

                // Final update with statistics - create isolated copies
                let finalResults = uids.enumerated().map { (idx, uid) -> BatchResult in
                    let tokens = generatedTokens[uid] ?? []
                    let text = context.tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                    let finish = finishReasons[uid]?.rawValue ?? "complete"

                    return BatchResult(
                        prompt: prompts[idx],
                        response: text,
                        tokenCount: tokens.count,
                        finishReason: finish,
                        tokensPerSecond: nil
                    )
                }
                
                Task { @MainActor [finalResults] in
                    self.batchResults = finalResults

                    // Also set output for non-batch mode fallback
                    var formattedOutput = "# Batch Generation Results\n\n"
                    for (idx, batch) in self.batchResults.enumerated() {
                        formattedOutput += "### Prompt \(idx + 1)\n"
                        formattedOutput += "> \(batch.prompt)\n\n"
                        formattedOutput += "**Response:**\n"
                        formattedOutput += "\(batch.response)\n\n"
                        var metrics = "\(batch.tokenCount) tokens • \(batch.finishReason)"
                        if let tps = batch.tokensPerSecond {
                            metrics += " • \(String(format: "%.2f", tps)) tokens/s"
                        }
                        formattedOutput += "*\(metrics)*\n\n"
                        if idx < self.batchResults.count - 1 {
                            formattedOutput += "---\n\n"
                        }
                    }

                    // Add statistics
                    formattedOutput += "\n---\n\n## 📊 Statistics\n\n"
                    formattedOutput += "| Metric | Value |\n"
                    formattedOutput += "|--------|-------|\n"
                    formattedOutput += "| Prompt Tokens | \(stats.promptTokenCount) |\n"
                    formattedOutput += "| Prompt TPS | \(String(format: "%.2f", stats.promptTokensPerSecond)) |\n"
                    formattedOutput += "| Prompt Time | \(String(format: "%.2f", stats.promptTime))s |\n"
                    formattedOutput += "| Generation Tokens | \(stats.generationTokenCount) |\n"
                    formattedOutput += "| Generation TPS | \(String(format: "%.2f", stats.generationTokensPerSecond)) |\n"
                    formattedOutput += "| Generation Time | \(String(format: "%.2f", stats.generationTime))s |\n"
                    formattedOutput += "| Peak Memory | \(String(format: "%.2f", stats.peakMemoryGB)) GB |\n"

                    self.output = formattedOutput
                    self.stat = "\(String(format: "%.2f", stats.generationTokensPerSecond)) tokens/s (batch)"
                }
            }

        } catch {
            output = "Failed: \(error)"
        }
    }

    private func generateContinuousBatchPrompts(promptText: String) async {
        self.output = ""

        let prompts = promptText
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !prompts.isEmpty else {
            self.output = "No prompts provided. Enter prompts separated by newlines."
            return
        }

        self.output = "Starting continuous batching for \(prompts.count) prompts...\n\n"

        do {
            let modelContainer = try await load()

            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let aggregatedStats = try await modelContainer.perform { (context: ModelContext) -> AggregatedStats? in
                let tokenizer = context.tokenizer
                let baseParameters = generateParameters
                let maxTokens = baseParameters.maxTokens ?? 128

                let tokenized: [[Int]] = try prompts.map { prompt in
                    let messages = [[
                        "role": "user",
                        "content": prompt,
                    ]]
                    return try tokenizer.applyChatTemplate(messages: messages)
                }

                let staggerNanoseconds = UInt64(continuousStaggerSeconds * 1_000_000_000.0)

                Task { @MainActor in
                    self.batchResults = prompts.enumerated().map { (idx, prompt) in
                        let delaySeconds = Double(idx) * continuousStaggerSeconds
                        let status: String
                        if idx == 0 {
                            status = "scheduled"
                        } else {
                            status = String(format: "queued (%.1fs)", delaySeconds)
                        }

                        return BatchResult(
                            prompt: prompt,
                            response: "",
                            tokenCount: 0,
                            finishReason: status,
                            tokensPerSecond: nil
                        )
                    }
                }

                let schedulerConfig = SchedulerConfig(
                    maxConcurrentRequests: min(max(1, prompts.count), 8),
                    completionBatchSize: 32,
                    prefillBatchSize: 8,
                    prefillStepSize: 2_048,
                    returnLogProbs: false
                )

                // SAFETY: This is safe because we're within ModelContainer.perform{} isolation boundary.
                // The ModelContext is designed to be used within this actor context, and InferenceScheduler
                // is an actor that will maintain proper isolation of the context. The 'unsafe' annotation
                // acknowledges that we're intentionally transferring the context across concurrency boundaries
                // in a controlled manner that maintains thread safety.
                nonisolated(unsafe) let contextForScheduler = context
                let scheduler = InferenceScheduler(context: contextForScheduler, config: schedulerConfig)
                defer {
                    Task { await scheduler.shutdown() }
                }

                try await withThrowingTaskGroup(of: Void.self) { group in
                    for (index, tokens) in tokenized.enumerated() {
                        let prompt = prompts[index]

                        group.addTask {
                            if Task.isCancelled { return }

                            if index > 0 {
                                let delay = UInt64(Double(index) * Double(staggerNanoseconds))
                                if delay > 0 {
                                    try await Task.sleep(nanoseconds: delay)
                                    if Task.isCancelled { return }
                                }
                            }

                            var requestParameters = baseParameters
                            if requestParameters.maxTokens == nil {
                                requestParameters.maxTokens = maxTokens
                            }

                            let request = InferenceRequest(
                                tokens: tokens,
                                params: requestParameters,
                                maxTokens: requestParameters.maxTokens ?? maxTokens
                            )

                            let stream = await scheduler.submit(request)

                            var accumulatedText = ""
                            var generatedTokens = 0
                            var finishReason: BatchGenerateResult.FinishReason?
                            var started = false

                            for await event in stream {
                                if Task.isCancelled { break }

                                if !started {
                                    started = true
                                    let currentText = accumulatedText
                                    let currentTokens = generatedTokens
                                    Task { @MainActor [currentText, currentTokens] in
                                        guard index < self.batchResults.count else { return }
                                        self.batchResults[index] = BatchResult(
                                            prompt: prompt,
                                            response: currentText,
                                            tokenCount: currentTokens,
                                            finishReason: "running",
                                            tokensPerSecond: nil
                                        )
                                    }
                                }

                                if event.token != nil {
                                    generatedTokens += 1
                                }

                                if let delta = event.textDelta, !delta.isEmpty {
                                    accumulatedText += delta
                                } else if event.token != nil {
                                    let fragment = tokenizer.decode(
                                        tokens: [event.token!],
                                        skipSpecialTokens: true
                                    )
                                    accumulatedText += fragment
                                }

                                if let reason = event.finishReason {
                                    finishReason = reason
                                }

                                // Create isolated copies before sending to MainActor
                                let currentText = accumulatedText
                                let currentTokens = generatedTokens
                                let status = finishReason?.rawValue ?? "generating"

                                Task { @MainActor [currentText, currentTokens, status] in
                                    guard index < self.batchResults.count else { return }
                                    self.batchResults[index] = BatchResult(
                                        prompt: prompt,
                                        response: currentText,
                                        tokenCount: currentTokens,
                                        finishReason: status,
                                        tokensPerSecond: nil
                                    )
                                }
                            }

                            let wasCancelled = Task.isCancelled
                            let finalStatus: String
                            if let finishReason {
                                finalStatus = finishReason.rawValue
                            } else if wasCancelled {
                                finalStatus = "cancelled"
                            } else if started {
                                finalStatus = "complete"
                            } else {
                                finalStatus = "cancelled"
                            }
                            // Create final isolated copies
                            let finalText = accumulatedText
                            let finalTokens = generatedTokens
                            let requestID = request.id
                            let tpsStats = await scheduler.stats(for: requestID)
                            let perStreamTPS = (tpsStats?.tokensPerSecond ?? 0) > 0
                                ? tpsStats?.tokensPerSecond
                                : nil
                            Task { @MainActor [finalText, finalTokens, finalStatus, perStreamTPS] in
                                guard index < self.batchResults.count else { return }
                                self.batchResults[index] = BatchResult(
                                    prompt: prompt,
                                    response: finalText,
                                    tokenCount: finalTokens,
                                    finishReason: finalStatus,
                                    tokensPerSecond: perStreamTPS
                                )
                            }
                        }
                    }

                    try await group.waitForAll()
                }

                Stream().synchronize()

                let aggregated = await scheduler.aggregatedStats()
                return aggregated
            }

            if let aggregated = aggregatedStats {
                self.stat = "\(String(format: "%.2f", aggregated.averageTPS)) tokens/s (continuous)"
            }

            await Task.yield()

            var formattedOutput = "# Continuous Batch Results\n\n"
            for (idx, batch) in batchResults.enumerated() {
                formattedOutput += "### Prompt \(idx + 1)\n"
                formattedOutput += "> \(batch.prompt)\n\n"
                formattedOutput += "**Response:**\n"
                formattedOutput += "\(batch.response)\n\n"
                var metrics = "\(batch.tokenCount) tokens • \(batch.finishReason)"
                if let tps = batch.tokensPerSecond {
                    metrics += " • \(String(format: "%.2f", tps)) tokens/s"
                }
                formattedOutput += "*\(metrics)*\n\n"
                if idx < batchResults.count - 1 {
                    formattedOutput += "---\n\n"
                }
            }

            if let aggregated = aggregatedStats {
                formattedOutput += "---\n\n## 📊 Scheduler Statistics\n\n"
                formattedOutput += "| Metric | Value |\n"
                formattedOutput += "|--------|-------|\n"
                formattedOutput += "| Active Requests | \(aggregated.activeRequests) |\n"
                formattedOutput += "| Queued Requests | \(aggregated.queuedRequests) |\n"
                formattedOutput += "| Total Processed | \(aggregated.totalRequestsProcessed) |\n"
                formattedOutput += "| Average TPS | \(String(format: "%.2f", aggregated.averageTPS)) |\n"
                formattedOutput += "| Peak Memory (GB) | \(String(format: "%.2f", aggregated.peakMemoryGB)) |\n"
            }

            self.output = formattedOutput

        } catch is CancellationError {
            output = "Cancelled"
        } catch {
            output = "Failed: \(error)"
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }

    private func handleToolCall(_ toolCall: ToolCall, prompt: String) async throws {
        let result =
            switch toolCall.function.name {
            case currentWeatherTool.name:
                try await toolCall.execute(with: currentWeatherTool).toolResult
            case addTool.name:
                try await toolCall.execute(with: addTool).toolResult
            case timeTool.name:
                try await toolCall.execute(with: timeTool).toolResult
            default:
                "No tool match"
            }

        await generate(prompt: prompt, toolResult: result)
    }
}

struct WeatherInput: Codable {
    let location: String
    let unit: String?
}

struct WeatherOutput: Codable {
    let temperature: Double
    let conditions: String
}

struct AddInput: Codable {
    let first: Int
    let second: Int
}

struct AddOutput: Codable {
    let result: Int
}

struct EmptyInput: Codable {}

struct TimeOutput: Codable {
    let time: String
}
