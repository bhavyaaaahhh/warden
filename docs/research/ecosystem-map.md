# Ecosystem map: offline evaluation of LLM apps and agents

**Framed question.** What is the best known approach to offline evaluation of LLM apps and agents, especially multi-turn conversations and tool use? That covers the test-case model, how conversations are executed or simulated, scorers and judges, and regression comparison between runs.

**Pass.** Breadth only. Written 2026-09-29. Identity data (license, language, last push, archived flag) came from the GitHub REST API (`gh api repos/<owner>/<repo>`) on 2026-09-29. Capability claims come from official docs or READMEs fetched the same day, or from arXiv abstracts. All URLs are in `sources.md` under "Ecosystem map".

**Legend**
- Tags: [doc] = official docs or README, [paper] = academic or technical report, [inferred] = my interpretation, [hypothesis] = needs validation. No capability claim here comes from code. Those claims wait for the deep-dive dossiers (`projects/<name>.md`).
- Mode: Offline = batch or experiment evaluation against a dataset. Online = scoring production traffic or traces. Both = both.
- Multi-turn (MT): does the tool accept a *multi-turn test case*, meaning a conversation or a simulated user? Yes / Partial / No / Unknown, with a tag. "Partial" means multi-turn data can be scored, but there is no first-class conversation test case or simulator, or the docs are ambiguous.
- Last activity = last `pushed_at` on the default repo. "n/a" = closed source.
- License "other" = GitHub could not classify it. Known cases: Langfuse is MIT core plus an `ee/` directory; Phoenix is Elastic License 2.0 [inferred, from repo license file naming; verify in dossier].
- Stars are left out on purpose (weak signal). A few are noted only where they help tell forks apart.

---

## 1. Open-source eval frameworks and libraries

| Name | Mode | License | Lang | Last activity | Purpose | MT test cases |
|---|---|---|---|---|---|---|
| DeepEval (confident-ai/deepeval) | Offline (+ Confident AI cloud) | Apache-2.0 | Python | 2026-09-28 | Pytest-style LLM eval framework with a metric library | **Yes**: `ConversationalTestCase` plus a multi-turn simulation guide [doc] |
| Ragas (vibrantlabsai/ragas; moved from explodinggradients) | Offline | Apache-2.0 | Python | 2026-02-24 | RAG and agent metrics | **Yes**: `MultiTurnSample`, Tool Call Accuracy/F1, Agent Goal Accuracy [doc] |
| promptfoo | Offline (CLI/CI), red-team | MIT | TypeScript | 2026-09-29 | Declarative YAML test matrix across prompts, providers and assertions | **Yes**: chat/conversation config plus a "Simulated User" provider "inspired by Tau-bench" [doc] |
| OpenEvals (langchain-ai/openevals) | Offline + online | MIT | Python | 2026-09-29 | Ready-made LLM-as-judge and other evaluators | **Yes**: README documents multi-turn simulation utilities [doc] |
| AgentEvals (langchain-ai/agentevals) | Offline | MIT | Python | 2026-07-14 | Trajectory evaluators (match modes, trajectory LLM judge) | **Partial**: scores message/tool trajectories, no user simulator [doc] |
| autoevals (braintrustdata/autoevals) | Offline | MIT | Python/TS | 2026-09-23 | Scorer library (LLM classifiers, heuristics, embeddings) used by Braintrust | Unknown |
| Inspect AI (UKGovernmentBEIS/inspect_ai) | Offline | MIT | Python | 2026-09-29 | Task/solver/scorer eval framework, agents, sandboxes, structured eval logs | **Yes**: README cites "multi-turn dialog", agent docs [doc] |
| OpenAI Evals (openai/evals) | Offline | other | Python | 2026-04-14 | Eval framework plus benchmark registry | **Partial**: chat-format samples; multi-turn simulation not documented [inferred] |
| simple-evals (openai/simple-evals) | Offline | MIT | Python | 2026-04-22 | Reference implementations of OpenAI's reported evals (hosts HealthBench) | **Partial**: HealthBench items are multi-turn conversations [paper] |
| lm-evaluation-harness (EleutherAI) | Offline | MIT | Python | 2026-09-14 | Few-shot model benchmark harness | **No**: model-level single-prompt tasks [inferred] |
| lighteval (huggingface) | Offline | MIT | Python | 2026-09-28 | Model benchmark toolkit over many backends | **No/Partial** [inferred] |
| HELM (stanford-crfm/helm) | Offline | Apache-2.0 | Python | 2026-09-01 | Holistic model benchmarking (scenarios × metrics) | **No/Partial** [inferred] |
| Giskard OSS v3 (Giskard-AI/giskard-oss) | Offline | Apache-2.0 | Python | 2026-09-29 | Checks/suites, LLM judge, vulnerability scan; v3 rewrite targets agents | **Yes**: "Evaluate multi-turn agents — test full conversations"; `LLMGenerator` synthesizes user messages [doc] |
| TruLens | Both | MIT | Python | 2026-09-29 | Feedback functions plus tracing for apps and agents | Unknown (agent tracing documented; conversation test case not verified) |
| Guardrails AI | Online (runtime validation) | Apache-2.0 | Python | 2026-09-25 | Input/output validators | **No**: runtime guard, not a test harness [inferred] |
| Pydantic Evals (pydantic/pydantic-ai) | Offline | MIT | Python | 2026-09-29 | Typed `Dataset`/`Case`/evaluators, span-based evaluation | **Partial**: span-based evaluation of agent runs; no simulator found in docs [doc] |
| Strands Evals (strands-agents/evals) | Offline | Apache-2.0 | Python | 2026-09-28 | Output, trajectory and interaction evaluators, experiment generation, chaos testing | **Yes**: "Multi-turn conversation simulation with realistic user behavior ... LLM-powered tool simulation with shared state" [doc] |
| Google ADK eval (google/adk-python) | Offline | Apache-2.0 | Python | 2026-09-29 | Agent eval sets, trajectory + response scoring, user simulator | **Yes**: user simulator documented (audio-capable) [doc] |
| Vertex Gen AI evaluation (googleapis/python-aiplatform) | Offline (managed) | Apache-2.0 (SDK) | Python | 2026-09-28 | Managed eval service: computation + model-based metrics, agent/trajectory metrics | **Yes**: official notebook for "multi-turn agent evaluation with user simulation" [doc] |
| Azure AI Evaluation SDK (in Azure/azure-sdk-for-python) | Both | MIT (SDK) | Python | 2026-09-29 | Evaluators incl. `IntentResolution`, `ToolCallAccuracy`, `AIAgentConverter` | **Partial**: converts agent threads for evaluation; simulator status not verified [doc] |
| Mastra evals (mastra-ai/mastra) | Both | other | TypeScript | 2026-09-29 | Built-in/custom scorers, datasets, experiments over traces | Unknown |
| promptflow (microsoft) | Offline | MIT | Python | 2026-08-26 | Flow authoring plus batch evaluation flows | **Partial** [inferred] |
| Harbor (harbor-framework/harbor; laude-institute) | Offline | Apache-2.0 | Python | 2026-09-28 | Framework for evaluating and improving agents in containerized tasks (runs Terminal-Bench) | **No** user sim; agentic task episodes [inferred] |

## 2. Platforms (hosted or self-hostable, with experiments and run comparison)

| Name | Mode | License | Lang | Last activity | Purpose | MT test cases |
|---|---|---|---|---|---|---|
| LangSmith (+ langsmith-sdk) | Both | Proprietary; SDK MIT | Py/TS | 2026-09-29 (SDK) | Datasets, experiments, comparison views, tracing | **Yes**: multi-turn simulation docs [doc] |
| Braintrust (braintrust-sdk-python/-javascript) | Both | Proprietary; SDK Apache-2.0 | Py/TS | 2026-09-29 | `Eval()` experiments with diff against a baseline experiment | **Partial**: docs mention multi-step/multi-turn; simulator not verified [doc] |
| Arize Phoenix | Both | other (ELv2 [inferred]) | Python | 2026-09-29 | OTel/OpenInference tracing, evals, datasets, experiments | **Partial**: sessions not confirmed in fetched eval page; simulator not verified [inferred] |
| Langfuse | Both | other (MIT + ee) | TypeScript | 2026-09-29 | Tracing, sessions, datasets, experiments, LLM-judge evaluators | **Partial**: sessions are first-class; dataset items are single I/O [doc]/[inferred] |
| MLflow GenAI eval | Both | Apache-2.0 | Python | 2026-09-29 | `mlflow.genai.evaluate`, scorers, judges, tracing | **Yes**: multi-turn evaluation and conversation simulation (MLflow blog/docs) [doc] |
| W&B Weave | Both | Apache-2.0 | Python | 2026-09-28 | `Evaluation` objects, compare evaluations, "identify regressions" | Unknown |
| Opik (comet-ml) | Both | Apache-2.0 | Python | 2026-09-29 | Tracing, experiments, thread evaluation | **Yes**: "Evaluate threads", conversation-level GEval [doc] |
| Confident AI (DeepEval cloud) | Both | Proprietary | n/a | n/a | Hosted DeepEval: regression views, persona conversation simulation | **Yes** [doc] |
| LangWatch (langwatch/langwatch) | Both | Apache-2.0 | TypeScript | 2026-09-29 | Eval + agent-testing platform (hosts Scenario) | **Yes** [doc] |
| Agenta | Both | other | TypeScript | 2026-09-29 | Prompt/agent workspace with evaluation | Unknown |
| Latitude (latitude-llm) | Both | MIT | TypeScript | 2026-09-29 | Observability + evaluations for agents | Unknown |
| Laminar (lmnr) | Both | Apache-2.0 | TypeScript | 2026-09-29 | Agent observability + evals | Unknown |
| OpenLIT | Both | Apache-2.0 | TypeScript | 2026-09-29 | OTel-native observability + evaluation | Unknown |
| Rhesis (rhesis-ai/rhesis) | Offline | other | Python | 2026-09-29 | Collaborative test sets, review; "Penelope" conversation simulator, "Polyphemus" adversarial probe | **Yes**: simulates multi-turn conversations, per-turn feedback [doc] |
| Evidently | Both | Apache-2.0 | Python | 2026-09-29 | ML/LLM evals, test suites, monitoring | Unknown |
| Arthur (arthur-engine) | Both (monitoring-first) | MIT | Python | 2026-09-29 | Monitoring, governance, guardrails, evals | Unknown |
| Patronus AI (patronus-py SDK not found at guessed path) | Both | Proprietary | n/a | n/a | Evaluators/judges (Lynx, GLIDER), experiments | Unknown (docs returned HTTP 401) |
| Galileo (galileo-python) | Both | Proprietary; SDK Apache-2.0 | Python | 2026-08-18 | Eval + observability; sessions | **Partial**: "Sessions" in docs [doc] |
| Maxim (maxim-py) | Both | Proprietary; SDK Apache-2.0 | Python | 2026-09-02 | Eval, text + voice simulation, observability | **Yes**: "Text Simulation", "Voice Simulation" [doc] |
| Future AGI (agent-learning-kit; simulate-sdk archived) | Both | other / Apache-2.0 | Python | 2026-09-29 | Evaluation + simulation environment | **Yes**: "Simulation" in docs [doc] |
| Scorecard | Offline-first | Proprietary; SDK Apache-2.0 | Python | 2026-09-22 | "Simulation platform" for running agents through scenarios | **Yes** [doc] |

## 3. Conversation and user simulators, agent-testing harnesses

These tools execute a test case by having a simulated user talk to the agent. That is the central mechanism for multi-turn offline eval.

| Name | Mode | License | Lang | Last activity | Purpose | MT test cases |
|---|---|---|---|---|---|---|
| LangWatch Scenario (langwatch/scenario) | Offline | Apache-2.0 | TypeScript (+Py) | 2026-09-29 | Simulation-based agent testing; user simulator + judge agent inside a test runner | **Yes**: "evaluate and judge at any point of the conversation" [doc] |
| IntellAgent (plurai-ai/intellagent) | Offline | Apache-2.0 | Python | 2026-09-14 | Generates policy-graph scenarios, simulates users at scale, diagnoses failures | **Yes** [doc]/[paper] |
| ArkSim (arklexai/arksim) | Offline | Apache-2.0 | Python | 2026-08-05 | Synthetic users with profile/goal/knowledge, scores every turn | **Yes** [doc] |
| AWS agent-evaluation (awslabs) | Offline | Apache-2.0 | Python | 2025-12-15 | An evaluator LLM runs concurrent multi-turn conversations with the target agent | **Yes** [doc] |
| TRACER (Chatbot-TRACER) | Offline | GPL-3.0 | Python | 2026-06-02 | Explores a chatbot, models workflows as graphs, generates user profiles for the Sensei simulator | **Yes** [doc] |
| Coval | Offline + online | Proprietary | n/a | n/a | Voice/chat agent simulation and evaluation | **Yes**: "Simulated Conversations" [doc] |
| Cekura | Offline + online | Proprietary | n/a | n/a | Voice agent scenario simulation and testing | **Yes** [doc] |
| Bluejay | Offline | Proprietary | n/a | n/a | Agent simulation testing | **Yes**: "Simulations" [doc] |
| Egma (egma-ai/egma) | Both | other | TypeScript | 2026-09-28 | OSS voice/text simulation regression suites with mocked tool responses, personas, graders, pass thresholds | **Yes** [doc] |
| voicetest | Offline | Apache-2.0 | Python | 2026-07-20 | Voice agent harness (imports Retell/VAPI/Bland/LiveKit), LLM users + LLM judges | **Yes** [doc] |
| SOTOPIA (sotopia-lab) | Offline (research) | MIT | Python | 2026-06-05 | Multi-agent social-interaction simulation and evaluation | **Yes** [paper] |
| Meta ARE / Gaia2 | Offline | MIT | Python | 2026-08-26 | Research environment for dynamic, asynchronous agent scenarios | **Yes**: dynamic scenarios [doc] |

Red-team and adversarial (multi-turn attacks are the relevant part):

| Name | Mode | License | Lang | Last activity | Purpose | MT |
|---|---|---|---|---|---|---|
| PyRIT (microsoft/PyRIT; Azure/PyRIT is archived) | Offline | MIT | Python | 2026-09-29 | Risk identification / red-teaming orchestrators | **Yes** (multi-turn attack orchestrators; not confirmed in README text) [inferred] |
| garak (NVIDIA) | Offline | Apache-2.0 | Python | 2026-09-16 | LLM vulnerability scanner (probes/detectors) | **Partial** [inferred] |

## 4. Benchmarks (multi-turn, tool use, agents)

| Name | Mode | License | Lang | Last activity | Purpose | MT |
|---|---|---|---|---|---|---|
| MT-Bench (lm-sys/FastChat `llm_judge`) | Offline | Apache-2.0 | Python | 2026-05-01 | 80 two-turn questions judged by GPT-4 | **Yes** (2 turns, fixed) [paper] |
| MT-Bench-101 | Offline | Apache-2.0 | n/a | 2024-07-24 | Fine-grained multi-turn dialogue taxonomy | **Yes** [paper] |
| MT-Eval | Offline | n/a (repo not located) | n/a | n/a | Multi-turn capability benchmark (recollection, expansion, refinement, follow-up) | **Yes** [paper] |
| MultiChallenge (Scale) | Offline | n/a (repo not located) | n/a | n/a | Realistic multi-turn challenges that are hard for frontier models; instance-level rubric judge | **Yes** [paper] |
| Lost in Conversation (microsoft/lost_in_conversation) | Offline | MIT | Python | 2026-06-09 | "Sharded" instructions revealed across turns; measures multi-turn degradation | **Yes** [paper] |
| MINT | Offline | Apache-2.0 | Python | 2024-06-04 | Multi-turn tool use with simulated language feedback | **Yes** [paper] |
| ToolTalk (microsoft) | Offline | MIT | Python | 2024-05-31 | Tool use in conversational settings | **Yes** [paper] |
| τ-bench (sierra-research/tau-bench) | Offline | MIT | Python | 2026-03-18 | Tool-agent-user benchmark with LLM user simulator; DB-state check; introduced pass^k | **Yes** [paper] |
| τ²-bench (sierra-research/tau2-bench) | Offline | MIT | Python | 2026-09-28 | Dual-control (user also has tools); text half-duplex + voice full-duplex | **Yes** [doc]/[paper] |
| BFCL v3+ (ShishirPatil/gorilla) | Offline | Apache-2.0 | Python | 2026-04-13 | Function-calling leaderboard; v3 adds multi-turn, multi-step with state checks | **Yes** [doc] |
| ToolSandbox (apple-aiml-research) | Offline | other | Python | 2026-09-11 | Stateful, conversational tool-use benchmark with user simulator and milestones | **Yes** [paper] |
| AppWorld | Offline | Apache-2.0 | Python | 2026-09-04 | Controllable app world, state-based unit-test grading | **Partial** (task-level, not dialog) [paper] |
| AgentBench (THUDM) | Offline | Apache-2.0 | Python | 2026-02-08 | 8 environment agent benchmark | **Partial** (multi-turn env interaction) [paper] |
| WebArena | Offline | Apache-2.0 | Python | 2025-11-26 | Self-hosted realistic websites, functional grading | **Partial** [paper] |
| BrowserGym / WorkArena (ServiceNow) | Offline | other | Python | 2026-09-27 | Unified web-agent gym; enterprise knowledge-work tasks | **Partial** [paper] |
| OSWorld | Offline | Apache-2.0 | Python | 2026-09-14 | Real computer-environment tasks | **Partial** [paper] |
| GAIA | Offline | n/a (HF dataset) | n/a | n/a | General assistant questions with exact-answer grading | **No** (single task) [paper] |
| SWE-bench | Offline | MIT | Python | 2026-09-18 | Resolve GitHub issues, graded by tests | **No** user dialog [paper] |
| Terminal-Bench (harbor-framework/terminal-bench-1) | Offline | Apache-2.0 | Python | 2026-07-11 | Terminal tasks in containers | **No** user dialog [doc] |
| Toolathlon (hkust-nlp) | Offline | none detected | Python | 2026-08-18 | Diverse, long-horizon tasks over MCP servers | **Partial** [doc] |
| MCP-Bench | Offline | n/a | n/a | n/a | Tool-using agents over real MCP servers | **Partial** [paper] |
| ToolBench (OpenBMB) | Offline | Apache-2.0 | Python | 2025-05-21 | Large-scale API tool use; ToolEval judge | **Partial** [paper] |
| HealthBench (in simple-evals) | Offline | MIT | Python | 2026-04-22 | 5k multi-turn health conversations with physician-written per-conversation rubrics | **Yes** [paper] |
| Arena-Hard-Auto / AlpacaEval (LC) | Offline | Apache-2.0 | Python | 2025-06 / 2025-08 | Pairwise-judge chat benchmarks; length-controlled debiasing | **No** (single-turn) [paper] |
| HAL: Holistic Agent Leaderboard (princeton-pli/hal-harness, archived) | Offline | none detected | Python | 2026-07-01 | Cross-benchmark agent harness reporting cost and accuracy | **Partial** [paper] |
| AgentRewardBench (McGill-NLP) | Offline (meta-eval) | none detected | Python | 2026-08-16 | Benchmarks LLM judges on web-agent trajectories | n/a (judge meta-eval) [paper] |
| JudgeBench | Offline (meta-eval) | n/a | n/a | n/a | Benchmarks LLM judges on hard pairs | n/a [paper] |

## 5. Methods (papers)

| Method | Year | Source | Relevance |
|---|---|---|---|
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | 2023 | arXiv 2306.05685 | Founding study of pairwise/single judging; finds position, verbosity and self-enhancement biases [paper] |
| G-Eval | 2023 | arXiv 2303.16634 | CoT + form-filling judge; probability-weighted scores [paper] |
| Prometheus / Prometheus 2 | 2023/2024 | arXiv 2310.08491, 2405.01535; repo prometheus-eval (Apache-2.0, last push 2025-04-25) | Open judge models for rubric-based absolute + pairwise grading [paper] |
| Large Language Models are not Fair Evaluators | 2023 | arXiv 2305.17926 | Position bias; calibration by swapping order [paper] |
| Justice or Prejudice? (CALM) | 2024 | arXiv 2410.02736 | Taxonomy of 12 judge biases [paper] |
| Judging the Judges | 2024 | arXiv 2406.12624 | Judge alignment and vulnerabilities vs humans [paper] |
| ChatEval | 2023 | arXiv 2308.07201 | Multi-agent debate judges [paper] |
| A Survey on LLM-as-a-Judge | 2024 | arXiv 2411.15594 | Survey of judge reliability techniques [paper] |
| Length-Controlled AlpacaEval | 2024 | arXiv 2404.04475 | Regression-based length debiasing [paper] |
| Agent-as-a-Judge | 2024 | arXiv 2410.10934; repo metauto-ai/agent-as-a-judge (MIT) | Agentic judge that inspects intermediate steps [paper] |
| AdaRubric | 2026 | arXiv 2603.21362; repo alphadl/AdaRubrics (Apache-2.0) | Task-adaptive rubrics for agent trajectories [paper] |
| Rubrics as Rewards | 2025 | arXiv 2507.17746 | Rubric-based grading beyond verifiable domains [paper] |
| HealthBench | 2025 | arXiv 2505.08775 | Per-conversation rubric criteria with weights, graded by a model [paper] |
| τ-bench pass^k | 2024 | arXiv 2406.12045 | Reliability metric: probability that all k i.i.d. trials succeed [paper] |
| Adding Error Bars to Evals | 2024 | arXiv 2411.00640 | CLT standard errors, clustered SEs, paired-difference tests for comparing runs [paper] |
| Holistic Agent Leaderboard | 2025 | arXiv 2510.11977 | Standardized, cost-aware, multi-benchmark agent eval [paper] |
| User Simulation in the Era of Generative AI (Balog & Zhai) | 2025 | arXiv 2501.04410 | Survey: user simulation for evaluation [paper] |
| A Survey on LLM-based Conversational User Simulation | 2026 | arXiv 2604.24977 | Survey specific to LLM user simulators [paper] |
| Reliable LLM-based User Simulator for TOD | 2024 | arXiv 2402.13374 | Simulator reliability for task-oriented dialog [paper] |
| Goal Alignment in LLM-Based User Simulators | 2025 | arXiv 2507.20152 | Simulators drift from their goals; tracking fixes [paper] |
| SAGE | 2025 | arXiv 2510.11997 | Knowledge-grounded user simulator for multi-turn agent eval [paper] |
| IntellAgent | 2025 | arXiv 2501.11067 | Policy-graph-driven scenario generation + simulation [paper] |
| Efficient Agent Evaluation via Diversity-Guided User Simulation | 2026 | arXiv 2604.21480 | Coverage/diversity of simulated users [paper] |
| VISTA | 2026 | arXiv 2606.11079 | Interactive user-simulation toolkit for agent evaluation [paper] |
| GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation | 2026 | arXiv 2609.12191 | Judge failure modes specific to simulated-user evals [paper] |
| Catching One in Five: judge blind spots in multi-turn transaction agents | 2026 | arXiv 2606.10315 | Judge recall on production multi-turn agents [paper] |
| Designing Reliable LLM-as-a-Judge Measurement Systems for Multi-Turn Business Agents | 2026 | arXiv 2609.33955 | Measurement-system design for multi-turn judges [paper] |
| LLMs Get Lost in Multi-Turn Conversation | 2025 | arXiv 2505.06120 | Large performance drop when instructions are spread over turns [paper] |

## 6. Standards and instrumentation (context only)

| Name | License | Last activity | Purpose |
|---|---|---|---|
| OpenTelemetry GenAI semantic conventions | Apache-2.0 | active | Standard span attributes for LLM/agent calls; the likely interchange format for trace-based eval [doc] |
| OpenLLMetry (traceloop) | Apache-2.0 | 2026-09-29 | OTel instrumentation for LLM SDKs [doc] |

## 7. Candidates found but not verified (for a follow-up pass)

Snowglobe (guardrails-ai), Hamming, Kolena (repo exists: kolenaIO/kolena, Apache-2.0, 2026-07-17), Sensei (satori-chatbots/user-simulator; repo path 404), Goose (Raff-dev/goose, MIT, a conversational agent test library), CRAB-Bench (arXiv 2606.01815), KnowSim (arXiv 2608.17150), AgentJudgeBench (arXiv 2608.26623), MCP-GRANITE. Their docs either did not load or were not checked. Do not cite them as evidence yet.

## Counts

| Category | Count |
|---|---|
| Frameworks / libraries | 23 |
| Platforms | 21 |
| Simulators / agent-testing harnesses (incl. 2 red-team) | 14 |
| Benchmarks (incl. 2 judge meta-benchmarks) | 27 |
| Methods papers | 28 |
| Standards / instrumentation | 2 |
| **Total mapped** | **115** (plus 9 unverified candidates) |

## Early observations (for the deep-dive selection, not conclusions)

- **Simulated-user execution is converging on one shape.** The pattern: a scenario (persona + goal + hidden info), a user-simulator LLM, a stop condition or max turns, then a judge and/or an environment-state check. Many unrelated tools document it: τ-bench, ToolSandbox, promptfoo's simulated user ("inspired by Tau-bench"), LangWatch Scenario, Strands Evals, ADK, Vertex, MLflow, DeepEval, LangSmith/OpenEvals, IntellAgent and ArkSim [doc]/[inferred].
- **The best benchmarks grade outcomes with deterministic state checks, not only judges.** Examples: τ-bench's DB-state comparison, BFCL v3 state checks, AppWorld unit tests, ToolSandbox milestones. Judges are used for the conversational quality that state checks cannot see [paper]/[inferred].
- **Reliability metrics beyond pass@1 exist but are rare in products.** pass^k (τ-bench) and paired/clustered standard errors (Miller 2024) are well defined, but most platform docs describe experiment diffing, not statistical comparison [paper]/[hypothesis; verify in dossiers].
- **Simulator validity is its own open research area in 2025-2026.** Examples: goal drift, persona neutrality gaps, and judge blind spots under simulation (GAUGE, "Catching One in Five") [paper].
- **Suggested deep-dive set, ranked by relevance to the framed question:** τ²-bench, LangWatch Scenario, DeepEval (conversational + simulator), Inspect AI (eval log + scorer model), Strands Evals, promptfoo (simulated user + comparison), Braintrust or LangSmith (experiment comparison), MLflow GenAI (multi-turn + simulation) [inferred].
