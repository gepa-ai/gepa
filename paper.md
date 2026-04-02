Under review as a conference paper at COLM 2026
Combee: Scaling Prompt Learning for Self-Improving Language Model Agents
Anonymous authors
Paper under double-blind review
Abstract
1 Recent advances in prompt learning allow large language model agents
2 to acquire task-relevant knowledge from inference-time context without
3 parameter changes. For example, existing methods (like ACE or GEPA) can
4 learn system prompts to improve accuracy based on previous agent runs.
5 However, these methods primarily focus on single-agent or low-parallelism
6 settings. This fundamentally limits their ability to efficiently learn from
7 a large set of collected agentic traces. It would be efficient and beneficial
8 to run prompt learning in parallel to accommodate the growing trend of
9 learning from many agentic traces or parallel agent executions. Yet with10 out a principled strategy for scaling, current methods suffer from quality
11 degradation with high parallelism. To improve both the efficiency and
12 quality of prompt learning, we propose Combee, a novel framework to
13 scale parallel prompt learning for self-improving agents. Combee speeds
14 up learning and enables running many agents in parallel while learning
15 from their aggregate traces without quality degradation. To achieve this,
16 Combee leverages parallel scans and employs an augmented shuffle mech17 anism; Combee also introduces a dynamic batch size controller to balance
18 quality and delay. Evaluations on AppWorld, Terminal-Bench, Formula,
19 and FiNER demonstrate that Combee achieves up to 17× speedup over
20 previous methods with comparable or better accuracy and equivalent cost.
21 1 Introduction
0 2000 4000
Training time (s)
50
55
60
65
70
Avg score
bs 40
bs 20
bs 10
bs 5
bs 1
Quality-Delay Tradeoff
BaseLLM
ACE
Combee
ACE (bs 1) ACE (bs 5) ACE (bs 10) ACE (bs 20) ACE (bs 40) Combee (bs 40)
0
2000
4000
6000
8000
Context size (tokens)
Improved Prompt Learning
ACE (bs 1) ACE (bs 5) ACE (bs 10) ACE (bs 20) ACE (bs 40) Combee (bs 40)
0.0
0.5
1.0
1.5
2.0
2.5
Training Cost ($)
$1.62 $1.68
$1.50 $1.40 $1.40
$1.67
Cost Comparison
Figure 1: Summary of improvement snapshot. Combee achieves close-to-optimal quality
with significantly reduced training time by increasing the content in the prompt learnt under
high parallelism. Experiments with DeepSeek-V3.1 on AppWorld.
22 Large language models (LLMs) achieve strong performance in tasks such as mathematics
23 and programming (Lu et al., 2024; Jimenez et al., 2024; Agarwal et al., 2024). However,
24 real-world problem solving usually requires learning from information that is only available
25 at inference time (Dou et al., 2026; Mang et al., 2025). This information is typically provided
26 as context (such as documents, examples, tool traces, or execution histories), serving as
27 additional input at inference time that cannot be incorporated through offline training alone.
28 Recent work has shown that language agents can engage in prompt learning to improve cur29 rent or future task performance: extracting task-relevant knowledge from rich inference-time
30 inputs (trajectories, documents, tool traces) and consolidating it into reusable artifacts such
1
Under review as a conference paper at COLM 2026
31 as playbooks or rules, without any weight updates (Khattab et al., 2024; Zhang et al., 2025a;
32 Agrawal et al., 2025; Shinn et al., 2023; Wang et al., 2023). For example, ACE (Zhang et al.,
33 2025a) enables agents to adapt during inference by consolidating experience into structured
34 playbooks, while GEPA (Agrawal et al., 2025) optimizes prompts based on performance
35 feedback from contextual examples. These approaches demonstrate that inference-time
36 context can serve as a powerful learning medium without updating parameters.
37 However, existing prompt learning methods (including ACE and GEPA) were designed
38 around sequential or low-parallelism updates, where one or a small number of trajectories
39 are reflected on and consolidated at a time, and thus provide no principled strategy for
40 scaling the reflection-and-aggregation step to high parallelism. This is increasingly limiting:
41 as agentic systems grow in scale, agents produce large volumes of interaction traces (Wang
42 et al., 2024b; Zhao et al., 2024; Yang et al., 2024) that ideally would be learned from concur43 rently, and parallel multi-agent deployments are becoming standard practice in academia (Li
44 et al., 2024; Hong et al., 2023; Qian et al., 2024a) and industry (Cursor, 2026; Anthropic, 2026).
45 Yet naively increasing parallelism creates a bottleneck: the aggregator LLM responsible for
46 consolidating many reflections must process increasingly long-horizon reflective context
47 at once, and becomes overwhelmed. We refer to this context overload. Concretely (§2.2),
48 scaling from batch 1 to batch 100 on the Formula dataset (Wang et al., 2025a) drops accuracy
49 from 87.0% to 72.5%, and qualitative analysis reveals that the aggregator only retained
50 generic patterns while discarding the specific, high-value entries that drive downstream
51 performance. Prompt-level mitigations such as summarization and top-K retrieval do not
52 resolve this (§4). This bottlenecks both efficient learning from large-scale agent traces and
53 timely adaptation in parallel agent scenarios (Snell et al., 2024; Li et al., 2024).
54 To address this problem, we propose Combee, a distributed framework for scalable prompt
55 learning. Combee adopts a Map-Shuffle-Reduce paradigm: multiple agents process distinct
56 context shards in parallel (Map), reflections are duplicated and shuffled to prevent infor57 mation loss (Shuffle), and a hierarchical parallel scan algorithm aggregates local updates
58 into a coherent global context without overloading the LLM context curator (Reduce). A dy59 namic batch size controller further balances quality and training delay automatically across
60 iterations. Combee is framework-agnostic and integrates with existing prompt learning
61 methods with minimal changes: we prototype it on both ACE and GEPA, and expect it to
62 generalize to other generate-reflect-update frameworks. Evaluations on agentic benchmarks,
63 AppWorld (Trivedi et al., 2024) and Terminal-Bench (Merrill et al., 2026) and domain-specific
64 tasks, FiNER (Loukas et al., 2022) and Formula (Wang et al., 2025a) demonstrate that Combee
65 achieves up to 17× speedup over baselines with comparable or improved accuracy while
66 maintaining equivalent cost.
67 To summarize, our contributions are to:
68 • Identify the problem of efficient scaling for prompt learning and failure of previous
69 methods (§2.2).
70 • Design Combee, a novel framework for scalable prompt learning featuring parallel scan
71 aggregation, augmented shuffling, and a dynamic batch size controller (§3).
72 • Prototype Combee on top of ACE and GEPA, and expect it to generalize to other generate73 reflect-update frameworks with minimal changes.
74 • Perform evaluations on Terminal-Bench 2.0, AppWorld, Formula, and FiNER to show
75 that Combee achieves up to 17× speedup with comparable or improved accuracy and
76 equivalent cost over baselines (§4).
77 2 Background and Motivation
78 2.1 Prompt Learning
79 Prompt learning is an emerging inference-time learning paradigm in which an agent extracts
80 task-relevant knowledge from rich inputs (execution trajectories, tool traces, documents)
81 and consolidates it into reusable artifacts such as playbooks, memories, or skill libraries that
82 improve current or future performance without any weight updates (Zhang et al., 2025a;
83 Agrawal et al., 2025; Suzgun et al., 2025; Wang et al., 2025b).
2
Under review as a conference paper at COLM 2026
1 5 10 20 50 100
Batch size
0
100
200
# of Context Updates
264
199
121
97
47
21
Accuracy: 87.0
Accuracy: 72.5
Formula
1 5 20 50 125
Batch size
246
184
110
39
11
Accuracy: 76.0
Accuracy: 70.6
FiNER
Figure 2: Context overload from naive scaling. As batch size increases, the aggregator LLM
produces monotonically fewer and less useful context updates, directly degrading final
accuracy across benchmarks.
84 In this work, we focus on methods that follow a generate-reflect-update loop: an agent executes
85 a task, reflects on its trajectory to extract useful insights, and updates a shared context artifact
86 for future iterations. Our goal is to scale this loop to high parallelism, spinning up multiple
87 agents concurrently per iteration, while preserving the quality of the resulting context
88 updates. This paradigm is broadly adopted: some systems distill skills or programs from
89 trajectories into reusable libraries (Wang et al., 2023; 2025b; Zhang et al., 2026; Xia et al., 2026),
90 while others evolve structured memories, playbooks, or system prompts from accumulated
91 experience (Zhang et al., 2025a; Agrawal et al., 2025; Zhao et al., 2024; Suzgun et al., 2025;
92 Ouyang et al., 2025; Shinn et al., 2023; Zhou et al., 2025). The breadth of this family confirms
93 that generate-reflect-update is a well-established foundation, making its parallel scaling a
94 natural and important problem. We also note that “prompt learning” has been used with
95 varying scope in recent work (Dou et al., 2026); throughout this paper, we use it specifically
96 to refer to this generate-reflect-update paradigm.
97 Relationship to Prompt Engineering Prompt engineering methods focus on crafting a
98 fixed prompt a priori, either manually or through an offline search procedure (Wei et al.,
99 2022; Zhou et al., 2022), that is then deployed without further modification at inference time.
100 In contrast, prompt learning as studied in this work treats the prompt (or more broadly, the
101 context artifact) as a living object that evolves during deployment through a generate-reflect102 update loop: agents interact with tasks, reflect on outcomes, and iteratively revise the shared
103 context based on accumulated experience. The key distinction is that prompt engineering
104 optimizes what to say to the model before deployment, whereas prompt learning optimizes
105 what the model knows from experience as it runs.
106 2.2 The Problem: Context Overload from Naive Parallel Scaling
107 A natural approach to scale prompt learning for the general generate-reflect-update paradigm
108 mentioned above is to increase the batch size of reflections before generating context updates,
109 which can aggregate more feedback signals before updating the context. Here, batch size
110 refers to the number of parallel agent trajectories or reflections aggregated before producing
111 one context update in an iteration. This is appealingly simple, but fails in practice due to a
112 phenomenon we call context overload: as batch size grows, the aggregator LLM must distill an
113 increasingly large volume of reflections into a single context update, producing far fewer and
114 lower-quality entries. Critically, this degradation occurs even when all reflections fit within
115 the model’s context window (we use DeepSeek-V3.1 with 128K context), ruling out simple
116 truncation as the cause. Instead, the aggregator appears to perform lossy compression:
117 when presented with many reflections simultaneously, it defaults to retaining broad, generic
118 patterns while discarding the specific, high-value insights that disproportionately drive
119 downstream accuracy.
120 Quantitative Evidence Figure 2 demonstrates the information loss from naive scaling
121 across Formula (numerical reasoning) (Wang et al., 2025a) and FiNER (financial entity
122 recognition) (Loukas et al., 2022). In both settings, the number of context updates drops
123 monotonically with batch size: on Formula from 264 (batch 1) to 21 (batch 100), on FiNER
3
Under review as a conference paper at COLM 2026
Query
Context
ÆAgent Reflect
ÆAgent Reflect
ÆAgent Reflect
ÆAgent Reflect
Augmented Shuffle
¶
Prompt Update
Prompt Update
.
.
.
Combine Context
Update
ÔBatch Size Controller
next iteration
Map (bs = n)
Augment
Reduce (Parallel Scan)
Figure 3: Overall design of Combee, following a Map-Shuffle-Reduce paradigm. In the
Map phase, n (bs) parallel agents execute queries and reflect on their trajectories. The
Shuffle phase applies augmented shuffling to the reflections. In the Reduce phase, a parallel
scan aggregation hierarchically combines the reflections into a context update. Naive scaling
only contains the Map phase and directly generates context updates after reflection.
124 from 246 to 11. Accuracy follows the same trend: Formula drops from 87.0% to 72.5%,
125 FiNER from 76.0% to 70.6%. The same pattern holds on agentic tasks: on AppWorld (Trivedi
126 et al., 2024), scaling from batch 1 to batch 40 reduces accuracy from 58.1 to 55.7, approaching
127 the no-context-learning baseline of 53.3 (Table 1).
128 Qualitative Evidence The degradation goes beyond quantity. In ACE, the final system
129 prompt learnt is a playbook with many entries. Each entry is marked helpful (h) or harmful
130 (r) during inference, providing a measure of entry utility. Under sequential learning (batch
131 1), the Formula playbook accumulates 174 total helpful hits across 264 entries, with 19
132 entries reaching h ≥ 3 and a maximum of h = 16; the FiNER playbook accumulates 331
133 helpful hits across 246 entries, with 38 entries reaching h ≥ 3. Under naive scaling, these
134 high-value entries vanish entirely: the batch 100 Formula playbook retains zero entries with
135 h ≥ 3 (total helpful hits: 5), and the batch 125 FiNER playbook retains zero (total helpful
136 hits: 4). Appendix E provides concrete playbook snapshots illustrating how task-specific
137 strategies (e.g., formula-specific edge-case handling, precise rounding protocols) collapse
138 into generic reminders under high parallelism.
139 This reveals a fundamental tension: increasing parallelism reduces wall-clock training time,
140 but naive aggregation destroys the fine-grained knowledge that makes prompt learning
141 effective. The sweet spot for naive scaling, small batch sizes that partially avoid overload,
142 yields only modest speedups, while the large batch sizes needed for meaningful acceleration
143 collapse quality toward the no-context-learning baseline.
144 3 Design of Combee
145 We present Combee, a framework that enables scalable prompt learning through parallel
146 generation and adaptation. Combee extends prompt learning from previous work (Zhang
147 et al., 2025a; Agrawal et al., 2025; Li et al., 2024) to support high degrees of parallelism while
148 maintaining quality.
149 To address the challenges of context overload, Combee introduces three key components:
150 parallel scan aggregation, augmented shuffling and dynamic batch size controller. These compo151 nents work together to ensure that the learning process remains stable and efficient under
152 high parallelism: (1) To solve the context overload problem, we employ a parallel scan
153 algorithm for aggregating learned experience from multiple trajectories. (2) To make sure
154 that important information is not missed out, Combee applies augmented shuffling before
155 dispatching reflections to the aggregation tree, giving each reflection multiple chances to be
156 incorporated. (3) For learning from a large number of traces, we introduce a dynamic batch
4
Under review as a conference paper at COLM 2026
157 size controller that dynamically determines an efficient yet safe batch size at run time. These
158 optimizations allow Combee to coordinate between multiple parallel agents and improve
159 itself over time without manual tuning, just like a bee colony where agents work together to
160 efficiently build and maintain the system.
161 3.1 Parallel Scan Aggregation
162 One core design inside Combee is the parallel scan aggregation algorithm, which is designed to
163 efficiently aggregate learning experience from multiple parallel trajectories while avoiding
164 the context overload problem observed in §2.2.
165 In order to solve this problem, Combee employs a multi-level parallel scan algorithm to
166 aggregate learning experience from multiple trajectories in a way that prevents overloading
167 the aggregator. Given the n generated trajectories, Combee can choose to first separate
168 them into k mini-batches, each containing n/k trajectories. Instead of directly feeding all
169 the reflections from the n trajectories to the aggregator, Combee can first aggregate the
170 reflections within each mini-batch of trajectories into context updates. Then, Combee can
171 further aggregate the k context updates into a single update for this round. Conceptually,
172 this is similar to the parallel scan algorithm used in parallel computing for performing
173 prefix sum operations (Blelloch, 1990) and recently adopted in sequence modeling (Gu &
174 Dao, 2024). This approach also draws inspiration from MapReduce-style decomposition for
175 LLM processing of long documents (Zhou et al., 2024). By default, Combee sets k equal to
⌊
√
176 n⌋ so that each level of the aggregation tree processes similar count of entries : the first
level generates context updates based on n/k =
√
177 n reflections per group, while the second
level aggregates k =
√
178 n updates. Moreover, this design empirically achieves better quality
179 as shown later in Figure 8.
180 3.2 Augmented Shuffling
181 Popular context engineering methods, including GEPA (Agrawal et al., 2025) and
182 ACE (Zhang et al., 2025a), all incorporate reflection steps to extract past insights from
183 rollouts. These reflections usually have higher information density: although they consist
184 of a small number of tokens, they contain crucial information necessary for the agent’s
185 improvement. To fully leverage this dense information during parallel learning without
186 losing vital insights, Combee introduces an augmented shuffling mechanism. Specifically,
187 given a set of x generated reflections, Combee duplicates each reflection p times (default
188 p = 2) and shuffles the augmented set before issuing them to the worker nodes. By giving
189 each reflection multiple opportunities to contribute to the learning process, echoing the
190 principle behind self-consistency (Wang et al., 2022), Combee increases the chances that the
191 aggregator can learn from the reflections even under large batch sizes. This improves the
192 robustness of the parallel learning pipeline despite increased batch size.
193 3.3 Dynamic Batch Size Controller
194 Parallel scan aggregation and augmented shuffling ensure that learning quality is main195 tained across a wide range of batch sizes. The batch size selection therefore primarily reduces
196 to a speed question: as the batch size increases, per-epoch delay decreases (more samples
197 are processed in parallel); but with diminishing returns, analogous to the critical batch size
198 concept from distributed training (McCandlish et al., 2018). That said, excessively large
199 batch sizes may still degrade learning quality, so we would like to stay within a reasonable
200 range. We therefore select the largest batch size that still yields meaningful delay reduction,
201 while enforcing an upper bound to avoid unnecessary risk of quality degradation (Smith
202 et al., 2017; Goyal et al., 2017).
203 To find this point, we profile the delay by running trial iterations at a set of default candidate
batch sizes {bs1, bs2, . . . , bsk}. For each candidate bsi
204 , we run one iteration to measure the
delay d(bs 205 i) and convert it to estimated epoch time:
Tepoch(bs) = d(bs) ·
Ntrain
bs ,
206 where Ntrain is the training set size. We fit a power-law delay curve through measurements:
Tepoch(bs) = A · bs−α
.
5
Under review as a conference paper at COLM 2026
207 Given the fitted curve, we select the batch size at which the marginal delay reduction falls
below a fixed threshold τ
1
. Solving



dTepoch
d bs


208  = τ yields:
plateau bs =

αA
τ
 1
α+1
.
209
210 4 Results
211 Our main takeaways from evaluating Combee are:
212 • Combee integrates with existing prompt learning methods such as ACE and GEPA to
213 enable efficient learning at scale, achieving comparable or even better performance with
214 significantly reduced training time.
215 • Combee’s specialized design including parallel scan aggregation and augmented shuffle
216 prevents context overload and improves on previous parallel methods.
217 • Combee remains robust across different models, tasks, and learning settings with cost
218 comparable to previous methods.
219 4.1 Experiment Setup
220 Tasks and Datasets We evaluate Combee on agentic and domain-specific benchmarks.
221 • Agentic Benchmarks: AppWorld (Trivedi et al., 2024) evaluates multi-step API tasks
222 via Task Goal Completion (TGC) and Scenario Goal Completion (SGC). We reuse the
223 training set of 90 tasks and evaluate on the held out Test-Normal dataset. Terminal-Bench
224 2.0 (Merrill et al., 2026) contains 89 command-line tasks testing software engineering
225 capabilities. We train on 60 Deepseek 3.2 trajectories released on huggingface (Lee, 2026)
226 and evaluate average Accuracy@1 across three runs on 29 held-out tasks.
227 • Domain-Specific Benchmarks: We use two finance NLP datasets: FiNER (Loukas et al.,
228 2022) for fine-grained entity typing in XBRL documents, and Formula (Wang et al., 2025a)
229 for numerical reasoning over structured filings.
230 Frameworks and Baselines For majority of experiments, we use DeepSeek-V3.1 provided
231 by Together AI as the base LLM. Combee is agnostic to the base prompt learning method:
232 We implement Combee on top of two prompt learning: ACE (Zhang et al., 2025a), which
233 accumulates strategies into text-based playbooks, and GEPA (Agrawal et al., 2025), which
234 optimizes system prompts via evolutionary search. Both follow a generate-reflect-update
235 loop that Combee extends for parallel scaling. We also compared with two methods on
236 top of ACE and GEPA: Top-K Retrieval and Summarization. Top-K Retrieval embeds
237 reflections, clusters them into K groups, and feeds one reflection from each group to the
238 curator. Summarization summarizes reflections before feeding them into the curator.
239 4.2 Results on Agent Benchmarks
240 Table 1 shows results on AppWorld. The results reveal a clear trade-off in naive parallel
241 scaling: the sequential baseline (batch 1) takes 86 minutes to complete one epoch, whereas
242 increasing the batch size reduces training time but suffers from context overload. The sweet
243 spot for naive scaling is batch 10; beyond this point, quality degrades sharply, and batch 40
244 drops to barely above the no-context-learning baseline. This confirms that increasing
245 parallelism without proper aggregation is harmful.
246 Combee breaks this trade-off. At batch size 40, where naive scaling degrades severely,
247 Combee achieves the highest average score and SGC across all methods, with a 12× speedup
248 over the sequential baseline at comparable cost. A key reason is that Combee’s playbook
1
In our experiments, we set τ to 1.6% of the peak slope size. This means we stop increasing batch
size once each new unit reduces epoch time by less than 1.6% of the steepest improvement rate.
6
Under review as a conference paper at COLM 2026
Method Batch Playbook Training Training Test-Normal
Size Size (tokens) Time (min) Cost TGC↑ SGC↑ Avg↑
ReAct – – 0 $0 63.7 42.9 53.3
ReAct + ACE 1 1,578 86 $1.62 66.1 50.0 58.1
Parallel Prompt Learning
ReAct + ACE 5 4,697 30 $1.68 70.2 57.1 63.7
ReAct + ACE 10 2,329 19 $1.50 72.0 58.9 65.4
ReAct + ACE 20 954 10 $1.40 67.9 48.2 58.1
ReAct + ACE 40 526 5 $1.40 66.7 44.6 55.7
ReAct + Combee 40 6,887 7 $1.67 70.8 60.7 65.8
Table 1: Parallel prompt learning results with ReAct agent for AppWorld.
Method Batch Playbook Training Training Average
Size Size (tokens) Time (min) Cost Accuracy @ 1
Terminus-2 – – 0 $0 32.2%
Terminus-2 + ACE 1 9,067 42.4 $0.24 37.9%
Parallel Prompt Learning
Terminus-2 + ACE 5 4,983 10.2 $0.17 29.9%
Terminus-2 + ACE 10 3,967 5.6 $0.15 33.3%
Terminus-2 + ACE 30 3,150 2.1 $0.13 31.0%
Terminus-2 + Combee 30 8,023 2.4 $0.17 35.6%
Table 2: Parallel prompt learning results with Terminus-2 agent for Terminal-Bench 2.0.
We trained on existing open-source traces instead of generating trajectories on the fly. We
report the average accuracy over three runs.
249 retains 6,887 tokens compared to only 526 for naive batch 40, indicating that parallel scan
250 aggregation preserves far more information from reflections.
251 Table 2 shows results on Terminal-Bench 2.0. The same pattern emerges: the sequential
252 baseline achieves the highest accuracy but requires 42 minutes of training, while larger batch
253 sizes degrade due to context overload: batch 5 even falls below the no-context-learning
254 baseline. Combee at batch 30 recovers most of the sequential quality while reducing training
255 time by over 17×. Notably, Combee’s playbook size (8,023 tokens) is much closer to the
256 sequential baseline (9,067 tokens) than other batched variants, again confirming that parallel
257 scan aggregation retains more information.
258 4.3 Results on Domain Specific Benchmarks
259 Figure 4 and Figure 5 show results on the two finance benchmarks, FiNER and Formula,
260 using GEPA and ACE respectively. Since there are a large number of training samples
261 in Formula (500) and FiNER (1000), we employ the dynamic batch size controller to dy262 namically adjust the batch size during training. For summarization and Top K retrieval
263 baselines, we use batch size 50 as it has similar delay with Combee. We set K = 5 and use
264 openai/text-embedding-3-large as the embedding model.
265 The same quality–speed trade-off from the agent benchmarks persists across both tasks
266 and both frameworks: small batch sizes yield higher accuracy but require long training
267 times, while large batch sizes are fast but suffer from context overload: on GEPA FiNER,
268 batch 100 even falls below the base LLM. Combee consistently reaches the Pareto frontier,
269 matching or exceeding the best fixed-batch accuracy while training significantly faster than
270 quality-matching setups. With GEPA (Figure 4), Combee matches the best fixed-batch
7
Under review as a conference paper at COLM 2026
Base LLM Summarization + GEPA TopK + GEPA GEPA ⋆ Combee + GEPA
0 1000 2000 3000
Training Time (s)
0.6
0.7
0.8
0.9
1.0
Accuracy
100
50
25 10
5
Formula
0 1000 2000 3000
Training Time (s)
0.60
0.65
0.70
0.75
0.80
0.85
100
50
25 10
5
FiNER
Figure 4: Combee achieves superior quality-delay trade off on GEPA for finance benchmarks.
Base LLM Summarization + ACE TopK + ACE ACE ⋆ Combee + ACE
0 2000 4000 6000 8000
Training Time (s)
0.6
0.7
0.8
0.9
1.0
Accuracy
60
50
20 10
5
Formula
0 1000 2000 3000 4000 5000
Training Time (s)
0.60
0.65
0.70
0.75
0.80
0.85
60 50
20
10
5
FiNER
Figure 5: Combee achieves superior quality-delay trade off on ACE for finance benchmarks.
271 accuracy on FiNER and achieves competitive accuracy on Formula with less than half of the
272 time by fixed-batch baseline. With ACE (Figure 5), Combee achieves the highest accuracy
273 on Formula and FiNER, while training more than 2.4× faster than the quality-comparable
274 baselines. The Top K and Summarization baselines achieved much worse generation quality
275 compared with Combee or naive ACE methods. These results confirm that Combee’s design
276 is framework-agnostic and effective for domain-specific tasks.
277 4.4 Extended Analysis
278 We conduct ablation study and robustness analysis on the Formula dataset.
Baseline Normal parallel ⋆ Combee
0 10000 20000
Training Time (s)
0.50
0.75
1.00
Accuracy
100
50
20 10
5
ACE
0 2500 5000
Training time(s)
0.50
0.75
1.00
100
5020
10 5
GEPA
Figure 6: Combee achieves similar improvement with GPT-OSS 120B.
279
280 Figure 6 evaluates Combee on top of GPT281 OSS 120B on the same Formula dataset. The
282 batch size controller and parallel scan aggre283 gator transfer seamlessly across model fami284 lies, and Combee with GPT-OSS follows the
285 same pattern: superior quality over fixed286 batch baselines with much reduced training
287 time.
288 Figure 7 ablates the dynamic batch size con289 troller by comparing Combee against a vari290 ant that uses a fixed batch size throughout
291 training on the Formula dataset. Without a dynamic controller, Combee with a fixed batch
292 size may choose a necessarily small batch size, causing a delay increase with little quality
293 change. This demonstrates the effectiveness of the batch size controller of Combee.
294 Figure 8 demonstrates the effectiveness of the augmented shuffling for Combee. We com295 pared Combee against the plain parallel scan variant across different group sizes. The batch
296 size is set to be 50. Without augmented shuffling, quality fluctuates and is significantly
8
Under review as a conference paper at COLM 2026
Baseline Normal parallel ⋆ Combee
0 5000
Training time (s)
0.50
0.75
1.00
Accuracy
40
60
20
ACE
0 2000
Training time (s)
0.50
0.75
1.00
6050 25
20
GEPA
Figure 7: Combee’s dynamic batch size
adaptation saves delay besides maintaining
high accuracy
w/ augmented shuffling
w/o augmented shuffling
6 8 10 15 20
Minibatch Size
0.75
0.80
0.85
Accuracy
ACE
6 8 10 15 20
Minibatch Size
0.825
0.850
GEPA
Figure 8: Combee’s augmented shuffling
improves learning robustness across minibatch sizes.
297 worse than Combee, confirming the necessity of our design. Moreover, when mini-batch
size is around √
298 bs, the quality is usually higher, which validates our design in §3.1.
299 5 Related Work
300 Memory Mechanism for LLMs and Agents Prompt learning has been extensively used to
301 help language models and language agents improve over time by maintaining and updating
302 an external non-parametric memory. Dynamic Cheatsheet (Suzgun et al., 2025; Xu et al.,
303 2025; Shinn et al., 2023) demonstrates that compact, evolving textual memory can help
304 language agents adapt at inference time by accumulating reusable guidance from past
305 experience. ReasoningBank (Ouyang et al., 2025) similarly investigates how agents can
306 store and reuse distilled reasoning traces or experience to support future problem solving.
307 Agentic Plan Caching (Zhang et al., 2025b) extends this direction by caching reusable plans
308 from prior executions to reduce repeated reasoning and improve efficiency. These works,
309 along with ACE (Zhang et al., 2025a), GEPA (Agrawal et al., 2025), ExpeL (Zhao et al., 2024),
310 Voyager (Wang et al., 2023), Agent-Pro (Zhang et al., 2024a), and TextGrad (Yuksekgonul
311 et al., 2024), primarily focus on what information to store, retrieve, or reuse across tasks.
312 In contrast, our work focuses on how to scale the context-learning process itself: rather than
313 proposing a new memory abstraction, we study how multiple workers can learn in parallel
314 and how context updates can be aggregated effectively under high concurrency. The parallel
315 abstraction we proposed is expected to work along with existing memory frameworks.
316 Parallel Agents Recent work has explored parallel agent systems, where multiple agents
317 or workers collaborate to solve tasks concurrently (Hong et al., 2023; Qian et al., 2024a).
318 Learning to Share (LTS) (Fioresi et al., 2026) studies how agents can share useful intermediate
319 information while avoiding redundant computation. In practice, modern agentic coding
320 systems such as Claude Code (Anthropic, 2025), OpenHands (Wang et al., 2024b), and SWE321 agent (Yang et al., 2024) also increasingly rely on parallel task decomposition and concurrent
322 execution to improve throughput on complex workloads (Li et al., 2024; Qian et al., 2024b).
323 However, these systems are primarily concerned with parallelizing task solving (Zhang et al.,
324 2024b; Zhou et al., 2024; Wang et al., 2024a). Our focus is orthogonal: we study how to
325 parallelize learning from experience, i.e., how multiple workers can independently produce
326 local context updates and how those updates can be merged into a coherent global context.
327 Thus, while prior work on parallel agents improves execution efficiency, our work addresses
328 the systems challenges of scalable context adaptation.
329 6 Conclusion
330 We presented Combee, a novel framework for scalable context learning that enables parallel
331 agents to acquire and consolidate knowledge efficiently. By combining parallel-scan aggre332 gation, augmented shuffling, and dynamic batch-size control, Combee addresses the context
333 overload issue that emerges when existing context learning methods are scaled naively.
334 Across agentic benchmarks (AppWorld and Terminal-Bench 2.0) and domain-specific tasks
335 (FiNER and Formula), Combee delivers substantial speedups and maintains or improves
336 quality with negligible cost variations. We believe prompt learning is entering a new era of
337 scale, and Combee is a first step toward making that possible.
9
Under review as a conference paper at COLM 2026
338 Ethics Statement
339 This paper and the associated research do not raise any specific ethical concerns.
340 Reproducibility Statement
341 We clearly describe the experimental setup used in our study, including the language models,
342 datasets, and hyperparameters, so that readers with appropriate compute resources should
343 be able to reproduce our results. All experiments in this paper are conducted on publicly
344 available benchmarks. The source code will be released upon publication.
345 References
346 Mart´ın Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean,
347 Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. {TensorFlow}:
348 a system for {Large-Scale} machine learning. In 12th USENIX symposium on operating
349 systems design and implementation (OSDI 16), pp. 265–283, 2016.
350 Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao
351 Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot in-context learning.
352 Advances in Neural Information Processing Systems, 37:76930–76966, 2024.
353 Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl354 Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. GEPA:
355 Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv preprint
356 arXiv:2507.19457, 2025.
357 Anthropic. Claude code. https://claude.ai/code, 2025.
358 Anthropic. Building a c compiler with a team of parallel claudes. https://www.anthropic.
359 com/engineering/building-c-compiler, February 2026. Anthropic Engineering Blog.
360 Guy E Blelloch. Prefix sums and their applications. 1990.
361 Cursor. Scaling long-running autonomous coding. https://cursor.com/blog/
362 scaling-agents, January 2026. Accessed: 2026-02-25.
363 Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao,
364 Marc’aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, et al. Large scale distributed
365 deep networks. Advances in neural information processing systems, 25, 2012.
366 Shihan Dou, Ming Zhang, Zhangyue Yin, Chenhao Huang, Yujiong Shen, Junzhe Wang,
367 Jiayi Chen, Yuchen Ni, Junjie Ye, Cheng Zhang, et al. CL-bench: A Benchmark for Context
368 Learning. arXiv preprint arXiv:2602.03587, 2026.
369 Joseph Fioresi, Parth Parag Kulkarni, Ashmal Vayani, Song Wang, and Mubarak Shah.
370 Learning to Share: Selective Memory for Efficient Parallel Agentic Systems. arXiv preprint
371 arXiv:2602.05965, 2026.
372 Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo ´
373 Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd:
374 Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
375 Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces.
376 In First conference on language modeling, 2024.
377 Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang,
378 Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta program379 ming for a multi-agent collaborative framework. In The twelfth international conference on
380 learning representations, 2023.
10
Under review as a conference paper at COLM 2026
381 Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and
382 Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?,
383 2024. URL https://arxiv.org/abs/2310.06770.
384 Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri
385 Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather
386 Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language
387 model calls into self-improving pipelines. 2024.
388 Yoonho Lee. Terminalbench trajectories, 2026. URL https://huggingface.co/datasets/
389 yoonholee/terminalbench-trajectories. Dataset, accessed 2026.
390 Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need.
391 arXiv preprint arXiv:2402.05120, 2024.
392 Mu Li, David G. Andersen, Jun Woo Park, Alexander J. Smola, Amr Ahmed, Vanja Josifovski,
393 James Long, Eugene J. Shekita, and Bor-Yiing Su. Scaling distributed machine learning
394 with the parameter server. In 11th USENIX Symposium on Operating Systems Design and
395 Implementation (OSDI 14), 2014.
396 Lefteris Loukas, Manos Fergadiotis, Ilias Chalkidis, Eirini Spyropoulou, Prodromos Malaka397 siotis, Ion Androutsopoulos, and Georgios Paliouras. Finer: Financial numeric entity
398 recognition for xbrl tagging. arXiv preprint arXiv:2203.06482, 2022.
399 Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi,
400 Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evalu401 ating mathematical reasoning of foundation models in visual contexts, 2024. URL
402 https://arxiv.org/abs/2310.02255.
403 Qiuyang Mang, Wenhao Chai, Zhifei Li, Huanzhi Mao, Shang Zhou, Alexander Du,
404 Hanchen Li, Shu Liu, Edwin Chen, Yichuan Wang, et al. Frontiercs: Evolving challenges
405 for evolving intelligence. arXiv preprint arXiv:2512.15699, 2025.
406 Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical
407 model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.
408 Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich,
409 Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench:
410 Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint
411 arXiv:2601.11868, 2026.
412 Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le,
413 Samira Daruki, Xiangru Tang, et al. ReasoningBank: Scaling Agent Self-Evolving with
414 Reasoning Memory. arXiv preprint arXiv:2509.25140, 2025.
415 Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang,
416 Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software
417 development. In Proceedings of the 62nd annual meeting of the association for computational
418 linguistics (volume 1: Long papers), pp. 15174–15186, 2024a.
419 Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang,
420 Zhuoyun Du, Weize Chen, Cheng Yang, et al. Scaling large language model-based
421 multi-agent collaboration. arXiv preprint arXiv:2406.07155, 2024b.
422 Benjamin Recht, Christopher Re, Stephen Wright, and Feng Niu. Hogwild!: A lock-free
423 approach to parallelizing stochastic gradient descent. Advances in neural information
424 processing systems, 24, 2011.
425 Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.
426 Reflexion: Language Agents with Verbal Reinforcement Learning. Advances in Neural
427 Information Processing Systems, 36:8634–8652, 2023.
428 Samuel L Smith, Pieter-Jan Kindermans, Chris Ying, and Quoc V Le. Don’t decay the
429 learning rate, increase the batch size. arXiv preprint arXiv:1711.00489, 2017.
11
Under review as a conference paper at COLM 2026
430 Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute opti431 mally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314,
432 2024.
433 Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic
434 Cheatsheet: Test-Time Learning with Adaptive Memory. arXiv preprint arXiv:2504.07952,
435 2025.
436 Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward
437 Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. AppWorld: A
438 Controllable World of Apps and People for Benchmarking Interactive Coding Agents.
439 arXiv preprint arXiv:2407.18901, 2024.
440 Dannong Wang, Jaisal Patel, Daochen Zha, Steve Y Yang, and Xiao-Yang Liu. Finlora:
441 Benchmarking lora methods for fine-tuning llms on financial datasets. arXiv preprint
442 arXiv:2505.19819, 2025a.
443 Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi
444 Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large
445 language models. arXiv preprint arXiv:2305.16291, 2023.
446 Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents
447 enhances large language model capabilities. arXiv preprint arXiv:2406.04692, 2024a.
448 Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi
449 Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai
450 software developers as generalist agents. arXiv preprint arXiv:2407.16741, 2024b.
451 Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha
452 Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in
453 language models. arXiv preprint arXiv:2203.11171, 2022.
454 Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. Inducing program455 matic skills for agentic tasks. arXiv preprint arXiv:2504.06821, 2025b.
456 Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V
457 Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language
458 models. Advances in neural information processing systems, 35:24824–24837, 2022.
459 Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han,
460 Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive
461 skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.
462 Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem:
463 Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.
464 John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik
465 Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated
466 software engineering. Advances in Neural Information Processing Systems, 37:50528–50652,
467 2024.
468 Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin,
469 and James Zou. Textgrad: Automatic” differentiation” via text. arXiv preprint
470 arXiv:2406.07496, 2024.
471 Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and
472 Wenya Wang. Memskill: Learning and evolving memory skills for self-evolving agents.
473 arXiv preprint arXiv:2602.02474, 2026.
474 Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar
475 Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic Context
476 Engineering: Evolving Contexts for Self-Improving Language Models. arXiv preprint
477 arXiv:2510.04618, 2025a.
12
Under review as a conference paper at COLM 2026
478 Qizheng Zhang, Michael Wornow, Gerry Wan, and Kunle Olukotun. Agentic Plan Caching:
479 Test-Time Memory for Fast and Cost-Efficient LLM Agents. arXiv preprint arXiv:2506.14852,
480 2025b.
481 Wenqi Zhang, Ke Tang, Hai Wu, Mengna Wang, Yongliang Shen, Guiyang Hou, Zeqi Tan,
482 Peng Li, Yueting Zhuang, and Weiming Lu. Agent-pro: Learning to evolve via policy-level
483 reflection and optimization. In Proceedings of the 62nd Annual Meeting of the Association for
484 Computational Linguistics (Volume 1: Long Papers), pp. 5348–5375, 2024a.
Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan O Arık. Chain ¨ 485
486 of agents: Large language models collaborating on long-context tasks. Advances in Neural
487 Information Processing Systems, 37:132208–132237, 2024b.
488 Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang.
489 Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on
490 Artificial Intelligence, volume 38, pp. 19632–19642, 2024.
491 Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee,
492 Guchun Zhang, Kun Shao, Linyi Yang, et al. Memento: Fine-tuning llm agents without
493 fine-tuning llms. arXiv preprint arXiv:2508.16153, 2025.
494 Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan,
495 and Jimmy Ba. Large language models are human-level prompt engineers. In The eleventh
496 international conference on learning representations, 2022.
497 Zihan Zhou, Chong Li, Xinyi Chen, Shuo Wang, Yu Chao, Zhili Li, Haoyu Wang, Rongqiao
498 An, Qi Shi, Zhixing Tan, et al. LLMxMapReduce: Simplified long-sequence processing
499 using large language models. arXiv preprint arXiv:2410.09342, 2024.
13
Under review as a conference paper at COLM 2026
500 A Use of Large Language Models (LLMs)
501 This work focuses on developing algorithms and system frameworks for effective context
502 adaptation in large language models (LLMs). Accordingly, our experiments employ LLMs
503 for the empirical evaluation of the proposed methods. For paper preparation, we used
504 LLMs only to polish writing (e.g., correcting grammatical errors), and not to generate
505 new text from scratch. We also use Claude Code and Cursor in the empirical experiment
506 development process.
507 B Limitations and Future Work
508 While Combee demonstrates consistent improvements across our evaluation settings, sev509 eral aspects remain open for future work. First, our experiments focus on two base prompt
510 learning frameworks (ACE and GEPA). Although both follow the generate-reflect-update
511 paradigm and Combee’s design is intended to be framework-agnostic, we plan to further
512 validate integration with methods that maintain structurally different context artifacts (e.g.,
513 program libraries or retrieval-augmented skill stores). Second, the dynamic batch size
514 controller relies on a power-law delay model with a fixed marginal-reduction threshold
515 (τ), which performed well in our settings but may require adjustment for workloads with
516 substantially different latency profiles. Finally, the current design assumes synchronous
517 parallel execution within each iteration; exploring asynchronous or partially-synchronous
518 variants, analogous to asynchronous SGD in distributed training, could further improve
519 throughput in heterogeneous deployment environments and is an interesting direction we
520 leave for future investigation.
521 C Extended Problem Formulation
522 We formalize the problem of Prompt Learning at Scale here. For the previous single-threaded
523 prompt learning pipeline, agents first execute the task, and then reflect upon their execution
524 to update their context. Mathematically, let Ct denote the agent context at iteration t, E the
525 environment, τt the interaction trajectory, and Ft the feedback extracted from the trajectory.
526 The process can be written as
τt ∼ Exec(A, E | Ct), Ft = Reflect(τt
, E | Ct), Ct+1 = Update(Ct
, Ft).
527 For the new paradigm of prompt learning at scale, we adopt a Map–Reduce style approach.
For each iteration, we spin up parallel agents (A1, A2, . . . , Abs 528 ) to interact with the environment and collect feedback (F
(1)
t
, F
(2)
t
, . . . , F
(bs)
t
). Each agent produces its own trajectory τ
(i)
t
529
530 and feedback signal. We then aggregate the feedback through an aggregation function to
531 update the global agent context. Mathematically, the pipeline can be represented as
τ
(i)
t ∼ Exec(Ai
, E | Ct), F
(i)
t = Reflect(τ
(i)
t
, E | Ct), i = 1, . . . , bs,
532
F¯
t = Agg
F
(1)
t
, F
(2)
t
, . . . , F
(bs)
t

, Ct+1 = Update(Ct
, F¯
t).
533 D Analogy to Distributed Training
534 We motivate the research vision by drawing an analogy between parallel prompt learning
535 and distributed training of machine learning models. In distributed training, learning
536 is parallelized across multiple workers, each processing a shard of data and computing
537 local gradients. These gradients are periodically aggregated, either synchronously or asyn538 chronously, by a parameter server or through collective communication, yielding a globally
539 improved model without requiring any single worker to observe the full dataset (Recht
540 et al., 2011; Dean et al., 2012; Li et al., 2014; Abadi et al., 2016).
541 Prompt Learning at Scale follows a similar pattern, but replaces parameter updates with
542 contextual adaptation. Instead of updating shared weights, multiple agents or workers
14
Under review as a conference paper at COLM 2026
543 independently interact with tasks, environments, or documents, and learn from their local
544 contexts during inference. Each worker acquires task-relevant knowledge, such as rules,
545 heuristics, plans, or summaries, which can be represented as prompts, memories, files, or
546 structured artifacts like playbooks (Zhang et al., 2025a). These context-level updates can then
547 be optionally consolidated, accumulated, or shared across workers, enabling downstream
548 agents to benefit from experience they did not directly observe.
549 Under this analogy, contexts play a role similar to gradients: they are locally generated
550 learning signals that encode how an agent should behave on future inputs. Accumulating
551 contexts across workers resembles gradient aggregation, while curating, compressing, or
552 distilling these artifacts parallels techniques such as gradient averaging, compression, or
553 delayed synchronization in distributed systems. Crucially, this process scales learning
554 capacity without modifying model parameters, allowing systems to improve continuously
555 under strict inference-time and deployment constraints.
556 This highlights context as a first-class medium for scalable learning, suggesting that many
557 principles from distributed training, such as parallelism, aggregation strategies, communi558 cation efficiency, and consistency trade-offs, can inspire the design of large-scale prompt
559 learning systems.
560 E Qualitative Examples of Context Overload
ACE Final Playbook — Formula, Batch Size = 100
Total context entries: 21 — Test accuracy: 72.5%
FORMULAS & CALCULATIONS
• [calc-00009] h=0 r=2 For relative purchasing power parity (PPP) with changing exchange rates: P1 = P0 * (S1/S0), where P1 is the new price level in the quote country,
P0 is the initial price level in the base country, ...
• [calc-00013] h=0 r=0 For put-call parity with discrete compounding, use P = C - S + X
/ (1 + r)ˆt to find put price, where C is call price, S is spot price, X is strike price, r is
risk-free rate, and t is time. This ...
• [calc-00014] h=2 r=0 For annuity due (cash flows at beginning of periods), use PV = C
* [1 - (1 + r)ˆ(-n)] / r * (1 + r). For ordinary annuity (end of periods), use PV = C * [1 -
(1 + r)ˆ(-n)] / r. Verify timing from ...
• [calc-00015] h=0 r=0 For Interest Rate Parity (IRP), the forward rate formula is F = S *
(1 + r base) / (1 + r quote), where base is the numerator currency and quote is the
denominator currency in the exchange rate ...
• [calc-00020] h=0 r=0 For annuity due calculations where the initial investment is not
explicitly given (e.g., rental properties), the present value of cash inflows alone may
be the expected answer. Use PV = C * [1 - ...
COMMON MISTAKES TO AVOID
• [err-00001] h=1 r=2 Avoid automatically converting decimal results to percentages.
When the question asks for a ’plain floating point number’, output the decimal
equivalent (e.g., 0.05 for 5%) rather than the ...
• [err-00002] h=0 r=2 For annuity calculations, carefully determine the timing of cash
flows. If payments occur at the beginning of periods (e.g., grants, rents, or insurance
premiums), use the annuity due formula: PV ...
• [err-00005] h=0 r=0 For financial ratios like ROA, ROE, profit margins, and other
performance metrics, output the percentage value (e.g., 5.0) rather than the decimal
equivalent (0.05) when the context indicates ...
• [err-00006] h=0 r=0 When calculating option price differences using put-call parity,
remember to multiply by the standard contract multiplier of 100 shares for equity
options. The formula gives per-share values, but ...
561
15
Under review as a conference paper at COLM 2026
• [err-00007] h=0 r=2 Verify the compounding assumption in option pricing formulas.
Put-call parity typically uses continuous compounding (eˆ(-r*t)), but some contexts
may expect discrete compounding ((1+r)ˆ(-t)). ...
• [err-00008] h=0 r=0 For ROI calculations on investments with ongoing benefits, clarify
whether to use incremental profit or (new profit - cost) in the numerator. When in
doubt, use incremental profit (increase from ...
• [err-00010] h=0 r=0 For financial ratios and rates (e.g., ROA, ROE, profit margins,
inflation rate, ROI), output the percentage value as a floating point number (e.g., 5.0)
rather than the decimal (0.05), as they are ...
• [err-00011] h=0 r=2 Maintain high precision in intermediate calculations, especially
for exponentiation (e.g., (1+r)ˆn), division, and financial models like Black-Scholes.
Avoid rounding until the final result to ...
• [err-00012] h=0 r=0 In ROI calculations, clearly distinguish whether ’gain’ refers to
incremental income increase or total new income. When uncertain, use (incremental
income - cost) / cost, but verify against ...
• [err-00016] h=0 r=0 For financial ratios and rates, output percentages (e.g., 5.0) for
ROA, ROE, profit margins, inflation rate, and ROI, as they are conventionally reported
as percentages. Output decimals (e.g., ...
• [err-00017] h=0 r=0 When the question explicitly specifies rounding precision (e.g.,
’round to nearest hundredth’), always apply the rounding directive to the final result,
regardless of financial conventions. For ...
• [err-00018] h=0 r=0 For financial ratios like profit margins, ROE, ROA, and efficiency
ratios, output the percentage value (e.g., 15.0) rather than the decimal (0.15) unless
explicitly instructed otherwise. These ...
• [err-00019] h=0 r=0 When asked how much a value ’changes’ or the ’change’ in a ratio,
determine if the context requires the absolute magnitude of change (e.g., decreased
by 1.5) or the directional change with sign ...
OTHERS
• [misc-00003] h=1 r=0 When calculating financial ratios or rates, first compute the
decimal result, then check the required output format. Only convert to percentage if
explicitly requested or if the context clearly ...
• [misc-00004] h=1 r=0 The phrase ’plain floating point number’ in questions typically
indicates that results should be in decimal form (e.g., 0.05) rather than percentage
form (5.0). Examples like ’5 million should be ...
• [misc-00021] h=0 r=0 When time values are calculated and the result seems unusually
small or large, consider whether unit conversion is expected (e.g., years to days by
multiplying by 365). Check ground truth patterns ...
562
ACE Final Playbook — FiNER, Batch Size = 125
Total context entries: 11 — Test accuracy: 70.6%
STRATEGIES & INSIGHTS
• [sai-00010] h=0 r=0 When tagging line of credit facilities, distinguish between current
borrowing capacity (for existing facilities - use LineOfCreditFacilityCurrentBorrowingCapacity) and maximum borrowing capacity ...
• [sai-00011] h=0 r=0 For business acquisitions, use BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued for equity instruments issued as consideration, in addition to distinguishing between total ...
COMMON MISTAKES TO AVOID
• [err-00003] h=0 r=0 Avoid confusing share-based compensation expense recognition
with grant measurement. Use AllocatedShareBasedCompensationExpense for recognized expenses during a period, and reserve grant-related ...
563
16
Under review as a conference paper at COLM 2026
• [err-00007] h=0 r=0 Avoid confusing debt instrument components: use DebtInstrumentFaceAmount for principal/face value, DebtInstrumentCarryingAmount for net
book value after discounts/premiums, and ...
• [err-00008] h=0 r=0 Avoid over-specifying share issuance tags: use StockIssuedDuringPeriodSharesNewIssues for general period-level reporting of shares issued, and
reserve SaleOfStockNumberOfSharesIssuedInTransaction ...
CONTEXT CLUES & INDICATORS
• [ctx-00009] h=0 r=0 Key phrases indicating specific tag requirements: ’borrowings’ or
’outstanding borrowings’ → DebtInstrumentCarryingAmount, ’available’ or ’unused
capacity’ → ...
OTHERS
• [misc-00001] h=1 r=0 When tagging business acquisition transactions, distinguish
between total consideration (BusinessCombinationConsiderationTransferred1) and
cash components (PaymentsToAcquireBusinessesGross). Use ...
• [misc-00002] h=1 r=0 For debt instruments, prefer specific tags over general ones when
context supports it. Use LongTermDebtFairValue for long-term debt fair values rather
than the generic DebtInstrumentFairValue, and ...
• [misc-00004] h=0 r=0 Key phrases indicating specific tag requirements: ’undrawn
amounts’ → LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage, ’net
of actual and estimated forfeitures’ → ...
• [misc-00005] h=1 r=0 When multiple tags could apply, always choose the most specific
tag that matches the exact context. Verify tag names precisely against available
options (e.g., OperatingLeasesRentExpenseNet vs ...
• [misc-00006] h=1 r=0 For ownership percentages, use MinorityInterestOwnershipPercentageByParent for parent’s ownership in consolidated subsidiaries, and EquityMethodInvestmentOwnershipPercentage for investments with ...
564
ACE Final Playbook — Formula, Batch Size = 1
Total context entries: 264 — Test accuracy: 87.0%
FORMULAS & CALCULATIONS
• [calc-00001] h=6 r=0 Current Ratio = Current Assets / Current Liabilities. Always
verify both values are in the same units before calculating. Round the result to two
decimal places unless otherwise specified.
• [calc-00002] h=1 r=1 ROI = (Net Profit / Total Investment) * 100. Convert percentage
to decimal by dividing by 100. Note: 0.20 and 0.2 are numerically equivalent; trailing
zeros after decimal don’t change value.
• [calc-00003] h=8 r=1 NPV = Σ [CF t / (1 + r)ˆt]. Carefully check cash flow timing: if
described as ’for the next year’ or ’starting now’, the first cash flow may be at t=0
(immediate, no discounting). Subsequent cash ...
• [calc-00004] h=2 r=0 Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio
Standard Deviation. Convert all percentage inputs to decimal form before calculation
(e.g., 7% = 0.07). Round the final result to two ...
• [calc-00005] h=1 r=0 CAPM: Expected Return = Risk-Free Rate + Beta * (Market Return
- Risk-Free Rate). All inputs and outputs should be in decimal form (not percentages).
Round the final result to the specified ...
• [calc-00006] h=2 r=0 DSO (Days Sales Outstanding) = (Accounts Receivable / Credit
Sales) * Number of Days in Period. For a quarter, assume 90 days unless specified
otherwise. Ensure both Accounts Receivable and Credit ...
• [calc-00007] h=0 r=0 Sortino Ratio = (Portfolio Return - Risk-Free Rate) / Downward
Volatility. Convert all percentage inputs to decimal form before calculation (e.g., 10%
= 0.10). Round the final result to two ...
565
17
Under review as a conference paper at COLM 2026
• [calc-00008] h=0 r=0 Accounts Receivable Turnover = Net Credit Sales / Average
Accounts Receivable. Ensure both values are in the same currency units. The result is
typically expressed as a number (not percentage). ...
• [calc-00009] h=7 r=4 For NPV of annuity cash flows: if cash flows occur at period
end (ordinary annuity), use NPV = CF * [1 - (1 + r)ˆ(-n)] / r; if at period beginning
(annuity due), use NPV = CF * [1 - (1 + r)ˆ(-n)] ...
• [calc-00010] h=4 r=0 Modigliani-Miller Proposition I with taxes: V L = V U + (T c *
D), where V L is levered firm value, V U is unlevered firm value, T c is corporate tax
rate (as decimal), and D is debt value. The ...
• [calc-00011] h=2 r=0 Gross Profit Margin = (Total Sales - Cost of Goods Sold) / Total
Sales * 100. Ensure both sales and COGS values are in the same currency units and
from the same period. The result should be ...
• [calc-00012] h=0 r=0 WACC = (E/V) * Re + (D/V) * Rd * (1 - Tc). Always use decimal
form for all inputs (Re, Rd, Tc) and output. When the question specifies ’a plain
floating point number’, it means the decimal form ...
• [calc-00013] h=1 r=0 Inventory Turnover = Cost of Goods Sold (COGS) / Average
Inventory. Ensure both values are in the same currency units. The result is a ratio
(number of times), not a percentage. Round to the ...
• [calc-00014] h=6 r=2 For NPV of project cash flows described as ’annual cash inflows’
without explicit timing, assume annuity due (cash flows at beginning of period) as
projects often start immediately. Use ...
• [calc-00015] h=0 r=0 Beta = Covariance(Stock, Market) / Variance(Market). Ensure
both covariance and variance are calculated from the same data period and in consistent units. Round the final result to the required ...
• [calc-00016] h=0 r=0 For continuous compounding FV = P * eˆ(r*t), use eˆ(r*t) rounded
to 6 decimal places by default (e.g., eˆ(0.21) ≈ 1.233678) unless higher precision is
specified. Financial contexts may expect this ...
• [calc-00017] h=3 r=3 Put-Call Parity: S = C - P + X * (1 + r)ˆ{-t} for discrete compounding (annual) or S = C - P + X * eˆ{-r*t} for continuous compounding. Default to
discrete compounding ((1+r)ˆ{-t}) unless the ...
• [calc-00018] h=0 r=0 For put-call parity problems, if the standard formula (C - P = S -
X * (1 + r)ˆ{-t}) yields a result significantly different from expected values, verify the
arrangement. In some contexts, the ...
• [calc-00020] h=1 r=0 For relative Purchasing Power Parity (PPP) problems where the
exchange rate is quoted as EUR/CHF (domestic/foreign), the formula for the final
domestic price level (P 1 EUR) is: P 1 EUR = P 0 CHF ...
• [calc-00021] h=1 r=1 For APR calculations: APR = (Total Finance Charge / Principal)
* (365 / Term in Days). Output must be in decimal form (e.g., 1.22 for 122%) unless
specified otherwise. Always adhere to ’plain ...
• [calc-00022] h=3 r=0 Gordon Growth Model (Constant Growth DDM): P0 = D1 / (r -
g). D1 is the expected dividend next period, r is the required rate of return (cost of
equity), and g is the constant growth rate. Convert ...
• [calc-00023] h=3 r=0 Black-Scholes Call Option Price: C = S 0*N(d1) - X*eˆ(-r*t)*N(d2).
Always convert percentage inputs (r, σ) to decimal form. Calculate d1 = [ln(S 0/X) +
(r + σ
2/2)*t] / (σ*
√t) and d2 = d1 - σ*
√t ...
• [calc-00024] h=0 r=0 P/B Ratio = Market Price per Share / Book Value per Share.
Ensure both values are in the same currency units (e.g., dollars). The result is a ratio
and should be output as a plain floating point ...
• [calc-00025] h=1 r=0 Treynor Ratio = (Portfolio Return - Risk-Free Rate) / Beta. Convert all percentage inputs to decimal form before calculation (e.g., 6% = 0.06). Round
the final result to two decimal places unless ...
• [calc-00026] h=0 r=0 DSCR = Net Operating Income / Annual Debt Service. Use the
values as provided without unit conversion or unnecessary adjustments. The result
is typically a plain floating point number; round only ...
566
18
Under review as a conference paper at COLM 2026
• [calc-00027] h=4 r=0 Free Cash Flow (FCF) = Operational Cash Flow - Capital Expenditures (CAPEX). CAPEX includes expenditures that maintain or enhance long-term
assets (e.g., machinery maintenance). Ensure both values ...
• [calc-00028] h=2 r=0 Net Profit Margin = (Net Profit / Total Revenue) * 100. When the
question asks for a ’plain floating point number’, output the percentage value (e.g.,
15.0 for 15%) rather than the decimal ratio ...
• [calc-00029] h=0 r=0 Interest Coverage Ratio = EBIT / Interest Expense. Ensure both
EBIT and Interest Expense are in the same currency units. The result is a plain floating
point number; no rounding is necessary if ...
• [calc-00030] h=0 r=0 Simple Interest = Principal × Rate (as decimal). When compounding is not specified (e.g., ’annual interest rate’ without mention of compounding
periods), assume simple interest. Convert the ...
• [calc-00031] h=2 r=0 Operating Margin = (Operating Income / Sales) * 100. Ensure
both values are in the same currency units. When output is requested as a ’plain
floating point number’, provide the percentage value ...
• [calc-00032] h=1 r=0 Debt-to-Equity Ratio = Total Liabilities / Shareholder Equity.
Ensure both values are in the same currency units. The result is a plain floating point
number; no rounding is needed for integer ...
• [calc-00033] h=1 r=0 Unemployment Rate = (Number of Unemployed / Labor Force) *
100. When the problem states the number of unemployed ’rises by’ a specific amount
and the labor force is stagnant, and no initial ...
• [calc-00034] h=0 r=0 P/E Ratio = Market Price per Share / Earnings per Share. Ensure
both values are in the same currency units. The result is a plain floating point number;
no unit conversion or percentage formatting ...
• [calc-00035] h=0 r=0 Return on Assets (ROA) = Net Income / Total Assets. The result
is typically reported as a percentage. When the question asks for a ’plain floating
point number’, output the percentage value (e.g., ...
• [calc-00036] h=3 r=0 For Black-Scholes calculations, compute cumulative normal
distribution values N(d1) and N(d2) with high precision (at least 4-5 decimal places).
Use precise methods like Excel’s NORM.S.DIST, ...
• [calc-00037] h=1 r=0 Return on Equity (ROE) = Net Income / Shareholder’s Equity.
The result is typically expressed as a percentage in financial contexts. After calculating
the decimal value, multiply by 100 to get the ...
• [calc-00038] h=2 r=0 Market Capitalization = Shares Outstanding × Price per Share.
Ensure both inputs are in consistent units (e.g., shares and dollars per share). The
result is a currency value; output as a plain ...
• [calc-00039] h=0 r=0 Gordon Growth Model for cost of equity: r = (D1 / P0) + g. Use
this rearrangement when solving for required return rather than stock price. Convert
percentage growth rates (g) to decimal form ...
• [calc-00040] h=2 r=0 Working Capital = Current Assets - Current Liabilities. Ensure
both values are in the same currency units. The result is a currency value; output as a
plain floating point number (e.g., 70000.0) ...
• [calc-00041] h=0 r=0 EPS = (Net Income - Preferred Dividends) / Weighted Average
Shares Outstanding. Always subtract preferred dividends from net income first to
get earnings available to common shareholders. Round ...
• [calc-00042] h=1 r=0 Trade Balance = Exports - Imports. Convert all values to the same
units (e.g., millions to actual numbers: $120 million = 120,000,000) before subtracting.
Output the result as a plain floating ...
• [calc-00043] h=3 r=0 For present value calculations using PV = FV / (1 + r)ˆt, ensure high numerical precision in both exponentiation and division. Avoid manual
multiplication steps for (1+r)ˆt as they can introduce ...
• [calc-00044] h=1 r=0 Discrete Compound Interest: FV = PV * (1 + r)ˆn where r is the
periodic interest rate (convert percentage to decimal by dividing by 100) and n is the
number of compounding periods. Ensure n ...
567
19
Under review as a conference paper at COLM 2026
• [calc-00045] h=1 r=0 Enterprise Value (EV) = Market Capitalization + Total Debt -
Cash & Cash Equivalents. Ensure all values are in consistent units (e.g., convert
millions to actual numbers: 70 million = 70,000,000). ...
• [calc-00047] h=0 r=0 For quarterly compounding interest: FV = PV * (1 + r/4)ˆ(4*t).
Always convert the annual rate (r) to a quarterly rate by dividing by 4. Calculate the
total number of quarters (4*t) for the ...
• [calc-00048] h=1 r=0 Dividend Payout Ratio = Dividends per Share / Earnings per
Share. Output as a plain floating point number (decimal ratio, not percentage). Both
0.2 and 0.20 are numerically equivalent and ...
• [calc-00049] h=1 r=0 Dividend Yield = Annual Dividend per Share / Share Price.
When the question asks for a ’plain floating point number’, output the decimal ratio
(e.g., 0.02) without multiplying by 100. This aligns ...
• [calc-00050] h=1 r=0 Inflation Rate = ((Current CPI - Previous CPI) / Previous CPI)
* 100. Ensure both CPI values are from consecutive periods (e.g., previous year vs.
current year). Convert the result to a percentage ...
• [calc-00051] h=0 r=0 For high-precision compound interest calculations, especially
with small interest rates over many periods, use FV = PV * eˆ(n * ln(1 + r)) to minimize
numerical errors. This method is more stable ...
• [calc-00052] h=1 r=0 Jensen’s Alpha = Actual Portfolio Return - [Risk-Free Rate + Beta
* (Market Return - Risk-Free Rate)]. Convert all percentage inputs to decimal form
before calculation. First calculate the CAPM ...
• [calc-00053] h=2 r=0 For Treynor Ratio and similar financial ratios, note that numerical
outputs like 0.10 and 0.1 are equivalent; trailing zeros after the decimal do not change
the value. This is particularly ...
• [calc-00054] h=0 r=0 For semi-annual compounding interest: FV = PV * (1 + r/2)ˆ(2*t).
Always convert the annual rate (r) to a semi-annual rate by dividing by 2. Calculate
the total number of semi-annual periods (2*t) ...
• [calc-00055] h=5 r=1 EAR (Effective Annual Rate) = (1 + r/n)ˆn - 1, where r is the
nominal annual rate (as decimal) and n is the number of compounding periods per
year. Output the result as a plain floating point ...
• [calc-00056] h=8 r=1 For financial ratios that are conventionally reported as percentages (e.g., ROE, ROA, Net Profit Margin, Operating Margin), output the percentage
value (e.g., 20.0) when the question requests a ...
• [calc-00057] h=12 r=0 For ratios that yield exact, integer-like results (e.g., Current Ratio
= 150000 / 75000 = 2.0), output the result without applying additional rounding. The
plain floating point format (2.0) is ...
• [calc-00058] h=5 r=1 For EAR (Effective Annual Rate) calculations: when rounding
the decimal result to hundredths (two decimal places), examine the thousandths digit.
If the thousandths digit is 5 or greater, round ...
• [calc-00060] h=1 r=3 For relative PPP problems, first identify the domestic and foreign
countries from the exchange rate quote: if given as A/B, A is the domestic currency
and B is the foreign currency. The formula ...
• [calc-00063] h=2 r=0 For the Gordon Growth Model (P0 = D1 / (r - g)), the output
stock price should be rounded to two decimal places (e.g., 8.33) to match standard
financial reporting conventions for currency values, ...
• [calc-00064] h=0 r=0 For Jensen’s Alpha, round the final result to two decimal places
as standard practice (e.g., -0.0195 → -0.02). This aligns with typical financial reporting
precision for alpha values and ensures ...
• [calc-00065] h=0 r=0 Interest Rate Parity (Forward Rate): Forward = Spot * (1 +
r domestic) / (1 + r foreign). Higher foreign interest rates lead to forward depreciation of the foreign currency. Always verify the ...
• [calc-00066] h=0 r=0 For service companies, ’fees’ typically represent revenue (Total
Sales) and ’delivery and personnel costs’ typically represent Cost of Goods Sold
(COGS) when calculating Gross Profit Margin. ...
568
20
Under review as a conference paper at COLM 2026
• [calc-00067] h=3 r=0 Quick Ratio = (Current Assets - Inventories - Prepaid Expenses)
/ Current Liabilities. Exclude inventories and any prepaid assets to focus on the most
liquid assets. If prepaid expenses are not ...
• [calc-00068] h=1 r=0 For trade balance calculations, always subtract imports from
exports (Exports - Imports). When values are given in billions (e.g., $2 billion), convert
to actual numbers (2,000,000,000) before ...
• [calc-00070] h=3 r=1 For annuity due and ordinary annuity calculations, avoid rounding intermediate values (e.g., [1 - (1 + r)ˆ(-n)] / r) to prevent cumulative errors. Use
maximum precision (at least 6 decimal places) ...
• [calc-00071] h=16 r=5 For rounding to the nearest hundredth (two decimal places),
examine the thousandths digit (third decimal). If the thousandths digit is 5 or greater,
round the hundredths digit up (e.g., 1.335 → ...
• [calc-00072] h=2 r=0 For present value calculations PV = FV / (1 + r)ˆt, ensure high
numerical precision in both the exponentiation (1+r)ˆt and the subsequent division.
Use a calculator or computational tool to ...
• [calc-00073] h=1 r=1 Operating Cash Flow (OCF) - Indirect Method = Net Income +
Non-cash Expenses (e.g., Depreciation) - Increase in Working Capital (or + Decrease
in Working Capital). An increase in working capital ...
• [calc-00074] h=0 r=1 For APR calculations, the Total Finance Charge includes both
interest and any fees (e.g., service fees, origination fees). Always sum all charges
before applying the formula APR = (Total Finance ...
• [calc-00075] h=0 r=0 For WACC calculations, perform all intermediate steps with
maximum precision (e.g., use exact fractions like 2/3 instead of 0.6667) to minimize
rounding errors. Round only the final result to the ...
• [calc-00076] h=0 r=0 For annuity calculations (ordinary or due), compute (1 + r)ˆn and
(1 + r)ˆ(-n) with high precision (at least 6 decimal places) to avoid rounding errors in
intermediate exponentiation. Use exact ...
• [calc-00077] h=0 r=0 For put-call parity calculations (P = C + X * (1 + r)ˆ{-t} - S or
continuous equivalent), compute intermediate values like (1 + r)ˆ{-t} and X * (1 +
r)ˆ{-t} with high precision (at least 6 decimal ...
• [calc-00079] h=0 r=1 For relative PPP problems with asymmetric price level data (e.g.,
only initial domestic and final foreign price levels provided), assume constant price
levels for the country with missing data. If ...
• [calc-00080] h=1 r=0 For the Gordon Growth Model (P0 = D1 / (r - g)), always verify
that the required rate of return (r) is greater than the growth rate (g) before applying
the formula. If r ¡= g, the model is invalid ...
• [calc-00081] h=2 r=2 For put-call parity with discrete compounding (C = P + S - X * (1
+ r)ˆ{-t}): 1) Convert the time to expiration from months to years (t = months / 12). 2)
Calculate the discount factor (1 + ...
• [calc-00082] h=0 r=0 GDP Expenditure Approach: GDP = Consumption + Investment
+ Government Spending + (Exports - Imports). Ensure all components are converted
to the same units (e.g., millions to actual numbers: $4.3 ...
• [calc-00083] h=2 r=0 When components of financial formulas are not explicitly provided (e.g., prepaid expenses in quick ratio calculations), assume they are zero unless
context suggests otherwise. This applies to ...
• [calc-00084] h=1 r=0 For Interest Rate Parity calculations involving exchange rates
with small magnitudes (e.g., JPY/EUR), the standard rounding rule to the nearest
hundredth is absolute: always examine the third ...
• [calc-00085] h=0 r=0 For Interest Rate Parity (Forward = Spot * (1 + r domestic) / (1
+ r foreign)), when rounding the result to two decimal places (hundredths), always
examine the thousandths digit (third decimal). ...
• [calc-00086] h=0 r=0 For direct ratio formulas (e.g., P/E, P/B, Current Ratio), apply
the formula directly using the provided values. Ensure units are consistent (e.g., both
in dollars per share) but avoid making ...
569
21
Under review as a conference paper at COLM 2026
• [calc-00091] h=0 r=0 For ROA and similar ratios, always convert all inputs to consistent base units (e.g., convert millions to actual numbers: $1.3 million = 1,300,000)
before applying the formula. This ensures ...
• [calc-00092] h=0 r=0 When using Gordon Growth Model for cost of equity (r = D1/P0
+ g), verify D1 is the expected next dividend (not current dividend), P0 is the current
stock price, and g is the constant growth rate. ...
• [calc-00093] h=0 r=0 Quick Ratio can be fundamentally expressed as (Cash + Marketable Securities + Accounts Receivable) / Current Liabilities. This simplifies to
(Current Assets - Inventories) / Current Liabilities ...
• [calc-00095] h=1 r=0 For compound interest FV = PV * (1 + r)ˆn, calculate the exponentiation (1 + r)ˆn with maximum available precision (e.g., 8-9 decimal places) and
do not round this intermediate value. Perform the ...
• [calc-00096] h=6 r=0 For Current Ratio calculations, when the division yields an exact
integer result (e.g., 850000 / 425000 = 2.0), output the result as a floating point number
without additional rounding. This ...
• [calc-00097] h=1 r=0 For Beta = Covariance / Variance, explicitly confirm both inputs
are in decimal form (not percentages) before division. This avoids unit conversion
errors and ensures the result is a unitless ...
• [calc-00098] h=2 r=0 Equipment upgrades (e.g., new machinery, technology improvements) are capital expenditures (CAPEX) and should be subtracted from operating
cash flow when calculating Free Cash Flow (FCF). This ...
• [calc-00099] h=0 r=0 For P/E Ratio calculations, ensure both market price per share
and earnings per share are in the same currency units (e.g., both in dollars). The
result should be output as a plain floating point ...
• [calc-00100] h=0 r=0 For put-call parity and other time-value calculations, always
verify the compounding period of the risk-free rate. If time to expiration is given in
months (e.g., 24 months), ensure the rate is ...
• [calc-00102] h=1 r=0 For Treynor Ratio outputs, note that 0.06 and 0.060 are numerically equivalent; trailing zeros after the decimal do not change the value. When
rounding to the nearest hundredth, follow standard ...
• [calc-00103] h=0 r=0 The Quick Ratio (acid-test ratio) excludes inventories from current assets because they are the least liquid component and may not be easily converted to cash to meet short-term obligations. This ...
• [calc-00104] h=0 r=1 For relative PPP problems where the domestic price level is
constant and the foreign price level is to be found, use P foreign = P domestic *
(S new / S old), where S is the exchange rate in ...
• [calc-00105] h=1 r=2 For multi-step financial calculations like put-call parity (S = C - P
+ X * (1 + r)ˆ{-t}), perform all intermediate calculations (e.g., discount factor, present
value of X) with maximum precision ...
• [calc-00106] h=1 r=0 For Treynor Ratio calculations, after computing (Portfolio Return
- Risk-Free Rate) / Beta, round the result to the nearest hundredth (two decimal
places) using standard rounding rules: examine ...
• [calc-00107] h=1 r=0 For Jensen’s Alpha calculations, follow this sequence: 1) Convert
all percentage inputs to decimals. 2) Calculate CAPM expected return: R f + β*(R m -
R f). 3) Compute alpha: Actual Return - CAPM ...
• [calc-00108] h=0 r=0 EV/EBITDA = Enterprise Value / EBITDA. Ensure both values
are in the same units (e.g., both in millions) before calculating. The result is a unitless
multiple; output as a plain floating point ...
• [calc-00109] h=0 r=0 For EPS and other ratios, if the calculation yields an exact value
(e.g., 4.5, 2.0), output the concise form without unnecessary trailing zeros (e.g., 4.5,
not 4.50) unless rounding to hundredths ...
• [calc-00111] h=0 r=0 For P/B Ratio, if the division yields an exact integer result (e.g.,
50/25 = 2.0), output the result without applying additional rounding. The plain
floating point format (2.0) is correct and ...
570
22
Under review as a conference paper at COLM 2026
• [calc-00112] h=0 r=0 For EV/EBITDA ratio, ensure both Enterprise Value and EBITDA
are in the same units (e.g., both in millions) before division, as the ratio is unitless.
Round the result to the required decimal ...
• [calc-00113] h=0 r=0 For Debt-to-Equity Ratio calculations, if only current liabilities
are provided and no other liabilities (e.g., long-term debt) are mentioned, assume
current liabilities represent total ...
• [calc-00114] h=0 r=0 For Current Ratio calculations, if liabilities are provided without specification (e.g., only ’liabilities’ mentioned without ’current’ or ’long-term’
qualifiers), assume the given liabilities are ...
• [calc-00116] h=0 r=0 For ratio calculations (e.g., Debt-to-Equity, Current Ratio) where
inputs are given in scaled units (e.g., millions, billions), convert all values to actual
numerical form before division to ...
• [calc-00117] h=0 r=0 For trade balance and similar calculations involving monetary
values, always convert all inputs to consistent base units (e.g., convert millions to
actual numbers: $300 million = 300,000,000) ...
• [calc-00118] h=0 r=0 For Interest Rate Parity (Forward = Spot * (1 + r domestic) / (1 +
r foreign)), the domestic currency is always the base currency in the exchange rate
quote (e.g., for BRL/EUR, BRL is domestic, ...
• [calc-00119] h=1 r=0 For EAR calculations with low nominal rates (e.g., 2%), the
rounded result (e.g., 0.02) may appear counterintuitively low because the unrounded
value (e.g., 0.020184) is greater than the rounded ...
• [calc-00120] h=0 r=0 ROI = (Net Profit / Total Investment), where Net Profit = Revenue
- Total Investment (cost). Output as a plain floating point number (decimal ratio, not
percentage). For example, an ROI of 2.5 ...
• [calc-00121] h=0 r=0 For multi-component formulas like GDP (C + I + G + (X - M)),
always convert ALL components to the same base units (e.g., millions to actual
numbers) BEFORE performing any arithmetic operations. ...
• [calc-00122] h=0 r=0 For multi-year financial calculations with a ’periodic interest rate’
and no explicit compounding frequency (e.g., ’annual interest rate of 4% for 3 years’),
assume annual compounding unless ...
• [calc-00123] h=0 r=0 For put-call parity rearranged to solve for strike price (X) with
discrete compounding: X = (S + C - P) * (1 + r)ˆt. Steps: 1) Convert time to expiration
to years (t = months / 12). 2) Convert ...
• [calc-00127] h=1 r=0 For present value calculations PV = FV / (1 + r)ˆt, use the exact
value of (1 + r)ˆt without any intermediate rounding. Even slight rounding of this
intermediate factor (e.g., using 1.557967 ...
• [calc-00128] h=0 r=0 For annual DSO calculations, default to 365 days unless a different period is specified. Always ensure accounts receivable and credit sales are from
the same period and in the same currency units. ...
• [calc-00129] h=0 r=0 When numerical values are given with magnitude terms (e.g.,
’million’, ’billion’), convert them to actual numbers before calculation (e.g., ’5 million’
= 5,000,000). This ensures precision and ...
• [calc-00130] h=0 r=0 For formulas involving monetary values with magnitude terms
(e.g., GDP = C + I + G + (X - M) with inputs in billions), first calculate the result in the
given units (e.g., 640 billion dollars), ...
• [calc-00131] h=0 r=0 For monetary values given in millions (e.g., $1.2 million), convert
to base units (1,200,000) before performing division or other operations to ensure
scale accuracy. This applies to ratios like ...
• [calc-00132] h=1 r=0 For Operating Cash Flow (OCF) using the indirect method (OCF
= Net Income + Non-cash Expenses + Changes in Working Capital), add the ’Changes
in Working Capital’ value directly with its sign as ...
• [calc-00133] h=0 r=0 For Jensen’s Alpha rounding, apply standard hundredths rounding: examine the thousandths digit and round up if ≥5 (e.g., 0.0056 → 0.01). This
ensures alignment with financial reporting precision ...
571
23
Under review as a conference paper at COLM 2026
• [calc-00134] h=0 r=0 GDP Expenditure Approach: GDP = Consumption + Investment
+ Government Spending + (Exports - Imports). When inputs are in billions (e.g., $650
billion), first compute GDP in billions, then convert ...
• [calc-00135] h=0 r=0 For compound interest FV = PV * (1 + r)ˆn, perform the entire exponentiation (1 + r)ˆn in a single, high-precision calculation (e.g., using a calculator’s
power function) and do not round this ...
• [calc-00136] h=0 r=0 When a call or put option price is explicitly provided in a put-call
parity problem, the question typically asks for the price of the other option, not the
difference C-P. Use P = C - S + PV(X) ...
• [calc-00137] h=0 r=0 For NPV calculations in project finance contexts (e.g., ’project
returns’, ’annual cash inflows from investment’), prioritize annuity due (cash flows
at beginning of periods) over ordinary annuity ...
• [calc-00138] h=0 r=0 For Free Cash Flow (FCF) calculations, when expenditures are
ambiguously grouped (e.g., ’spends $X on A and B’) and context suggests capital
expenditure (CAPEX) treatment, treat the entire amount ...
• [calc-00139] h=0 r=1 For compound interest FV = PV * (1 + r)ˆn, calculate the exponentiation (1 + r)ˆn with at least 8-9 decimal places of precision for the base value
(e.g., 1.075ˆ20 ≈ 4.247851096) before multiplying ...
• [calc-00140] h=0 r=0 When using Gordon Growth Model for cost of equity (r = D1/P0
+ g), perform operations in the correct sequence: 1) Convert percentage growth rate
(g) to decimal, 2) Calculate D1/P0 (dividend ...
• [calc-00141] h=0 r=0 For Sortino Ratio calculations, when rounding to the nearest
hundredth, apply the standard rule: examine the thousandths digit and round up if
≥5 (e.g., 1.2857 → 1.29). This ensures alignment with ...
• [calc-00142] h=0 r=0 Net Profit Margin = (Net Profit / Total Revenue) * 100. Output
the result as a percentage value (e.g., 5.0 for 5%) when a ’plain floating point number’
is requested. For exact integer results ...
• [calc-00143] h=0 r=0 For Inflation Rate calculations, if the result is an exact integer
(e.g., 5.0), output without additional rounding. This aligns with the ’plain floating
point number’ requirement and matches ...
• [calc-00144] h=0 r=0 For EV/EBITDA and similar ratios, when values are given
with magnitude terms (e.g., ’billion’, ’million’), convert to actual numbers before
calculation (e.g., $1 billion = 1,000,000,000; $125 ...
• [calc-00145] h=0 r=0 For monthly compounding interest: FV = PV * (1 + r)ˆn, where r
is the monthly interest rate (convert percentage to decimal, e.g., 1.5% = 0.015) and n
is the total number of months. Ensure n ...
• [calc-00146] h=0 r=0 For P/E Ratio calculations, when both market price per share
and earnings per share are provided in the same currency units (e.g., both in dollars)
and no unit conversion is specified, apply the ...
• [calc-00147] h=1 r=1 For present value calculations PV = FV / (1 + r)ˆt, compute the
exponentiation (1 + r)ˆt with high precision (at least 6-8 decimal places) to avoid
rounding errors in the intermediate factor. Even ...
• [calc-00148] h=0 r=0 The Quick Ratio (acid-test ratio) excludes inventories and prepaid
expenses from current assets because they are the least liquid components and may
not be easily converted to cash to meet ...
• [calc-00149] h=0 r=0 For Return on Assets (ROA) calculations, always convert monetary values from millions to base units before applying the formula (Net Income /
Total Assets). For example, convert $2.2 million to ...
• [calc-00150] h=0 r=0 For Purchasing Power Parity (PPP) problems where initial and
final domestic price levels (P 0 domestic, P 1 domestic) and exchange rates (S 0, S 1)
are provided, and the initial foreign price ...
• [calc-00152] h=0 r=0 For present value calculations PV = FV / (1 + r)ˆt, avoid truncating or rounding the intermediate exponentiation result (1 + r)ˆt to a fixed number of
decimal places (e.g., 9 decimal places). Use ...
572
24
Under review as a conference paper at COLM 2026
• [calc-00153] h=0 r=0 For Black-Scholes calculations, the sensitivity to N(d1) and N(d2)
approximations is highest for out-of-the-money options (where S 0 ¡ X for calls or S 0
¿ X for puts). A difference of 0.0002 in ...
• [calc-00154] h=1 r=0 For present value calculations PV = FV / (1 + r)ˆt, always convert
the interest rate (r) to decimal form (e.g., 5% = 0.05) before computing the compounding factor (1 + r)ˆt. Calculate the ...
• [calc-00155] h=0 r=0 For monetary value outputs (e.g., Market Cap, Working Capital,
Free Cash Flow, Enterprise Value), if the calculation yields an exact integer result,
output it as a floating point number (e.g., ...
• [calc-00156] h=1 r=0 For Jensen’s Alpha calculations, follow this explicit sequence: 1)
Convert all percentage inputs (actual return, risk-free rate, market return) to decimals.
2) Calculate the CAPM expected return: ...
• [calc-00157] h=0 r=0 For Free Cash Flow (FCF) calculations, R&D investments may be
treated as CAPEX if the context implies capitalization (e.g., ’investment in intangible
assets’ or under IFRS standards) rather than ...
• [calc-00158] h=0 r=0 For APR and other financial calculations where precision is
critical, use ’round half to even’ (banker’s rounding) when the thousandths digit is
exactly 5. For example, 0.145 rounds to 0.14 ...
• [calc-00159] h=0 r=0 ROI = (Total Net Profit - Investment) / Investment, where Total
Net Profit is the sum of net profits over the investment’s entire useful life. When
output is requested as a ’plain floating point ...
• [calc-00161] h=0 r=0 For exchange rate calculations (e.g., forward rates via Interest
Rate Parity), if the ground truth or context implies rounding to the nearest tenth (one
decimal place), apply tenths rounding: ...
• [calc-00162] h=0 r=0 For DSCR calculations, explicitly verify that Net Operating
Income and Annual Debt Service are in the same currency units (e.g., both annualized,
both in dollars) before performing the division. ...
• [calc-00163] h=0 r=0 For EAR calculations, ensure precise rounding to the required
decimal places: compute EAR = (1 + r/n)ˆn - 1 with high precision, then round the
decimal result (e.g., 0.050625 → 0.0506 when ...
• [calc-00164] h=0 r=0 For rounding to hundredths, even if the value appears to have
only three decimal places (e.g., 0.006), treat it as having implicit trailing zeros (0.0060)
and apply standard rounding: examine the ...
• [calc-00165] h=0 r=0 For Black-Scholes calculations, the option price is highly sensitive
to small errors in N(d1) and N(d2), especially for near-the-money options. Use
computational libraries (e.g., ...
• [calc-00166] h=0 r=0 For financial ratios involving monetary values with magnitude
terms (e.g., ’million’, ’billion’), convert all inputs to consistent base units (e.g., $2
million = 2,000,000) before performing ...
• [calc-00167] h=0 r=0 For Dividend Payout Ratio and similar per-share ratios, use the
provided values directly without unit conversion (e.g., dollars per share) unless the
problem specifies otherwise (e.g., values in ...
• [calc-00168] h=0 r=0 For Working Capital calculations, always verify that all components are properly classified as current assets or current liabilities before applying
the formula (Current Assets - Current ...
• [calc-00169] h=0 r=1 Before applying put-call parity (discrete compounding: C - P = S
- X * (1 + r)ˆ{-t}), first check for arbitrage: if S - C + P ¿ X, then (1 + r)ˆ{-t} ¿ 1, which is
impossible for t ¿= 0. This ...
• [calc-00171] h=0 r=0 For Inventory Turnover, if the calculation yields an exact integer
result (e.g., 3.0), output without applying additional rounding. Only round to the
specified precision if the result is not an ...
• [calc-00172] h=0 r=0 For monthly compounding interest with an annual term: convert
the term from years to months (years × 12) to match the monthly interest rate
frequency. Then apply FV = PV × (1 + r)ˆn, where r is ...
573
25
Under review as a conference paper at COLM 2026
• [calc-00174] h=0 r=0 For Sortino Ratio and similar financial ratios, if the calculated
result is already at the required precision (e.g., exactly two decimal places like 1.9),
output the value without applying ...
• [calc-00175] h=0 r=0 For APR calculations, output the result as a decimal (e.g., 0.97)
unless the problem explicitly requests a percentage. This aligns with the common
instruction for ’plain floating point number’ and ...
• [calc-00176] h=0 r=0 For present value calculations involving multiple exponents
(e.g., PV = FV1/(1+r)ˆt1 + FV2/(1+r)ˆt2), compute each (1+r)ˆt term directly with
maximum precision rather than deriving one exponent ...
• [calc-00177] h=0 r=0 For EPS outputs, trailing zeros after the decimal do not change
the numerical value (e.g., 3.20 and 3.2 are equivalent). Output the concise form unless
the problem specifies a particular precision ...
• [calc-00178] h=0 r=0 For Dividend Payout Ratio and similar financial ratios, if the
calculated result is already at the required precision (e.g., 0.25 for hundredths), output
the value without applying additional ...
• [calc-00180] h=0 r=0 For Sortino Ratio and similar financial ratios, if the calculated
result is already at the required precision (e.g., exactly two decimal places like 2.5),
output the value without applying ...
• [calc-00181] h=1 r=1 For Black-Scholes calculations, compute d1 and d2 with high
precision (at least 6 decimal places) and verify all input components (S 0, X, r, t, σ)
are correctly applied and converted to decimals. ...
• [calc-00182] h=0 r=1 For present value calculations PV = FV / (1 + r)ˆt, never round
the intermediate exponentiation result (1 + r)ˆt to a fixed number of decimal places.
Use the full-precision value (e.g., from a ...
• [calc-00183] h=0 r=0 For all financial outputs, recognize that trailing zeros after the
decimal point do not change the numerical value (e.g., 7.50 and 7.5 are equivalent).
The concise form (e.g., 7.5) is often ...
• [calc-00184] h=0 r=0 For any financial calculation that yields an exact integer result
(e.g., 850000 / 425000 = 2.0), output the result as a floating point number without
applying additional rounding. This applies ...
• [calc-00186] h=0 r=0 For GDP expenditure approach (GDP = C + I + G + (X - M)),
negative net exports (when imports exceed exports) reduce the total GDP. This occurs
when (X - M) is negative, and the subtraction is ...
• [calc-00187] h=0 r=0 For Accounts Receivable Turnover, perform the division with
exact values (no intermediate rounding) to ensure accuracy, especially when the
result is an integer. Output as a floating point number ...
• [calc-00188] h=0 r=0 For EAR and other decimal outputs, ’round to the nearest hundredth’ means round the decimal value itself to two decimal places (hundredths
place), not the percentage equivalent. Always apply ...
• [calc-00189] h=0 r=0 For P/B Ratio and other per-share ratios (e.g., P/E, EPS), explicitly verify that both inputs are ’per share’ values (e.g., market price per share, book
value per share) and not aggregate values ...
• [calc-00190] h=0 r=0 For ROI = (Net Profit / Total Investment), output as a plain
floating point number (decimal form, e.g., 2.25 for 225% return). If the calculation
yields an exact value (e.g., 2.0, 2.25), do not ...
• [calc-00191] h=0 r=0 For Inflation Rate calculations, always verify that the CPI values
are from consecutive periods (e.g., previous year vs. current year) and are based on
the same index base year to ensure ...
• [calc-00192] h=0 r=0 For exchange rate quotes in the form A/B (e.g., ZAR/GBP),
currency A is the domestic currency and currency B is the foreign currency when
applying Interest Rate Parity (IRP) and other parity ...
• [calc-00194] h=0 r=0 For annual compounding (n=1), the EAR formula simplifies to
EAR = r (the nominal rate). This is because (1 + r/1)ˆ1 - 1 = r. Deposit amounts or
other principal values are never relevant for EAR ...
574
26
Under review as a conference paper at COLM 2026
• [calc-00195] h=0 r=0 For EAR calculations, always trust the precise computed value
and apply rounding rules strictly to the specified decimal places. Do not second-guess
the result based on common approximations ...
• [calc-00196] h=0 r=0 For EAR outputs, remember that 0.2 and 0.20 are numerically
identical (both represent 20%). Focus on calculation accuracy and apply rounding
rules precisely, but avoid overcomplicating ...
• [calc-00197] h=0 r=0 For exponential functions in financial formulas (e.g., eˆx in continuous compounding, discount factors), always use high-precision calculations with
reliable tools (e.g., financial calculators, ...
• [calc-00198] h=0 r=0 For Jensen’s Alpha, always convert all percentage inputs (actual
return, risk-free rate, market return) to decimal form before any calculations. After
computing alpha = Actual Return - [R f + ...
• [calc-00199] h=0 r=0 For Sortino Ratio calculations, after computing (Portfolio Return
- Risk-Free Rate) / Downward Volatility with decimal inputs, round the final result to
the nearest hundredth using standard rules: ...
• [calc-00200] h=0 r=0 For GDP expenditure approach (GDP = C + I + G + (X - M)),
always calculate net exports (X - M) first before summing with other components.
This ensures accuracy, especially when imports and ...
• [calc-00201] h=0 r=0 ROI = (Net Profit / Total Investment), where Net Profit = (Cumulative Returns Over Investment Life) - Total Investment Cost. For multi-period
investments, ensure net profit is calculated by ...
• [calc-00202] h=1 r=0 For Black-Scholes calculations, the option price is highly sensitive
to small errors in N(d1) and N(d2), especially for near-the-money or out-of-the-money
options (where —S 0 - X— is small). A ...
• [calc-00204] h=0 r=0 For WACC calculations, follow this sequential approach: 1)
Convert all percentage inputs (Re, Rd, Tc) to decimals, 2) Calculate total firm value V
= E + D, 3) Compute equity weight (E/V) and debt ...
• [calc-00205] h=0 r=0 For Interest Rate Parity (Forward = Spot * (1 + r domestic) / (1 +
r foreign)), always first identify the base currency in the exchange rate quote (e.g., for
GBP/USD, GBP is the base/domestic ...
• [calc-00208] h=0 r=0 Quick Ratio (acid-test ratio) = (Current Assets - Inventories) /
Current Liabilities. This ratio measures a company’s immediate liquidity by excluding
less liquid assets like inventory. Always ...
• [calc-00209] h=0 r=0 Future Value with Annual Compounding: FV = PV * (1 + r)ˆn,
where r is the annual interest rate converted to decimal form (e.g., 4.5% = 0.045) and
n is the number of years. Calculate (1 + r)ˆn with ...
• [calc-00210] h=0 r=0 For NPV calculations, when a problem describes cash flows ’over
n years’ with an ’initial’ cash flow at time 0, interpret this as n total cash flows from
Year 0 to Year n-1. For example, ’initial ...
• [calc-00211] h=0 r=0 For Jensen’s Alpha and similar metrics, when rounding a very
small negative value (e.g., -0.001) to the nearest hundredth results in -0.00, output 0.0
instead. Numerically equivalent, but 0.0 is ...
• [calc-00212] h=1 r=0 For Black-Scholes calculations, convert time to expiration from
months to years by dividing by 12 (e.g., 9 months → 9/12 = 0.75 years). Calculate
ln(S0/X) with high precision (at least 6 decimal ...
• [calc-00213] h=0 r=0 For ROI = (Net Profit / Total Investment), Net Profit is calculated
as Revenue - Total Investment (cost). Ensure both Revenue and Total Investment are
in the same currency units. Output the result ...
• [calc-00214] h=0 r=0 Operating Cash Flow (OCF) - Indirect Method = Net Income +
Non-cash Expenses (e.g., Depreciation) + Decrease in Working Capital (or - Increase
in Working Capital). A decrease in working capital is ...
• [calc-00215] h=0 r=0 When numerical values are given with magnitude terms (e.g.,
’million’, ’billion’), convert them to actual numbers before calculation (e.g., ’5 million’
= 5,000,000). This ensures precision and ...
575
27
Under review as a conference paper at COLM 2026
• [calc-00216] h=0 r=0 For EAR and other decimal outputs, ’round to the nearest hundredth’ means round the decimal value itself to two decimal places (hundredths
place), not the percentage equivalent. Always apply ...
• [calc-00217] h=0 r=0 For direct ratio calculations like P/E Ratio (Price per Share /
Earnings per Share), focus on precise division without overcomplicating the process.
Ensure both inputs are in the same units and ...
• [calc-00218] h=0 r=0 For Return on Equity (ROE), when the question requests a ’plain
floating point number’, output the percentage value (e.g., 15.0 for 15%) rather than
the decimal ratio (0.15). This aligns with ...
• [calc-00219] h=0 r=0 For Inventory Turnover, if COGS is not provided, sales revenue
may be used as a proxy only if the problem context implies minimal profit margins
(e.g., ’sold $X worth of product’ suggesting cost ...
• [calc-00221] h=0 r=0 For relative Purchasing Power Parity (PPP) problems where the
initial price level of the country being solved for is implicitly set to 100 (e.g., ’price
level in the UK is to be calculated as an ...
• [calc-00222] h=0 r=0 For relative PPP problems where the foreign price level is constant and the initial foreign price level (P 0 foreign) is unknown, use P 0 foreign =
P 1 domestic * (S 0 / S 1). This formula is ...
• [calc-00223] h=0 r=0 For compound interest with small periodic interest rates (e.g.,
¡1%) and many periods (e.g., ¿30), use the logarithmic method FV = PV * eˆ(n * ln(1 +
r)) to minimize numerical errors. This is more ...
• [calc-00224] h=0 r=0 Debt-to-Equity Ratio = Total Liabilities / Shareholders’ Equity.
Always verify that Shareholders’ Equity is not zero to avoid division by zero errors.
Ensure both values are in the same currency ...
• [calc-00225] h=0 r=0 For Gordon Growth Model cost of equity (r = D1/P0 + g), output
the result as a decimal without unnecessary trailing zeros (e.g., 0.1 instead of 0.10)
unless rounding to a specific precision is ...
• [calc-00226] h=0 r=0 For Black-Scholes calculations, use cumulative normal distribution values N(d1) and N(d2) with at least 6 decimal places (e.g., 0.667296 instead of
0.6673) to prevent rounding errors that can ...
• [calc-00227] h=0 r=0 When the labor force is constant, the change in employment
equals the negative change in unemployment: ∆Employed = - (∆Unemployment
Rate) × Labor Force. Use this to directly compute the number of ...
• [calc-00228] h=0 r=0 For future value calculations (FV = PV * (1 + r)ˆn), trailing zeros
after the decimal point do not change the numerical value (e.g., 172,877.70 and
172,877.7 are equivalent). When rounding to the ...
• [calc-00229] h=0 r=0 For compound interest FV = PV * (1 + r)ˆn, when calculating (1 +
r)ˆn manually without a calculator, use step-by-step exponentiation via successive
squaring to maintain precision. For example: ...
• [calc-00230] h=0 r=0 When calculating financial ratios like EV/EBITDA, use directly
provided values if available (e.g., explicit EBITDA amount). Ignore extraneous details
(e.g., EBIT and depreciation add-backs) if the ...
• [calc-00231] h=0 r=0 GDP Expenditure Approach: GDP = Consumption + Investment
+ Government Spending + (Exports - Imports). Ensure all components are converted
to the same units (e.g., all in billions or all in base ...
• [calc-00232] h=0 r=0 When converting monetary values from billions to base units
(e.g., for GDP, trade balance), multiply by 1,000,000,000 (10ˆ9). Double-check the
number of zeros in the result to avoid placement ...
• [calc-00233] h=0 r=0 For Quick Ratio calculations, always verify the composition
of current assets to ensure only highly liquid assets (cash, marketable securities,
accounts receivable) are included. Exclude ...
• [calc-00234] h=0 r=0 For Black-Scholes calculations, the option price is most sensitive
to precision errors in N(d1) and N(d2) when the option is near-the-money (S 0 ≈ X).
A difference of 0.0001 in these values can ...
576
28
Under review as a conference paper at COLM 2026
• [calc-00235] h=0 r=0 Accounts Receivable Turnover = Net Credit Sales / Average
Accounts Receivable. Ensure both values are in the same currency units. The result is
a ratio (number of times). If the division yields an ...
• [calc-00236] h=0 r=0 For ROI = (Net Profit / Total Investment), carefully interpret
terms: ’earnings’, ’returns’, or ’amount received’ typically refer to the total amount
received from the investment, not the net ...
• [calc-00237] h=0 r=0 For multi-component formulas like GDP (C + I + G + (X - M)),
always convert ALL input values to the same base units (e.g., billions to actual
numbers: $1 billion = 1,000,000,000) BEFORE performing ...
• [calc-00238] h=0 r=0 For Purchasing Power Parity (PPP) problems where the domestic
price level (P domestic) is constant and the foreign price level (P foreign) is to be
found, use P 1 foreign = P 0 domestic * (S 1 / ...
• [calc-00239] h=0 r=0 For all multi-step financial calculations (e.g., Black-Scholes, NPV,
annuity valuations), maintain high precision (at least 6 decimal places) in all intermediate steps and avoid rounding until the ...
• [calc-00241] h=0 r=0 For Dividend Yield and other financial ratios where the calculated
value is exactly halfway between two hundredths (e.g., 0.075), apply round-half-down
(round to 0.07) instead of round-half-up, as ...
• [calc-00242] h=0 r=0 For multi-component summation formulas like GDP (C + I + G +
(X - M)) where inputs are in large units (e.g., billions), first compute the result in the
given units to avoid handling very large ...
• [calc-00244] h=0 r=0 For put-call parity and other time-value calculations, always
convert the annual risk-free rate from percentage to decimal form (e.g., 1.5% → 0.015)
and convert the time to expiration to years ...
• [calc-00245] h=0 r=0 For present value calculations PV = FV / (1 + r)ˆt, avoid stepwise
manual multiplication to compute (1 + r)ˆt (e.g., 1.06 * 1.06 * 1.06 * 1.06), as this can
introduce rounding errors at each step. ...
• [calc-00246] h=0 r=0 For Beta and similar ratio calculations (e.g., Covariance / Variance), if the division yields a result already at the required decimal precision (e.g., 0.6
for hundredths), output the value ...
• [calc-00247] h=0 r=0 P/E Ratio = Market Price per Share / Earnings per Share. Ensure
both values are in the same currency units. Round the result to the nearest hundredth
(two decimal places) using standard rounding ...
• [calc-00248] h=0 r=0 For relative Purchasing Power Parity (PPP) problems with constant foreign price level and unknown initial domestic price level, use P 1 domestic
= P 0 foreign * (S 0 / S 1), where S is the ...
• [calc-00250] h=0 r=0 For GDP expenditure approach (GDP = C + I + G + (X - M)), after
converting all components to the same base units and summing, output the result as
a plain floating point number (e.g., ...
• [calc-00251] h=0 r=0 For Purchasing Power Parity (PPP) problems, always verify the
exchange rate quote convention (domestic/foreign) before applying formulas. For
absolute PPP, if S is quoted as domestic/foreign ...
• [calc-00252] h=0 r=0 Simple Interest = Principal × Rate (as decimal) × Time. For exact
integer results (e.g., 180.0), output as a floating point number without additional
rounding to satisfy ’plain floating point ...
• [calc-00253] h=0 r=0 For bi-annual compounding interest: FV = PV * (1 + r)ˆn, where
r is the bi-annual interest rate converted to decimal (e.g., 2.5% = 0.025) and n is the
total number of bi-annual periods (e.g., 12 ...
• [calc-00254] h=0 r=0 For annuity due calculations (PV due = C * [1 - (1+r)ˆ-n] / r *
(1+r)), compute (1+r)ˆ-n with high precision (at least 8 decimal places) and avoid
rounding any intermediate values. Perform all ...
• [calc-00255] h=0 r=0 For Accounts Receivable Turnover, if the division yields an exact
result (e.g., 7.5), output the value without applying additional rounding. This aligns
with the ’plain floating point number’ ...
577
29
Under review as a conference paper at COLM 2026
• [calc-00256] h=0 r=0 For Operating Cash Flow (OCF) using the indirect method,
remember the cash flow implications: an increase in working capital represents a
cash outflow (subtracted from net income), while a ...
• [calc-00258] h=0 r=0 For put-call parity used to solve for time to expiration (t), first
check for arbitrage: if (S - C + P) / X ¿ 1, then eˆ{-rt} ¿ 1, which is impossible for t ¿=
0. This indicates the given option ...
• [calc-00259] h=0 r=0 For WACC calculations, after computing the unrounded decimal
value, apply rounding to the required precision (typically hundredths) by examining
the thousandths digit: if it is 5 or greater, round ...
• [calc-00260] h=0 r=0 For Gordon Growth Model cost of equity (r = D1/P0 + g), if
the calculated result is already at the required decimal precision (e.g., 0.11 for hundredths), output the value without applying ...
• [calc-00262] h=0 r=0 For Interest Rate Parity calculations, if a forward rate is provided
in the problem but contradicts the calculated theoretical rate (Forward = Spot * (1 +
r domestic) / (1 + r foreign)), ignore ...
• [calc-00263] h=0 r=0 For Dividend Yield and similar financial ratios, if the calculated
result is already at the required precision (e.g., 0.05 for hundredths), output the value
without applying additional rounding. ...
• [calc-00264] h=0 r=0 For ratio calculations involving proportional relationships (e.g.,
’liabilities are twice equity’ for Debt-to-Equity Ratio), use the given multiplier to
compute the missing value directly: if A = ...
COMMON MISTAKES TO AVOID
• [err-00019] h=1 r=0 Avoid assuming the standard put-call parity rearrangement is
always correct without validation. If the computed strike price (X) is far from intuitive
values (e.g., S + C - P ≈ 71 vs. S - C + P ≈ ...
• [err-00046] h=0 r=0 Avoid including unnecessary trailing zeros after the decimal point
in floating point outputs (e.g., output 3.2 instead of 3.20 for EPS) unless specified
otherwise, as they are numerically ...
• [err-00059] h=0 r=0 When calculating DSO, ensure Accounts Receivable and Credit
Sales are from the same accounting period and in the same currency units. Mismatched periods (e.g., using quarterly sales with annual ...
• [err-00061] h=0 r=0 For the Dividend Payout Ratio, avoid applying additional rounding if the calculated result is already at the required precision (e.g., 0.15 is already at
the hundredth place). Only round if the ...
• [err-00062] h=0 r=1 For EAR calculations, avoid outputting the result as a percentage
(e.g., 6.70%) when the instruction specifies a ’plain floating point number’. The
decimal form (e.g., 0.067) is required. This ...
• [err-00078] h=0 r=0 For the Debt-to-Equity Ratio and other ratios that yield a result
already at the required precision (e.g., 0.5), avoid applying unnecessary rounding.
Only round if the result has more decimal ...
• [err-00087] h=2 r=1 Avoid converting decimal outputs to percentages when the question specifies a ’plain floating point number’. For ratios like CAPM expected return,
WACC, or Dividend Yield, the output should be in ...
• [err-00088] h=0 r=0 Avoid taking unemployment rate problems at face value without
verifying context. If the calculated employed population (labor force * (1 - unemployment rate)) is positive but the ground truth is ...
• [err-00090] h=0 r=1 Avoid defaulting to ordinary annuity (end-of-period) for project
cash flows described as ’annual’ without explicit timing. In project finance, returns
often start immediately, so annuity due ...
• [err-00094] h=0 r=0 Avoid second-guessing standard rounding rules in financial contexts. When instructed to round to a specific decimal place (e.g., nearest hundredth),
apply the universal rule: examine the digit ...
• [err-00115] h=0 r=0 For ROI calculations, be aware that some contexts may incorrectly
use (Total Profit After Investment - Investment Cost) / Investment Cost instead of
the standard (Incremental Profit Gain / ...
578
30
Under review as a conference paper at COLM 2026
• [err-00125] h=0 r=0 Avoid automatically applying annuity formulas for multi-period
cash flows without first verifying the number of payments intended. If the computed
result (e.g., using ordinary or due annuity) ...
• [err-00160] h=0 r=0 Avoid outputting ROI as a percentage when the instruction specifies a ’plain floating point number’. The decimal form (e.g., 2.857 for 285.7% return)
is required unless percentage output is ...
• [err-00170] h=0 r=0 Avoid strictly adhering to ’end of each year’ wording for annuity
timing if the calculated PV (ordinary annuity) significantly diverges from the expected
answer. In such cases, verify if the ...
• [err-00179] h=0 r=0 Avoid using imprecise exponentiation methods for compound
interest calculations, especially with small interest rates over many periods. Even
slight rounding in the base (1 + r)ˆn can lead to ...
• [err-00185] h=0 r=0 Avoid overcomplicating straightforward rounding tasks by introducing extraneous financial context (e.g., percentage interpretation) when the
instruction explicitly requests a plain floating point ...
• [err-00203] h=0 r=0 Avoid altering numerically equivalent floating point outputs (e.g.,
changing 0.10 to 0.1) for financial calculations when the output format is specified as
a ’plain floating point number’. ...
• [err-00207] h=0 r=0 When using the Gordon Growth Model to calculate cost of equity
(r = D1/P0 + g), use only the current stock price (P0) as the denominator. Do not use
historical or previous stock prices, as they ...
• [err-00240] h=0 r=0 Always remember the fundamental working capital formula:
Working Capital = Current Assets - Current Liabilities. Ensure both values are in the
same currency units before subtracting, and follow ...
• [err-00243] h=0 r=0 For ROI calculations, avoid using revenue directly without subtracting the investment cost to compute net profit first. ROI = (Revenue - Cost) /
Cost, not Revenue / Cost. This ensures the net gain ...
• [err-00249] h=0 r=1 Avoid using P 1 domestic = P 0 foreign * (S 1 / S 0) for relative
PPP problems with constant foreign price level. This incorrect formula assumes
domestic prices change proportionally with the ...
• [err-00261] h=0 r=0 For ratio change questions (e.g., P/B, P/E), carefully distinguish
between calculating the new ratio value versus the change in the ratio. The question
often asks for the new value after a ...
CONTEXT CLUES & INDICATORS
• [ctx-00069] h=0 r=0 For unemployment rate problems: when the question states
unemployed ’rises by’ a specific number and the labor force is stagnant with no initial
unemployed value provided, assume initial ...
• [ctx-00089] h=0 r=0 For unemployment rate problems, a ground truth of 0.0 when the
calculated employed is positive may indicate a misstated problem or trick question.
Look for keywords like ’all’ or ’entire’ labor ...
• [ctx-00110] h=0 r=0 For DSO calculations, carefully extract the period length from the
problem context (e.g., ’45 days’ explicitly stated). If not specified, default to standard
periods (e.g., 90 days for a quarter, ...
• [ctx-00126] h=0 r=0 Phrases like ’due to receive’ often indicate that a payment is
immediate or very near-term, potentially implying only one cash flow should be
considered for present value calculations, even if ...
• [ctx-00206] h=0 r=0 For lottery or prize contexts, phrases like ’pay you $X annually for
Y years’ may be misleading. If the computed present value using annuity formulas is
significantly higher than the prize amount ...
• [ctx-00220] h=0 r=0 When questions ask for ’impact on [metric]’ due to a parameter change, carefully determine whether they require the resulting value or the
change/difference. If the calculated change is small ...
OTHERS
579
31
Under review as a conference paper at COLM 2026
• [misc-00101] h=0 r=0 When encountering phrases like ’decreased market cap’ vs ’decrease in market cap’, interpret carefully: ’decreased market cap’ typically refers to
the new market value after the decrease, while ...
• [misc-00124] h=0 r=0 When both revenue and net profit are provided in a question
asking for ’Net Profit from selling products’ or similar phrasing, interpret this as a
request for the net profit margin percentage (Net ...
• [misc-00151] h=1 r=0 For financial ratios output as plain floating point numbers, recognize that trailing zeros after the decimal point do not change the numerical value
(e.g., 0.2, 0.20, and 0.200 are all ...
• [misc-00173] h=0 r=0 Investments in software development tools, technology upgrades, or intangible assets that enhance long-term operational capacity are typically
classified as capital expenditures (CAPEX) and should ...
• [misc-00193] h=0 r=0 For DSO calculations, the period length (number of days) must
be inferred from context: ’half-year’ or ’semi-annual’ typically implies 180 days,
’quarter’ implies 90 days, and ’annual’ implies 365 ...
• [misc-00257] h=0 r=0 For Free Cash Flow calculations, classify expenditures as CAPEX
if they involve acquiring, upgrading, or maintaining long-term assets (e.g., machinery,
buildings, software) that provide benefits ...
580
ACE Final Playbook — FiNER, Batch Size = 1
Total context entries: 246 — Test accuracy: 76.0%
STRATEGIES & INSIGHTS
• [sai-00006] h=21 r=0 When identifying US GAAP tags for numerical entities, focus
on the contextual meaning: credit facility amounts relate to borrowing capacity tags
(e.g., ...
• [sai-00007] h=1 r=0 When the same numerical value appears in multiple questions
about the same financial facility, it typically represents the same fundamental metric.
Phrases like ’may be less than $X’ indicate that ...
• [sai-00008] h=3 r=4 For interest expense amounts specifically related to debt instruments, use InterestExpenseDebt. This tag is appropriate for both current period
expense amounts and comparative period amounts when ...
• [sai-00009] h=3 r=0 For concentration risk percentages, particularly those related to
revenue from major customers, use ConcentrationRiskPercentage1. This tag applies
to percentages that quantify the proportion of ...
• [sai-00010] h=22 r=0 When identifying stated interest rates for debt instruments (expressed as percentages), use DebtInstrumentInterestRateStatedPercentage. This tag
is appropriate for fixed rates or the stated ...
• [sai-00011] h=11 r=0 For share-based compensation and common stock metrics, carefully distinguish between: par value per share (CommonStockParOrStatedValuePerShare), shares outstanding (CommonStockSharesOutstanding), ...
• [sai-00012] h=14 r=3 For credit facility amounts, carefully distinguish between current
and maximum borrowing capacity: ’borrowing base’ refers to the LineOfCreditFacilityCurrentBorrowingCapacity (the currently ...
• [sai-00013] h=19 r=7 For debt instruments, distinguish between principal/face
amounts and fair value measurements: use DebtInstrumentFaceAmount for the
original principal value (e.g., ’principal amount of notes’) and ...
• [sai-00014] h=1 r=1 For insurance company disclosures about ’development’ in prior
year claims reserves (adverse or favorable), use the specialized tag ...
• [sai-00015] h=11 r=0 For share-based compensation events, precisely match the context
to the GAAP tag: options exercised (intrinsic value) use ...
• [sai-00016] h=3 r=0 For debt-related fair value measurements, carefully distinguish
context: use LongTermDebtFairValue when the context refers to ’debt obligations’ at
a balance sheet date or specifically mentions ...
581
32
Under review as a conference paper at COLM 2026
• [sai-00017] h=1 r=2 For lease and rental expenses, use LeaseAndRentalExpense as the
default tag when the context mentions ’rent expense’ without specifying lease type
(operating vs capital). Reserve ...
• [sai-00018] h=5 r=0 For statutory tax rates mentioned in effective tax rate reconciliation disclosures, use EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate regardless of whether it represents a ...
• [sai-00019] h=8 r=0 The ConcentrationRiskPercentage1 tag applies broadly to various
concentration risk disclosures, including revenue concentration (percentage of total
revenue from major customers), accounts ...
• [sai-00020] h=1 r=0 For US GAAP tag identification, first analyze the sentence context to determine the exact nature of the financial metric (expense type, liability
characteristic, or measurement basis), then select ...
• [sai-00021] h=15 r=1 For share-based compensation disclosures, carefully distinguish
between recognized expense and other metrics: use AllocatedShareBasedCompensationExpense for compensation ’recognized’ during a ...
• [sai-00022] h=1 r=0 For treasury stock transactions, carefully distinguish between pershare and aggregate measurements: TreasuryStockAcquiredAverageCostPerShare
for average price per share of repurchased shares, and ...
• [sai-00023] h=0 r=0 For revolving credit facilities described at inception (e.g., ’entered
into a $X million facility’), the stated amount represents the current borrowing
capacity available at that time before any ...
• [sai-00024] h=6 r=2 For letters of credit under credit facilities, use LettersOfCreditOutstandingAmount for outstanding balances, regardless of currency denomination or
sub-limit context. This tag applies to the ...
• [sai-00025] h=9 r=0 For share-based compensation disclosures, precisely distinguish
between the quantity of awards granted (use GrantsInPeriod tags like ...
• [sai-00026] h=1 r=0 When classifying debt-related amounts, distinguish between
instrument characteristics and balance sheet presentation: use DebtInstrumentFaceAmount for the original principal value of specific debt ...
• [sai-00027] h=0 r=0 When encountering incomplete financial context with multiple
unlabeled amounts (e.g., ’Includes of $X million, and of $Y million’), analyze the
relative magnitudes of the numbers. Large amounts ...
• [sai-00028] h=2 r=0 For tax-related amounts, distinguish between current period
income tax expenses (use IncomeTaxExpenseBenefit) and uncertain tax positions (use
UnrecognizedTaxBenefits for gross amounts and ...
• [sai-00029] h=1 r=0 For utility companies filing rate change applications with regulatory commissions, use PublicUtilitiesRequestedRateIncreaseDecreaseAmount for the
dollar amount of the requested change. This tag ...
• [sai-00030] h=0 r=0 For lease accounting under ASC 842, when a numerical range is
provided for ’lease assets and lease liabilities’ together, carefully analyze the context
to determine which component is being ...
• [sai-00031] h=7 r=0 For share-based compensation plan limits, precisely distinguish
between three key metrics: 1) Total authorized shares for the plan ...
• [sai-00032] h=0 r=0 For XBRL tag selection, pay meticulous attention to capitalization
patterns as they are case-sensitive and must exactly match the standard taxonomy.
Common patterns include camelCase for ...
• [sai-00033] h=0 r=0 Unit appreciation rights (UARs) and stock appreciation rights
(SARs) are classified as ’equity instruments other than options’ under US GAAP, not
as options. Use ...
• [sai-00034] h=1 r=4 For equity share reservations, distinguish between share-based
compensation authorizations and general future issuances: Use ...
• [sai-00035] h=12 r=0 For stock option disclosures, carefully distinguish between vesting period (time until options become exercisable) and expiration period (total time
options remain valid). Use ...
582
33
Under review as a conference paper at COLM 2026
• [sai-00036] h=2 r=0 For amortization of intangible assets (e.g., patents, trademarks),
use AmortizationOfIntangibleAssets. For amortization of deferred contract costs
under ASC 340-40 (e.g., capitalized incremental ...
• [sai-00037] h=0 r=1 For counts of primary business lines or segments that meet the
definition of operating segments under ASC 280, use NumberOfOperatingSegments.
This tag applies to the quantitative disclosure of how ...
• [sai-00038] h=1 r=0 When selecting US GAAP tags, systematically evaluate three dimensions: 1) Financial instrument type (debt, equity, derivative, etc.), 2) Measurement
context (face amount, fair value, outstanding ...
• [sai-00039] h=1 r=0 For debt-related payments, carefully distinguish between transaction types: use RepaymentsOfDebt for installment payments or scheduled reductions
of debt principal, and ...
• [sai-00040] h=4 r=0 For intangible assets, distinguish between amortization expense
and useful life: use AmortizationOfIntangibleAssets for the periodic expense amount,
and FiniteLivedIntangibleAssetUsefulLife for ...
• [sai-00041] h=3 r=1 For property, plant and equipment (PP&E), distinguish between
depreciation expense and useful life: use Depreciation for the periodic expense
amount, and PropertyPlantAndEquipmentUsefulLife for ...
• [sai-00042] h=0 r=4 For segment reporting under ASC 280, carefully distinguish between operating segments and reportable segments: use NumberOfOperatingSegments for counts of segments identified by management before ...
• [sai-00043] h=0 r=0 For Dividend Reinvestment Plans (DRIPs), use ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized
for shares authorized for issuance under such plans. DRIPs are ...
• [sai-00044] h=5 r=1 For effective tax rate disclosures, use EffectiveIncomeTaxRateContinuingOperations when the context mentions ’effective tax rate’ or ’effective income
tax rate’ for continuing operations. This tag ...
• [sai-00045] h=6 r=0 For antidilutive securities excluded from EPS calculations, use
AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount when
the context specifies securities ’excluded from ...
• [sai-00046] h=8 r=0 When distinguishing between common and preferred stock metrics, carefully analyze the context to identify the specific stock type: CommonStockParOrStatedValuePerShare for common stock par value, ...
• [sai-00047] h=0 r=5 For share-based compensation, carefully distinguish between total
recognized expense and allocated amounts: use ShareBasedCompensation for the
total compensation expense recognized during a period ...
• [sai-00049] h=0 r=0 When analyzing credit facility structures with multiple tranches,
carefully distinguish between overall facility borrowing capacity (LineOfCreditFacilityMaximumBorrowingCapacity) and specific ...
• [sai-00050] h=0 r=0 For lease-related charges, carefully distinguish between ongoing
operational expenses and restructuring activities: use LeaseAndRentalExpense for
normal recurring lease/rent expenses, but use ...
• [sai-00051] h=0 r=0 For share issuance disclosures indicating no new issuances (e.g.,
’no common units issued’), use StockIssuedDuringPeriodSharesNewIssues with a
value of zero. This tag applies to both positive ...
• [sai-00052] h=0 r=0 For employer contributions to defined benefit pension plans,
use DefinedBenefitPlanContributionsByEmployer. This tag applies to cash funding
amounts made by the employer to pension plans, distinct ...
• [sai-00053] h=0 r=0 When tagging numerical values in financial contexts, prioritize
the economic substance over surface features. For debt instruments, treat units (e.g.,
’years’) as integral to the measurement ...
• [sai-00054] h=0 r=3 For operating lease rent expenses, always prefer OperatingLeasesRentExpenseNet over the more general OperatingLeaseExpense when the context
specifically mentions ’rent expense for operating ...
583
34
Under review as a conference paper at COLM 2026
• [sai-00056] h=1 r=0 For unrecognized tax benefits, carefully analyze contextual
phrases that indicate impact on the effective tax rate. Phrases like ’will reduce
our effective tax rate’, ’would impact the effective ...
• [sai-00060] h=0 r=1 When derivative instruments (e.g., bond hedges, warrants) have
terms explicitly defined by reference to a primary financial instrument’s characteristic
(e.g., conversion price of convertible ...
• [sai-00062] h=0 r=0 When encountering benchmark rates (e.g., LIBOR, SOFR) in debt
instrument contexts, carefully analyze whether they are presented as standalone
reference rates or as components of basis spread ...
• [sai-00063] h=2 r=0 For stock issuances during a period, default to ’StockIssuedDuringPeriodSharesNewIssues’ for both common and preferred shares unless specific
context requires more granular tags. Reserve ...
• [sai-00064] h=0 r=0 For credit facility disclosures, carefully distinguish between actual
debt drawn and available capacity: ’borrowings under [facility]’ indicates actual debt
incurred (use LongTermDebt or ...
• [sai-00065] h=3 r=0 For share-based compensation, distinguish between the quantity of awards granted (ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod) and ...
• [sai-00066] h=0 r=0 When analyzing business transaction consideration, carefully
determine the directional context: if the company is paying consideration (e.g., ’we
paid’, ’purchase price’), use acquisition tags ...
• [sai-00067] h=1 r=0 When identifying transaction types, carefully analyze contextual
clues for related party relationships. Phrases like ’with related parties’, ’transactions
with affiliates’, or specific entity ...
• [sai-00068] h=0 r=1 For debt instruments appearing in balance sheet contexts (e.g.,
’notes payable’, ’outstanding debt’), prefer LongTermDebt or ShortTermDebt over
DebtInstrumentFaceAmount. Use ...
• [sai-00069] h=0 r=0 When both ’operating’ and ’reportable’ are used together to describe segments (e.g., ’operating and reportable segments’), prefer NumberOfOperatingSegments as it captures the fundamental segments ...
• [sai-00070] h=3 r=1 For loss contingency disclosures, carefully distinguish between
recognized amounts and estimated ranges: use LossContingencyAccrualAtCarryingValue for liabilities already recognized on the balance ...
• [sai-00071] h=0 r=2 For business acquisition disclosures involving ownership percentages, use BusinessAcquisitionPercentageOfVotingInterestsAcquired when the
context describes the percentage of voting interests ...
• [sai-00072] h=0 r=0 For amortization expenses specifically related to intangible assets
from lease acquisitions (e.g., in-place leases), use ’AmortizationOfIntangibleAssets’.
This tag applies to lease-related ...
• [sai-00073] h=1 r=0 For weighted average interest rates on debt obligations, use
’DebtWeightedAverageInterestRate’. This tag specifically applies to the calculated
average rate across multiple debt instruments, ...
• [sai-00074] h=0 r=0 For share-based compensation expenses, carefully analyze contextual modifiers that indicate allocation or adjustment: use AllocatedShareBasedCompensationExpense when phrases like ’net of ...
• [sai-00075] h=1 r=0 For revenue transactions involving related parties (e.g., equity
method investees, affiliates), prioritize the transaction nature over the relationship
when selecting US GAAP tags. Use Revenues ...
• [sai-00076] h=1 r=0 When tagging measurement units (e.g., ’year’, ’month’) in sharebased compensation contexts, treat them as integral to the measurement they describe
rather than separate entities. For example, ...
• [sai-00077] h=1 r=0 In share-based compensation disclosures, the phrase ’from the
grant date’ specifically indicates expiration period (total term of the award) rather
than vesting period. Use ...
584
35
Under review as a conference paper at COLM 2026
• [sai-00078] h=0 r=1 When encountering fair value hierarchy disclosures (Level 1, 2,
or 3) for debt instruments, carefully distinguish between the measurement basis
disclosed (fair value) and the actual balance sheet ...
• [sai-00081] h=2 r=0 For warrant exercise price disclosures, use ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1 when the context describes the price at which
warrants can be exercised to purchase underlying ...
• [sai-00082] h=0 r=0 When identifying useful life measurements for property, plant
and equipment, use PropertyPlantAndEquipmentUsefulLife for the estimated service
life period (e.g., ’20 years’ for buildings). This ...
• [sai-00083] h=0 r=0 For revenue amounts specifically described in context (e.g., ’reimbursable revenues’, ’contract revenue’, ’customer revenue’), prefer the specific tag
...
• [sai-00086] h=1 r=0 For restructuring costs, carefully distinguish temporal context:
use RestructuringCharges for expenses that have been recognized and incurred (e.g.,
’expense totaled’, ’charges recognized’), but ...
• [sai-00088] h=1 r=2 For debt facilities, critically distinguish between term loans and
revolving credit lines: term loans represent fixed principal amounts requiring DebtInstrumentCarryingAmount (e.g., ’$20.0 million ...
• [sai-00089] h=2 r=0 When analyzing numerical values, carefully distinguish between
percentage-based metrics and dollar amount metrics. Percentage values (e.g., ’0.5%
commitment fee’) require percentage tags like ...
• [sai-00090] h=0 r=0 For insurance risk concentration percentages (e.g., ceded risk
percentages, maximum exposure percentages, risk allocation percentages), use ConcentrationRiskPercentage1. This tag applies to ...
• [sai-00091] h=3 r=0 For stock repurchase program disclosures, when both the authorized amount and actual repurchased amount reference the same authorization
program, use StockRepurchaseProgramAuthorizedAmount1 for ...
• [sai-00092] h=6 r=0 For share-based compensation disclosures, carefully distinguish
between different award phases: use grants tags (e.g., ...
• [sai-00093] h=0 r=0 For lease and rental contexts, critically distinguish between cash
payment language (’paid in rent’, ’rental payments’) and expense recognition language (’rent expense’, ’lease expense’). Use ...
• [sai-00095] h=0 r=0 For business acquisition disclosures, carefully distinguish between
cash flow perspective and accounting perspective: use PaymentsToAcquireBusinessesNetOfCashAcquired for net cash outflow amounts ...
• [sai-00096] h=0 r=0 For segment reporting under ASC 280, use NumberOfOperatingSegments when the context describes operational structure or management approach (e.g., ’conduct operations in/through segments’, ’managed ...
• [sai-00097] h=6 r=1 For fair value measurements of debt instruments, always analyze
the maturity date to determine debt classification: use LongTermDebtFairValue for
obligations due beyond one year (e.g., ’Senior ...
• [sai-00098] h=0 r=0 For rent expense contexts that mention contingent rentals or
multiple lease components, prefer LeaseAndRentalExpense over OperatingLeasesRentExpenseNet. The broader tag better captures ...
• [sai-00099] h=0 r=0 For business acquisition payments, carefully distinguish between
cash consideration and total consideration: use PaymentsToAcquireBusinessesGross
for specific cash payments made to acquire ...
• [sai-00100] h=1 r=2 For stock repurchase transactions, carefully distinguish between
different measurement contexts: use StockRepurchasedDuringPeriodShares for total
shares repurchased (e.g., ’repurchased X shares’), ...
• [sai-00101] h=0 r=0 For XBRL tagging of time-based measurements in share-based
compensation (e.g., vesting periods, expiration periods), treat the measurement unit
(e.g., ’years’, ’months’) as an integral part of the ...
• [sai-00102] h=1 r=1 When analyzing debt-related contexts, carefully distinguish between measurement tags that describe instrument characteristics (e.g., DebtInstrumentFaceAmount for original principal value, ...
585
36
Under review as a conference paper at COLM 2026
• [sai-00104] h=2 r=0 When analyzing ownership percentages, carefully distinguish
between acquisition events and resulting ownership structures: use BusinessAcquisitionPercentageOfVotingInterestsAcquired for the ...
• [sai-00106] h=0 r=1 When multiple tags could potentially apply to a financial context,
always prefer the most specific tag that precisely matches the economic substance of
the transaction or measurement. For example, ...
• [sai-00108] h=0 r=0 For operating lease rent expenses, use OperatingLeasesRentExpenseNet when the context explicitly mentions ’rent expense for operating leases’ or
similar phrasing that specifically ties the rent ...
• [sai-00109] h=0 r=0 For letters of credit under credit facilities, use LettersOfCreditOutstandingAmount when the context describes the actual drawn or outstanding
balance, including initial amounts at issuance. ...
• [sai-00110] h=2 r=0 For credit facility borrowing capacity, carefully distinguish three
concepts: 1) Maximum authorized limit (LineOfCreditFacilityMaximumBorrowingCapacity), 2) Current capacity based on collateral ...
• [sai-00111] h=1 r=4 For commitment fees on credit facilities, use LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage when the context specifically mentions fees ’on undrawn amounts’ or ’on unused capacity’. ...
• [sai-00112] h=8 r=1 For business acquisition disclosures, carefully distinguish between
total consideration transferred (BusinessCombinationConsiderationTransferred1) and
net cash payments ...
• [sai-00113] h=0 r=1 For stock repurchase disclosures, carefully distinguish between
shares merely repurchased (use StockRepurchasedDuringPeriodShares) and shares
both repurchased and retired (use ...
• [sai-00114] h=0 r=0 For the weighted-average grant date fair value of stock options
granted during a period, use ...
• [sai-00115] h=0 r=0 For remaining performance obligation disclosures under ASC 606,
use RevenueRemainingPerformanceObligation for both the total amount and any
segment-specific portions. The same tag applies to the ...
• [sai-00116] h=1 r=0 When the context specifies ’net of cash acquired’ in business
acquisition disclosures, use PaymentsToAcquireBusinessesNetOfCashAcquired. This
tag specifically applies to the net cash outflow ...
• [sai-00117] h=4 r=3 For debt instruments, carefully distinguish between ’outstanding’
amounts and ’principal’ amounts: ’outstanding’ refers to the current carrying amount
on the balance sheet (use ...
• [sai-00118] h=0 r=0 When ’principal amount’ is mentioned in credit facility contexts
(e.g., ’aggregate principal amount of up to €65 million’), it refers to the face value of
debt instruments that can be drawn under ...
• [sai-00119] h=4 r=1 For depreciation expenses specifically related to property, plant
and equipment (e.g., buildings, machinery, equipment), use the ’Depreciation’ tag.
This applies to both current period expense ...
• [sai-00120] h=2 r=0 For goodwill amounts that represent the asset value recorded
during acquisition transactions (not impairment charges), use the ’Goodwill’ tag.
This applies when the context describes goodwill ...
• [sai-00121] h=3 r=0 For interest rate spreads on variable rate debt instruments, use
DebtInstrumentBasisSpreadOnVariableRate1 regardless of the benchmark rate (e.g.,
LIBOR, prime, SOFR) specified in the context. The ...
• [sai-00122] h=0 r=2 When tagging ownership percentages, carefully analyze legal
structure and control indicators beyond just the percentage value. For limited liability
companies (LLCs), joint ventures, or entities ...
• [sai-00123] h=1 r=0 In fair value hierarchy disclosures, the dollar amounts presented
represent fair value measurements even when the context mentions that fair value
approximates carrying amount. The fair value ...
• [sai-00124] h=0 r=0 When share-based compensation expense is explicitly broken
down by specific award types (e.g., options, restricted stock awards, restricted stock
units) with separate dollar amounts for each ...
586
37
Under review as a conference paper at COLM 2026
• [sai-00125] h=1 r=0 For share-based compensation fair value measurements, critically
distinguish between total aggregate amounts and weighted average amounts: use
’TotalFairValue’ tags (e.g., ...
• [sai-00126] h=0 r=0 For equity interests issued in business acquisitions, use BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued regardless of
whether the context describes shares being issued or ...
• [sai-00127] h=0 r=0 When the entity receives equity investments from third parties
(rather than issuing its own stock), use EquityMethodInvestments instead of ProceedsFromIssuanceOfCommonStock. The key distinction is ...
• [sai-00128] h=0 r=0 When a percentage appears in the name of a debt instrument (e.g.,
’6.75 % Notes’), it typically refers to the stated interest rate that characterizes the
instrument, not the face amount. Use ...
• [sai-00129] h=3 r=2 For share-based compensation, precisely match the specific type
of equity instrument mentioned in the context to the most specific tag available.
Restricted share units (RSUs) and other equity ...
• [sai-00131] h=1 r=1 For specific US GAAP tag applications: use BusinessCombinationAcquisitionRelatedCosts for acquisition-related costs (e.g., legal, accounting,
advisory fees), ...
• [sai-00132] h=0 r=0 For warrant quantity disclosures (e.g., ’4.6 million warrants issued’), use StockIssuedDuringPeriodSharesNewIssues as warrants are equity instruments representing potential future shares issued ...
• [sai-00134] h=0 r=0 For share-based compensation fair value measurements, critically
distinguish temporal context: use grant-date weighted average fair value tags (e.g., ...
• [sai-00135] h=0 r=0 For business acquisition disclosures, carefully distinguish between
cash consideration and total consideration: use PaymentsToAcquireBusinessesGross
for specific cash payments made to acquire ...
• [sai-00136] h=0 r=0 For stock repurchase transactions, distinguish between authorization context (’up to $X’) which requires StockRepurchaseProgramAuthorizedAmount1 and acquisition context (’received X shares’) which ...
• [sai-00137] h=2 r=0 For loss contingency disclosures, always check if the context
specifies a particular type of contingency (environmental, litigation, etc.) and prefer
the most specific tag available. When ...
• [sai-00138] h=0 r=0 For share-based compensation metrics, systematically match the
specific measurement context (unrecognized compensation cost, weighted-average
recognition period, grant date fair value per share, ...
• [sai-00139] h=0 r=0 For warrants and similar instruments, carefully analyze whether
periods until exercisability (e.g., ’exercisable after X months’) should be tagged as
part of the expiration period rather than ...
• [sai-00140] h=0 r=0 For share-based compensation disclosures, critically distinguish
between quantitative measurements (count/number of shares) and monetary values
(fair value amounts). Pay close attention to ...
• [sai-00141] h=0 r=0 For counts of operational groupings used in internal reporting,
such as grouping tenants by activity segments or other management-defined categories, use NumberOfOperatingSegments. This tag ...
• [sai-00142] h=0 r=0 When share-based compensation expense is recognized and allocated to specific income statement line items (e.g., cost of sales, selling, general and
administrative expenses), use ...
• [sai-00143] h=2 r=0 For debt issuance costs, carefully distinguish between initial
capitalization and subsequent amortization: use DeferredFinanceCostsGross for the
initial payment/capitalization of financing costs ...
• [sai-00144] h=0 r=0 For equity share reservations, carefully distinguish between shares
reserved for future issuance and proceeds from actual issuances: use CommonStockCapitalSharesReservedForFutureIssuance for ...
• [sai-00145] h=0 r=0 For lease expense tagging, follow a specificity hierarchy: use
the general LeaseAndRentalExpense tag when the context mentions ’rent expense’
without specifying lease type (operating vs capital). ...
587
38
Under review as a conference paper at COLM 2026
• [sai-00146] h=0 r=0 When model predictions exactly match ground truth answers
and environment feedback confirms no errors occurred, continue applying the same
successful approach: maintain contextual analysis, ...
• [sai-00147] h=1 r=0 For percentage values describing vesting conditions of share-based
awards (e.g., ’40% of the restricted share units vest based on market conditions’), use
...
• [sai-00148] h=2 r=0 For weighted average grant date fair value per unit measurements
of equity instruments other than options (e.g., ’$19.28 per restricted share unit’), use
...
• [sai-00149] h=0 r=0 For debt instruments issued at a discount or premium, the ’aggregate principal amount’ mentioned in issuance contexts may be tagged as DebtInstrumentCarryingAmount when it refers to the amount ...
• [sai-00150] h=0 r=0 When interpreting share-based compensation plan descriptions,
carefully analyze modifying words that indicate maximum capacity: phrases like
’ceiling of X shares available for issuance,’ ’maximum ...
• [sai-00151] h=0 r=0 For operating lease rent expenses, prefer ’LeaseAndRentalExpense’
when the context mentions ’rent expense’ without indicating sublease income or
netting. Reserve ’OperatingLeasesRentExpenseNet’ ...
• [sai-00152] h=0 r=0 For business acquisition disclosures, carefully distinguish between
total consideration transferred (BusinessCombinationConsiderationTransferred1) and
specific cash payments ...
• [sai-00153] h=0 r=0 For debt-related transactions, carefully distinguish between instrument characteristics and repayment actions: use DebtInstrumentFaceAmount for
describing the original principal value ...
• [sai-00154] h=0 r=0 For cumulative effects of adopting new accounting principles (e.g.,
’cumulative effect of adopting ASC 842’), use CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption. This tag applies to ...
• [sai-00155] h=3 r=1 For debt instrument measurements on the balance sheet, prioritize
DebtInstrumentCarryingAmount over LongTermDebt when the context describes
the carrying value of specific debt instruments (e.g., ...
• [sai-00156] h=10 r=1 For revolving credit facilities, always use LineOfCreditFacilityMaximumBorrowingCapacity for the maximum authorized borrowing limit. Never
use DebtInstrumentFaceAmount for revolving facilities, as ...
• [sai-00157] h=0 r=0 For loss contingency disclosures, always check if more specific
tags exist for particular contingency types (environmental, litigation damages, etc.)
before defaulting to general tags. The phrase ...
• [sai-00158] h=0 r=0 When shares are issued in business acquisitions or other transactions, prioritize the stock type context (common vs. preferred) over the transaction
context for tag selection. Use ...
• [sai-00159] h=0 r=0 When derivative instruments (e.g., warrants, hedges) have terms
explicitly defined by reference to a primary financial instrument’s characteristic,
carefully analyze whether the numerical value ...
• [sai-00160] h=0 r=0 For credit facilities with sub-facilities (e.g., letter of credit subfacilities, swingline sub-facilities), carefully distinguish between the maximum authorized capacity of the sub-facility and ...
• [sai-00161] h=0 r=0 For fair value measurements of debt instruments, default to
DebtInstrumentFairValue when maturity context is absent or unspecified. Only use
LongTermDebtFairValue when the debt is explicitly ...
• [sai-00162] h=0 r=0 For loss contingency disclosures, critically distinguish between
accrued amounts (already recorded on balance sheet) and estimates of possible losses
(potential future exposures). Use accrual tags ...
• [sai-00163] h=0 r=0 For stock repurchase transactions, carefully distinguish between
shares repurchased and held as treasury stock versus shares repurchased and retired:
use TreasuryStockSharesAcquired when shares ...
588
39
Under review as a conference paper at COLM 2026
• [sai-00165] h=0 r=0 For ownership percentages in collaborative arrangements, critically distinguish between consortiums and joint ventures: consortiums may involve
subsidiary structures with minority interests ...
• [sai-00167] h=0 r=0 When shares are repurchased under a formal stock repurchase
program, they are typically retired as part of that program unless explicitly stated
otherwise. Use ...
• [sai-00169] h=0 r=0 For tax benefits derived from operating loss carryforwards that
reduce income tax expense, use OperatingLossCarryforwards. This tag applies when
the context describes tax benefits from net ...
• [sai-00170] h=0 r=0 Maintain consistent tagging for the same financial metric across
different periods, including when values are zero (’none’ or ’0’). The same US GAAP
tag should be applied to represent the ...
• [sai-00171] h=0 r=0 For deferred financing costs, carefully distinguish between gross
and net carrying amounts: ’unamortized’ costs refer to the net carrying amount after
accumulated amortization, requiring ...
• [sai-00172] h=0 r=0 For share-based compensation disclosures, critically analyze verb
phrases to distinguish between similar numerical contexts: ’recognize over a period’
indicates weighted-average recognition period ...
• [sai-00173] h=0 r=0 When multiple numerical values appear in the same sentence
context (e.g., ’$172.5 million aggregate principal amount of 3.25% convertible senior
notes’), carefully analyze which specific entity ...
• [sai-00174] h=0 r=0 For debt issuance costs, critically distinguish between the initial
cost amount (use DeferredFinanceCostsNet for the net carrying amount) and the
periodic amortization expense (use ...
• [sai-00175] h=0 r=1 For ownership percentages in consolidated subsidiaries, use MinorityInterestOwnershipPercentageByNoncontrollingOwners when the parent company consolidates the entity despite owning less than 100%. ...
• [sai-00176] h=0 r=0 For specific stock offering transactions (e.g., follow-on offerings,
public offerings), prefer transaction-specific tags like SaleOfStockNumberOfSharesIssuedInTransaction and ...
• [sai-00178] h=0 r=0 For ownership percentages, use EquityMethodInvestmentOwnershipPercentage for stakes typically between 20-50% (indicating significant influence),
and MinorityInterestOwnershipPercentageByParent for ...
• [sai-00179] h=0 r=0 For tax rate reconciliation disclosures, carefully distinguish between reconciliation components and the final effective rate: use ...
• [sai-00180] h=0 r=0 For loss contingency disclosures, critically distinguish between
estimated ranges of possible outcomes and specific claimed amounts: use LossContingencyEstimateOfPossibleLoss for ranges of ...
• [sai-00181] h=0 r=0 For intersegment revenues in segment reporting, use ’Revenues’
rather than ’RevenueFromRelatedParties’. Intersegment revenues represent internal
transactions between business segments and are ...
• [sai-00182] h=0 r=0 For new stock issuances during a period (e.g., common stock
offerings, equity raises), use ’StockIssuedDuringPeriodSharesNewIssues’. Reserve
’SaleOfStockNumberOfSharesIssuedInTransaction’ for ...
• [sai-00184] h=0 r=0 When the context explicitly states that operating segments ’represent our reportable segments’ or similar phrasing that directly equates operating
segments with reportable segments, use ...
• [sai-00185] h=0 r=0 When term loans are presented as components of credit facilities
within borrowing capacity context (e.g., ’Amended Credit Facilities... which includes
the $X million Term Loan Facility’), use ...
• [sai-00187] h=0 r=0 For share-based compensation disclosures, critically distinguish
between shares available within existing compensation arrangements and shares
reserved from capital stock: use ...
• [sai-00188] h=0 r=0 For business acquisition cash payments, use PaymentsToAcquireBusinessesGross when the context specifies cash paid at closing without mentioning
netting against cash acquired (e.g., ’was paid at ...
589
40
Under review as a conference paper at COLM 2026
• [sai-00189] h=0 r=0 When identifying US GAAP tags for numerical entities, focus on
the contextual meaning: property counts use NumberOfRealEstateProperties, share
grant quantities use ...
• [sai-00190] h=0 r=0 When identifying US GAAP tags for numerical entities, focus
on the specific accounting context and purpose of the number rather than just its
numerical value or general financial statement ...
• [sai-00191] h=0 r=0 For intangible asset useful life disclosures, critically distinguish
between general reporting (FiniteLivedIntangibleAssetUsefulLife) and business combination contexts: use ...
• [sai-00192] h=0 r=0 For business acquisition cash payments, carefully distinguish
between gross cash consideration and total consideration: use PaymentsToAcquireBusinessesGross for specific cash payments made to ...
• [sai-00193] h=0 r=0 For credit arrangements described with ’not to exceed’ language,
this indicates maximum borrowing capacity rather than fixed principal amounts.
Both revolving credit facilities and term loan ...
• [sai-00194] h=0 r=2 For business acquisition contingent consideration, use BusinessCombinationContingentConsiderationLiability for both the fair value at acquisition
date and subsequent measurement dates. The same ...
• [sai-00195] h=0 r=0 In US GAAP financial reporting contexts, ’rent expense’ disclosures typically refer to operating leases unless explicitly stated otherwise. Always
prefer the more specific tag ...
• [sai-00196] h=0 r=0 When backward references (e.g., ’such services’, ’these transactions’) clearly refer to previously described related party transactions, they should
trigger RelatedPartyTransaction tags instead of ...
• [sai-00197] h=1 r=0 For share-based compensation expense disclosures, verbs like
’recorded’, ’recognized’, or ’expensed’ during a period specifically indicate expense
allocation to the income statement, requiring ...
• [sai-00198] h=1 r=0 For tax benefits specifically related to stock-based compensation
arrangements (e.g., ’Income tax benefits related to stock-based compensation’), use ...
• [sai-00199] h=0 r=0 When selecting between similar XBRL tags, prioritize the most
specific tag that directly matches contextual language. If the context contains precise
terminology (e.g., ’unused portion’) that ...
• [sai-00200] h=0 r=0 When multiple numerical values appear in close proximity, carefully analyze each value’s specific context to determine the appropriate XBRL tag.
For tax benefits, use ...
• [sai-00201] h=0 r=0 For revenue recognition under ASC 606, critically distinguish
between: 1) RevenueRemainingPerformanceObligation - for unsatisfied performance
obligations that represent future revenue to be ...
• [sai-00202] h=0 r=0 For interest expense amounts, default to the general InterestExpense tag unless the context explicitly requires distinguishing between different types
of interest (e.g., debt interest vs. lease ...
• [sai-00203] h=0 r=0 For contingent consideration liabilities, carefully distinguish between liability measurement contexts and accretion contexts: use BusinessCombinationContingentConsiderationLiability for the fair ...
• [sai-00204] h=0 r=0 For stock issuance contexts, use SaleOfStockNumberOfSharesIssuedInTransaction when specific transaction details are present, including: named
counterparties (e.g., ’Mill Road Capital’), explicit ...
• [sai-00205] h=0 r=0 When encountering vague terms like ’these plans’ without explicit
specification, carefully analyze the broader context for clues about plan type. Annual
expense amounts for employee benefit ...
• [sai-00206] h=0 r=0 For business acquisition disclosures, critically distinguish between
cash consideration and total consideration: use PaymentsToAcquireBusinessesGross
when the context describes only cash payments ...
• [sai-00207] h=0 r=0 When the context explicitly describes ’stock award expense’ recognized in the income statement, use AllocatedShareBasedCompensationExpense
rather than the general ShareBasedCompensation tag. The ...
590
41
Under review as a conference paper at COLM 2026
• [sai-00208] h=0 r=0 For unrecognized tax benefits, use UnrecognizedTaxBenefits when
the context refers to the gross amount without any mention of impact on effective tax
rate. Reserve ...
• [sai-00209] h=0 r=0 For segment counts under ASC 280, use NumberOfReportableSegments when the context refers to the final aggregated segments presented in financial
statements after applying quantitative thresholds ...
• [sai-00210] h=0 r=0 Before selecting a US GAAP tag, always verify that the tag exists
in the provided options list. If a more specific tag is recommended by context but not
available, use the most appropriate general ...
• [sai-00212] h=0 r=0 For share issuances during a period, default to StockIssuedDuringPeriodSharesNewIssues as the primary tag for all new shares issued, including
those issued in business combinations. Reserve ...
• [sai-00213] h=0 r=0 For share-based compensation, restricted stock units (RSUs) are
treated identically to restricted stock awards as both are ’equity instruments other
than options.’ Use the same tags for RSUs as ...
• [sai-00214] h=0 r=0 For business acquisition disclosures: use FiniteLivedIntangibleAssetUsefulLife for amortization periods of acquired intangible assets, Goodwill for
recognized goodwill amounts from acquisitions, ...
• [sai-00215] h=2 r=0 For debt issuance contexts, carefully distinguish between debt
discounts and deferred financing costs: use DebtInstrumentUnamortizedDiscount
for debt discounts (difference between face value and ...
• [sai-00216] h=0 r=0 When analyzing debt-related contexts, critically distinguish between instrument characteristic tags (e.g., DebtInstrumentFaceAmount, DebtInstrumentUnamortizedDiscount) that describe features of ...
• [sai-00217] h=0 r=0 For interest expense amounts, default to the general InterestExpense tag unless the context explicitly specifies debt instruments or requires
distinction between different interest types. ...
• [sai-00218] h=0 r=0 Before finalizing tag selection, always verify that the chosen tag
exists in the provided options list. If a more specific tag is recommended by context
but not available, select the most ...
• [sai-00219] h=0 r=0 For share-based compensation and equity disclosures, critically
distinguish between grant date measurements (e.g., weighted-average grant date fair
value of options) and current market ...
• [sai-00220] h=0 r=0 For warrants issued as compensation, treat them as share-based
payment arrangements under ASC 718 rather than standalone derivatives. Use ...
• [sai-00221] h=0 r=0 For tax authority claims involving unpaid taxes, penalties, and
interest where the outcome is uncertain (e.g., show cause notices, regulatory assessments), use ...
• [sai-00222] h=0 r=0 For proceeds from debt instrument issuances (including convertible notes, promissory notes, and other debt forms), always use ’DebtInstrumentFaceAmount’ to represent the principal amount received. ...
• [sai-00224] h=0 r=0 In US GAAP share-based compensation disclosures, the verb
’recognized’ specifically indicates expense allocation to accounting periods, requiring
the ’AllocatedShareBasedCompensationExpense’ tag. ...
• [sai-00225] h=0 r=0 For share-based compensation disclosures, carefully distinguish
between vested fair value measurements ...
• [sai-00226] h=0 r=0 For revolving credit facilities, carefully distinguish between initial
establishment and ongoing characteristics: use DebtInstrumentFaceAmount for the
original principal amount authorized when a ...
• [sai-00228] h=0 r=0 For financing costs that are capitalized (added to an asset’s value
rather than expensed), use DeferredFinanceCostsNet regardless of whether the context mentions amortization. ...
• [sai-00229] h=0 r=0 For share issuance disclosures, critically distinguish between
general issuances (e.g., for compensation, acquisitions, conversions) and specific sale
transactions (e.g., IPOs, secondary ...
591
42
Under review as a conference paper at COLM 2026
• [sai-00230] h=0 r=0 When distinguishing between LineOfCreditFacilityCurrentBorrowingCapacity and LineOfCreditFacilityMaximumBorrowingCapacity, carefully
analyze verb tense and temporal context: past tense verbs like ...
• [sai-00231] h=0 r=0 For convertible note transactions with Original Issue Discount
(OID), consistently apply DebtInstrumentFaceAmount for the principal/face amount
of consideration tranches and ...
• [sai-00232] h=0 r=0 For revenue recognition under ASC 606, carefully distinguish
between total revenue from contracts with customers and revenue recognized from
contract liabilities: use ...
• [sai-00233] h=0 r=0 For debt instruments, the original principal value is always tagged
as DebtInstrumentFaceAmount, and the stated annual interest rate percentage is
always tagged as ...
• [sai-00234] h=0 r=0 For tax-related numerical entities, use unit context as a key differentiator: dollar amounts representing tax benefits or expenses require IncomeTaxExpenseBenefit, while percentage values ...
• [sai-00235] h=0 r=0 For interest rate spreads on variable rate debt instruments, use
DebtInstrumentBasisSpreadOnVariableRate1 for all margin components regardless
of whether they represent minimum or maximum values ...
• [sai-00236] h=0 r=0 For term loans, prioritize LongTermDebt as the primary tag for the
outstanding amount on the balance sheet, as this represents the standard classification
for aggregated term debt. Reserve ...
• [sai-00237] h=0 r=0 For business acquisition disclosures, critically distinguish between
cash consideration and total consideration: use PaymentsToAcquireBusinessesGross
for specific cash payments made to acquire ...
• [sai-00238] h=0 r=0 For ownership percentages, apply a clear threshold-based approach: use EquityMethodInvestmentOwnershipPercentage for investments with
significant influence (typically 20-50% ownership), and ...
• [sai-00239] h=0 r=0 When interpreting share-based compensation plan descriptions,
carefully analyze phrases like ’authorized for awards that may be granted’ - this
typically refers to shares currently available for ...
• [sai-00240] h=0 r=0 For shares reserved for future issuance under stock plans, prefer
the broader tag ’CommonStockCapitalSharesReservedForFutureIssuance’ over ...
• [sai-00241] h=0 r=0 For credit facility disclosures, recognize that ’available borrowing capacity’ consistently refers to the remaining unused portion after deducting
outstanding borrowings and letters of credit, ...
• [sai-00242] h=0 r=0 When bank guarantees are issued under a credit facility as part of
its components (alongside borrowings and available capacity), use LineOfCredit for
the guarantee amount rather than ...
• [sai-00243] h=0 r=0 For share price disclosures, prefer the general ’SharePrice’ tag for
per-share price measurements in various contexts including stock issuances, rather
than transaction-specific tags like ...
• [sai-00244] h=0 r=0 For share repurchase disclosures, critically distinguish temporal
context: use StockRepurchaseProgramAuthorizedAmount1 for authorized amounts
for future repurchases (e.g., ’entered into an ASR to ...
• [sai-00245] h=0 r=0 For commitment fees on credit facilities, prefer the general LineOfCreditFacilityCommitmentFeePercentage tag as the standard choice, even when
the fee calculation is based on unused amounts. The ...
COMMON MISTAKES TO AVOID
• [err-00048] h=0 r=0 Avoid using incomplete or abbreviated tag names. Always match
the exact tag format from the provided US GAAP taxonomy list, including full
compound words and proper suffixes (e.g., ...
• [err-00055] h=3 r=2 Avoid selecting general tags when more specific tags exist that
exactly match the context. For operating lease rent expenses, using OperatingLeaseExpense instead of OperatingLeasesRentExpenseNet ...
592
43
Under review as a conference paper at COLM 2026
• [err-00059] h=0 r=0 Avoid over-specifying tags when the context does not explicitly
justify it. For example, prefer ’LeaseAndRentalExpense’ for general rent expense
contexts unless the sentence explicitly ...
• [err-00061] h=0 r=0 Avoid treating derivative instruments as independent when their
terms are directly derived from and identical to characteristics of an underlying
primary financial instrument. For example, when ...
• [err-00079] h=0 r=0 Avoid assuming that dollar amounts associated with fair value
hierarchy disclosures (Level 1, 2, or 3) automatically represent fair values. Balance
sheet amounts at specific dates typically ...
• [err-00080] h=4 r=0 Avoid using DebtInstrumentFaceAmount for revolving credit
facilities. The term ’revolving credit facility’ specifically indicates a line of credit
arrangement that requires ...
• [err-00087] h=0 r=0 Avoid treating depreciation within restructuring contexts as regular depreciation. When depreciation is part of estimated restructuring program costs
(e.g., ’Non-cash accelerated depreciation of ...
• [err-00094] h=0 r=0 Avoid using OperatingLeasesRentExpenseNet when the context
describes cash payments (’paid in rent’) rather than expense recognition. The phrase
’paid...in rent’ indicates cash outflow for rental ...
• [err-00103] h=0 r=1 Avoid using DebtInstrumentFaceAmount for repayment transactions. The phrase ’principal repayment of $X’ describes an action (repayment
transaction) requiring RepaymentsOfDebt, not a measurement of ...
• [err-00105] h=0 r=2 Avoid using BusinessAcquisitionPercentageOfVotingInterestsAcquired for total ownership percentages in subsidiaries or joint ventures. This tag
specifically applies to the percentage acquired in a ...
• [err-00107] h=0 r=0 Avoid misclassifying balance sheet amounts as transactional
events. When amounts represent outstanding balances or carrying values reported
on the balance sheet (e.g., ’outstanding term loan of ...
• [err-00130] h=0 r=2 Avoid using generic share-based compensation tags when more
specific tags exist that distinguish between different types of equity instruments. For
restricted share units (RSUs) and other equity ...
• [err-00133] h=0 r=0 Avoid using share-based compensation tags (e.g., ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross)
for warrant quantity disclosures. Warrants are distinct equity ...
• [err-00164] h=0 r=0 Avoid using tags that are not present in the provided US GAAP
taxonomy list, even if they seem semantically correct based on context. Always verify
that the selected tag exists in the available ...
• [err-00166] h=0 r=0 Avoid assuming all ownership percentages in collaborative arrangements are equity method investments. For consortium structures described
with ownership percentages below 50%, do not default to ...
• [err-00168] h=0 r=0 Avoid using ’CommonStockCapitalSharesReservedForFutureIssuance’ for Employee Stock Purchase Plan (ESPP) shares. ESPP shares represent
share-based compensation arrangements and should use ...
• [err-00177] h=0 r=0 Avoid using EquityMethodInvestmentOwnershipPercentage for
consolidated subsidiaries. When an entity is described as a subsidiary and consolidated in financial statements, the non-controlling ...
• [err-00183] h=0 r=0 Avoid using ’SaleOfStockNumberOfSharesIssuedInTransaction’
for new stock issuances that increase outstanding shares. This tag is more appropriate
for treasury stock sales or specific disposal ...
• [err-00186] h=0 r=0 Avoid automatically using DebtInstrumentFaceAmount for term
loan amounts when they are presented as components within credit facility structures
described with borrowing capacity language. Even ...
• [err-00211] h=0 r=0 Avoid selecting US GAAP tags that are not present in the provided
options list, even if they seem conceptually correct based on context or playbook
recommendations. Always verify tag availability ...
593
44
Under review as a conference paper at COLM 2026
• [err-00227] h=0 r=0 Avoid applying absolute prohibitions against using DebtInstrumentFaceAmount for revolving credit facilities. While LineOfCreditFacilityMaximumBorrowingCapacity is appropriate for describing the ...
• [err-00246] h=0 r=0 Avoid using LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage for commitment fees unless the context explicitly requires distinguishing between fees on the entire facility versus only the ...
OTHERS
• [misc-00001] h=2 r=4 When shares are issued to acquire ownership interests in another
entity (e.g., in an exchange offer), use BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued instead of ...
• [misc-00002] h=6 r=0 For cash payments in business acquisitions, use PaymentsToAcquireBusinessesGross when the context specifies ’cash paid’ or ’purchase price’ without mentioning netting against cash acquired. ...
• [misc-00003] h=3 r=0 In business combination transactions, carefully analyze the form
of consideration: equity interests issued, cash payments (gross vs net), and total
consideration. Each has distinct tags - equity ...
• [misc-00004] h=0 r=0 For mandatory convertible preferred stock and other hybrid
instruments, use tags specific to convertible features (e.g., DebtInstrumentConvertibleConversionPrice1) rather than simple stock sale ...
• [misc-00005] h=0 r=0 When dealing with preferred stock issuances, use broader stock
issuance tags like StockIssuedDuringPeriodSharesNewIssues instead of SaleOfStockNumberOfSharesIssuedInTransaction, which is typically ...
• [misc-00057] h=6 r=0 For amortization expense related to intangible assets, always
use ’AmortizationOfIntangibleAssets’ as the appropriate US GAAP tag. This tag
specifically applies to the periodic expense recognition ...
• [misc-00058] h=2 r=2 For shares available for issuance under share-based compensation plans, always use ’ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant’. This tag specifically ...
• [misc-00084] h=1 r=0 For pension and benefit plan disclosures, critically distinguish
between Defined Contribution (DC) and Defined Benefit (DB) plans as they have
fundamentally different accounting treatments and ...
• [misc-00085] h=1 r=0 Avoid using DefinedBenefitPlanContributionsByEmployer for
Defined Contribution (DC) plan contexts. These are distinct plan types with separate
accounting treatments and tags. The tag ...
• [misc-00223] h=0 r=0 When multiple debt-related measurements appear together in
disclosures, systematically distinguish between: (1) ’outstanding balance’ or ’principal balance’ at a point in time indicating carrying ...
594
45