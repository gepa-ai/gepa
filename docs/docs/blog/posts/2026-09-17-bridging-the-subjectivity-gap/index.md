---
date:
  created: 2026-09-17
authors:
  - seth
slug: bridging-the-subjectivity-gap
readtime: 8
title: "Bridging the Subjectivity Gap: How APO Helps Teams Build Expert-Aligned AI Functions"
description: "A Sutro case study showing how automated prompt optimization aligns repeated AI decisions with organization-specific judgment using only 30 annotations."
social_image: blog/2026-09-17-bridging-the-subjectivity-gap/images/accuracy-lift.png
external_links_new_tab: true
---

# Bridging the Subjectivity Gap: How APO Helps Teams Build Expert-Aligned AI Functions

<div class="blog-stat-strip" markdown>
  <div><strong>30</strong><span><b>annotations</b></span></div>
  <div><strong>+42.9pp</strong><span><b>average accuracy increase</b></span></div>
  <div><strong>+30.6pp</strong><span><b>average consistency increase</b></span></div>
  <div><strong>79%</strong><span><b>lower inference cost for frontier performance</b></span></div>
</div>

Most applied AI organizations build systems for recurring tasks, often taking the form of scorers, judges, classifiers, summarizers, matchers, and more. These tasks are typically subjective in nature, and the goal is to align the judgment of a model with that of a domain expert.

While foundation models have a lot of general world knowledge, they don't know how *your organization* wants a particular decision made. That gap between general intelligence and organization-specific judgment is the **subjectivity gap**.

At <a href="https://sutro.sh/" target="_blank" rel="noopener noreferrer">Sutro</a>, we refer to models that make repeated decisions as **AI Functions**. They make up a less-visible category of intelligence relative to agentic coding, world models, and robotics - but are perhaps just as important, if not more so.

<!-- more -->

<figure markdown="span">
  ![A two-axis diagram categorizing AI tasks by how often they repeat and whether correctness is objective or organization-specific. AI Functions, such as judging whether a company is a good lead, occupy the highly repeated and organization-specific quadrant.](images/subjectivity-gap.png){ style="width: 100%;" }
  <figcaption>AI Functions are repeated tasks whose correct output depends on organization-specific judgment rather than a universal answer.</figcaption>
</figure>

Building AI Functions is easy, but calibrating them to your judgment remains difficult. Common options include hand-tuning prompts, having a coding agent scan a dataset and write a prompt that encodes its own judgment, or collecting labeled data and fine-tuning your own model.

However, we think automated prompt and harness optimization (APO) is a superior tool for building reliable AI Functions for two reasons:

- **APO works well in sparse-data environments.** You only need to capture enough representative samples to demonstrate the general decision procedure, not thousands of "easy" cases that are already in-distribution for a foundation model.
- **Modern foundation models are adept at instruction following.** They can follow a well-defined, in-context decision policy. Instead of encoding that policy in model weights through fine-tuning or reinforcement learning, APO can encode it in context - through prompts and, optionally, tools - making the resulting AI Function much more portable.

It's like teaching a new employee or intern how work gets done at your company. If you can give them the right context, instructions, and a handful of tricky examples plus their solutions, they can be surprisingly effective at carrying out specific organizational tasks right out of the gate.

And, of course, <a href="https://gepa-ai.github.io/gepa/" target="_blank" rel="noopener noreferrer">GEPA</a> is perhaps the best open-source APO tool a team can reach for today. Its flexibility, speed, and abstractions such as <a href="https://gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything/" target="_blank" rel="noopener noreferrer"><em>optimize_anything</em></a> make it ideal for adapting to new scenarios.

## A simple demonstration

At Sutro, we have an internal lead scorer. We built it on top of the Sutro platform, which materializes a dataset of "hard" cases to annotate and optimize against.

We used a simple starting prompt that briefly described what Sutro does and asked models to classify leads as *strong*, *medium*, or *weak* fits. We then evaluated eleven frontier models on a validation set of 30 cases representing Sutro's judgment of lead quality.

Separately, we used GEPA to optimize against an additional 30 annotated cases, allowing each model to independently learn an optimized system prompt for the task. We show the striking results below, plotted against the extrapolated cost to process 1,000 records.

<figure markdown="span">
  ![Scatter plot of held-out lead-scoring accuracy against estimated inference cost for eleven models. Arrows connect each model's default-prompt accuracy to its optimized-prompt accuracy, showing gains of 20 to 73 percentage points.](images/accuracy-lift.png){ style="width: 100%;" }
  <figcaption>GEPA improved held-out accuracy for every model. The chart plots default and optimized prompts against the estimated inference cost per 1,000 records.</figcaption>
</figure>

<figure markdown="span">
  ![Dumbbell chart comparing default-prompt and optimized-prompt consistency for eleven models over ten runs. Every optimized prompt exceeds 90 percent consistency, and the all-model average rises by 30.6 percentage points.](images/consistency-lift.png){ style="width: 100%;" }
  <figcaption>Optimization also made model judgments substantially more repeatable. Every model exceeded 90% consistency after optimization.</figcaption>
</figure>

**Accuracy**

<div class="benchmark-results" markdown>

| Model | Default prompt | Optimized prompt | Gain |
| --- | ---: | ---: | ---: |
| **`gpt-oss-120b`** | 23% | 77% | +54pp |
| **`gpt-oss-20b`** | 20% | 73% | +53pp |
| **`nemotron-3-nano`** | 0% | 73% | **+73pp** |
| **`nemotron-3-super`** | 37% | **90%** | +53pp |
| **`claude-haiku-4-5`** | 30% | 83% | +53pp |
| **`claude-sonnet-4-5`** | 47% | 73% | +26pp |
| **`gemini-3.5-flash`** | **57%** | 77% | +20pp |
| **`gemma-4-26b-a4b`** | 43% | 73% | +30pp |
| **`gemma-4-31b`** | 30% | 77% | +47pp |
| **`openai-gpt-5.6-luna`** | **57%** | **90%** | +33pp |
| **`openai-gpt-5.6-terra`** | 43% | 73% | +30pp |

</div>

**Consistency across 10 runs**

<div class="benchmark-results" markdown>

| Model | Default prompt | Optimized prompt | Gain |
| --- | ---: | ---: | ---: |
| **`gpt-oss-120b`** | 53% | 93% | +40pp |
| **`gpt-oss-20b`** | 41% | 91% | +50pp |
| **`nemotron-3-nano`** | 35% | 92% | **+57pp** |
| **`nemotron-3-super`** | 60% | 91% | +31pp |
| **`claude-haiku-4-5`** | 62% | 92% | +30pp |
| **`claude-sonnet-4-5`** | 65% | 95% | +30pp |
| **`gemini-3.5-flash`** | 77% | **97%** | +20pp |
| **`gemma-4-26b-a4b`** | 80% | 94% | +14pp |
| **`gemma-4-31b`** | 59% | 95% | +36pp |
| **`openai-gpt-5.6-luna`** | **91%** | **97%** | +6pp |
| **`openai-gpt-5.6-terra`** | 72% | 95% | +23pp |

</div>

**Every single model sees a sizable accuracy gain.** Small, open-weight models often exceed the task quality of much larger proprietary models at a fraction of the cost.

Perhaps equally important, response consistency also dramatically improves, **lifting all models above 90%** across ten runs for each model-prompt combination.

You may scoff and think, "Of course accuracy goes up; the system was optimized against your subjective decision criteria." If so, you understand the point. We are measuring decision *alignment* against an organization-defined reference set, not objectively verifiable outcomes. This is the nature of LLM judges and of many applied AI evaluations without verifiable rewards.

### Cost improvements

The average cost of scoring each additional lead with the optimized prompt **grew** by **15.4%**. However, optimized Nemotron 3 Super and GPT 5.6 Luna matched the highest observed accuracy while costing **54%** and **79% less** per lead, respectively, than Gemini 3.5 Flash. These savings far outweigh the cost increase from longer input prompts, which can often be cached.

## Becoming model-agnostic

It's worth reiterating that we treated this task as model-agnostic. We created an annotation set of hard cases, gave each model the opportunity to adapt to the task with APO, and then let our own evaluation tell us which models were best suited for the job. We used our own data, representing our own decision criteria, to make an informed choice about which model to deploy.

This becomes particularly important when there are real deployment constraints. If a team needs to use an open-weight model for security reasons or stay below a particular inference cost, a task-specific evaluation like this can show which models actually satisfy those constraints **after adaptation**, rather than forcing the team to infer suitability from general-purpose benchmarks.

## Alternatives

**Manual prompt engineering.** If we were to do the same exercise with manual prompt engineering, it would require the time-intensive process of:

- Gathering hard and representative samples by hand and setting up tooling to track performance quality on each iteration.
- Manually looking for error modes in failed cases and appending them as new rules to the prompt. This could take hours, days, or weeks, depending on quality needs.
- Repeating this for every model we want to test - or hoping a single prompt generalizes well to all of them. Spoiler: it often doesn't.

**Model-written prompts.** We could ask Claude or another auto-grader to populate our annotations instead of humans, but this defeats the purpose of learning our subjective rules.

**Fine-tuning or reinforcement learning.** Weight-based adaptation can be powerful, particularly when sufficient training data and verifiable rewards are available. For this task, however, APO produced substantial improvement from only 30 annotations and required no weight updates.

In our case, 30 annotations earned us an average of **42.9** percentage points of task accuracy, or **1.43 percentage points per annotation**.

Because APO doesn't require weight updates, it's easy to run AI Functions using off-the-shelf, serverless inference providers. It also becomes trivial to re-optimize whenever labeled data arrives.

## Own your own evals, own your own intelligence

As more public benchmarks become saturated and the frontier landscape fragments, more applied AI teams are asking:

- Is this model performant on *our task*, not just a public benchmark?
- How can we have greater sovereignty over the models that run *our tasks* on *our data*?
- How can we become *model-agnostic* and adapt our tasks to the best model for the job?

We believe APO - and, in turn, GEPA - can help deliver these answers for many companies. Many enterprise tasks take the form of AI Functions, where subjective judgment represents the last mile in making a task work well, as we demonstrated with our internal lead scorer.

Moreover, APO is a way to lift the performance of open-weight models to or above that of their proprietary counterparts without needing to train custom models. As intelligence proliferates, we believe cheap, efficient, and portable ways to adapt models will be extremely helpful alongside alternatives such as reinforcement learning and fine-tuning.

## Limitations

- As mentioned, one of the most challenging aspects of using APO is gathering high-quality annotations on difficult and representative samples. In practice, teams need infrastructure for finding hard examples, collecting annotations, and maintaining those datasets over time. This is one of the problems we work on at <a href="https://sutro.sh/" target="_blank" rel="noopener noreferrer">Sutro</a>.
- APO and GEPA require good scorers and objective functions to hill-climb successfully. Sometimes these are straightforward to set up; other times, strong results require meta-judges or other complex scorers.

## Addendum: TypeSafe's Jev

We ran this benchmark before the release of <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev" target="_blank" rel="noopener noreferrer">TypeSafe's Jev model</a>. Instead of including it head-to-head with the other models, we include its results here because Jev cannot yet be optimized directly in the same way Sutro optimizes other LLMs with GEPA.

To run what we think is a fair benchmark, we transferred the optimized prompts from each of the eleven models we tested and reformatted them as TypeSafe's documentation suggests. We evaluated all eleven transferred prompts on held-out cases, then compared their average and best performance with the unoptimized prompt.

| Metric | Result |
| --- | ---: |
| **Zero-shot accuracy** | 80% |
| **Average post-optimization accuracy** | 77% |
| **Highest accuracy from a specifically transferred prompt** | 86.7% |
| **Estimated cost per 1,000 records, unoptimized** | $0.07 |
| **Estimated cost per 1,000 records, optimized** | $0.097 |
| **Consistency** | 100% over 10 runs |

Notably:

- It had the best zero-shot performance across the models benchmarked.
- Its best-case optimization outcome was below the best accuracy of the other models.
- It was approximately 5× less expensive than Luna, the next-best model on price, and leading overall performance.
- It's very fast, though we did not benchmark the latency of the other models.
- It's deterministic, so no variance appears between runs.

Overall, while its accuracy was below the best-case performance in this benchmark, we're extremely excited by Jev and instruction-following discriminators as a model class. We should not treat its optimization accuracy as a ceiling - better methods for adapting this type of model will arise, and this benchmark covers only one task.

Jev suggests an interesting direction for AI Functions: models purpose-built for instruction-following and decision-making may dramatically improve the baseline economics and consistency of these workloads. But the subjectivity gap remains: deciding what "correct" means for a particular task, measuring it, and adapting the system accordingly.

One thing is clear: as intelligence becomes too cheap to meter and the number of tasks we can automate grows dramatically, more efficient tools for last-mile alignment will become increasingly important. We've only just seen the start.

## Reach out

If you have questions about this post, GEPA, APO, or getting started with AI Functions in general, email [team@sutro.sh](mailto:team@sutro.sh) or join the GEPA community on <a href="https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w" target="_blank" rel="noopener noreferrer">Slack</a>.
