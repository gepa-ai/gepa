---
date:
  created: 2026-02-18
authors:
  - shangyin
  - lakshya
  - rohit
  - dan
  - koushik
  - alex
  - matei
equal_contribution:
  - "Shangyin Tan"
  - "Lakshya A Agrawal"
slug: learning-skills-for-any-repository
title: "Learning Skills for Any Repository"
description: "Introducing gskill, a fully automated pipeline that uses GEPA and SWE-smith to learn repository-specific skills for coding agents."
---

# Learning Skills for Any Repository

Today, we are introducing **gskill**, a fully automated pipeline to learn skills for any repository. Given any GitHub repository, gskill learns important skills for coding agents to better work with the repository.

<!-- more -->

## The Recipe

gskill consists of two main components:

1. **GEPA's [optimize_anything](https://gepa-ai.github.io/gepa/)** -- a powerful API to optimize any textual artifact, including skills.
2. **SWE-smith** -- a data generation pipeline that creates arbitrary verifiable tasks for any GitHub repository, providing meaningful training and validation data for the optimization process.

<figure markdown="span">
  ![gskill pipeline: Target Repo feeds into SWE-smith for task generation, then enters the GEPA Optimization Loop where Agent Rollouts are evaluated for fitness, a Reflective Proposer generates candidate skills, and the best candidates are selected into a pool, ultimately producing Learned Skills.](gskill-pipeline.png)
  <figcaption>The gskill pipeline: SWE-smith generates tasks from a target repository, and the GEPA optimization loop iteratively evolves skills through agent evaluation and reflective proposal.</figcaption>
</figure>

Learning skills requires feedback, and feedback requires tasks. This is where **SWE-smith** comes in. Given a GitHub repository, SWE-smith automatically generates a diverse set of verifiable software engineering tasks. These tasks are grounded in the real codebase and come with verifiable tests. Think of SWE-smith as converting a static repository into an active training environment.

Once we have tasks, we can optimize. We build gskill based on the belief that there are common skills about a repository that can be crucial for most tasks within the repository, and these skills should be learned, automatically. To achieve this goal, `optimize_anything`[^gepa-docs] provides the backbone of the learning loop.

[^gepa-docs]: For readers who are not familiar with how `optimize_anything` and GEPA works, we highly recommend checking out the [official documentation](https://gepa-ai.github.io/gepa/).

In a nutshell, the `optimize_anything` loop starts with a (possibly empty) set of skills, evaluates the agent with a chosen skill, and then updates the skill by employing another more powerful LLM to reflect on the evaluation results and feedback. This process is repeated until some budget is exhausted.

## Experiments

To evaluate gskill, we chose two popular repositories: [jinja](https://github.com/pallets/jinja) and [bleve](https://github.com/blevesearch/bleve). Our evaluation setup:

- Start with an empty **mini-swe-agent** powered by gpt-5-mini
- Generate ~250 SWE-smith tasks per repository
- Use gskill to learn skills for the agent
- Evaluate performance on a holdout test set
- Transfer the **learned skills** to Claude Code and evaluate with both claude-haiku-4-5 and claude-sonnet-4-5

### Mini-SWE-Agent Results

<figure markdown="span">
  ![Bar chart comparing Mini-SWE-Agent (gpt-5-mini) test set resolve rates. Jinja: baseline 0.55 vs with GEPA skills 0.82. Bleve: baseline 0.24 vs with GEPA skills 0.93.](mini-swe-agent-results.png)
  <figcaption>Mini-SWE-Agent performance with GEPA-evolved skills. Skills learned via gskill dramatically improve resolve rates on both repositories.</figcaption>
</figure>

### Claude Code Transfer Results

<figure markdown="span">
  ![Bar chart showing Claude Code evaluation on Bleve (n=58). Pass rates: Claude Haiku 4.5 at 79.3% (173s), Claude Haiku 4.5 + Skills at 98.3% (142s), Claude Sonnet 4.5 at 94.8% (285s), Claude Sonnet 4.5 + Skills at 100.0% (169s).](claude-code-bleve.png)
  <figcaption>Claude Code evaluation on Bleve (n=58). Adding learned skills boosts pass rates while also reducing average task duration.</figcaption>
</figure>

<figure markdown="span">
  ![Bar chart showing Claude Code evaluation on Jinja (n=66). Pass rates: Claude Haiku 4.5 at 93.9% (177s), Claude Haiku 4.5 + Skills at 100.0% (148s), Claude Sonnet 4.5 at 100.0% (254s), Claude Sonnet 4.5 + Skills at 98.5% (225s).](claude-code-jinja.png)
  <figcaption>Claude Code evaluation on Jinja (n=66). Skills learned on a weaker model (gpt-5-mini) transfer effectively to Claude Code, achieving near-perfect pass rates.</figcaption>
</figure>
