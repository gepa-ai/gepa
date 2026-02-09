# Proposers

Proposers are strategies that generate new candidate programs during optimization. GEPA provides two main proposer types:

## Reflective Mutation Proposer

The primary proposer that uses LLM-based reflection to improve candidates based on execution feedback.

- [`ReflectiveMutationProposer`](ReflectiveMutationProposer.md) - Main reflective mutation proposer

## Merge Proposer

A proposer that combines successful candidates from the Pareto frontier.

- [`MergeProposer`](MergeProposer.md) - Merge-based candidate proposer

## Base Classes and Protocols

- [`CandidateProposal`](CandidateProposal.md) - Data class for candidate proposals
- [`ProposeNewCandidate`](ProposeNewCandidate.md) - Protocol for proposer strategies
- [`Signature`](Signature.md) - Base class for LLM prompt signatures
- [`LanguageModel`](LanguageModel.md) - Protocol for language models
