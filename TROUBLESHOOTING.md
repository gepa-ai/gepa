# GEPA Troubleshooting Guide

This guide helps you resolve common issues when using GEPA.

## AWS Bedrock Issues

### Bedrock Inference Profile Error

**Error Message:**
```
BedrockException - {'message':'Invocation of model ID anthropic.claude-sonnet-4-5-20250929-v1:0 with on-demand throughput isn't supported. 
Retry your request with the ID or ARN of an inference profile that contains this model.'}
```

**Problem:**
Some AWS Bedrock models (like Claude Sonnet 4.5) require using an inference profile instead of the direct model ID. This is a requirement from AWS Bedrock for certain newer models.

**Solution:**
Use an inference profile ARN or cross-region inference profile ID instead of the direct model ID.

**Examples:**

Instead of:
```python
reflection_lm="bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"
```

Use one of these:

1. **Cross-region inference profile** (recommended for most users):
```python
reflection_lm="bedrock/us.anthropic.claude-sonnet-4-5-v2:0"
```

2. **Inference profile ARN** (for specific configurations):
```python
reflection_lm="bedrock/arn:aws:bedrock:us-west-2:123456789012:inference-profile/your-profile-id"
```

**Additional Resources:**
- [AWS Bedrock Cross-Region Inference Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
- [AWS Bedrock Inference Profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html)

**How GEPA Handles This:**
When GEPA encounters this error, it will:
1. Log the error with helpful guidance
2. Continue gracefully without crashing the optimization process
3. Skip the current iteration and move to the next one

The error message will appear in your logs with instructions on how to fix it.

---

## General LLM Provider Issues

### Authentication Errors

If you encounter authentication errors with any LLM provider:

1. **Check your API keys** are properly set in environment variables:
   - OpenAI: `OPENAI_API_KEY`
   - AWS Bedrock: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
   - Anthropic: `ANTHROPIC_API_KEY`

2. **Verify the model name** is correct for your provider:
   - OpenAI: `openai/gpt-4`, `openai/gpt-3.5-turbo`
   - Bedrock: Use inference profiles as described above
   - Anthropic: `anthropic/claude-3-opus-20240229`

### Rate Limiting

If you hit rate limits:

1. **Reduce the `reflection_minibatch_size`** parameter to make fewer LLM calls per iteration
2. **Implement retry logic** in your custom adapters
3. **Use a different model** with higher rate limits

---

## Need More Help?

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/gepa-ai/gepa/issues) for similar problems
2. Join our [Discord community](https://discord.gg/A7dABbtmFw) for support
3. Open a new issue with:
   - Full error message and traceback
   - Your GEPA version
   - Code snippet showing how you're using GEPA
   - LLM provider and model name
