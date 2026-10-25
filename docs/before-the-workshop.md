# Before the workshop

About 5 minutes. Do it the day before, not weeks ahead.

1. **Get a free Gemini API key.** Go to [Google AI Studio](https://aistudio.google.com/apikey),
   sign in with a Google account, and choose **Create API key**. No credit card is needed.
2. **Colab track:** open any lab from [`notebooks/`](../notebooks/) and add
   secrets with the 🔑 icon:
   - `GOOGLE_API_KEY`: every lab needs it.
   - `LASP_MCP_TOKEN`: Labs 2 and up. The token is given out in the session.
3. **n8n track:** accept your invite to the workshop's n8n, then follow
   [`n8n/README.md`](../n8n/README.md) to add the two credentials.

## Gemini or Claude?

Both tracks default to **Gemini** because it's free, and each attendee uses
their own key. If you have an Anthropic key, you can use **Claude** instead:
- **Colab:** set `PROVIDER = "claude"` and add an `ANTHROPIC_API_KEY` secret.
- **n8n:** import from `n8n/claude/`.

The Claude API has no free tier. If you hand out one shared workshop key, set a
spending limit in the [Anthropic Console](https://console.anthropic.com) first.
The agent labs (2–4) use the most tokens.

## Free-tier limits

The labs use `gemini-3.5-flash-lite`, which allows **15 requests a minute and
500 a day** on the free tier. All four labs together use about 45 requests.
Every agent step is one request.

The notebooks pace themselves to 12 a minute, so you won't hit the limit.

> ⚠️ **Use workshop or public data only.** The free tier of the Gemini API may
> use your prompts to improve Google's products. Keep unpublished results and
> anything under NDA out of it.
