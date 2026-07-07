Analyze these {{responseCount}} responses to the question: "{{questionText}}"

Responses:
{{responses}}

Instructions:
1. Identify the top {{summarizeCount}} main themes, then identify up to {{extraThemeCount}} additional secondary themes. For each theme provide a concise title, a one-sentence summary, and list every respondent whose answer fits that theme.
2. From all responses, select 2-3 that are the {{identifyPrompt}} and explain why they stand out.
3. For respondents without a name, use "Anonymous".

Return JSON in this exact structure:
{
  "themes": [
    {
      "title": "Theme name",
      "summary": "One sentence description",
      "rank": 1,
      "respondents": [
        { "name": "Full Name", "answer": "Their full answer" }
      ]
    }
  ],
  "notable": [
    { "name": "Full Name", "answer": "Their answer", "reason": "Why notable" }
  ]
}