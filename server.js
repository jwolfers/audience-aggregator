require('dotenv').config();
const express = require('express');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { createPromptStore, kitMiddleware, modelRegistry } = require('prompt-store');

const app = express();
const PORT = process.env.PORT || 3005;

// Data paths — use Render persistent disk if available, else local data/
const DATA_DIR = process.env.DATA_DIR
    ? path.join(process.env.DATA_DIR, 'audience-aggregator')
    : path.join(__dirname, 'data');
const RESPONSES_DIR = path.join(DATA_DIR, 'responses');
const QUESTIONS_FILE = path.join(DATA_DIR, 'questions.json');
const SETTINGS_FILE = path.join(DATA_DIR, 'settings.json');

// Default settings
const DEFAULT_SETTINGS = {
    provider: (process.env.GOOGLE_AI_KEY || process.env.GOOGLE_API_KEY) ? 'google' : process.env.OPENAI_API_KEY ? 'openai' : process.env.ANTHROPIC_API_KEY ? 'anthropic' : 'google',
    model: (process.env.GOOGLE_AI_KEY || process.env.GOOGLE_API_KEY) ? 'gemini-2.5-flash' : process.env.OPENAI_API_KEY ? 'gpt-4.1-mini' : process.env.ANTHROPIC_API_KEY ? 'claude-haiku-4-5-20251001' : 'gemini-2.5-flash',
    apiKeys: { openai: '', anthropic: '', google: '' },
    defaultNameDisplay: 'first',
    systemPrompt: `You are an expert at synthesizing audience responses during live lectures and presentations. You analyze free-text answers and produce clear, concise summaries organized by theme. You always return valid JSON.`,
    analysisPromptTemplate: `Analyze these {{responseCount}} responses to the question: "{{questionText}}"

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
}`
};

// ─── Git-backed prompt store (Platypus prompt architecture) ─────────
// Prose prompts live as one file per prompt in prompts/*.md; the GitHub repo
// is the source of truth (web edits commit via the Contents API, local edits
// are plain git). Config (provider, model, API keys, name display) stays in
// DATA_DIR settings.json exactly as before. Factory reset = the
// DEFAULT_SETTINGS constants above; finer-grained undo = git history on
// prompts/*.md.
const store = createPromptStore({
    owner: 'jwolfers',
    repo: 'audience-aggregator',
    branch: 'master',
    dir: 'prompts',
    // Contents:R/W PAT on the gateway (PROMPT_EDIT_TOKEN), falling back to
    // GITHUB_TOKEN. With neither set (local dev), writes go to the local
    // prompts/ files for you to git-commit.
    token: () => process.env.PROMPT_EDIT_TOKEN || process.env.GITHUB_TOKEN || '',
    secret: () => process.env.PROMPT_EDIT_SECRET || '',
    localDir: path.join(__dirname, 'prompts'), // the git checkout — fast reads + write-through
    author: { name: 'Platypus Prompt Editor', email: 'bot@platypuseconomics.com' },
    allowedOrigins: ['https://tools.platypuseconomics.com'],
});

// settings field -> git-backed prompt file
const PROSE_FIELDS = {
    systemPrompt: 'system-prompt.md',
    analysisPromptTemplate: 'analysis-prompt-template.md',
};

// Config from DATA_DIR settings.json with the prose prompts overlaid from the
// git-backed prompts/*.md files (falling back to whatever is in the JSON,
// then to the defaults, so the app still works if a file is missing).
function getSettings() {
    const cfg = readJSON(SETTINGS_FILE) || { ...DEFAULT_SETTINGS };
    for (const [field, file] of Object.entries(PROSE_FIELDS)) {
        const md = store.readSync(file);
        if (md !== null) cfg[field] = md;
        else if (typeof cfg[field] !== 'string') cfg[field] = DEFAULT_SETTINGS[field];
    }
    return cfg;
}

// Split an incoming settings blob: prose prompts → git-backed store (commit on
// the gateway, local file in dev); config fields → DATA_DIR settings.json.
async function saveSettingsSplit(updates) {
    const current = readJSON(SETTINGS_FILE) || { ...DEFAULT_SETTINGS };
    const proseWrites = [];

    for (const [field, file] of Object.entries(PROSE_FIELDS)) {
        if (typeof updates[field] === 'string') {
            proseWrites.push(store.write(file, updates[field], {
                message: `Edit ${field} via Audience Aggregator settings`,
            }));
        }
    }

    // Merge API keys carefully (don't overwrite with masked values)
    if (updates.apiKeys) {
        current.apiKeys = current.apiKeys || {};
        Object.keys(updates.apiKeys).forEach(key => {
            if (updates.apiKeys[key] && !updates.apiKeys[key].startsWith('••••')) {
                current.apiKeys[key] = updates.apiKeys[key];
            }
        });
    }

    // Merge other config settings (prompts are handled by the store above)
    const allowed = ['provider', 'model', 'defaultNameDisplay'];
    allowed.forEach(key => {
        if (updates[key] !== undefined) current[key] = updates[key];
    });

    await Promise.all(proseWrites);
    writeJSON(SETTINGS_FILE, current);
}

// Bootstrap data directories and files
function bootstrap() {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    if (!fs.existsSync(RESPONSES_DIR)) fs.mkdirSync(RESPONSES_DIR, { recursive: true });
    if (!fs.existsSync(QUESTIONS_FILE)) writeJSON(QUESTIONS_FILE, []);
    if (!fs.existsSync(SETTINGS_FILE)) writeJSON(SETTINGS_FILE, DEFAULT_SETTINGS);
}

// JSON helpers
function readJSON(filePath) {
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    } catch {
        return null;
    }
}

function writeJSON(filePath, data) {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

function getResponsesFile(questionId) {
    return path.join(RESPONSES_DIR, `${questionId}.json`);
}

function getResponses(questionId) {
    const file = getResponsesFile(questionId);
    return fs.existsSync(file) ? readJSON(file) : { responses: [], analysis: null };
}

function saveResponses(questionId, data) {
    writeJSON(getResponsesFile(questionId), data);
}

// SSE clients per question
const sseClients = {}; // { questionId: [res, res, ...] }

function notifyClients(questionId, count) {
    const clients = sseClients[questionId] || [];
    clients.forEach(res => {
        res.write(`data: ${JSON.stringify({ count })}\n\n`);
    });
}

// ─── AI Provider Calls ──────────────────────────────────────────────

async function callOpenAI(systemPrompt, userPrompt, model, apiKey) {
    const resp = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: model,
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: userPrompt }
            ],
            response_format: { type: 'json_object' },
            temperature: 0.3
        })
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error.message);
    return data.choices[0].message.content;
}

async function callAnthropic(systemPrompt, userPrompt, model, apiKey) {
    const resp = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
            model: model,
            max_tokens: 4096,
            system: systemPrompt,
            messages: [{ role: 'user', content: userPrompt }]
        })
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error.message);
    return data.content[0].text;
}

async function callGoogle(systemPrompt, userPrompt, model, apiKey) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            system_instruction: { parts: [{ text: systemPrompt }] },
            contents: [{ parts: [{ text: userPrompt }] }],
            generationConfig: {
                responseMimeType: 'application/json',
                temperature: 0.3
            }
        })
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error.message);
    return data.candidates[0].content.parts[0].text;
}

async function callAI(systemPrompt, userPrompt) {
    const settings = readJSON(SETTINGS_FILE);
    const { provider, model, apiKeys } = settings;

    // Check for API key from settings or env
    let apiKey;
    if (provider === 'openai') {
        apiKey = apiKeys.openai || process.env.OPENAI_API_KEY;
        if (!apiKey) throw new Error('OpenAI API key not configured');
        return callOpenAI(systemPrompt, userPrompt, model, apiKey);
    } else if (provider === 'anthropic') {
        apiKey = apiKeys.anthropic || process.env.ANTHROPIC_API_KEY;
        if (!apiKey) throw new Error('Anthropic API key not configured');
        return callAnthropic(systemPrompt, userPrompt, model, apiKey);
    } else if (provider === 'google') {
        apiKey = apiKeys.google || process.env.GOOGLE_AI_KEY || process.env.GOOGLE_API_KEY;
        if (!apiKey) throw new Error('Google AI API key not configured');
        return callGoogle(systemPrompt, userPrompt, model, apiKey);
    }
    throw new Error(`Unknown provider: ${provider}`);
}

// ─── Middleware ──────────────────────────────────────────────────────

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ─── Questions API ──────────────────────────────────────────────────

app.get('/api/questions', (req, res) => {
    const questions = readJSON(QUESTIONS_FILE) || [];
    res.json(questions);
});

app.post('/api/questions', (req, res) => {
    const questions = readJSON(QUESTIONS_FILE) || [];
    const question = {
        id: uuidv4(),
        text: req.body.text || '',
        nameDisplay: req.body.nameDisplay || 'first',
        summarizeCount: req.body.summarizeCount || 4,
        identifyPrompt: req.body.identifyPrompt || 'most interesting',
        includeExplanations: req.body.includeExplanations !== false,
        active: false,
        createdAt: new Date().toISOString()
    };
    questions.push(question);
    writeJSON(QUESTIONS_FILE, questions);
    res.status(201).json(question);
});

app.put('/api/questions/:id', (req, res) => {
    const questions = readJSON(QUESTIONS_FILE) || [];
    const idx = questions.findIndex(q => q.id === req.params.id);
    if (idx === -1) return res.status(404).json({ error: 'Question not found' });

    const allowed = ['text', 'nameDisplay', 'summarizeCount', 'identifyPrompt', 'includeExplanations'];
    allowed.forEach(key => {
        if (req.body[key] !== undefined) questions[idx][key] = req.body[key];
    });
    writeJSON(QUESTIONS_FILE, questions);
    res.json(questions[idx]);
});

app.delete('/api/questions/:id', (req, res) => {
    let questions = readJSON(QUESTIONS_FILE) || [];
    questions = questions.filter(q => q.id !== req.params.id);
    writeJSON(QUESTIONS_FILE, questions);
    // Also clean up responses file
    const respFile = getResponsesFile(req.params.id);
    if (fs.existsSync(respFile)) fs.unlinkSync(respFile);
    res.json({ success: true });
});

app.post('/api/questions/:id/activate', (req, res) => {
    const questions = readJSON(QUESTIONS_FILE) || [];
    questions.forEach(q => q.active = (q.id === req.params.id));
    writeJSON(QUESTIONS_FILE, questions);
    const active = questions.find(q => q.active);
    res.json(active || { error: 'Question not found' });
});

app.get('/api/active', (req, res) => {
    const questions = readJSON(QUESTIONS_FILE) || [];
    const active = questions.find(q => q.active);
    if (!active) return res.json({ active: false });
    res.json(active);
});

// ─── Responses API ──────────────────────────────────────────────────

app.post('/api/respond', (req, res) => {
    const questions = readJSON(QUESTIONS_FILE) || [];
    const active = questions.find(q => q.active);
    if (!active) return res.status(400).json({ error: 'No active question' });

    const data = getResponses(active.id);
    const response = {
        id: uuidv4(),
        name: (req.body.name || '').trim() || null,
        answer: (req.body.answer || '').trim(),
        timestamp: new Date().toISOString()
    };
    if (!response.answer) return res.status(400).json({ error: 'Answer is required' });

    data.responses.push(response);
    saveResponses(active.id, data);

    // Notify SSE clients
    notifyClients(active.id, data.responses.length);

    res.status(201).json({ success: true, count: data.responses.length });
});

app.get('/api/responses/:id', (req, res) => {
    const data = getResponses(req.params.id);
    res.json(data);
});

app.get('/api/responses/:id/stream', (req, res) => {
    const questionId = req.params.id;

    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });

    // Send current count immediately
    const data = getResponses(questionId);
    res.write(`data: ${JSON.stringify({ count: data.responses.length })}\n\n`);

    // Register client
    if (!sseClients[questionId]) sseClients[questionId] = [];
    sseClients[questionId].push(res);

    // Cleanup on close
    req.on('close', () => {
        sseClients[questionId] = (sseClients[questionId] || []).filter(c => c !== res);
    });
});

app.post('/api/responses/:id/clear', (req, res) => {
    saveResponses(req.params.id, { responses: [], analysis: null });
    notifyClients(req.params.id, 0);
    res.json({ success: true });
});

app.get('/api/responses/:id/csv', (req, res) => {
    const data = getResponses(req.params.id);
    const rows = [['Name', 'Answer', 'Timestamp']];
    data.responses.forEach(r => {
        rows.push([
            `"${(r.name || 'Anonymous').replace(/"/g, '""')}"`,
            `"${(r.answer || '').replace(/"/g, '""')}"`,
            `"${r.timestamp}"`
        ]);
    });
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="responses-${req.params.id}.csv"`);
    res.send(rows.map(r => r.join(',')).join('\n'));
});

// ─── Analysis API ───────────────────────────────────────────────────

app.post('/api/analyze/:id', async (req, res) => {
    try {
        const questions = readJSON(QUESTIONS_FILE) || [];
        const question = questions.find(q => q.id === req.params.id);
        if (!question) return res.status(404).json({ error: 'Question not found' });

        const data = getResponses(req.params.id);
        if (data.responses.length === 0) return res.status(400).json({ error: 'No responses to analyze' });

        const settings = getSettings();

        // Build responses text
        const responsesText = data.responses.map((r, i) => {
            const name = r.name || 'Anonymous';
            return `${i + 1}. [${name}]: ${r.answer}`;
        }).join('\n');

        // Build prompt from template
        const summarizeCount = question.summarizeCount || 4;
        const extraThemeCount = Math.max(2, Math.ceil(summarizeCount * 0.75));
        let prompt = settings.analysisPromptTemplate
            .replace(/\{\{responseCount\}\}/g, data.responses.length)
            .replace(/\{\{questionText\}\}/g, question.text)
            .replace(/\{\{responses\}\}/g, responsesText)
            .replace(/\{\{summarizeCount\}\}/g, summarizeCount)
            .replace(/\{\{extraThemeCount\}\}/g, extraThemeCount)
            .replace(/\{\{identifyPrompt\}\}/g, question.identifyPrompt || 'most interesting');

        const raw = await callAI(settings.systemPrompt, prompt);

        // Parse JSON from AI response
        let analysis;
        try {
            analysis = JSON.parse(raw);
        } catch {
            // Try to extract JSON from markdown code block
            const jsonMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
            if (jsonMatch) {
                analysis = JSON.parse(jsonMatch[1]);
            } else {
                throw new Error('AI did not return valid JSON');
            }
        }

        // Save analysis
        data.analysis = {
            result: analysis,
            timestamp: new Date().toISOString(),
            provider: settings.provider,
            model: settings.model
        };
        saveResponses(req.params.id, data);

        res.json(data.analysis);
    } catch (err) {
        console.error('Analysis error:', err);
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/analysis/:id', (req, res) => {
    const data = getResponses(req.params.id);
    if (!data.analysis) return res.json({ analysis: null });
    res.json(data.analysis);
});

// ─── Settings API ───────────────────────────────────────────────────

app.get('/api/settings', (req, res) => {
    const settings = getSettings();
    // Mask API keys for security
    const masked = { ...settings };
    if (masked.apiKeys) {
        masked.apiKeys = {
            openai: masked.apiKeys.openai ? '••••' + masked.apiKeys.openai.slice(-4) : '',
            anthropic: masked.apiKeys.anthropic ? '••••' + masked.apiKeys.anthropic.slice(-4) : '',
            google: masked.apiKeys.google ? '••••' + masked.apiKeys.google.slice(-4) : ''
        };
    }
    res.json(masked);
});

app.put('/api/settings', async (req, res) => {
    try {
        await saveSettingsSplit(req.body || {});
        res.json({ success: true });
    } catch (err) {
        console.error('Settings save error:', err);
        res.status(500).json({ error: err.message });
    }
});

// Reset prompts to factory defaults (the DEFAULT_SETTINGS constants).
// For finer-grained undo, use the git history of prompts/*.md.
app.post('/api/settings/reset-prompts', async (req, res) => {
    try {
        await Promise.all(Object.entries(PROSE_FIELDS).map(([field, file]) =>
            store.write(file, DEFAULT_SETTINGS[field], {
                message: `Reset ${field} to default via Audience Aggregator settings`,
            })
        ));
        res.json({ success: true });
    } catch (err) {
        console.error('Prompt reset error:', err);
        res.status(500).json({ error: err.message });
    }
});

// ─── Prompt store & model registry ──────────────────────────────────

app.use('/api/prompts', store.router);   // list/get/put/history/revert
app.use('/prompt-kit', kitMiddleware()); // serves settings-kit.css/js
app.get('/api/model-registry', (_req, res) => res.json(modelRegistry.registryJson()));

// ─── Start ──────────────────────────────────────────────────────────

bootstrap();
app.listen(PORT, () => {
    console.log(`Audience Aggregator running at http://localhost:${PORT}`);
});
