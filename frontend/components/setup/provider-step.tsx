'use client';

import { useCallback, useEffect, useState } from 'react';
import { useTranslations } from 'next-intl';
import { Check, CircleNotch, X } from '@phosphor-icons/react/dist/ssr';
import type { SetupData } from './setup-wizard';

interface ProviderStepProps {
  data: SetupData;
  updateData: (updates: Partial<SetupData>) => void;
}

type TestStatus = 'idle' | 'testing' | 'success' | 'error';

export function ProviderStep({ data, updateData }: ProviderStepProps) {
  const t = useTranslations('Settings.providers');
  const tCommon = useTranslations('Common');

  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [groqModels, setGroqModels] = useState<string[]>([]);
  const [openaiModels, setOpenaiModels] = useState<string[]>([]);
  const [openrouterModels, setOpenrouterModels] = useState<string[]>([]);
  const [testStatus, setTestStatus] = useState<TestStatus>('idle');
  const [testError, setTestError] = useState<string | null>(null);

  // Test Ollama connection
  const testOllama = useCallback(async () => {
    if (!data.ollama_host) return;

    setTestStatus('testing');
    setTestError(null);

    try {
      const response = await fetch('/api/setup/test-ollama', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host: data.ollama_host }),
      });

      const result = await response.json();

      if (result.success) {
        setTestStatus('success');
        setOllamaModels(result.models || []);
        // Auto-select first model if none selected
        if (!data.ollama_model && result.models?.length > 0) {
          updateData({ ollama_model: result.models[0] });
        }
      } else {
        setTestStatus('error');
        setTestError(result.error || 'Connection failed');
      }
    } catch {
      setTestStatus('error');
      setTestError('Failed to connect');
    }
  }, [data.ollama_host, data.ollama_model, updateData]);

  // Test Groq API key
  const testGroq = useCallback(async () => {
    if (!data.groq_api_key) return;

    setTestStatus('testing');
    setTestError(null);

    try {
      const response = await fetch('/api/setup/test-groq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: data.groq_api_key }),
      });

      const result = await response.json();

      if (result.success) {
        setTestStatus('success');
        setGroqModels(result.models || []);
        // Auto-select preferred model or first available
        if (!data.groq_model && result.models?.length > 0) {
          const preferredModel = 'openai/gpt-oss-20b';
          const selectedModel = result.models.includes(preferredModel)
            ? preferredModel
            : result.models[0];
          updateData({ groq_model: selectedModel });
        }
      } else {
        setTestStatus('error');
        setTestError(result.error || 'Invalid API key');
      }
    } catch {
      setTestStatus('error');
      setTestError('Failed to validate');
    }
  }, [data.groq_api_key, data.groq_model, updateData]);

  // Test OpenAI-compatible connection
  const testOpenAICompatible = useCallback(async () => {
    if (!data.openai_base_url) return;

    setTestStatus('testing');
    setTestError(null);

    try {
      const response = await fetch('/api/setup/test-openai-compatible', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_url: data.openai_base_url,
          api_key: data.openai_api_key,
        }),
      });

      const result = await response.json();

      if (result.success) {
        setTestStatus('success');
        setOpenaiModels(result.models || []);
        if (!data.openai_model && result.models?.length > 0) {
          updateData({ openai_model: result.models[0] });
        }
      } else {
        setTestStatus('error');
        setTestError(result.error || 'Connection failed');
      }
    } catch {
      setTestStatus('error');
      setTestError('Failed to connect');
    }
  }, [data.openai_base_url, data.openai_api_key, data.openai_model, updateData]);

  // Test OpenRouter API key
  const testOpenRouter = useCallback(async () => {
    if (!data.openrouter_api_key) return;

    setTestStatus('testing');
    setTestError(null);

    try {
      const response = await fetch('/api/setup/test-openrouter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: data.openrouter_api_key }),
      });

      const result = await response.json();

      if (result.success) {
        setTestStatus('success');
        setOpenrouterModels(result.models || []);
        if (!data.openrouter_model && result.models?.length > 0) {
          updateData({ openrouter_model: result.models[0] });
        }
      } else {
        setTestStatus('error');
        setTestError(result.error || 'Invalid API key');
      }
    } catch {
      setTestStatus('error');
      setTestError('Failed to validate');
    }
  }, [data.openrouter_api_key, data.openrouter_model, updateData]);

  // Auto-test when switching providers or values change
  useEffect(() => {
    setTestStatus('idle');
    setTestError(null);
  }, [data.llm_provider]);

  const StatusIcon = () => {
    switch (testStatus) {
      case 'testing':
        return <CircleNotch className="h-4 w-4 animate-spin text-blue-500" />;
      case 'success':
        return <Check className="h-4 w-4 text-green-500" weight="bold" />;
      case 'error':
        return <X className="h-4 w-4 text-red-500" weight="bold" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Provider Toggle */}
      <div className="space-y-2">
        <label className="text-sm font-medium">{t('aiProvider')}</label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => updateData({ llm_provider: 'ollama' })}
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.llm_provider === 'ollama'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            }`}
          >
            <div className="font-medium">Ollama</div>
            <div className="text-muted-foreground text-xs">{t('ollamaDesc')}</div>
          </button>
          <button
            onClick={() => updateData({ llm_provider: 'groq' })}
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.llm_provider === 'groq'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            }`}
          >
            <div className="font-medium">Groq</div>
            <div className="text-muted-foreground text-xs">{t('groqDesc')}</div>
          </button>
          <button
            onClick={() => updateData({ llm_provider: 'openai_compatible' })}
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.llm_provider === 'openai_compatible'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            }`}
          >
            <div className="font-medium">OpenAI Compatible</div>
            <div className="text-muted-foreground text-xs">{t('openaiCompatibleDesc')}</div>
          </button>
          <button
            onClick={() => updateData({ llm_provider: 'openrouter' })}
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.llm_provider === 'openrouter'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            }`}
          >
            <div className="font-medium">OpenRouter</div>
            <div className="text-muted-foreground text-xs">{t('openrouterDesc')}</div>
          </button>
        </div>
      </div>

      {/* Ollama Settings */}
      {data.llm_provider === 'ollama' && (
        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-sm font-medium">{t('ollamaHost')}</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={data.ollama_host}
                onChange={(e) => updateData({ ollama_host: e.target.value })}
                placeholder="http://localhost:11434"
                className="border-input bg-background flex-1 rounded-md border px-3 py-2"
              />
              <button
                onClick={testOllama}
                disabled={!data.ollama_host || testStatus === 'testing'}
                className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-md px-3 py-2 text-sm disabled:opacity-50"
              >
                <StatusIcon />
                {tCommon('test')}
              </button>
            </div>
            {testError && <p className="text-xs text-red-500">{testError}</p>}
            {testStatus === 'success' && (
              <p className="text-xs text-green-500">
                {tCommon('connected')} - {t('modelsAvailable', { count: ollamaModels.length })}
              </p>
            )}
          </div>

          {ollamaModels.length > 0 && (
            <div className="space-y-1">
              <label className="text-sm font-medium">{t('model')}</label>
              <select
                value={data.ollama_model}
                onChange={(e) => updateData({ ollama_model: e.target.value })}
                className="border-input bg-background w-full rounded-md border px-3 py-2"
              >
                <option value="">{t('selectModel')}</option>
                {ollamaModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          )}

          <p className="text-muted-foreground text-xs">{t('ollamaSttNote')}</p>
        </div>
      )}

      {/* Groq Settings */}
      {data.llm_provider === 'groq' && (
        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-sm font-medium">{t('groqApiKey')}</label>
            <div className="flex gap-2">
              <input
                type="password"
                value={data.groq_api_key}
                onChange={(e) => updateData({ groq_api_key: e.target.value })}
                placeholder="gsk_..."
                className="border-input bg-background flex-1 rounded-md border px-3 py-2"
              />
              <button
                onClick={testGroq}
                disabled={!data.groq_api_key || testStatus === 'testing'}
                className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-md px-3 py-2 text-sm disabled:opacity-50"
              >
                <StatusIcon />
                {tCommon('test')}
              </button>
            </div>
            {testError && <p className="text-xs text-red-500">{testError}</p>}
            {testStatus === 'success' && (
              <p className="text-xs text-green-500">
                {tCommon('connected')} - {t('modelsAvailable', { count: groqModels.length })}
              </p>
            )}
            <p className="text-muted-foreground text-xs">
              {t('getApiKeyAt')}{' '}
              <a
                href="https://console.groq.com/keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline"
              >
                console.groq.com
              </a>
            </p>
          </div>

          {groqModels.length > 0 && (
            <div className="space-y-1">
              <label className="text-sm font-medium">{t('model')}</label>
              <select
                value={data.groq_model}
                onChange={(e) => updateData({ groq_model: e.target.value })}
                className="border-input bg-background w-full rounded-md border px-3 py-2"
              >
                <option value="">{t('selectModel')}</option>
                {groqModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          )}

          <p className="text-muted-foreground text-xs">{t('groqSttNote')}</p>
        </div>
      )}

      {/* OpenAI-compatible Settings */}
      {data.llm_provider === 'openai_compatible' && (
        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-sm font-medium">{t('baseUrl')}</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={data.openai_base_url}
                onChange={(e) => updateData({ openai_base_url: e.target.value })}
                placeholder="http://localhost:8000/v1"
                className="border-input bg-background flex-1 rounded-md border px-3 py-2"
              />
              <button
                onClick={testOpenAICompatible}
                disabled={!data.openai_base_url || testStatus === 'testing'}
                className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-md px-3 py-2 text-sm disabled:opacity-50"
              >
                <StatusIcon />
                {tCommon('test')}
              </button>
            </div>
            {testError && <p className="text-xs text-red-500">{testError}</p>}
            {testStatus === 'success' && (
              <p className="text-xs text-green-500">
                {tCommon('connected')} - {t('modelsAvailable', { count: openaiModels.length })}
              </p>
            )}
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t('apiKey')} ({t('optional')})
            </label>
            <input
              type="password"
              value={data.openai_api_key}
              onChange={(e) => updateData({ openai_api_key: e.target.value })}
              placeholder="sk-..."
              className="border-input bg-background w-full rounded-md border px-3 py-2"
            />
            <p className="text-muted-foreground text-xs">{t('openaiApiKeyNote')}</p>
          </div>

          {openaiModels.length > 0 && (
            <div className="space-y-1">
              <label className="text-sm font-medium">{t('model')}</label>
              <select
                value={data.openai_model}
                onChange={(e) => updateData({ openai_model: e.target.value })}
                className="border-input bg-background w-full rounded-md border px-3 py-2"
              >
                <option value="">{t('selectModel')}</option>
                {openaiModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          )}

          <p className="text-muted-foreground text-xs">{t('openaiCompatibleSttNote')}</p>
        </div>
      )}

      {/* OpenRouter Settings */}
      {data.llm_provider === 'openrouter' && (
        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-sm font-medium">{t('apiKey')}</label>
            <div className="flex gap-2">
              <input
                type="password"
                value={data.openrouter_api_key}
                onChange={(e) => updateData({ openrouter_api_key: e.target.value })}
                placeholder="sk-or-..."
                className="border-input bg-background flex-1 rounded-md border px-3 py-2"
              />
              <button
                onClick={testOpenRouter}
                disabled={!data.openrouter_api_key || testStatus === 'testing'}
                className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-md px-3 py-2 text-sm disabled:opacity-50"
              >
                <StatusIcon />
                {tCommon('test')}
              </button>
            </div>
            {testError && <p className="text-xs text-red-500">{testError}</p>}
            {testStatus === 'success' && (
              <p className="text-xs text-green-500">
                {tCommon('connected')} - {t('modelsAvailable', { count: openrouterModels.length })}
              </p>
            )}
            <p className="text-muted-foreground text-xs">
              {t('getApiKeyAt')}{' '}
              <a
                href="https://openrouter.ai/keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline"
              >
                openrouter.ai
              </a>
            </p>
          </div>

          {openrouterModels.length > 0 && (
            <div className="space-y-1">
              <label className="text-sm font-medium">{t('model')}</label>
              <select
                value={data.openrouter_model}
                onChange={(e) => updateData({ openrouter_model: e.target.value })}
                className="border-input bg-background w-full rounded-md border px-3 py-2"
              >
                <option value="">{t('selectModel')}</option>
                {openrouterModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {/* Groq API key for STT when LLM is not Groq */}
      {data.stt_provider === 'groq' && data.llm_provider !== 'groq' && (
        <div className="space-y-1 border-t pt-3">
          <label className="text-sm font-medium">{t('groqApiKey')} (STT)</label>
          <div className="flex gap-2">
            <input
              type="password"
              value={data.groq_api_key}
              onChange={(e) => updateData({ groq_api_key: e.target.value })}
              placeholder="gsk_..."
              className="border-input bg-background flex-1 rounded-md border px-3 py-2"
            />
            <button
              onClick={testGroq}
              disabled={!data.groq_api_key || testStatus === 'testing'}
              className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-md px-3 py-2 text-sm disabled:opacity-50"
            >
              <StatusIcon />
              {tCommon('test')}
            </button>
          </div>
          {testError && <p className="text-xs text-red-500">{testError}</p>}
          <p className="text-muted-foreground text-xs">{t('groqSttNote')}</p>
        </div>
      )}
    </div>
  );
}
