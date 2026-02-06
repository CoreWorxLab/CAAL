'use client';

import { useState } from 'react';
import { createPortal } from 'react-dom';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/livekit/button';
import { IntegrationsStep } from './integrations-step';
import { ProviderStep } from './provider-step';
import { SttProviderStep } from './stt-provider-step';
import { TtsStep } from './tts-step';

export interface SetupData {
  // STT Provider
  stt_provider: 'speaches' | 'groq';
  // LLM Provider
  llm_provider: 'ollama' | 'groq' | 'openai_compatible' | 'openrouter';
  ollama_host: string;
  ollama_model: string;
  groq_api_key: string;
  groq_model: string;
  // OpenAI-compatible
  openai_base_url: string;
  openai_api_key: string;
  openai_model: string;
  // OpenRouter
  openrouter_api_key: string;
  openrouter_model: string;
  // TTS Provider
  tts_provider: 'kokoro' | 'piper';
  tts_voice_kokoro: string;
  tts_voice_piper: string;
  // Integrations
  hass_enabled: boolean;
  hass_host: string;
  hass_token: string;
  n8n_enabled: boolean;
  n8n_url: string;
  n8n_token: string;
}

interface SetupWizardProps {
  onComplete: () => void;
}

const INITIAL_DATA: SetupData = {
  stt_provider: 'speaches',
  llm_provider: 'ollama',
  ollama_host: 'http://localhost:11434',
  ollama_model: '',
  groq_api_key: '',
  groq_model: '',
  openai_base_url: '',
  openai_api_key: '',
  openai_model: '',
  openrouter_api_key: '',
  openrouter_model: '',
  tts_provider: 'kokoro',
  tts_voice_kokoro: 'am_puck',
  tts_voice_piper: 'speaches-ai/piper-en_US-ryan-high',
  hass_enabled: false,
  hass_host: '',
  hass_token: '',
  n8n_enabled: false,
  n8n_url: '',
  n8n_token: '',
};

const TOTAL_STEPS = 4;

// Per-language Piper voice mapping (mirrors PIPER_VOICE_MAP in settings.py)
const PIPER_MODELS: Record<string, string> = {
  fr: 'speaches-ai/piper-fr_FR-siwis-medium',
  it: 'speaches-ai/piper-it_IT-paola-medium',
};

export function SetupWizard({ onComplete }: SetupWizardProps) {
  const t = useTranslations('Setup');
  const tCommon = useTranslations('Common');
  const tStatus = useTranslations('Settings.status');

  const [step, setStep] = useState(1);
  const [data, setData] = useState<SetupData>(() => {
    // Read language from cookie to pre-select appropriate TTS provider/voice
    const locale =
      typeof document !== 'undefined'
        ? document.cookie.match(/CAAL_LOCALE=(\w+)/)?.[1] || 'en'
        : 'en';

    if (locale === 'en') return INITIAL_DATA;

    return {
      ...INITIAL_DATA,
      tts_provider: 'piper' as const,
      tts_voice_piper: PIPER_MODELS[locale] || INITIAL_DATA.tts_voice_piper,
    };
  });
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const updateData = (updates: Partial<SetupData>) => {
    setData((prev) => ({ ...prev, ...updates }));
  };

  const handleNext = () => {
    setStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setStep((prev) => prev - 1);
  };

  // Build full n8n MCP URL from host
  const getN8nMcpUrl = (host: string) => {
    if (!host) return '';
    // Remove trailing slash if present
    const baseUrl = host.replace(/\/$/, '');
    // If user already included the path, use as-is
    if (baseUrl.includes('/mcp-server')) return baseUrl;
    // Otherwise append the standard MCP path
    return `${baseUrl}/mcp-server/http`;
  };

  const handleComplete = async () => {
    setSaving(true);
    setError(null);

    // Transform n8n_url to include full MCP path if enabled
    const finalData = {
      ...data,
      n8n_url: data.n8n_enabled ? getN8nMcpUrl(data.n8n_url) : data.n8n_url,
    };

    try {
      const response = await fetch('/api/setup/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalData),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message || 'Setup failed');
      }

      // Trigger prewarm if using Ollama
      if (finalData.llm_provider === 'ollama') {
        fetch('/api/prewarm', { method: 'POST' }).catch(() => {
          // Ignore prewarm errors - it's non-critical
        });
      }

      // Download Piper model if using Piper
      if (finalData.tts_provider === 'piper' && finalData.tts_voice_piper) {
        setSaveStatus(tStatus('downloadingVoice'));
        await fetch('/api/download-piper-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: finalData.tts_voice_piper }),
        });
      }

      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Setup failed');
    } finally {
      setSaving(false);
    }
  };

  // Check if current step is valid to proceed
  const canProceed = () => {
    // Step 1: STT provider - always valid (default is speaches)
    if (step === 1) {
      return true;
    }
    // Step 2: LLM provider
    if (step === 2) {
      switch (data.llm_provider) {
        case 'ollama':
          return data.ollama_host && data.ollama_model;
        case 'groq':
          return data.groq_api_key && data.groq_model;
        case 'openai_compatible':
          return data.openai_base_url && data.openai_model;
        case 'openrouter':
          return data.openrouter_api_key && data.openrouter_model;
        default:
          return false;
      }
    }
    // Step 3: TTS - both providers valid if voice selected
    if (step === 3) {
      if (data.tts_provider === 'piper') {
        return !!data.tts_voice_piper;
      }
      return !!data.tts_voice_kokoro;
    }
    // Step 4: Integrations - always valid
    return true;
  };

  const getStepTitle = () => {
    switch (step) {
      case 1:
        return t('step1Title');
      case 2:
        return t('step2Title');
      case 3:
        return t('step3Title');
      case 4:
        return t('step4Title');
      default:
        return '';
    }
  };

  const isLastStep = step === TOTAL_STEPS;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-background border-input dark:border-muted flex max-h-[85vh] w-full max-w-lg flex-col rounded-lg border shadow-xl">
        {/* Header */}
        <div className="border-input dark:border-muted shrink-0 border-b p-4">
          <h2 className="text-lg font-semibold">{t('welcome')}</h2>
          <p className="text-muted-foreground text-sm">
            {t('stepOf', { current: step, total: TOTAL_STEPS })} &mdash; {getStepTitle()}
          </p>
          {/* Progress bar */}
          <div className="bg-muted mt-3 h-1 w-full rounded-full">
            <div
              className="bg-primary h-1 rounded-full transition-all duration-300"
              style={{ width: `${(step / TOTAL_STEPS) * 100}%` }}
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {error && <div className="mb-4 rounded-md bg-red-500/10 p-3 text-red-500">{error}</div>}

          {step === 1 && <SttProviderStep data={data} updateData={updateData} />}
          {step === 2 && <ProviderStep data={data} updateData={updateData} />}
          {step === 3 && <TtsStep data={data} updateData={updateData} />}
          {step === 4 && <IntegrationsStep data={data} updateData={updateData} />}
        </div>

        {/* Footer */}
        <div className="border-input dark:border-muted flex shrink-0 justify-between border-t p-4">
          <div>
            {step > 1 && (
              <Button variant="secondary" onClick={handleBack} disabled={saving}>
                {tCommon('back')}
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            {isLastStep ? (
              <Button variant="primary" onClick={handleComplete} disabled={saving || !canProceed()}>
                {saving ? saveStatus || tCommon('saving') : t('finishSetup')}
              </Button>
            ) : (
              <Button variant="primary" onClick={handleNext} disabled={!canProceed()}>
                {tCommon('continue')}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
