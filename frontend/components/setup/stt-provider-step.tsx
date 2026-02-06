'use client';

import { useTranslations } from 'next-intl';
import type { SetupData } from './setup-wizard';

interface SttProviderStepProps {
  data: SetupData;
  updateData: (updates: Partial<SetupData>) => void;
}

export function SttProviderStep({ data, updateData }: SttProviderStepProps) {
  const t = useTranslations('Settings.pipeline');

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-2">
        <button
          onClick={() => updateData({ stt_provider: 'speaches' })}
          className={`rounded-lg border p-4 text-left transition-colors ${
            data.stt_provider === 'speaches'
              ? 'border-primary bg-primary/5'
              : 'border-input hover:border-muted-foreground'
          }`}
        >
          <div className="font-medium">Speaches</div>
          <div className="text-muted-foreground text-xs">{t('sttSpeachesDesc')}</div>
        </button>
        <button
          onClick={() => updateData({ stt_provider: 'groq' })}
          className={`rounded-lg border p-4 text-left transition-colors ${
            data.stt_provider === 'groq'
              ? 'border-primary bg-primary/5'
              : 'border-input hover:border-muted-foreground'
          }`}
        >
          <div className="font-medium">Groq Whisper</div>
          <div className="text-muted-foreground text-xs">{t('sttGroqDesc')}</div>
        </button>
      </div>

      {data.stt_provider === 'groq' && (
        <p className="text-muted-foreground text-xs">{t('sttGroqKeyNote')}</p>
      )}
    </div>
  );
}
