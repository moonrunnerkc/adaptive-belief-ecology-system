// Author: Bradley R. Kinnard
'use client';

import { RefreshCw, X } from 'lucide-react';
import { useEffect, useState } from 'react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SystemSettings {
  storage_backend: string;
  llm_provider: string;
  llm_fallback_enabled: boolean;
  decay_profile: string;
  embedding_model: string;
  total_beliefs: number;
  active_beliefs: number;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchSettings();
    }
  }, [isOpen]);

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch from backend stats endpoint
      const response = await fetch('http://localhost:8000/bel/stats');
      if (response.ok) {
        const data = await response.json();
        setSettings({
          storage_backend: data.storage_backend || 'memory',
          llm_provider: data.llm_provider || 'ollama',
          llm_fallback_enabled: data.llm_fallback_enabled ?? true,
          decay_profile: data.decay_profile || 'moderate',
          embedding_model: data.embedding_model || 'all-MiniLM-L6-v2',
          total_beliefs: data.total_beliefs || 0,
          active_beliefs: data.active_beliefs || 0,
        });
      } else {
        setError('Failed to load settings');
      }
    } catch (err) {
      setError('Cannot connect to backend');
    }
    setLoading(false);
  };

  const handleClearBeliefs = async () => {
    if (!confirm('Are you sure you want to clear all beliefs? This cannot be undone.')) {
      return;
    }
    try {
      const response = await fetch('http://localhost:8000/beliefs/clear', {
        method: 'POST',
      });
      if (response.ok) {
        fetchSettings();
      }
    } catch (err) {
      setError('Failed to clear beliefs');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border border-[#1f1f1f] rounded-xl w-full max-w-md mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1f1f1f]">
          <h2 className="text-lg font-semibold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-[#1f1f1f] text-neutral-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 text-neutral-500 animate-spin" />
            </div>
          ) : error ? (
            <div className="text-red-400 text-sm text-center py-4">{error}</div>
          ) : settings ? (
            <>
              {/* System Info */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wide">
                  System Configuration
                </h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-[#141414] rounded-lg p-3">
                    <div className="text-neutral-500 text-xs mb-1">Storage</div>
                    <div className="text-white capitalize">{settings.storage_backend}</div>
                  </div>
                  <div className="bg-[#141414] rounded-lg p-3">
                    <div className="text-neutral-500 text-xs mb-1">LLM Provider</div>
                    <div className="text-white capitalize">{settings.llm_provider}</div>
                  </div>
                  <div className="bg-[#141414] rounded-lg p-3">
                    <div className="text-neutral-500 text-xs mb-1">Decay Profile</div>
                    <div className="text-white capitalize">{settings.decay_profile}</div>
                  </div>
                  <div className="bg-[#141414] rounded-lg p-3">
                    <div className="text-neutral-500 text-xs mb-1">Fallback Mode</div>
                    <div className="text-white">{settings.llm_fallback_enabled ? 'Enabled' : 'Disabled'}</div>
                  </div>
                </div>
              </div>

              {/* Belief Stats */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wide">
                  Belief Store
                </h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-[#141414] rounded-lg p-3">
                    <div className="text-neutral-500 text-xs mb-1">Total Beliefs</div>
                    <div className="text-white text-lg font-semibold">{settings.total_beliefs}</div>
                  </div>
                  <div className="bg-[#141414] rounded-lg p-3">
                    <div className="text-neutral-500 text-xs mb-1">Active Beliefs</div>
                    <div className="text-white text-lg font-semibold">{settings.active_beliefs}</div>
                  </div>
                </div>
              </div>

              {/* Embedding Model */}
              <div className="bg-[#141414] rounded-lg p-3">
                <div className="text-neutral-500 text-xs mb-1">Embedding Model</div>
                <div className="text-white text-sm font-mono">{settings.embedding_model}</div>
              </div>

              {/* Actions */}
              <div className="pt-2 space-y-2">
                <button
                  onClick={handleClearBeliefs}
                  className="w-full px-4 py-2 bg-red-900/30 hover:bg-red-900/50 text-red-400 rounded-lg text-sm transition-colors"
                >
                  Clear All Beliefs
                </button>
              </div>

              {/* Note */}
              <p className="text-xs text-neutral-600 text-center">
                Configuration is set via environment variables. Restart server to apply changes.
              </p>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
