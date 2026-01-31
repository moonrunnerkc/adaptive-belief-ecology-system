// Author: Bradley R. Kinnard
'use client';

import { ChatSession } from '@/lib/api';
import clsx from 'clsx';
import { ChevronLeft, MessageSquare, Plus, Settings } from 'lucide-react';
import { useState } from 'react';
import { SettingsModal } from './SettingsModal';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
}

export function Sidebar({
  isOpen,
  onToggle,
  sessions,
  currentSessionId,
  onNewSession,
  onSelectSession,
}: SidebarProps) {
  const [showSettings, setShowSettings] = useState(false);

  if (!isOpen) return null;

  return (
    <>
      <div className="w-64 bg-[#0d0d0d] border-r border-[#1f1f1f] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-[#1f1f1f]">
        <span className="text-sm font-medium text-neutral-300">Conversations</span>
        <button
          onClick={onToggle}
          className="p-1.5 hover:bg-[#1a1a1a] rounded transition-colors text-neutral-500 hover:text-neutral-300"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-2">
        <button
          onClick={onNewSession}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg bg-[#1a1a1a] hover:bg-[#222] border border-[#2a2a2a] text-neutral-300 text-sm transition-colors"
        >
          <Plus className="w-4 h-4" />
          New conversation
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto px-2 py-1">
        {sessions.length === 0 ? (
          <p className="text-xs text-neutral-600 text-center py-4">
            No conversations yet
          </p>
        ) : (
          <div className="space-y-1">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                className={clsx(
                  "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors text-left",
                  session.id === currentSessionId
                    ? "bg-[#1a1a1a] text-neutral-200 border border-[#2a2a2a]"
                    : "text-neutral-500 hover:bg-[#141414] hover:text-neutral-300"
                )}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="truncate flex-1">
                  {session.turn_count > 0
                    ? `${session.turn_count} messages`
                    : 'New chat'
                  }
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-[#1f1f1f]">
        <button
          onClick={() => setShowSettings(true)}
          className="flex items-center gap-2 px-3 py-2 w-full rounded-lg text-neutral-500 hover:bg-[#141414] hover:text-neutral-400 text-sm transition-colors"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
      </div>
    </div>

    <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </>
  );
}
