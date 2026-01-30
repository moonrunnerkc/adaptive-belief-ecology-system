// Author: Bradley R. Kinnard
// TypeScript types for ABES Chat

export interface BeliefEvent {
  event_type: 'created' | 'reinforced' | 'mutated' | 'deprecated' | 'tension_changed';
  belief_id: string;
  content: string;
  confidence: number;
  tension: number;
  details: Record<string, unknown>;
  timestamp: string;
}

export interface ChatTurn {
  id: string;
  user_message: string;
  assistant_message: string;
  beliefs_created: string[];
  beliefs_reinforced: string[];
  beliefs_mutated: string[];
  beliefs_deprecated: string[];
  beliefs_used: string[];
  events: BeliefEvent[];
  duration_ms: number;
  timestamp: string;
}

export interface ChatSession {
  id: string;
  turn_count: number;
  created_at: string;
}

export interface Belief {
  id: string;
  content: string;
  confidence: number;
  status: 'active' | 'decaying' | 'deprecated' | 'mutated';
  tension: number;
  cluster_id: string | null;
  parent_id: string | null;
  use_count: number;
  tags: string[];
  source: string;
  created_at: string;
  updated_at: string;
}

export interface WebSocketMessage {
  type: 'belief_event' | 'chat_chunk' | 'chat_complete' | 'pong' | 'error';
  data: BeliefEvent | { content: string; done: boolean } | ChatTurn | { message: string };
}

export interface Stats {
  total_beliefs: number;
  active_beliefs: number;
  deprecated_beliefs: number;
  cluster_count: number;
  snapshot_count: number;
  avg_confidence: number;
  avg_tension: number;
}
