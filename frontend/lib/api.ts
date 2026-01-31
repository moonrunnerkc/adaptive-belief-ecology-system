// Author: Bradley R. Kinnard
// API client for ABES backend

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

export interface BeliefListResponse {
  beliefs: Belief[];
  total: number;
  page: number;
  page_size: number;
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

export interface Cluster {
  id: string;
  size: number;
  created_at: string;
  updated_at: string;
}

export interface ClusterListResponse {
  clusters: Cluster[];
  total: number;
}

export interface AgentStatus {
  name: string;
  phase: string;
  enabled: boolean;
  last_run: string | null;
  run_count: number;
}

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export async function fetchStats(): Promise<Stats> {
  return fetchJson<Stats>('/bel/stats');
}

export async function fetchHealth() {
  return fetchJson('/bel/health');
}

export async function fetchBeliefs(
  page = 1,
  pageSize = 50
): Promise<BeliefListResponse> {
  return fetchJson<BeliefListResponse>(
    `/beliefs?page=${page}&page_size=${pageSize}`
  );
}

export async function fetchBelief(id: string): Promise<Belief> {
  return fetchJson<Belief>(`/beliefs/${id}`);
}

export async function createBelief(data: {
  content: string;
  confidence?: number;
  tags?: string[];
}): Promise<Belief> {
  return fetchJson<Belief>('/beliefs', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateBelief(
  id: string,
  data: { confidence?: number; tags?: string[]; status?: string }
): Promise<Belief> {
  return fetchJson<Belief>(`/beliefs/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export async function reinforceBelief(
  id: string,
  boost = 0.1
): Promise<Belief> {
  return fetchJson<Belief>(`/beliefs/${id}/reinforce?boost=${boost}`, {
    method: 'POST',
  });
}

export async function fetchClusters(): Promise<ClusterListResponse> {
  return fetchJson<ClusterListResponse>('/clusters');
}

export async function fetchAgents(): Promise<{ agents: AgentStatus[] }> {
  return fetchJson('/agents');
}

export async function toggleAgent(
  name: string,
  enabled: boolean
): Promise<AgentStatus> {
  return fetchJson<AgentStatus>(`/agents/${name}`, {
    method: 'PATCH',
    body: JSON.stringify({ enabled }),
  });
}

export async function runIteration(context = ''): Promise<any> {
  return fetchJson('/bel/iterate', {
    method: 'POST',
    body: JSON.stringify({ context }),
  });
}

// === Chat API ===

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

export interface BeliefEvent {
  event_type: 'created' | 'reinforced' | 'mutated' | 'deprecated' | 'tension_changed';
  belief_id: string;
  content: string;
  confidence: number;
  tension: number;
  details: Record<string, unknown>;
  timestamp: string;
}

export interface ChatSession {
  id: string;
  turn_count: number;
  created_at: string;
}

export async function sendChatMessage(
  message: string,
  sessionId?: string,
  token?: string | null
): Promise<ChatTurn> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch('http://localhost:8000/chat/message', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      message,
      session_id: sessionId,
      stream: false,
    }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export async function createChatSession(): Promise<{ id: string; created_at: string }> {
  return fetchJson('/chat/sessions', { method: 'POST' });
}

export async function listChatSessions(): Promise<{ sessions: ChatSession[]; total: number }> {
  return fetchJson('/chat/sessions');
}

export async function getChatSession(
  sessionId: string
): Promise<{ id: string; created_at: string; turns: ChatTurn[] }> {
  return fetchJson(`/chat/sessions/${sessionId}`);
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  await fetchJson(`/chat/sessions/${sessionId}`, { method: 'DELETE' });
}

// WebSocket URL for chat
export function getChatWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // In dev, API is on port 8000
  const host = process.env.NODE_ENV === 'development'
    ? 'localhost:8000'
    : window.location.host;
  return `${protocol}//${host}/chat/ws`;
}
