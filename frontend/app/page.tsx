// Author: Bradley R. Kinnard
'use client';

import { BeliefList } from '@/components/BeliefList';
import { ClusterView } from '@/components/ClusterView';
import { StatsPanel } from '@/components/StatsPanel';
import { fetchBeliefs, fetchStats } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';

export default function HomePage() {
  const statsQuery = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
  });

  const beliefsQuery = useQuery({
    queryKey: ['beliefs'],
    queryFn: () => fetchBeliefs(),
  });

  return (
    <div className="min-h-screen p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">ABES Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Adaptive Belief Ecology System
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stats Panel */}
        <div className="lg:col-span-1">
          <StatsPanel
            stats={statsQuery.data}
            isLoading={statsQuery.isLoading}
          />
        </div>

        {/* Belief List */}
        <div className="lg:col-span-2">
          <BeliefList
            beliefs={beliefsQuery.data?.beliefs ?? []}
            isLoading={beliefsQuery.isLoading}
          />
        </div>
      </div>

      {/* Cluster Visualization */}
      <div className="mt-8">
        <ClusterView />
      </div>
    </div>
  );
}
