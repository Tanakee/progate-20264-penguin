"use client";

import type { ClapEvent } from "@/lib/types";

const EMOTION_COLORS: Record<string, string> = {
  HAPPY: "bg-yellow-500",
  SURPRISED: "bg-purple-500",
  CALM: "bg-blue-400",
  SAD: "bg-indigo-500",
  ANGRY: "bg-red-500",
  CONFUSED: "bg-orange-400",
  DISGUSTED: "bg-green-600",
  FEAR: "bg-gray-500",
};

const EMOTION_LABELS: Record<string, string> = {
  HAPPY: "笑顔",
  SURPRISED: "驚き",
  CALM: "穏やか",
  SAD: "悲しみ",
  ANGRY: "怒り",
  CONFUSED: "困惑",
  DISGUSTED: "嫌悪",
  FEAR: "恐怖",
};

interface EmotionAgg {
  type: string;
  avgConfidence: number;
  count: number;
}

function aggregateEmotions(events: ClapEvent[]): EmotionAgg[] {
  const map = new Map<string, { total: number; count: number }>();

  for (const ev of events) {
    if (!ev.emotions) continue;
    for (const e of ev.emotions) {
      const prev = map.get(e.type) || { total: 0, count: 0 };
      map.set(e.type, { total: prev.total + e.confidence, count: prev.count + 1 });
    }
  }

  return Array.from(map.entries())
    .map(([type, { total, count }]) => ({
      type,
      avgConfidence: total / count,
      count,
    }))
    .sort((a, b) => b.avgConfidence - a.avgConfidence);
}

export default function EmotionSummary({ events }: { events: ClapEvent[] }) {
  const agg = aggregateEmotions(events);

  if (agg.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400">
        感情データなし
      </div>
    );
  }

  const maxConf = Math.max(...agg.map((a) => a.avgConfidence));

  return (
    <div className="space-y-3">
      {agg
        .filter((a) => a.avgConfidence > 1)
        .map((a) => (
          <div key={a.type} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">
                {EMOTION_LABELS[a.type] || a.type}
              </span>
              <span className="text-gray-400">
                {a.avgConfidence.toFixed(1)}%
              </span>
            </div>
            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${EMOTION_COLORS[a.type] || "bg-gray-400"}`}
                style={{ width: `${(a.avgConfidence / maxConf) * 100}%` }}
              />
            </div>
          </div>
        ))}
    </div>
  );
}
