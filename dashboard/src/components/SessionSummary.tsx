"use client";

import type { ClapEvent } from "@/lib/types";

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

const EMOTION_EMOJI: Record<string, string> = {
  HAPPY: "😊",
  SURPRISED: "😲",
  CALM: "😌",
  SAD: "😢",
  ANGRY: "😠",
  CONFUSED: "😕",
  DISGUSTED: "🤢",
  FEAR: "😨",
};

function formatDuration(events: ClapEvent[]): string {
  if (events.length < 2) return "—";
  const sorted = [...events].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );
  const ms =
    new Date(sorted[sorted.length - 1].timestamp).getTime() -
    new Date(sorted[0].timestamp).getTime();
  const sec = Math.floor(ms / 1000);
  const min = Math.floor(sec / 60);
  const remSec = sec % 60;
  return min > 0 ? `${min}分${remSec}秒` : `${remSec}秒`;
}

function getTopEmotion(events: ClapEvent[]): { type: string; avg: number } | null {
  const map = new Map<string, { total: number; count: number }>();
  for (const ev of events) {
    if (!ev.emotions) continue;
    for (const e of ev.emotions) {
      const prev = map.get(e.type) || { total: 0, count: 0 };
      map.set(e.type, { total: prev.total + e.confidence, count: prev.count + 1 });
    }
  }
  if (map.size === 0) return null;
  const top = [...map.entries()]
    .map(([type, { total, count }]) => ({ type, avg: total / count }))
    .sort((a, b) => b.avg - a.avg)[0];
  return top;
}

function getPeakMinute(events: ClapEvent[]): string {
  if (events.length === 0) return "—";
  const buckets = new Map<string, number>();
  for (const ev of events) {
    const d = new Date(ev.timestamp);
    const key = d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
    buckets.set(key, (buckets.get(key) || 0) + 1);
  }
  const peak = [...buckets.entries()].sort((a, b) => b[1] - a[1])[0];
  return `${peak[0]} (${peak[1]}回)`;
}

export default function SessionSummary({
  events,
  onReset,
}: {
  events: ClapEvent[];
  onReset: () => void;
}) {
  const totalClaps = events.length;
  const duration = formatDuration(events);
  const topEmotion = getTopEmotion(events);
  const peakMinute = getPeakMinute(events);
  const happyEvents = events.filter((e) => e.dominantEmotion === "HAPPY");
  const happyRate = totalClaps > 0 ? ((happyEvents.length / totalClaps) * 100).toFixed(0) : "0";

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="max-w-2xl w-full space-y-8">
        <div className="text-center space-y-2">
          <span className="text-6xl">🐧</span>
          <h1 className="text-3xl font-bold">セッション結果</h1>
          <p className="text-gray-400">発表お疲れさまでした！</p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 text-center">
            <div className="text-sm text-gray-400">総拍手数</div>
            <div className="text-4xl font-bold text-blue-400 mt-2">{totalClaps}</div>
            <div className="text-sm text-gray-500 mt-1">回</div>
          </div>

          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 text-center">
            <div className="text-sm text-gray-400">セッション時間</div>
            <div className="text-4xl font-bold text-green-400 mt-2">{duration}</div>
          </div>

          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 text-center">
            <div className="text-sm text-gray-400">笑顔率</div>
            <div className="text-4xl font-bold text-yellow-400 mt-2">{happyRate}%</div>
          </div>

          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 text-center">
            <div className="text-sm text-gray-400">ピークタイム</div>
            <div className="text-2xl font-bold text-purple-400 mt-2">{peakMinute}</div>
          </div>
        </div>

        {topEmotion && (
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 text-center">
            <div className="text-sm text-gray-400">会場の主要感情</div>
            <div className="text-5xl mt-3">{EMOTION_EMOJI[topEmotion.type] || "👏"}</div>
            <div className="text-xl font-bold mt-2">
              {EMOTION_LABELS[topEmotion.type] || topEmotion.type}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              平均スコア {topEmotion.avg.toFixed(1)}%
            </div>
          </div>
        )}

        <button
          onClick={onReset}
          className="w-full py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-lg transition-colors"
        >
          終了してリセット
        </button>
      </div>
    </div>
  );
}
