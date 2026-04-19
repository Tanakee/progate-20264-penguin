"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { ClapEvent } from "@/lib/types";

interface BucketPoint {
  time: string;
  count: number;
  happiness: number;
}

function bucketEvents(events: ClapEvent[], intervalSec: number): BucketPoint[] {
  if (events.length === 0) return [];

  const sorted = [...events].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  const start = new Date(sorted[0].timestamp).getTime();
  const end = Date.now();
  const buckets: BucketPoint[] = [];

  for (let t = start; t <= end; t += intervalSec * 1000) {
    const bucketEnd = t + intervalSec * 1000;
    const inBucket = sorted.filter((ev) => {
      const ts = new Date(ev.timestamp).getTime();
      return ts >= t && ts < bucketEnd;
    });

    const happinessScores = inBucket
      .map((ev) => ev.emotions?.find((e) => e.type === "HAPPY")?.confidence ?? 0)
      .filter((v) => v > 0);

    buckets.push({
      time: new Date(t).toLocaleTimeString("ja-JP", {
        hour: "2-digit",
        minute: "2-digit",
      }),
      count: inBucket.length,
      happiness:
        happinessScores.length > 0
          ? happinessScores.reduce((a, b) => a + b, 0) / happinessScores.length
          : 0,
    });
  }

  return buckets;
}

export default function ExcitementChart({ events }: { events: ClapEvent[] }) {
  const data = bucketEvents(events, 30);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400">
        データ蓄積中...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="colorHappy" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
        <YAxis stroke="#9ca3af" fontSize={12} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1f2937",
            border: "1px solid #374151",
            borderRadius: "8px",
            color: "#fff",
          }}
        />
        <Area
          type="monotone"
          dataKey="count"
          name="拍手数"
          stroke="#3b82f6"
          fill="url(#colorCount)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="happiness"
          name="笑顔スコア"
          stroke="#f59e0b"
          fill="url(#colorHappy)"
          strokeWidth={2}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
