"use client";

import type { ClapEvent } from "@/lib/types";

export default function StatsBar({ events }: { events: ClapEvent[] }) {
  const totalClaps = events.length;

  const happyEvents = events.filter((e) => e.dominantEmotion === "HAPPY");
  const happyRate =
    totalClaps > 0 ? ((happyEvents.length / totalClaps) * 100).toFixed(0) : "—";

  const lastEvent = events.length > 0 ? events[0] : null;
  const lastTime = lastEvent
    ? new Date(lastEvent.timestamp).toLocaleTimeString("ja-JP")
    : "—";

  const stats = [
    { label: "総拍手数", value: String(totalClaps), unit: "回" },
    { label: "笑顔率", value: happyRate, unit: "%" },
    { label: "最終検知", value: lastTime, unit: "" },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {stats.map((s) => (
        <div
          key={s.label}
          className="bg-gray-800 rounded-xl p-4 border border-gray-700 text-center"
        >
          <div className="text-sm text-gray-400">{s.label}</div>
          <div className="text-2xl font-bold text-white mt-1">
            {s.value}
            {s.unit && <span className="text-sm text-gray-400 ml-1">{s.unit}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
