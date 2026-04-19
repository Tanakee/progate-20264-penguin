"use client";

import type { ClapEvent } from "@/lib/types";

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

export default function EventFeed({ events }: { events: ClapEvent[] }) {
  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400">
        拍手イベントを待機中...
      </div>
    );
  }

  return (
    <div className="space-y-3 max-h-[600px] overflow-y-auto">
      {events.map((ev) => {
        const time = new Date(ev.timestamp).toLocaleTimeString("ja-JP");
        const emoji = ev.dominantEmotion
          ? EMOTION_EMOJI[ev.dominantEmotion] || "👏"
          : "👏";

        return (
          <div
            key={ev.id}
            className="flex items-center gap-4 p-4 bg-gray-800 rounded-xl border border-gray-700 animate-fade-in"
          >
            <div className="text-3xl">{emoji}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="font-semibold text-white">拍手検知!</span>
                <span className="text-sm text-gray-400">{time}</span>
              </div>
              {ev.dominantEmotion && (
                <div className="text-sm text-gray-300 mt-1">
                  感情: {ev.dominantEmotion}
                  {ev.confidence && (
                    <span className="text-gray-500 ml-1">
                      ({ev.confidence.toFixed(1)}%)
                    </span>
                  )}
                </div>
              )}
              {ev.emotions && ev.emotions.length > 0 && (
                <div className="flex gap-1 mt-2 flex-wrap">
                  {ev.emotions
                    .filter((e) => e.confidence > 5)
                    .sort((a, b) => b.confidence - a.confidence)
                    .slice(0, 4)
                    .map((e) => (
                      <span
                        key={e.type}
                        className="text-xs px-2 py-0.5 bg-gray-700 rounded-full text-gray-300"
                      >
                        {EMOTION_EMOJI[e.type] || ""} {e.type}{" "}
                        {e.confidence.toFixed(0)}%
                      </span>
                    ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
